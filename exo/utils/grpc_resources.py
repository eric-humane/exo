"""
GRPC Resource Management for exo.

This module provides specialized resources for managing gRPC connections,
ensuring proper lifecycle management, error handling, and cleanup.

Key classes:
- GRPCChannelResource: Resource for managing gRPC channel lifecycle
- GRPCServiceResource: Resource for managing gRPC service client
- GRPCConnectionPool: Pool for managing and reusing gRPC connections

Usage:
```python
# Basic usage
channel = GRPCChannelResource(address="localhost:50051")
await channel.initialize()
try:
    # Use the channel
    stub = ServiceStub(channel.channel)
    response = await stub.SomeMethod(request)
finally:
    await channel.cleanup()

# Context manager usage
async with GRPCChannelContext(address="localhost:50051") as channel:
    stub = ServiceStub(channel.channel)
    response = await stub.SomeMethod(request)
    
# Service resource usage
service = GRPCServiceResource(
    service_stub_class=ServiceStub,
    address="localhost:50051"
)
await service.initialize()
try:
    response = await service.stub.SomeMethod(request)
finally:
    await service.cleanup()

# Pool usage
pool = GRPCConnectionPool(
    service_stub_class=ServiceStub,
    address="localhost:50051",
    max_size=10
)
await pool.start()
async with pool.acquire() as service:
    response = await service.stub.SomeMethod(request)
await pool.stop()
```
"""

import asyncio
import grpc
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union
import numpy as np

from .async_resources import (
    AsyncResource,
    AsyncResourceContext,
    AsyncResourcePool,
    NetworkResource,
    ResourceState,
    ResourceInitializationError,
)

T = TypeVar('T')
StubType = TypeVar('StubType')


class GRPCError(Exception):
    """Base exception for gRPC-related errors."""
    pass


class GRPCConnectionError(GRPCError):
    """Raised when a gRPC connection cannot be established."""
    pass


class GRPCChannelResource(NetworkResource):
    """
    Resource for managing a gRPC channel.
    
    This class provides:
    1. Lifecycle management for gRPC channels
    2. Automatic reconnection
    3. Health checking
    4. Error handling specific to gRPC
    """
    
    def __init__(
        self,
        address: str,
        resource_id: Optional[str] = None,
        options: Optional[List[tuple]] = None,
        compression: Optional[grpc.Compression] = None,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
        initialize_timeout: float = 15.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        health_check_interval: Optional[float] = 60.0,
        reconnect_on_error: bool = True
    ):
        """
        Initialize a new GRPCChannelResource.
        
        Args:
            address: The address of the gRPC server (host:port).
            resource_id: Unique identifier for this resource.
            options: List of gRPC channel options.
            compression: gRPC compression option.
            ssl_credentials: SSL credentials for secure connections.
            initialize_timeout: Timeout for channel initialization.
            max_retries: Maximum number of retries for operations.
            retry_delay: Initial delay between retries.
            max_retry_delay: Maximum delay between retries.
            health_check_interval: Interval for automatic health checks.
            reconnect_on_error: Whether to automatically reconnect on error.
        """
        super().__init__(
            resource_id=resource_id or f"grpc-channel-{address}",
            address=address,
            initialize_timeout=initialize_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_retry_delay=max_retry_delay,
            health_check_interval=health_check_interval,
            reconnect_on_error=reconnect_on_error
        )
        self._options = options or []
        self._compression = compression or grpc.Compression.NoCompression
        self._ssl_credentials = ssl_credentials
        self._channel: Optional[grpc.aio.Channel] = None
        
        # Default gRPC channel options if none provided
        if not self._options:
            self._options = [
                ("grpc.max_metadata_size", 16 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_ping_interval_without_data_ms", 5000),
            ]
            
    @property
    def channel(self) -> Optional[grpc.aio.Channel]:
        """Get the gRPC channel."""
        return self._channel
        
    async def _do_initialize(self) -> None:
        """
        Initialize the gRPC channel.
        
        Raises:
            GRPCConnectionError: If the channel cannot be established.
            asyncio.TimeoutError: If initialization times out.
        """
        self._connection_attempts += 1
        self._last_connection_time = asyncio.get_event_loop().time()
        
        try:
            # Create the channel
            if self._ssl_credentials:
                self._channel = grpc.aio.secure_channel(
                    self._address,
                    self._ssl_credentials,
                    options=self._options,
                    compression=self._compression
                )
            else:
                self._channel = grpc.aio.insecure_channel(
                    self._address,
                    options=self._options,
                    compression=self._compression
                )
                
            # Wait for the channel to be ready
            await self._channel.channel_ready()
            
            # Verify channel is in READY state with a small wait to ensure stability
            channel_state = self._channel.get_state()
            if channel_state != grpc.ChannelConnectivity.READY:
                await asyncio.sleep(0.1)  # Short wait to let channel stabilize
                channel_state = self._channel.get_state()
                
            if channel_state != grpc.ChannelConnectivity.READY:
                raise GRPCConnectionError(
                    f"Channel connected but not ready for {self._id}@{self._address}, "
                    f"state: {channel_state}"
                )
                
        except (grpc.RpcError, GRPCConnectionError) as e:
            if self._channel:
                await self._channel.close()
                self._channel = None
            raise GRPCConnectionError(f"Failed to connect to {self._address}: {str(e)}") from e
        except Exception as e:
            if self._channel:
                await self._channel.close()
                self._channel = None
            raise
            
    async def _do_cleanup(self) -> None:
        """Clean up the gRPC channel."""
        if self._channel:
            try:
                await self._channel.close()
            except Exception:
                # Ignore errors during cleanup
                pass
            self._channel = None
            
    async def _check_health(self) -> bool:
        """
        Check if the channel is healthy.
        
        Returns:
            True if the channel is healthy, False otherwise.
        """
        if not self._channel:
            return False
            
        try:
            channel_state = self._channel.get_state()
            if channel_state not in (
                grpc.ChannelConnectivity.READY,
                grpc.ChannelConnectivity.IDLE
            ):
                # Try to reconnect if in a bad state
                if self._reconnect_on_error:
                    await self.reconnect()
                    channel_state = self._channel.get_state() if self._channel else None
                    return channel_state == grpc.ChannelConnectivity.READY
                return False
                
            return channel_state == grpc.ChannelConnectivity.READY
            
        except Exception:
            return False
            
    async def call_unary(
        self,
        method: Callable,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[List[tuple]] = None
    ) -> Any:
        """
        Make a unary gRPC call with automatic retries and error handling.
        
        Args:
            method: The gRPC method to call.
            request: The request object.
            timeout: Timeout for the call.
            metadata: Optional metadata for the call.
            
        Returns:
            The response from the gRPC call.
            
        Raises:
            GRPCError: If the call fails after retries.
        """
        async def operation():
            if not self._channel:
                raise GRPCConnectionError("Channel not initialized")
                
            try:
                if metadata:
                    return await method(request, metadata=metadata, timeout=timeout)
                else:
                    return await method(request, timeout=timeout)
            except grpc.RpcError as e:
                status_code = e.code() if hasattr(e, 'code') else None
                status_details = e.details() if hasattr(e, 'details') else str(e)
                
                # Map gRPC errors to our exception hierarchy
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    raise GRPCConnectionError(f"Service unavailable: {status_details}")
                elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise asyncio.TimeoutError(f"gRPC call timed out: {status_details}")
                else:
                    raise GRPCError(f"gRPC error: {status_code}, {status_details}")
                    
        return await self.with_connection(
            operation,
            operation_name=method.__qualname__,
            timeout=timeout
        )


class GRPCServiceResource(AsyncResource, Generic[StubType]):
    """
    Resource for managing a gRPC service client (stub).
    
    This class provides:
    1. High-level interface for gRPC service clients
    2. Automatic channel management
    3. Service-specific operations
    
    This resource manages both the channel and the service stub.
    """
    
    def __init__(
        self,
        service_stub_class: Type[StubType],
        address: str,
        resource_id: Optional[str] = None,
        channel_options: Optional[List[tuple]] = None,
        compression: Optional[grpc.Compression] = None,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
        channel_args: Optional[Dict[str, Any]] = None,
        initialize_timeout: float = 15.0
    ):
        """
        Initialize a new GRPCServiceResource.
        
        Args:
            service_stub_class: The gRPC service stub class.
            address: The address of the gRPC server.
            resource_id: Unique identifier for this resource.
            channel_options: gRPC channel options.
            compression: gRPC compression option.
            ssl_credentials: SSL credentials for secure connections.
            channel_args: Additional arguments for the channel resource.
            initialize_timeout: Timeout for initialization.
        """
        super().__init__(
            resource_id=resource_id or f"grpc-service-{address}",
            initialize_timeout=initialize_timeout
        )
        self._service_stub_class = service_stub_class
        self._address = address
        self._channel_options = channel_options
        self._compression = compression
        self._ssl_credentials = ssl_credentials
        self._channel_args = channel_args or {}
        
        self._channel_resource: Optional[GRPCChannelResource] = None
        self._stub: Optional[StubType] = None
        
    @property
    def stub(self) -> Optional[StubType]:
        """Get the gRPC service stub."""
        return self._stub
        
    @property
    def address(self) -> str:
        """Get the server address."""
        return self._address
        
    @property
    def channel(self) -> Optional[grpc.aio.Channel]:
        """Get the underlying gRPC channel."""
        return self._channel_resource.channel if self._channel_resource else None
        
    async def _do_initialize(self) -> None:
        """
        Initialize the service by creating a channel and stub.
        
        Raises:
            GRPCConnectionError: If the channel cannot be established.
        """
        try:
            # Create and initialize the channel resource
            self._channel_resource = GRPCChannelResource(
                address=self._address,
                resource_id=f"{self._id}-channel",
                options=self._channel_options,
                compression=self._compression,
                ssl_credentials=self._ssl_credentials,
                **self._channel_args
            )
            
            await self._channel_resource.initialize()
            
            # Create the service stub
            if self._channel_resource.channel:
                self._stub = self._service_stub_class(self._channel_resource.channel)
            else:
                raise GRPCConnectionError(f"Failed to create channel for {self._address}")
                
        except Exception as e:
            if self._channel_resource:
                await self._channel_resource.cleanup()
                self._channel_resource = None
            self._stub = None
            raise ResourceInitializationError(f"Failed to initialize gRPC service: {str(e)}") from e
            
    async def _do_cleanup(self) -> None:
        """Clean up the service, closing the channel."""
        self._stub = None
        if self._channel_resource:
            await self._channel_resource.cleanup()
            self._channel_resource = None
            
    async def _check_health(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise.
        """
        return (
            self._channel_resource is not None and
            self._stub is not None and
            await self._channel_resource.check_health()
        )
        
    async def reconnect(self) -> bool:
        """
        Force a reconnection of this service.
        
        Returns:
            True if reconnection was successful, False otherwise.
        """
        self._stub = None
        if self._channel_resource:
            result = await self._channel_resource.reconnect()
            if result and self._channel_resource.channel:
                self._stub = self._service_stub_class(self._channel_resource.channel)
                return True
        return False
        
    async def call_unary(
        self,
        method_name: str,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[List[tuple]] = None,
        retry_on_error: bool = True
    ) -> Any:
        """
        Make a unary gRPC call to the service.
        
        Args:
            method_name: Name of the method on the stub to call.
            request: The request object.
            timeout: Timeout for the call.
            metadata: Optional metadata for the call.
            retry_on_error: Whether to retry on error.
            
        Returns:
            The response from the gRPC call.
            
        Raises:
            GRPCError: If the call fails after retries.
            AttributeError: If the method doesn't exist on the stub.
        """
        if not self._stub:
            if not await self.initialize():
                raise GRPCConnectionError(f"Could not initialize connection to {self._address}")
                
        if not hasattr(self._stub, method_name):
            raise AttributeError(f"Method {method_name} not found on stub {type(self._stub).__name__}")
            
        method = getattr(self._stub, method_name)
        
        async def operation():
            try:
                if metadata:
                    return await method(request, metadata=metadata, timeout=timeout)
                else:
                    return await method(request, timeout=timeout)
            except Exception as e:
                # If we need to reconnect
                if retry_on_error:
                    await self.reconnect()
                raise e
                
        # Retry logic
        retries = 3 if retry_on_error else 0
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e
                if attempt < retries:
                    # Exponential backoff with jitter
                    delay = min(30, 1.0 * (2 ** attempt) * (0.5 + 0.5 * np.random.random()))
                    await asyncio.sleep(delay)
                    
        if last_error:
            raise last_error
            
        raise GRPCError("Unexpected error in call_unary")


class GRPCConnectionPool(AsyncResourcePool[GRPCServiceResource]):
    """
    A pool of reusable gRPC service connections.
    
    This class provides:
    1. Connection pooling and reuse
    2. Automatic connection management
    3. Service instance health checking
    
    Usage:
    ```python
    # Create a pool
    pool = GRPCConnectionPool(
        service_stub_class=ServiceStub,
        address="localhost:50051",
        max_size=10
    )
    
    # Start the pool
    await pool.start()
    
    # Use a connection from the pool
    async with pool.acquire() as service:
        response = await service.stub.SomeMethod(request)
        
    # Stop the pool when done
    await pool.stop()
    ```
    """
    
    def __init__(
        self,
        service_stub_class: Type[StubType],
        address: str,
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: Optional[float] = 300.0,
        health_check_interval: Optional[float] = 60.0,
        channel_options: Optional[List[tuple]] = None,
        compression: Optional[grpc.Compression] = None,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
        channel_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new GRPCConnectionPool.
        
        Args:
            service_stub_class: The gRPC service stub class.
            address: The address of the gRPC server.
            max_size: Maximum number of connections in the pool.
            min_size: Minimum number of connections to keep in the pool.
            max_idle_time: Maximum time a connection can be idle before being closed.
            health_check_interval: Interval for health checks.
            channel_options: gRPC channel options.
            compression: gRPC compression option.
            ssl_credentials: SSL credentials for secure connections.
            channel_args: Additional arguments for the channel resource.
        """
        self._service_stub_class = service_stub_class
        self._address = address
        self._channel_options = channel_options
        self._compression = compression
        self._ssl_credentials = ssl_credentials
        self._channel_args = channel_args or {}
        
        super().__init__(
            factory=self._create_service_resource,
            max_size=max_size,
            min_size=min_size,
            max_idle_time=max_idle_time,
            health_check_interval=health_check_interval
        )
        
    def _create_service_resource(self) -> GRPCServiceResource:
        """Factory function to create a new service resource."""
        return GRPCServiceResource(
            service_stub_class=self._service_stub_class,
            address=self._address,
            channel_options=self._channel_options,
            compression=self._compression,
            ssl_credentials=self._ssl_credentials,
            channel_args=self._channel_args
        )


# Context managers for convenience

class GRPCChannelContext(AsyncResourceContext):
    """Context manager for GRPCChannelResource."""
    
    def __init__(
        self,
        address: str,
        options: Optional[List[tuple]] = None,
        compression: Optional[grpc.Compression] = None,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize a new GRPCChannelContext.
        
        Args:
            address: The address of the gRPC server.
            options: gRPC channel options.
            compression: gRPC compression option.
            ssl_credentials: SSL credentials for secure connections.
            timeout: Timeout for initialization and operations.
        """
        resource = GRPCChannelResource(
            address=address,
            options=options,
            compression=compression,
            ssl_credentials=ssl_credentials,
            initialize_timeout=timeout or 15.0
        )
        super().__init__(resource=resource, initialize=True, cleanup=True)


class GRPCServiceContext(AsyncResourceContext):
    """Context manager for GRPCServiceResource."""
    
    def __init__(
        self,
        service_stub_class: Type[StubType],
        address: str,
        channel_options: Optional[List[tuple]] = None,
        compression: Optional[grpc.Compression] = None,
        ssl_credentials: Optional[grpc.ChannelCredentials] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize a new GRPCServiceContext.
        
        Args:
            service_stub_class: The gRPC service stub class.
            address: The address of the gRPC server.
            channel_options: gRPC channel options.
            compression: gRPC compression option.
            ssl_credentials: SSL credentials for secure connections.
            timeout: Timeout for initialization and operations.
        """
        resource = GRPCServiceResource(
            service_stub_class=service_stub_class,
            address=address,
            channel_options=channel_options,
            compression=compression,
            ssl_credentials=ssl_credentials,
            initialize_timeout=timeout or 15.0
        )
        super().__init__(resource=resource, initialize=True, cleanup=True)
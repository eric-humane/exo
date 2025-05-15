"""
Example usage of async resource management in exo.

This module demonstrates how to use the AsyncResource classes to manage resources properly,
with examples focused on networking, connections, and proper cleanup.
"""

import asyncio
import grpc
import time
import numpy as np
from typing import Optional, Dict, Any

from exo.utils.async_resources import (
    NetworkResource, AsyncResourcePool, AsyncResourceContext,
    with_timeout, with_resource
)

# Example 1: gRPC Connection Resource
class GRPCConnectionResource(NetworkResource):
    """
    Manages a gRPC connection as an async resource.
    
    This ensures proper lifecycle management including:
    - Connection establishment with timeout and retry
    - Connection health monitoring
    - Proper cleanup
    - Automatic reconnection
    """
    
    def __init__(
        self, 
        address: str,
        options: Optional[list] = None,
        compression: Optional[grpc.Compression] = None,
        connection_timeout: float = 10.0,
        keepalive_interval: float = 30.0,
        max_idle_time: float = 300.0,  # 5 minutes
        max_age: float = 3600.0,  # 1 hour
        max_retries: int = 3
    ):
        """
        Initialize the gRPC connection resource.
        
        Args:
            address: gRPC server address (host:port)
            options: gRPC channel options
            compression: gRPC compression method
            connection_timeout: Timeout for connection attempts
            keepalive_interval: Interval for keepalive pings
            max_idle_time: Maximum idle time before disconnecting
            max_age: Maximum connection age before recycling
            max_retries: Maximum retry attempts for operations
        """
        super().__init__(
            name=f"grpc_{address}",
            resource_id=f"grpc_connection_{address}",
            connection_timeout=connection_timeout,
            keepalive_interval=keepalive_interval,
            max_idle_time=max_idle_time,
            max_age=max_age,
            max_retries=max_retries
        )
        self.address = address
        self.options = options or []
        self.compression = compression or grpc.Compression.NoCompression
        self.channel = None
        self.last_activity = time.time()
        
    async def _initialize(self) -> None:
        """Initialize the gRPC channel."""
        try:
            self.channel = grpc.aio.insecure_channel(
                self.address,
                options=self.options,
                compression=self.compression
            )
            # Ensure channel becomes ready
            await asyncio.wait_for(
                self.channel.channel_ready(),
                timeout=self.connection_timeout
            )
        except asyncio.TimeoutError:
            if self.channel:
                await self.channel.close()
                self.channel = None
            raise
        except Exception:
            if self.channel:
                await self.channel.close()
                self.channel = None
            raise
            
    async def _cleanup(self) -> None:
        """Clean up the gRPC channel."""
        if self.channel:
            try:
                await asyncio.wait_for(
                    self.channel.close(),
                    timeout=5.0
                )
            except (asyncio.TimeoutError, Exception):
                pass  # We did our best to clean up
            finally:
                self.channel = None
                
    async def check_health(self) -> bool:
        """Check if the gRPC channel is healthy."""
        if not self.channel:
            return False
            
        try:
            state = self.channel.get_state()
            return state in (
                grpc.ChannelConnectivity.READY, 
                grpc.ChannelConnectivity.IDLE
            )
        except Exception:
            return False
            
    async def send_keepalive(self) -> None:
        """Send a keepalive ping to keep the connection active."""
        if self.channel:
            try:
                # Get the current state, which may trigger a ping
                state = self.channel.get_state(try_to_connect=True)
            except Exception:
                # If we can't even get the state, the connection is unhealthy
                pass
                
    async def get_stub(self, stub_class):
        """
        Get a gRPC stub for the specified service.
        
        Args:
            stub_class: The gRPC stub class to instantiate
            
        Returns:
            An instance of the stub class
        """
        if not self.channel:
            raise ConnectionError("Channel not initialized")
            
        self.last_activity = time.time()
        return stub_class(self.channel)


# Example 2: Connection Pool
class GRPCConnectionPool(AsyncResourcePool):
    """
    Pool of gRPC connections for efficient reuse.
    
    This manages a collection of gRPC connections with:
    - Connection creation and initialization
    - Connection reuse
    - Connection validation
    - Connection cleanup
    """
    
    def __init__(
        self,
        min_size: int = 1,
        max_size: int = 10,
        max_idle: int = 2,
        default_options: Optional[list] = None
    ):
        """
        Initialize the gRPC connection pool.
        
        Args:
            min_size: Minimum number of connections to keep in the pool
            max_size: Maximum number of connections allowed in the pool
            max_idle: Maximum number of idle connections to keep
            default_options: Default gRPC channel options
        """
        self.default_options = default_options or []
        
        def create_connection(resource_id=None, address=None):
            """Factory function for creating gRPC connections."""
            if not address:
                raise ValueError("Address is required to create a connection")
                
            return GRPCConnectionResource(
                address=address,
                options=self.default_options,
                resource_id=resource_id
            )
            
        super().__init__(
            resource_factory=create_connection,
            min_size=min_size,
            max_size=max_size,
            max_idle=max_idle,
            ttl=600.0,  # 10 minutes
            validate_on_borrow=True,
            name="GRPCConnectionPool"
        )
        
        # Track connections by address
        self.connections_by_address = {}
        
    async def get_connection(self, address: str) -> GRPCConnectionResource:
        """
        Get a connection for the specified address.
        
        Args:
            address: gRPC server address (host:port)
            
        Returns:
            A GRPCConnectionResource for the address
        """
        # Check if we already have a connection for this address
        if address in self.connections_by_address:
            conn = self.connections_by_address[address]
            if await conn.check_health():
                return conn
                
        # Create a new connection
        conn = self.resource_factory(address=address)
        await conn.acquire()
        self.connections_by_address[address] = conn
        return conn
        
    @asynccontextmanager
    async def connection(self, address: str):
        """
        Context manager for getting and automatically releasing a connection.
        
        Args:
            address: gRPC server address (host:port)
            
        Yields:
            A GRPCConnectionResource for the address
        """
        conn = await self.get_connection(address)
        try:
            yield conn
        finally:
            # We don't release connections immediately, just note that they're idle
            conn.last_activity = time.time()


# Example 3: Practical usage in a service client
class NodeServiceClient:
    """
    Client for the Node service using async resource management.
    
    This demonstrates how to use the async resource classes in a real-world
    service client implementation.
    """
    
    def __init__(
        self,
        pool: Optional[GRPCConnectionPool] = None,
        default_timeout: float = 30.0
    ):
        """
        Initialize the Node service client.
        
        Args:
            pool: Optional connection pool to use (will create one if not provided)
            default_timeout: Default timeout for operations
        """
        self.pool = pool or GRPCConnectionPool()
        self.default_timeout = default_timeout
        
    async def start(self):
        """Start the client and initialize the connection pool."""
        await self.pool.start()
        
    async def stop(self):
        """Stop the client and cleanup resources."""
        await self.pool.stop()
        
    async def send_prompt(
        self,
        address: str,
        prompt: str,
        shard_info: Dict[str, Any],
        request_id: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        Send a prompt to a node.
        
        Args:
            address: Node address (host:port)
            prompt: The prompt to send
            shard_info: Information about the model shard
            request_id: Optional request identifier
            timeout: Operation timeout (uses default if not specified)
            
        Returns:
            The response from the service
        """
        async with self.pool.connection(address) as conn:
            # Import here to avoid circular imports
            from exo.networking.grpc import node_service_pb2_grpc, node_service_pb2
            
            # Get the stub
            stub = await conn.get_stub(node_service_pb2_grpc.NodeServiceStub)
            
            # Create the request
            request = node_service_pb2.PromptRequest(
                prompt=prompt,
                shard=node_service_pb2.Shard(
                    model_id=shard_info.get("model_id"),
                    start_layer=shard_info.get("start_layer", 0),
                    end_layer=shard_info.get("end_layer", 0),
                    n_layers=shard_info.get("n_layers", 0),
                ),
                request_id=request_id,
                inference_state=None  # Set inference state if needed
            )
            
            # Send the request with timeout
            return await with_timeout(
                stub,
                lambda s: s.SendPrompt(request),
                timeout or self.default_timeout
            )
            
    async def health_check(
        self,
        address: str,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Check if a node is healthy.
        
        Args:
            address: Node address (host:port)
            timeout: Operation timeout (uses default if not specified)
            
        Returns:
            True if the node is healthy, False otherwise
        """
        try:
            async with self.pool.connection(address) as conn:
                from exo.networking.grpc import node_service_pb2_grpc, node_service_pb2
                stub = await conn.get_stub(node_service_pb2_grpc.NodeServiceStub)
                
                # Create the request
                request = node_service_pb2.HealthCheckRequest()
                
                # Send the request with timeout
                response = await with_timeout(
                    stub,
                    lambda s: s.HealthCheck(request),
                    timeout or 5.0  # Use shorter timeout for health checks
                )
                
                return response.is_healthy
        except Exception:
            # Any exception means the node is not healthy
            return False


# Example 4: Demonstration of how this would improve the codebase
async def example_code_improvements():
    """
    Demonstrate how using async resources improves code quality.
    
    This contrasts the current approach with the improved approach.
    """
    # ------ CURRENT APPROACH ------
    # Lots of manual connection management, error handling, and cleanup
    
    # Creating and connecting
    channel = None
    try:
        channel = grpc.aio.insecure_channel(
            "example.com:8080",
            options=[...],
            compression=grpc.Compression.Gzip
        )
        await asyncio.wait_for(channel.channel_ready(), timeout=10.0)
        stub = node_service_pb2_grpc.NodeServiceStub(channel)
    except Exception as e:
        print(f"Connection failed: {e}")
        if channel:
            await channel.close()
        return
    
    # Using the connection
    try:
        request = node_service_pb2.PromptRequest(...)
        response = await stub.SendPrompt(request)
    except Exception as e:
        print(f"Request failed: {e}")
        await channel.close()
        return
    
    # Cleaning up
    finally:
        if channel:
            await channel.close()
    
    # ------ IMPROVED APPROACH ------
    # Using the new async resource management
    
    # Create client
    client = NodeServiceClient()
    await client.start()
    
    try:
        # Send prompt (connection management handled automatically)
        response = await client.send_prompt(
            address="example.com:8080",
            prompt="Hello, world!",
            shard_info={"model_id": "model1", "n_layers": 12}
        )
        
        # Process response
        print(f"Got response: {response}")
        
    finally:
        # Clean up all resources
        await client.stop()
    
    # The improved approach:
    # 1. Reduces boilerplate code
    # 2. Ensures proper cleanup in all cases
    # 3. Centralizes error handling and retry logic
    # 4. Improves resource reuse through connection pooling
    # 5. Makes the code more readable and maintainable


# Example 5: Implementing resource-aware operations
async def example_resource_aware_operations():
    """Demonstrate resource-aware operations using async context managers."""
    
    # Setup resource pool
    pool = GRPCConnectionPool()
    await pool.start()
    
    try:
        # Example operation with a resource
        async def process_with_connection(conn, data):
            # Use the connection to process data
            from exo.networking.grpc import node_service_pb2_grpc
            stub = await conn.get_stub(node_service_pb2_grpc.NodeServiceStub)
            # ... use the stub ...
            return f"Processed {data} using {conn.address}"
        
        # Use a connection with the 'with_resource' decorator
        @with_resource(lambda: pool.connection("example.com:8080"))
        async def process_data(conn, data):
            return await process_with_connection(conn, data)
        
        # Call the decorated function
        result = await process_data("sample data")
        print(result)
        
        # Use resource context for more complex scenarios
        async with AsyncResourceContext(
            await pool.get_connection("example.com:8081"),
            on_success=lambda conn, result: print(f"Success with {conn.address}: {result}"),
            on_error=lambda conn, error: print(f"Error with {conn.address}: {error}"),
            timeout=10.0
        ) as ctx:
            # Run an operation with the resource
            result = await ctx.run(
                lambda conn: process_with_connection(conn, "more data")
            )
            
    finally:
        await pool.stop()


# Command-line execution for example
if __name__ == "__main__":
    async def run_examples():
        print("Running examples of async resource management...")
        
        # Run the examples
        await example_code_improvements()
        await example_resource_aware_operations()
        
        print("Examples completed!")
    
    # Run the async examples
    asyncio.run(run_examples())
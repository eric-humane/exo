"""
Async Resource Management for exo.

This module provides classes and utilities for managing async resources in a structured way,
ensuring proper lifecycle management, error handling, and cleanup.

Key classes:
- AsyncResource: Base class for async resources with lifecycle management
- NetworkResource: Specialized resource for network connections
- AsyncResourcePool: Pool for managing and reusing resources
- AsyncResourceContext: Context manager for structured resource operations

Usage:
```python
# Basic usage
resource = AsyncResource(resource_id="db_connection")
try:
    await resource.initialize()
    # Use the resource...
finally:
    await resource.cleanup()

# Context manager usage
async with AsyncResourceContext(resource) as r:
    # Use the resource...
    
# Pool usage
pool = AsyncResourcePool(
    factory=lambda: DatabaseConnection(),
    max_size=10
)
async with pool.acquire() as connection:
    # Use the connection...
```
"""

import asyncio
import functools
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union, cast

T = TypeVar('T')
R = TypeVar('R')


class ResourceState(Enum):
    """
    States that a resource can be in during its lifecycle.
    """
    UNINITIALIZED = auto()  # Resource is created but not initialized
    INITIALIZING = auto()   # Resource initialization in progress
    READY = auto()          # Resource is initialized and ready to use
    DEGRADED = auto()       # Resource is initialized but operating in a degraded state
    ERROR = auto()          # Resource is in an error state
    CLOSING = auto()        # Resource cleanup in progress
    CLOSED = auto()         # Resource is cleaned up and no longer usable


class ResourceAcquireTimeout(Exception):
    """Raised when a resource cannot be acquired within the specified timeout."""
    pass


class ResourceInitializationError(Exception):
    """Raised when a resource fails to initialize correctly."""
    pass


class ResourceUnavailableError(Exception):
    """Raised when a resource is not in a usable state."""
    pass


class AsyncResource(ABC):
    """
    Base class for asynchronous resources that need proper lifecycle management.
    
    This class provides:
    1. Structured lifecycle management (initialize, use, cleanup)
    2. State tracking and validation
    3. Error handling patterns
    4. Last error tracking
    5. Health checking mechanism
    
    Subclasses should implement:
    - _do_initialize: Resource-specific initialization
    - _do_cleanup: Resource-specific cleanup
    - _check_health: Resource-specific health check (optional)
    """
    
    def __init__(self, resource_id: Optional[str] = None, initialize_timeout: float = 30.0):
        """
        Initialize a new AsyncResource.
        
        Args:
            resource_id: Unique identifier for this resource. If not provided, a UUID will be generated.
            initialize_timeout: Default timeout for initialization in seconds.
        """
        self._id = resource_id or str(uuid.uuid4())
        self._state = ResourceState.UNINITIALIZED
        self._state_lock = asyncio.Lock()
        self._last_error: Optional[Exception] = None
        self._last_error_time: Optional[float] = None
        self._initialize_timeout = initialize_timeout
        self._created_at = time.time()
        self._initialized_at: Optional[float] = None
        self._in_use = False
        self._in_use_lock = asyncio.Lock()
        
        # Create a future that will be resolved when the resource is ready
        self._ready_future: Optional[asyncio.Future] = None
        
    @property
    def id(self) -> str:
        """Get the resource id."""
        return self._id
    
    @property
    def state(self) -> ResourceState:
        """Get the current state of the resource."""
        return self._state
    
    @property
    def last_error(self) -> Optional[Exception]:
        """Get the last error encountered by this resource."""
        return self._last_error
    
    @property
    def is_initialized(self) -> bool:
        """Check if the resource is initialized."""
        return self._state in (ResourceState.READY, ResourceState.DEGRADED)
    
    @property
    def is_usable(self) -> bool:
        """Check if the resource is in a usable state."""
        return self._state in (ResourceState.READY, ResourceState.DEGRADED)
    
    @property
    def is_healthy(self) -> bool:
        """Check if the resource is in a healthy state."""
        return self._state == ResourceState.READY
    
    @property
    def in_use(self) -> bool:
        """Check if the resource is currently in use."""
        return self._in_use
    
    async def initialize(self, timeout: Optional[float] = None) -> bool:
        """
        Initialize the resource. This method is safe to call multiple times;
        it will only initialize once and return quickly on subsequent calls.
        
        Args:
            timeout: Maximum time to wait for initialization in seconds.
                   If None, uses the default timeout from the constructor.
                   
        Returns:
            True if the resource was initialized successfully, False otherwise.
            
        Raises:
            asyncio.TimeoutError: If initialization times out.
            ResourceInitializationError: If initialization fails.
        """
        timeout = timeout if timeout is not None else self._initialize_timeout
        
        # Fast path: If already initialized, return immediately
        if self.is_initialized:
            return True
            
        async with self._state_lock:
            # Check again with the lock held
            if self.is_initialized:
                return True
                
            # If initialization is already in progress, wait for it to complete
            if self._state == ResourceState.INITIALIZING:
                if self._ready_future is None:
                    self._ready_future = asyncio.Future()
                    
            # Start initialization
            else:
                self._state = ResourceState.INITIALIZING
                self._ready_future = asyncio.Future()
                
                # Start the initialization in a task so we can timeout
                init_task = asyncio.create_task(self._do_initialize())
                
                try:
                    # Run initialization with timeout
                    await asyncio.wait_for(init_task, timeout=timeout)
                    
                    # Mark as ready and resolve the future
                    self._state = ResourceState.READY
                    self._initialized_at = time.time()
                    self._ready_future.set_result(True)
                    return True
                    
                except asyncio.TimeoutError:
                    # Cancel the task if it's still running
                    if not init_task.done():
                        init_task.cancel()
                        
                    self._set_error(
                        ResourceInitializationError(f"Initialization timed out after {timeout} seconds")
                    )
                    self._ready_future.set_exception(asyncio.TimeoutError(
                        f"Resource {self._id} initialization timed out after {timeout} seconds"
                    ))
                    raise
                    
                except Exception as e:
                    self._set_error(e)
                    self._ready_future.set_exception(
                        ResourceInitializationError(f"Failed to initialize resource {self._id}: {str(e)}")
                    )
                    raise ResourceInitializationError(f"Failed to initialize resource {self._id}: {str(e)}") from e
        
        # If we reach here, initialization was already in progress, so wait for it
        try:
            await asyncio.wait_for(self._ready_future, timeout=timeout)
            return True
        except Exception as e:
            # The future already has the initialization error set
            if isinstance(e, asyncio.TimeoutError):
                raise asyncio.TimeoutError(
                    f"Timed out waiting for resource {self._id} to initialize"
                ) from e
            raise
            
    async def cleanup(self) -> None:
        """
        Clean up the resource, releasing any acquired resources.
        Safe to call multiple times.
        """
        # Fast path: If already closed, return immediately
        if self._state in (ResourceState.CLOSED, ResourceState.CLOSING):
            return
            
        async with self._state_lock:
            # Check again with the lock held
            if self._state in (ResourceState.CLOSED, ResourceState.CLOSING):
                return
                
            # Mark as closing
            self._state = ResourceState.CLOSING
            
            try:
                # Perform the actual cleanup
                await self._do_cleanup()
                self._state = ResourceState.CLOSED
                
            except Exception as e:
                self._set_error(e)
                # Still mark as closed even if cleanup fails
                self._state = ResourceState.CLOSED
                # Re-raise the error
                raise
                
    async def check_health(self) -> bool:
        """
        Check if the resource is healthy.
        
        Returns:
            True if the resource is healthy, False otherwise.
        """
        # If not initialized, it's definitely not healthy
        if not self.is_initialized:
            return False
            
        # If in error state, it's not healthy
        if self._state == ResourceState.ERROR:
            return False
            
        try:
            # Run the resource-specific health check
            is_healthy = await self._check_health()
            
            # Update state based on health check
            async with self._state_lock:
                if is_healthy and self._state == ResourceState.DEGRADED:
                    self._state = ResourceState.READY
                elif not is_healthy and self._state == ResourceState.READY:
                    self._state = ResourceState.DEGRADED
                    
            return is_healthy
            
        except Exception as e:
            self._set_error(e)
            return False
            
    async def _check_health(self) -> bool:
        """
        Resource-specific health check. By default, returns True.
        
        Override this in subclasses with specific health check logic.
        
        Returns:
            True if the resource is healthy, False otherwise.
        """
        return True
        
    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        Resource-specific initialization logic.
        
        This is where subclasses should implement their specific initialization.
        Must be implemented by subclasses.
        
        Raises:
            Various exceptions: Depending on what can go wrong during initialization.
        """
        raise NotImplementedError("Subclasses must implement _do_initialize")
        
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """
        Resource-specific cleanup logic.
        
        This is where subclasses should implement their specific cleanup.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _do_cleanup")
        
    def _set_error(self, error: Exception) -> None:
        """
        Record an error that occurred with this resource.
        
        Args:
            error: The exception that occurred.
        """
        self._last_error = error
        self._last_error_time = time.time()
        self._state = ResourceState.ERROR
        
    async def mark_in_use(self) -> None:
        """Mark this resource as being in use."""
        async with self._in_use_lock:
            self._in_use = True
            
    async def mark_not_in_use(self) -> None:
        """Mark this resource as no longer in use."""
        async with self._in_use_lock:
            self._in_use = False
            
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, state={self._state.name})"
        
    def __repr__(self) -> str:
        return self.__str__()


class AsyncManagedResource(AsyncResource):
    """
    Extension of AsyncResource with retry logic and automatic recovery.
    
    This class adds:
    1. Retry logic for initialization
    2. Automatic recovery from transient errors
    3. Exponential backoff for retries
    """
    
    def __init__(
        self, 
        resource_id: Optional[str] = None,
        initialize_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        health_check_interval: Optional[float] = None
    ):
        """
        Initialize a new AsyncManagedResource.
        
        Args:
            resource_id: Unique identifier for this resource. If not provided, a UUID will be generated.
            initialize_timeout: Default timeout for initialization in seconds.
            max_retries: Maximum number of retries for initialization and operations.
            retry_delay: Initial delay between retries in seconds.
            max_retry_delay: Maximum delay between retries in seconds.
            health_check_interval: Interval for automatic health checks in seconds. If None, no automatic health checks.
        """
        super().__init__(resource_id, initialize_timeout)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_retry_delay = max_retry_delay
        self._health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self, timeout: Optional[float] = None) -> bool:
        """
        Initialize the resource with retry logic.
        
        Args:
            timeout: Maximum time to wait for initialization in seconds.
                   If None, uses the default timeout from the constructor.
                   
        Returns:
            True if the resource was initialized successfully, False otherwise.
        """
        timeout = timeout if timeout is not None else self._initialize_timeout
        retry_count = 0
        last_error = None
        
        # Try to initialize with retries
        while retry_count <= self._max_retries:
            try:
                result = await super().initialize(timeout)
                
                # If initialization succeeded and health checks are configured, start the health check task
                if result and self._health_check_interval and self._health_check_task is None:
                    self._start_health_check_task()
                    
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # If we've exhausted retries, raise the error
                if retry_count > self._max_retries:
                    break
                    
                # Calculate backoff time with jitter
                backoff = min(
                    self._max_retry_delay,
                    self._retry_delay * (2 ** (retry_count - 1)) * (0.5 + 0.5 * (time.time() % 1))
                )
                
                # Wait before retrying
                await asyncio.sleep(backoff)
                
        # If we get here, all retries failed
        if last_error:
            raise ResourceInitializationError(
                f"Failed to initialize resource {self.id} after {self._max_retries} retries"
            ) from last_error
            
        return False
        
    async def cleanup(self) -> None:
        """Clean up the resource and stop health check task if running."""
        # Stop health check task first
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            
        # Then perform normal cleanup
        await super().cleanup()
        
    def _start_health_check_task(self) -> None:
        """Start the background health check task."""
        if self._health_check_task is not None:
            return
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        # Ensure the task doesn't keep the program alive
        self._health_check_task.add_done_callback(lambda _: None)
        
    async def _health_check_loop(self) -> None:
        """Run health checks at the configured interval."""
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                try:
                    await self.check_health()
                except Exception:
                    # Just log the error and continue; don't crash the health check task
                    pass
        except asyncio.CancelledError:
            # Normal cancellation
            return
            
    async def with_retry(
        self, 
        operation: Callable[[], Any], 
        operation_name: str = "operation",
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: The operation to execute. Can be a regular function or a coroutine.
            operation_name: Name of the operation for error reporting.
            max_retries: Maximum number of retries. If None, uses the instance default.
            retry_delay: Initial delay between retries. If None, uses the instance default.
            timeout: Timeout for the operation. If None, no timeout is applied.
            
        Returns:
            The result of the operation.
            
        Raises:
            Various exceptions: Depending on what can go wrong during the operation.
        """
        max_retries = max_retries if max_retries is not None else self._max_retries
        retry_delay = retry_delay if retry_delay is not None else self._retry_delay
        retry_count = 0
        last_error = None
        
        # Ensure the resource is initialized
        if not self.is_initialized:
            await self.initialize()
            
        # Try the operation with retries
        while retry_count <= max_retries:
            try:
                # Execute the operation, handling both coroutines and regular functions
                if asyncio.iscoroutinefunction(operation) or asyncio.iscoroutine(operation):
                    # For coroutines, we might need to apply a timeout
                    if timeout is not None:
                        return await asyncio.wait_for(operation(), timeout=timeout)
                    else:
                        return await operation()
                else:
                    # For regular functions, just call them
                    return operation()
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # If we've exhausted retries, raise the error
                if retry_count > max_retries:
                    break
                    
                # Calculate backoff time with jitter
                backoff = min(
                    self._max_retry_delay,
                    retry_delay * (2 ** (retry_count - 1)) * (0.5 + 0.5 * (time.time() % 1))
                )
                
                # Wait before retrying
                await asyncio.sleep(backoff)
                
        # If we get here, all retries failed
        if last_error:
            raise last_error
            
        # Should never reach here
        raise RuntimeError(f"Unexpected error in with_retry for {operation_name}")


class NetworkResource(AsyncManagedResource):
    """
    Specialized resource for network connections.
    
    This class adds:
    1. Connection state tracking
    2. Network-specific health checks
    3. Reconnection logic
    """
    
    def __init__(
        self,
        resource_id: Optional[str] = None,
        address: str = "",
        initialize_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        health_check_interval: Optional[float] = 60.0,
        reconnect_on_error: bool = True
    ):
        """
        Initialize a new NetworkResource.
        
        Args:
            resource_id: Unique identifier for this resource.
            address: Network address for this connection.
            initialize_timeout: Default timeout for initialization.
            max_retries: Maximum number of retries for initialization.
            retry_delay: Initial delay between retries.
            max_retry_delay: Maximum delay between retries.
            health_check_interval: Interval for automatic health checks.
            reconnect_on_error: Whether to automatically reconnect on error.
        """
        super().__init__(
            resource_id=resource_id,
            initialize_timeout=initialize_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_retry_delay=max_retry_delay,
            health_check_interval=health_check_interval
        )
        self._address = address
        self._reconnect_on_error = reconnect_on_error
        self._connection_attempts = 0
        self._last_connection_time: Optional[float] = None
        self._last_successful_operation_time: Optional[float] = None
        
    @property
    def address(self) -> str:
        """Get the network address for this connection."""
        return self._address
        
    @property
    def connection_attempts(self) -> int:
        """Get the number of connection attempts made."""
        return self._connection_attempts
        
    async def reconnect(self) -> bool:
        """
        Force a reconnection of this resource.
        
        Returns:
            True if reconnection was successful, False otherwise.
        """
        await self.cleanup()
        return await self.initialize()
        
    async def with_connection(
        self, 
        operation: Callable[[], Any],
        operation_name: str = "network_operation",
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        reconnect_on_error: Optional[bool] = None
    ) -> Any:
        """
        Execute an operation with the connection, handling reconnection if needed.
        
        Args:
            operation: The operation to execute. Can be a regular function or coroutine.
            operation_name: Name of the operation for error reporting.
            max_retries: Maximum number of retries.
            retry_delay: Initial delay between retries.
            timeout: Timeout for the operation.
            reconnect_on_error: Whether to reconnect on error. If None, uses instance default.
            
        Returns:
            The result of the operation.
        """
        reconnect_on_error = reconnect_on_error if reconnect_on_error is not None else self._reconnect_on_error
        
        async def operation_with_reconnect():
            try:
                result = await self.with_retry(
                    operation,
                    operation_name=operation_name,
                    max_retries=0,  # No retries at this level, we'll handle it
                    timeout=timeout
                )
                self._last_successful_operation_time = time.time()
                return result
            except Exception as e:
                # If configured to reconnect on error, try to reconnect
                if reconnect_on_error:
                    await self.reconnect()
                raise e
                
        # Use the standard retry logic with our reconnection wrapper
        return await self.with_retry(
            operation_with_reconnect,
            operation_name=operation_name,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
    async def _do_initialize(self) -> None:
        """
        Resource-specific initialization for network connections.
        
        Subclasses must override this with their specific connection logic.
        """
        self._connection_attempts += 1
        self._last_connection_time = time.time()
        # The default implementation does nothing
        # Subclasses must override this


class AsyncResourcePool(Generic[T]):
    """
    A pool of reusable async resources.
    
    This class provides:
    1. Resource pooling and reuse
    2. Automatic resource creation
    3. Resource health checking
    4. Efficient resource allocation
    
    Usage:
    ```python
    # Create a pool
    pool = AsyncResourcePool(
        factory=lambda: DatabaseConnection(db_url),
        max_size=10
    )
    
    # Use a resource from the pool
    async with pool.acquire() as connection:
        # Use the connection...
    ```
    """
    
    def __init__(
        self,
        factory: Callable[[], AsyncResource],
        max_size: int = 10,
        min_size: int = 0,
        max_idle_time: Optional[float] = 300.0,
        health_check_interval: Optional[float] = 60.0
    ):
        """
        Initialize a new resource pool.
        
        Args:
            factory: A callable that creates new resources.
            max_size: Maximum number of resources in the pool.
            min_size: Minimum number of resources to keep in the pool.
            max_idle_time: Maximum time in seconds a resource can be idle before being closed.
            health_check_interval: Interval for health checks in seconds.
        """
        self._factory = factory
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time
        self._health_check_interval = health_check_interval
        
        self._resources: Dict[str, AsyncResource] = {}
        self._available_resources: Set[str] = set()
        self._in_use_resources: Set[str] = set()
        
        self._lock = asyncio.Lock()
        self._resource_available = asyncio.Condition(self._lock)
        
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False
        
    async def start(self) -> None:
        """Start the pool, initializing minimum resources and maintenance tasks."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start a closed pool")
                
            # Initialize minimum number of resources
            for _ in range(self._min_size):
                resource_id = await self._create_resource()
                self._available_resources.add(resource_id)
                
            # Start maintenance tasks
            if self._health_check_interval:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                
            if self._max_idle_time:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                
    async def stop(self) -> None:
        """Stop the pool, cleaning up all resources and maintenance tasks."""
        async with self._lock:
            self._closed = True
            
            # Cancel maintenance tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
                
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
                self._cleanup_task = None
                
            # Close all resources
            resources = list(self._resources.values())
            for resource in resources:
                await resource.cleanup()
                
            self._resources.clear()
            self._available_resources.clear()
            self._in_use_resources.clear()
            
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None) -> AsyncResource:
        """
        Acquire a resource from the pool.
        
        This is a context manager, so use it in an async with statement:
        
        ```python
        async with pool.acquire() as resource:
            # Use the resource...
        ```
        
        Args:
            timeout: Maximum time to wait for a resource in seconds.
                   If None, waits indefinitely.
                   
        Returns:
            An async context manager that yields a resource.
            
        Raises:
            ResourceAcquireTimeout: If a resource couldn't be acquired within the timeout.
            RuntimeError: If the pool is closed.
        """
        resource = await self._acquire(timeout)
        try:
            yield resource
        finally:
            await self._release(resource.id)
            
    async def _acquire(self, timeout: Optional[float] = None) -> AsyncResource:
        """
        Internal method to acquire a resource from the pool.
        
        Args:
            timeout: Maximum time to wait for a resource.
            
        Returns:
            A resource from the pool.
            
        Raises:
            ResourceAcquireTimeout: If a resource couldn't be acquired within the timeout.
            RuntimeError: If the pool is closed.
        """
        start_time = time.time()
        
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot acquire resources from a closed pool")
                
            # First, try to get an available resource
            while not self._available_resources:
                # If we haven't reached max size, create a new resource
                if len(self._resources) < self._max_size:
                    resource_id = await self._create_resource()
                    self._available_resources.add(resource_id)
                    break
                    
                # Otherwise, wait for a resource to become available
                if timeout is not None:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        raise ResourceAcquireTimeout(
                            f"Timed out waiting for a resource after {timeout:.2f} seconds"
                        )
                        
                    try:
                        await asyncio.wait_for(
                            self._resource_available.wait(),
                            timeout=remaining_time
                        )
                    except asyncio.TimeoutError:
                        raise ResourceAcquireTimeout(
                            f"Timed out waiting for a resource after {timeout:.2f} seconds"
                        )
                else:
                    await self._resource_available.wait()
                    
            # Get an available resource
            resource_id = next(iter(self._available_resources))
            self._available_resources.remove(resource_id)
            self._in_use_resources.add(resource_id)
            
            resource = self._resources[resource_id]
            await resource.mark_in_use()
            
            return resource
            
    async def _release(self, resource_id: str) -> None:
        """
        Release a resource back to the pool.
        
        Args:
            resource_id: The ID of the resource to release.
            
        Raises:
            ValueError: If the resource is not in use or doesn't belong to this pool.
        """
        async with self._lock:
            if resource_id not in self._resources:
                raise ValueError(f"Resource {resource_id} does not belong to this pool")
                
            if resource_id not in self._in_use_resources:
                raise ValueError(f"Resource {resource_id} is not in use")
                
            # Release the resource
            self._in_use_resources.remove(resource_id)
            resource = self._resources[resource_id]
            
            # Check if the resource is still healthy
            if resource.is_usable:
                self._available_resources.add(resource_id)
                await resource.mark_not_in_use()
                self._resource_available.notify()
            else:
                # If not healthy, clean it up and create a new one if needed
                await resource.cleanup()
                del self._resources[resource_id]
                
                # Create a new resource if we're below min_size
                if len(self._resources) < self._min_size:
                    new_id = await self._create_resource()
                    self._available_resources.add(new_id)
                    self._resource_available.notify()
                    
    async def _create_resource(self) -> str:
        """
        Create a new resource and add it to the pool.
        
        Returns:
            The ID of the newly created resource.
        """
        resource = self._factory()
        
        try:
            await resource.initialize()
            self._resources[resource.id] = resource
            return resource.id
        except Exception:
            # If initialization fails, clean up and re-raise
            await resource.cleanup()
            raise
            
    async def _health_check_loop(self) -> None:
        """Background task for checking resource health."""
        try:
            while not self._closed:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_health()
        except asyncio.CancelledError:
            # Normal cancellation
            return
            
    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up idle resources."""
        try:
            while not self._closed:
                await asyncio.sleep(self._max_idle_time / 2)  # Check at half the idle time
                await self._cleanup_idle_resources()
        except asyncio.CancelledError:
            # Normal cancellation
            return
            
    async def _check_all_health(self) -> None:
        """Check the health of all resources in the pool."""
        async with self._lock:
            # Create a list to avoid modifying during iteration
            resources = list(self._resources.values())
            
            # First check available resources
            for resource_id in list(self._available_resources):
                resource = self._resources.get(resource_id)
                if resource is None:
                    continue
                    
                # Skip resources in use
                if resource.in_use:
                    continue
                    
                try:
                    is_healthy = await resource.check_health()
                    if not is_healthy and resource.state != ResourceState.DEGRADED:
                        # If unhealthy and not just degraded, remove from pool
                        self._available_resources.remove(resource_id)
                        await resource.cleanup()
                        del self._resources[resource_id]
                        
                        # Create a replacement if we're below min_size
                        if len(self._resources) < self._min_size:
                            new_id = await self._create_resource()
                            self._available_resources.add(new_id)
                except Exception:
                    # On error, remove from pool
                    self._available_resources.remove(resource_id)
                    await resource.cleanup()
                    del self._resources[resource_id]
                    
            # Next check in-use resources, but don't modify them
            # This just updates their health status
            for resource_id in self._in_use_resources:
                resource = self._resources.get(resource_id)
                if resource is None:
                    continue
                    
                try:
                    await resource.check_health()
                except Exception:
                    # Just ignore errors for in-use resources
                    pass
                    
    async def _cleanup_idle_resources(self) -> None:
        """Clean up resources that have been idle for too long."""
        async with self._lock:
            # Skip if we're already at min_size
            if len(self._resources) <= self._min_size:
                return
                
            now = time.time()
            # Check available resources for idleness
            for resource_id in list(self._available_resources):
                # Only cleanup if we're above min_size
                if len(self._resources) <= self._min_size:
                    break
                    
                resource = self._resources.get(resource_id)
                if resource is None or resource.in_use:
                    continue
                    
                # If resource has an initialized_at timestamp and has been idle too long
                if resource._initialized_at and (now - resource._initialized_at) > self._max_idle_time:
                    # Remove from pool
                    self._available_resources.remove(resource_id)
                    await resource.cleanup()
                    del self._resources[resource_id]


class AsyncResourceGroup:
    """
    A group of related resources that should be managed together.
    
    This is useful for complex dependencies where multiple resources
    need to be initialized and cleaned up as a unit.
    
    Usage:
    ```python
    # Create a resource group
    group = AsyncResourceGroup("auth_system")
    
    # Add resources to the group
    await group.add_resource("db", DatabaseConnection())
    await group.add_resource("cache", CacheConnection())
    
    # Initialize all resources in the group
    await group.initialize()
    
    # Use resources from the group
    db = await group.get_resource("db")
    cache = await group.get_resource("cache")
    
    # Clean up all resources in the group
    await group.cleanup()
    ```
    """
    
    def __init__(self, group_id: Optional[str] = None):
        """
        Initialize a new resource group.
        
        Args:
            group_id: Unique identifier for this group. If None, a UUID will be generated.
        """
        self._id = group_id or str(uuid.uuid4())
        self._resources: Dict[str, AsyncResource] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    @property
    def id(self) -> str:
        """Get the group ID."""
        return self._id
        
    @property
    def is_initialized(self) -> bool:
        """Check if the group is initialized."""
        return self._initialized
        
    async def add_resource(self, name: str, resource: AsyncResource, initialize: bool = False) -> None:
        """
        Add a resource to the group.
        
        Args:
            name: Name for this resource within the group.
            resource: The resource to add.
            initialize: Whether to initialize the resource now.
            
        Raises:
            ValueError: If a resource with this name already exists in the group.
        """
        async with self._lock:
            if name in self._resources:
                raise ValueError(f"Resource with name '{name}' already exists in group {self._id}")
                
            self._resources[name] = resource
            
            if initialize:
                await resource.initialize()
                
    async def remove_resource(self, name: str, cleanup: bool = True) -> AsyncResource:
        """
        Remove a resource from the group.
        
        Args:
            name: Name of the resource to remove.
            cleanup: Whether to clean up the resource.
            
        Returns:
            The removed resource.
            
        Raises:
            ValueError: If no resource with this name exists in the group.
        """
        async with self._lock:
            if name not in self._resources:
                raise ValueError(f"No resource with name '{name}' in group {self._id}")
                
            resource = self._resources.pop(name)
            
            if cleanup:
                await resource.cleanup()
                
            return resource
            
    async def get_resource(self, name: str) -> AsyncResource:
        """
        Get a resource from the group by name.
        
        Args:
            name: Name of the resource to get.
            
        Returns:
            The requested resource.
            
        Raises:
            ValueError: If no resource with this name exists in the group.
            ResourceUnavailableError: If the resource exists but is not initialized.
        """
        async with self._lock:
            if name not in self._resources:
                raise ValueError(f"No resource with name '{name}' in group {self._id}")
                
            resource = self._resources[name]
            
            if not resource.is_initialized:
                raise ResourceUnavailableError(f"Resource '{name}' in group {self._id} is not initialized")
                
            return resource
            
    async def initialize(self, parallel: bool = True) -> None:
        """
        Initialize all resources in the group.
        
        Args:
            parallel: Whether to initialize resources in parallel.
        """
        async with self._lock:
            if self._initialized:
                return
                
            if parallel:
                # Initialize resources in parallel
                tasks = [resource.initialize() for resource in self._resources.values()]
                await asyncio.gather(*tasks)
            else:
                # Initialize resources sequentially
                for resource in self._resources.values():
                    await resource.initialize()
                    
            self._initialized = True
            
    async def cleanup(self, parallel: bool = True) -> None:
        """
        Clean up all resources in the group.
        
        Args:
            parallel: Whether to clean up resources in parallel.
        """
        async with self._lock:
            if parallel:
                # Clean up resources in parallel
                tasks = [resource.cleanup() for resource in self._resources.values()]
                await asyncio.gather(*tasks)
            else:
                # Clean up resources sequentially
                for resource in self._resources.values():
                    await resource.cleanup()
                    
            self._resources.clear()
            self._initialized = False
            
    async def check_health(self) -> Dict[str, bool]:
        """
        Check the health of all resources in the group.
        
        Returns:
            A dictionary mapping resource names to health status.
        """
        async with self._lock:
            results = {}
            for name, resource in self._resources.items():
                try:
                    results[name] = await resource.check_health()
                except Exception:
                    results[name] = False
                    
            return results


class AsyncResourceContext:
    """
    Context manager for working with async resources.
    
    This class provides:
    1. Context management with automatic cleanup
    2. Exception handling and resource state tracking
    3. Support for nested contexts and resource dependencies
    
    Usage:
    ```python
    # Basic usage
    async with AsyncResourceContext(db_resource) as db:
        # Use the db...
        
    # With custom cleanup
    async with AsyncResourceContext(file_resource, cleanup=lambda r: r.close()) as file:
        # Use the file...
    ```
    """
    
    def __init__(
        self, 
        resource: AsyncResource,
        initialize: bool = True,
        cleanup: bool = True,
        cleanup_func: Optional[Callable[[AsyncResource], Any]] = None
    ):
        """
        Initialize a new AsyncResourceContext.
        
        Args:
            resource: The resource to manage.
            initialize: Whether to initialize the resource when entering the context.
            cleanup: Whether to clean up the resource when exiting the context.
            cleanup_func: Custom cleanup function. If None, uses resource.cleanup().
        """
        self._resource = resource
        self._initialize = initialize
        self._cleanup = cleanup
        self._cleanup_func = cleanup_func or (lambda r: r.cleanup())
        self._entered = False
        
    async def __aenter__(self) -> AsyncResource:
        """
        Enter the context, initializing the resource if needed.
        
        Returns:
            The managed resource.
        """
        if self._entered:
            raise RuntimeError("AsyncResourceContext already entered")
            
        self._entered = True
        
        if self._initialize and not self._resource.is_initialized:
            await self._resource.initialize()
            
        return self._resource
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context, cleaning up the resource if needed.
        
        Returns:
            True if the exception was handled, False otherwise.
        """
        if not self._entered:
            return False
            
        self._entered = False
        
        if self._cleanup:
            cleanup_func = self._cleanup_func
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func(self._resource)
            else:
                cleanup_func(self._resource)
                
        return False  # Don't suppress exceptions


async def with_timeout(coro, timeout: float, message: Optional[str] = None):
    """
    Run a coroutine with a timeout.
    
    Args:
        coro: The coroutine to run.
        timeout: The timeout in seconds.
        message: Optional message for the timeout error.
        
    Returns:
        The result of the coroutine.
        
    Raises:
        asyncio.TimeoutError: If the coroutine times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if message:
            raise asyncio.TimeoutError(message)
        raise


def with_resource(
    resource_factory: Callable[[], AsyncResource],
    cleanup: bool = True
):
    """
    Decorator for using a resource within a function.
    
    Usage:
    ```python
    @with_resource(lambda: DatabaseConnection("db_url"))
    async def get_user(db, user_id):
        return await db.query("SELECT * FROM users WHERE id = ?", user_id)
    ```
    
    Args:
        resource_factory: Factory function that creates the resource.
        cleanup: Whether to clean up the resource after use.
        
    Returns:
        A decorator function.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            resource = resource_factory()
            try:
                await resource.initialize()
                return await func(resource, *args, **kwargs)
            finally:
                if cleanup:
                    await resource.cleanup()
        return wrapper
    return decorator
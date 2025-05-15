# Async Resource Management in exo

This document describes the async resource management system in exo, which provides a structured way to handle lifecycle management, error handling, and cleanup for asynchronous resources like network connections, file handles, and other external resources.

## Overview

Asynchronous code in Python can be challenging to manage correctly, especially when dealing with external resources that require proper initialization, error handling, and cleanup. The `exo.utils.async_resources` module provides a set of classes and utilities to address these challenges.

Key benefits of the system:

- **Structured lifecycle management**: Clearly defined resource states and transitions
- **Automatic error handling and recovery**: Built-in retry logic and graceful error handling
- **Resource pooling and reuse**: Efficient management of expensive resources
- **Centralized cleanup**: Guaranteed resource cleanup, even in error cases
- **Proper health monitoring**: Automatic health checks and degraded state handling

## Core Components

### AsyncResource

The `AsyncResource` class is the foundation of the system, providing:

- Resource state tracking (`UNINITIALIZED`, `INITIALIZING`, `READY`, `DEGRADED`, `ERROR`, `CLOSING`, `CLOSED`)
- Concurrency safety with locks to prevent race conditions
- Error tracking and handling
- Health check mechanism
- Automatic resource cleanup

**Example:**

```python
class DatabaseConnection(AsyncResource):
    def __init__(self, db_url):
        super().__init__(resource_id=f"db-{db_url}")
        self._db_url = db_url
        self._connection = None
        
    async def _do_initialize(self):
        # Resource-specific initialization
        self._connection = await connect_to_db(self._db_url)
        
    async def _do_cleanup(self):
        # Resource-specific cleanup
        if self._connection:
            await self._connection.close()
            self._connection = None
            
    async def _check_health(self):
        # Resource-specific health check
        if not self._connection:
            return False
        try:
            await self._connection.ping()
            return True
        except Exception:
            return False
            
    async def query(self, sql, *args):
        # Business logic using the resource
        if not self.is_initialized:
            await self.initialize()
        return await self._connection.execute(sql, *args)
```

### AsyncManagedResource

The `AsyncManagedResource` extends `AsyncResource` with:

- Retry logic for initialization and operations
- Automatic health checks
- Exponential backoff for retries
- Recovery from transient errors

**Example:**

```python
class ResilientDatabaseConnection(AsyncManagedResource):
    def __init__(self, db_url):
        super().__init__(
            resource_id=f"db-{db_url}",
            max_retries=3,
            retry_delay=1.0,
            health_check_interval=60.0
        )
        self._db_url = db_url
        self._connection = None
        
    # ... implementation similar to DatabaseConnection ...
    
    async def query(self, sql, *args):
        # Use built-in retry logic
        return await self.with_retry(
            lambda: self._connection.execute(sql, *args),
            operation_name="query",
            max_retries=2,
            timeout=10.0
        )
```

### NetworkResource

The `NetworkResource` specializes `AsyncManagedResource` for network connections:

- Connection state tracking
- Network-specific health checks
- Reconnection logic for network failures
- Connection statistics for monitoring

**Example:**

```python
class APIClient(NetworkResource):
    def __init__(self, api_url):
        super().__init__(
            address=api_url,
            max_retries=5,
            retry_delay=1.0,
            reconnect_on_error=True
        )
        self._api_url = api_url
        self._session = None
        
    async def _do_initialize(self):
        self._session = aiohttp.ClientSession()
        
    async def _do_cleanup(self):
        if self._session:
            await self._session.close()
            self._session = None
            
    async def fetch_data(self, endpoint, params=None):
        return await self.with_connection(
            lambda: self._session.get(f"{self._api_url}/{endpoint}", params=params),
            operation_name=f"fetch_{endpoint}",
            timeout=30.0
        )
```

### AsyncResourcePool

The `AsyncResourcePool` manages a collection of similar resources:

- Efficient resource acquisition and release
- Automatic resource creation and disposal
- Resource health monitoring
- Idle resource cleanup
- Waiting for available resources with timeout

**Example:**

```python
# Create a pool of database connections
db_pool = AsyncResourcePool(
    factory=lambda: DatabaseConnection("postgres://localhost/mydb"),
    max_size=10,
    min_size=2,
    max_idle_time=300.0,  # 5 minutes
    health_check_interval=60.0  # 1 minute
)

# Start the pool
await db_pool.start()

# Use a connection from the pool
async with db_pool.acquire() as db:
    result = await db.query("SELECT * FROM users")
    
# Stop the pool when done
await db_pool.stop()
```

### AsyncResourceGroup

The `AsyncResourceGroup` manages a collection of different resources that should be initialized and cleaned up as a unit:

- Manages dependencies between resources
- Parallel or sequential initialization
- Centralized cleanup
- Health checks for all resources

**Example:**

```python
# Create a resource group
resources = AsyncResourceGroup("user_service")

# Add resources to the group
await resources.add_resource("db", DatabaseConnection("postgres://localhost/users"))
await resources.add_resource("cache", RedisConnection("redis://localhost:6379"))
await resources.add_resource("api", APIClient("https://api.example.com"))

# Initialize all resources (parallel by default)
await resources.initialize()

# Use resources from the group
db = await resources.get_resource("db")
cache = await resources.get_resource("cache")
api = await resources.get_resource("api")

# Later, clean up all resources
await resources.cleanup()
```

### Context Managers and Utilities

The system also provides context managers and utility functions:

- `AsyncResourceContext`: Context manager for automatic initialization and cleanup
- `with_timeout`: Run async operations with timeouts and custom error messages
- `with_resource`: Decorator for functions that need resource management

**Examples:**

```python
# Using AsyncResourceContext
async with AsyncResourceContext(DatabaseConnection("postgres://localhost/mydb")) as db:
    await db.query("SELECT * FROM users")
    
# Using with_timeout
result = await with_timeout(
    fetch_large_dataset(),
    timeout=60.0,
    message="Dataset fetch timed out after 60 seconds"
)

# Using with_resource decorator
@with_resource(lambda: DatabaseConnection("postgres://localhost/mydb"))
async def get_user(db, user_id):
    return await db.query("SELECT * FROM users WHERE id = ?", user_id)
```

## Best Practices

### 1. Choose the Right Base Class

- Use `AsyncResource` for simple resources with minimal error handling needs
- Use `AsyncManagedResource` for resources that need retry logic and health checks
- Use `NetworkResource` for any network-based resources

### 2. Properly Implement Abstract Methods

Always implement these methods in your resource classes:

- `_do_initialize()`: Resource-specific initialization
- `_do_cleanup()`: Resource-specific cleanup
- `_check_health()`: Resource-specific health check (optional for `AsyncResource`)

### 3. Use Context Managers

Prefer using context managers to ensure proper resource cleanup:

```python
# Good: Using context manager
async with AsyncResourceContext(resource) as r:
    await r.do_something()
    
# Better: Using pool's context manager
async with resource_pool.acquire() as r:
    await r.do_something()
    
# Avoid: Manual initialization/cleanup
resource = MyResource()
await resource.initialize()
try:
    await resource.do_something()
finally:
    await resource.cleanup()
```

### 4. Handle Resource States

Check resource states before performing operations:

```python
if not resource.is_initialized:
    await resource.initialize()
    
if not resource.is_usable:
    logger.warning("Resource is in a degraded or error state")
```

### 5. Configure Retry Parameters Appropriately

Set appropriate retry parameters based on the operation:

- Critical operations: More retries, longer max delay
- User-facing operations: Fewer retries, shorter delays
- Background operations: More retries, longer delays

```python
# User-facing operation with limited retries
result = await resource.with_retry(
    operation,
    max_retries=2,
    retry_delay=0.5,
    timeout=5.0
)

# Background operation with more patience
result = await resource.with_retry(
    operation,
    max_retries=10,
    retry_delay=1.0,
    max_retry_delay=30.0,
    timeout=120.0
)
```

### 6. Use Resource Pooling for Performance

Use `AsyncResourcePool` when:

- Creating resources is expensive
- You need to limit the number of concurrent resources
- You want automatic health checks and resource replacement
- You need to handle peak loads efficiently

### 7. Group Related Resources

Use `AsyncResourceGroup` when:

- Multiple resources need to be managed together
- Resources have dependencies on each other
- You want centralized initialization and cleanup

## Real-World Examples

### GRPC Connection Management

In the exo codebase, the `GRPCPeerHandle` class uses the async resource management system to handle gRPC connections:

```python
class GRPCPeerHandle(PeerHandle):
    def __init__(self, _id, address, desc, device_capabilities):
        self._id = _id
        self.address = address
        self.desc = desc
        self._device_capabilities = device_capabilities
        
        # Create the channel resource
        self._channel_resource = GRPCChannelResource(
            address=address,
            resource_id=f"peer-{_id}",
            options=[...],
            compression=grpc.Compression.Gzip,
            max_retries=5,
            retry_delay=1.0,
            max_retry_delay=30.0,
            health_check_interval=60.0,
        )
        self.stub = None

    async def connect(self):
        await self._channel_resource.initialize()
        if self._channel_resource.channel:
            self.stub = node_service_pb2_grpc.NodeServiceStub(self._channel_resource.channel)
        else:
            raise ConnectionError(f"Failed to create channel for {self._id}@{self.address}")

    async def send_tensor(self, shard, tensor, inference_state=None, request_id=None):
        request = node_service_pb2.TensorRequest(...)
        
        # Use the channel resource to handle the connection and retries
        response = await self._channel_resource.call_unary(
            lambda: self.stub.SendTensor(request),
            timeout=30
        )
        
        # Process response...
```

### Database Connection Pool

Here's how a database connection pool might be implemented:

```python
class DatabasePool:
    def __init__(self, db_url, max_connections=10):
        self._pool = AsyncResourcePool(
            factory=lambda: DatabaseConnection(db_url),
            max_size=max_connections,
            min_size=2,
            max_idle_time=300.0,
            health_check_interval=60.0
        )
        
    async def start(self):
        await self._pool.start()
        
    async def stop(self):
        await self._pool.stop()
        
    @asynccontextmanager
    async def connection(self):
        async with self._pool.acquire() as conn:
            yield conn
            
    async def execute(self, query, *args):
        async with self.connection() as conn:
            return await conn.query(query, *args)
```

## Migration Guide

### Before (without resource management)

```python
class OldConnection:
    def __init__(self, address):
        self.address = address
        self.channel = None
        self.stub = None
        
    async def connect(self):
        try:
            self.channel = grpc.aio.insecure_channel(self.address)
            self.stub = ServiceStub(self.channel)
            await asyncio.wait_for(self.channel.channel_ready(), timeout=10.0)
        except Exception as e:
            print(f"Failed to connect: {e}")
            if self.channel:
                await self.channel.close()
            self.channel = None
            self.stub = None
            raise
            
    async def call_service(self, request):
        if not self.channel:
            await self.connect()
        try:
            return await self.stub.Method(request)
        except grpc.RpcError as e:
            print(f"RPC error: {e}")
            await self.channel.close()
            self.channel = None
            self.stub = None
            raise
```

### After (with resource management)

```python
class NewConnection(NetworkResource):
    def __init__(self, address):
        super().__init__(
            address=address,
            max_retries=3,
            retry_delay=1.0,
            reconnect_on_error=True
        )
        self._stub = None
        
    async def _do_initialize(self):
        self._channel = grpc.aio.insecure_channel(self.address)
        self._stub = ServiceStub(self._channel)
        await self._channel.channel_ready()
        
    async def _do_cleanup(self):
        if self._channel:
            await self._channel.close()
            self._channel = None
        self._stub = None
        
    async def _check_health(self):
        if not self._channel:
            return False
        return self._channel.get_state() == grpc.ChannelConnectivity.READY
        
    async def call_service(self, request):
        return await self.with_connection(
            lambda: self._stub.Method(request),
            operation_name="call_service",
            timeout=30.0
        )
```

## Conclusion

The async resource management system in exo provides a comprehensive solution for managing asynchronous resources. By using this system, you can write more robust and maintainable code that properly handles resource lifecycle, errors, and cleanup.

For specific implementations, see:
- `exo/utils/async_resources.py`: Core resource management classes
- `exo/utils/grpc_resources.py`: gRPC-specific resource management
- `exo/networking/grpc/grpc_peer_handle.py`: Example usage in the peer handle
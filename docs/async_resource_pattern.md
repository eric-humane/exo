# AsyncResource Pattern in exo

## Overview

The AsyncResource pattern provides a structured approach to managing the lifecycle of asynchronous resources in the exo distributed system. This pattern ensures proper initialization, cleanup, and error handling for critical system resources like network connections, inference engines, and discovery services.

## Key Features

- **Lifecycle Management**: Standardized initialize/cleanup lifecycle
- **State Tracking**: Resources track their state (UNINITIALIZED, READY, DEGRADED, etc.)
- **Error Handling**: Automatic tracking of errors and failure states
- **Health Checking**: Built-in health checking and degradation tracking
- **Resource Groups**: Management of related resources as a unit
- **Monitoring**: System-wide resource monitoring and diagnostics

## Core Classes

### AsyncResource

Base class for all async resources with lifecycle management:

```python
class AsyncResource(ABC):
    def __init__(self, resource_id: Optional[str] = None, initialize_timeout: float = 30.0):
        # ...
        
    async def initialize(self) -> bool:
        # Initialize the resource with proper error handling
        
    async def cleanup(self) -> None:
        # Clean up the resource, releasing acquired resources
        
    async def check_health(self) -> bool:
        # Check if the resource is healthy
        
    # Abstract methods that must be implemented by subclasses
    async def _do_initialize(self) -> None:
        raise NotImplementedError
        
    async def _do_cleanup(self) -> None:
        raise NotImplementedError
        
    async def _check_health(self) -> bool:
        return True
```

### NetworkResource

Specialized resource for network connections:

```python
class NetworkResource(AsyncManagedResource):
    def __init__(self, resource_id: Optional[str] = None, address: str = "", ...):
        # ...
        
    async def reconnect(self) -> bool:
        # Force a reconnection
        
    async def with_connection(self, operation, ...):
        # Execute an operation with the connection, handling reconnection if needed
```

### AsyncResourceGroup

Manages a group of related resources:

```python
class AsyncResourceGroup:
    def __init__(self, group_id: Optional[str] = None):
        # ...
        
    async def add_resource(self, name: str, resource: AsyncResource, initialize: bool = False):
        # Add a resource to the group
        
    async def initialize(self, parallel: bool = True):
        # Initialize all resources in the group
        
    async def cleanup(self, parallel: bool = True):
        # Clean up all resources in the group
```

## Resource States

Resources can be in the following states:

- **UNINITIALIZED**: Resource is created but not initialized
- **INITIALIZING**: Resource initialization is in progress
- **READY**: Resource is initialized and ready to use
- **DEGRADED**: Resource is initialized but operating in a degraded state
- **ERROR**: Resource is in an error state
- **CLOSING**: Resource cleanup is in progress
- **CLOSED**: Resource is cleaned up and no longer usable

## Usage Examples

### Basic Usage

```python
# Create a resource
resource = MyNetworkResource(resource_id="db_connection")

# Initialize with error handling
try:
    await resource.initialize()
    # Use the resource...
finally:
    # Always clean up
    await resource.cleanup()
```

### With Resource Group

```python
# Create a resource group
group = AsyncResourceGroup("node_resources")

# Add resources to the group
await group.add_resource("discovery", UDPDiscovery(node_id))
await group.add_resource("server", Server(node_id, "localhost", 0))

# Initialize all resources in parallel
await group.initialize(parallel=True)

try:
    # Use the resources...
    discovery = await group.get_resource("discovery")
    server = await group.get_resource("server")
finally:
    # Clean up all resources
    await group.cleanup()
```

### Ensuring Resource Readiness

When using AsyncResource-based resources:

```python
# Using ensure_ready before operations
async def send_data(peer_handle, data):
    # This ensures the resource is initialized and healthy before use
    await peer_handle.ensure_ready()
    
    # Now use the resource
    await peer_handle.send_data(data)
```

## Integration in exo Components

The AsyncResource pattern is used throughout key exo components:

### PeerHandle

All peer connections implement AsyncResource:

```python
class PeerHandle(AsyncResource, ABC):
    # Resource type identifier for the AsyncResource system
    RESOURCE_TYPE: ClassVar[str] = "peer_handle"
    
    async def _do_initialize(self) -> None:
        """Initialize the peer connection."""
        await self.connect()
    
    async def _do_cleanup(self) -> None:
        """Clean up the peer connection."""
        await self.disconnect()
    
    async def _do_health_check(self) -> bool:
        """Check the health of the peer connection."""
        return await self.health_check()
```

### GRPCPeerHandle

GRPC-specific implementation:

```python
class GRPCPeerHandle(PeerHandle):
    def __init__(self, _id: str, address: str, ...):
        # Initialize the AsyncResource base class
        super().__init__(
          resource_id=f"peer-{_id}",
          resource_type=self.RESOURCE_TYPE,
          display_name=f"Peer {_id} ({address})"
        )
        # ...
    
    async def send_prompt(self, shard: Shard, prompt: str, ...):
        # Ensure we're initialized
        await self.ensure_ready()
        # Proceed with operation...
```

## Resource Monitoring

The exo framework includes tools for monitoring and diagnosing resource issues:

```python
# Start a resource monitor
monitor = ResourceMonitor(check_interval=30.0)
await monitor.start()

# Get the latest stats
stats = monitor.get_latest_stats()
problem_resources = monitor.get_problem_resources()

# Get a detailed health report
report = await monitor.get_detailed_report()
print(report)

# Stop the monitor
await monitor.stop()
```

You can also use the resource_cli.py script to diagnose issues:

```bash
# List all resources
python -m exo.utils.resource_cli list

# Run a health check
python -m exo.utils.resource_cli health

# Reset a specific resource
python -m exo.utils.resource_cli reset peer-abc123 --type peer_handle

# Export resource data to JSON
python -m exo.utils.resource_cli export --output resources.json
```

## Best Practices

1. **Always use ensure_ready()**:
   - Call `await resource.ensure_ready()` before using a resource
   - This ensures the resource is initialized and healthy

2. **Group related resources**:
   - Use AsyncResourceGroup to manage related resources
   - This ensures proper initialization and cleanup order

3. **Handle cleanup properly**:
   - Always call cleanup in a finally block
   - Or use AsyncResourceContext for automatic cleanup

4. **Monitor resource health**:
   - Use ResourceMonitor to track resource health
   - Respond to degraded or failed resources

5. **Use the pattern consistently**:
   - Apply the pattern to all resources that need lifecycle management
   - Use the same patterns for error handling and state tracking
# AsyncResource Standardization in exo

## Overview

This directory contains documentation and guides for the AsyncResource standardization effort in the exo project. The AsyncResource pattern provides a consistent, reliable approach to managing resource lifecycles, error handling, and health monitoring in our distributed system.

## Documentation Files

- [**async_resource_pattern.md**](async_resource_pattern.md): Comprehensive guide to the AsyncResource pattern in exo
- [**async_resource_standardization.md**](async_resource_standardization.md): Plan for standardizing AsyncResource usage across the codebase
- [**async_resource_action_plan.md**](async_resource_action_plan.md): Specific action items for implementing AsyncResource consistently

## Example Code

Check out these examples of proper AsyncResource usage:

- [`/examples/async_resource_example.py`](../examples/async_resource_example.py): Example showing how to convert a class to use AsyncResource
- [`/examples/resource_management.py`](../examples/resource_management.py): Example of managing resources with AsyncResource in a Node

## Key Principles

1. **Consistent Lifecycle Management**: All resources should have clear initialization and cleanup phases
2. **State Tracking**: Resources should track their state (UNINITIALIZED, READY, DEGRADED, etc.)
3. **Resource Readiness**: Operations should check resource readiness with `ensure_ready()`
4. **Error Handling**: Resources should handle errors consistently and provide recovery mechanisms
5. **Health Monitoring**: Resources should implement health checks and report degraded states

## Implementation Status

The AsyncResource pattern is being systematically implemented across the exo codebase. Current status:

| Component | Status | Notes |
|-----------|--------|-------|
| PeerHandle | ✅ Complete | Base class and GRPC implementation |
| InferenceEngine | 🔄 In Progress | Base class needs updating |
| Discovery | 🔄 In Progress | UDPDiscovery needs updates |
| Server | 🔄 In Progress | Base class needs updating |
| Node | 🔄 In Progress | Needs AsyncResource integration |

## Contribution Guidelines

When contributing to exo, please follow these guidelines for AsyncResource usage:

1. **Use AsyncResource base class** for any component that manages resources with lifecycles
2. **Implement required methods**:
   - `_do_initialize()`: Resource-specific initialization
   - `_do_cleanup()`: Resource-specific cleanup
   - `_do_health_check()`: Resource-specific health check
3. **Call ensure_ready()** at the beginning of public methods that require the resource to be initialized
4. **Maintain backward compatibility** with existing interfaces when possible
5. **Add proper error handling** using AsyncResource's state management
6. **Use ResourceGroups** for managing related resources

## Integration with Existing Code

For classes that already have start/stop or other lifecycle methods:

1. Keep existing methods for backward compatibility
2. Delegate to AsyncResource methods internally
3. Update internal code to use the AsyncResource pattern

Example:

```python
async def start(self):
    """Legacy method for backward compatibility."""
    await self.initialize()
    
async def stop(self):
    """Legacy method for backward compatibility."""
    await self.cleanup()
```

## Testing AsyncResource Implementations

When testing components that use AsyncResource:

1. Test initialization and cleanup paths
2. Test state transitions
3. Test error recovery
4. Test health checks
5. Test resource degradation and recovery

## References

- [AsyncResource Base Class](../exo/utils/async_resources.py)
- [Resource Monitor](../exo/utils/resource_monitor.py)
- [Resource CLI](../exo/utils/resource_cli.py)
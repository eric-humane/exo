# Async Resource Management Examples

This directory contains examples demonstrating the use of the async resource management system in exo.

## Overview

The async resource management system in exo provides a structured way to handle lifecycle management, error handling, and cleanup for asynchronous resources like network connections, file handles, and other external resources.

## Examples

### Before/After Comparison

The `before_after_comparison.py` file demonstrates how the async resource management system improves code organization, error handling, and reliability for network resources.

It includes:
1. A traditional gRPC client implementation with manual connection management and error handling
2. An improved gRPC client implementation using the AsyncResource system
3. A performance comparison between the two approaches

### Running the Example

To run the before/after comparison:

```bash
python examples/async_resources/before_after_comparison.py
```

This will simulate a series of gRPC calls with both client implementations and compare:
- Success rate
- Performance (calls per second)
- Connection attempt count
- Code maintainability

## Key Benefits Demonstrated

The examples showcase several key benefits of the async resource management system:

1. **Structured lifecycle management**: Clearly defined resource states and transitions
2. **Automatic error handling and recovery**: Built-in retry logic and graceful error handling
3. **Resource pooling and reuse**: Efficient management of expensive resources
4. **Centralized cleanup**: Guaranteed resource cleanup, even in error cases
5. **Proper health monitoring**: Automatic health checks and degraded state handling
6. **Code organization**: More maintainable and readable code

## Further Reading

For more information about the async resource management system, see:
- [Async Resource Management Documentation](../../docs/async_resource_management.md)
- `exo/utils/async_resources.py`: Core resource management classes
- `exo/utils/grpc_resources.py`: gRPC-specific resource management
- `exo/networking/grpc/grpc_peer_handle.py`: Example usage in the peer handle
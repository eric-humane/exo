# Standardizing AsyncResource Usage in exo

## Overview

This document outlines our commitment to standardize resource management across the exo codebase using the AsyncResource pattern. Consistent application of this pattern will improve reliability, error handling, and resource lifecycle management throughout the system.

## Components that Need AsyncResource Implementation

Based on our analysis, the following components should implement the AsyncResource pattern:

### 1. Server Classes
- **Current status**: Uses basic `start()/stop()` without standardized state tracking
- **Components to update**:
  - `exo/networking/server.py` (base interface)
  - `exo/networking/grpc/grpc_server.py` (implementation)

### 2. Discovery Classes
- **Current status**: Has lifecycle methods but lacks structured state management
- **Components to update**:
  - `exo/networking/discovery.py` (base interface)
  - `exo/networking/udp/udp_discovery.py` (implementation)
  - `exo/networking/tailscale/tailscale_discovery.py` (implementation)
  - `exo/networking/manual/manual_discovery.py` (implementation)

### 3. Inference Engine Classes
- **Current status**: Has cleanup methods but lacks standardized initialization/state tracking
- **Components to update**:
  - `exo/inference/inference_engine.py` (base class)
  - `exo/inference/mlx/sharded_inference_engine.py` (implementation)
  - `exo/inference/tinygrad/inference.py` (implementation)
  - `exo/inference/dummy_inference_engine.py` (implementation)

### 4. Connection Pool Classes
- **Current status**: Custom implementation without AsyncResourcePool
- **Components to update**:
  - `exo/networking/connection_pool.py`
  - `exo/networking/grpc/grpc_connection_pool.py`

### 5. ShardDownloader Classes
- **Current status**: Custom lifecycle management
- **Components to update**:
  - `exo/download/shard_download.py`
  - `exo/download/new_shard_download.py`

### 6. Peer Handles (Already Updated)
- **Current status**: ✅ Successfully implements AsyncResource pattern
- `exo/networking/peer_handle.py`
- `exo/networking/grpc/grpc_peer_handle.py`

## Implementation Guide

When implementing AsyncResource across these components, follow these guidelines:

### 1. Class Structure

```python
class MyComponent(AsyncResource):
    RESOURCE_TYPE: ClassVar[str] = "my_component"
    
    def __init__(self, resource_id=None, ...):
        super().__init__(
            resource_id=resource_id or str(uuid.uuid4()),
            resource_type=self.RESOURCE_TYPE,
            display_name=f"MyComponent({resource_id})"
        )
        # Component-specific initialization
        self._other_fields = ...
        
    # Implement required AsyncResource methods
    async def _do_initialize(self) -> None:
        # Component-specific initialization logic
        # e.g., open connection, initialize storage, etc.
        
    async def _do_cleanup(self) -> None:
        # Component-specific cleanup logic
        # e.g., close connection, release resources, etc.
        
    async def _do_health_check(self) -> bool:
        # Component-specific health check
        # Return True if healthy, False otherwise
        
    # Public API methods should use ensure_ready
    async def perform_operation(self, ...):
        await self.ensure_ready()
        # Perform the operation
```

### 2. Migrating Existing Classes

For existing classes with their own lifecycle methods:

1. **Keep backwards compatibility**:
   ```python
   async def start(self):
       """Legacy method for compatibility."""
       return await self.initialize()
   
   async def stop(self):
       """Legacy method for compatibility."""
       return await self.cleanup()
   ```

2. **Update internal implementation**:
   - Replace direct calls to `start()/stop()` with `initialize()/cleanup()`
   - Add proper state tracking using AsyncResource states
   - Ensure health checks are implemented

### 3. Ensuring Resource Readiness

All methods that use a resource should ensure it's ready:

```python
async def some_method(self):
    # Ensure the resource is initialized and healthy
    await self.ensure_ready()
    
    # Now perform the operation
    # ...
```

## Example: Updating Server Class

Here's how to update the Server base class:

```python
from abc import ABC, abstractmethod
from typing import ClassVar, Optional
import uuid

from exo.utils.async_resources import AsyncResource

class Server(AsyncResource, ABC):
    """
    Base class for server implementations.
    
    Implements AsyncResource pattern for consistent lifecycle management.
    """
    RESOURCE_TYPE: ClassVar[str] = "server"
    
    def __init__(self, server_id: str, host: str, port: int):
        super().__init__(
            resource_id=f"server-{server_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Server {server_id} ({host}:{port})"
        )
        self._id = server_id
        self._host = host
        self._port = port
        self._real_port = port
        
    def id(self) -> str:
        return self._id
        
    def host(self) -> str:
        return self._host
        
    def port(self) -> int:
        return self._real_port
        
    # Legacy methods for backwards compatibility
    
    async def start(self) -> None:
        """Legacy method for backward compatibility."""
        await self.initialize()
        
    async def stop(self) -> None:
        """Legacy method for backward compatibility."""
        await self.cleanup()
        
    # AsyncResource implementation
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """Start the server, binding to the specified host and port."""
        pass
        
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """Stop the server, releasing all resources."""
        pass
        
    async def _do_health_check(self) -> bool:
        """Check if the server is healthy and responsive."""
        # Default implementation
        return True
```

## Example: Updating InferenceEngine Class

Here's how to update the InferenceEngine base class:

```python
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Dict, Any
import uuid

from exo.utils.async_resources import AsyncResource

class InferenceEngine(AsyncResource, ABC):
    """
    Base class for inference engine implementations.
    
    Implements AsyncResource pattern for consistent lifecycle management.
    """
    RESOURCE_TYPE: ClassVar[str] = "inference_engine"
    
    def __init__(self, engine_id: Optional[str] = None):
        super().__init__(
            resource_id=engine_id or str(uuid.uuid4()),
            resource_type=self.RESOURCE_TYPE,
            display_name=f"InferenceEngine({engine_id or 'unknown'})"
        )
    
    # Legacy method for backward compatibility
    async def cleanup(self) -> None:
        """Clean up resources used by this inference engine."""
        await super().cleanup()
    
    # AsyncResource implementation
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """Initialize the inference engine."""
        pass
    
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """Clean up resources used by this inference engine."""
        pass
    
    async def _do_health_check(self) -> bool:
        """Check if the inference engine is healthy."""
        # Default implementation checks basic functionality
        return True
    
    # Public inference methods should use ensure_ready
    
    async def infer_prompt(self, request_id: str, shard: Any, prompt: str,
                           inference_state: Optional[Dict[str, Any]] = None):
        """Run inference on a prompt."""
        await self.ensure_ready()
        return await self._infer_prompt(request_id, shard, prompt, inference_state)
    
    @abstractmethod
    async def _infer_prompt(self, request_id: str, shard: Any, prompt: str,
                            inference_state: Optional[Dict[str, Any]] = None):
        """Implementation-specific prompt inference."""
        pass
```

## Implementation Timeline

To ensure a consistent approach, we should implement these changes in the following order:

1. Base interfaces (Server, Discovery, InferenceEngine)
2. Common implementations 
3. Specialized components

## Testing Approach

When updating each component:

1. Add unit tests specifically for AsyncResource functionality:
   - Test initialization and cleanup
   - Test state transitions
   - Test error handling
   - Test health checking

2. Add integration tests for component interactions:
   - Test resource groups
   - Test health monitoring 
   - Test recovery from degraded states

## Benefits

Standardizing around AsyncResource will provide:

1. **Consistent lifecycle management** across all components
2. **Improved error handling** and recovery
3. **Automatic health monitoring** 
4. **Better observability** through standardized resource tracking
5. **Reduced resource leaks** thanks to structured cleanup
6. **Simplified code** with standardized patterns

## Conclusion

The AsyncResource pattern should be our standard approach for all components that manage resources with lifecycles. This includes network connections, services, engines, and stateful components. Consistently applying this pattern will improve the reliability and maintainability of the exo codebase.
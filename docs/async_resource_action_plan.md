# AsyncResource Implementation Action Plan

This document outlines specific actions needed to ensure consistent AsyncResource pattern usage across the exo codebase, based on our audit findings.

## High Priority Tasks

### 1. Update InferenceEngine Base Class

**File**: `exo/inference/inference_engine.py`

**Changes required**:
- Make InferenceEngine inherit from AsyncResource
- Implement `_do_initialize`, `_do_cleanup`, and `_do_health_check` methods
- Add ensure_ready() calls to all public methods
- Maintain backward compatibility by keeping existing cleanup methods

```python
class InferenceEngine(AsyncResource, ABC):
    RESOURCE_TYPE: ClassVar[str] = "inference_engine"
    
    # Make sure existing interface works while adding AsyncResource pattern
    async def cleanup(self) -> None:
        """Legacy method for backward compatibility."""
        await super().cleanup()
        
    async def _do_initialize(self) -> None:
        """Initialize the inference engine."""
        # Default implementation for backward compatibility
        pass  
    
    async def _do_cleanup(self) -> None:
        """Implementation-specific cleanup."""
        pass
        
    # Update public methods to use ensure_ready()
    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, 
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        await self.ensure_ready()
        return await self._infer_prompt(request_id, shard, prompt, inference_state)
        
    @abstractmethod
    async def _infer_prompt(self, request_id: str, shard: Shard, prompt: str,
                           inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Implementation-specific infer_prompt logic."""
        pass
```

### 2. Fix Node Class

**File**: `exo/orchestration/node.py`

**Changes required**:
- Make Node inherit from AsyncResource
- Convert start/stop methods to use initialize/cleanup
- Add ensure_ready() checks to key operations
- Implement proper state management

```python
class Node(AsyncResource):
    RESOURCE_TYPE: ClassVar[str] = "node"
    
    def __init__(self, _id: str, server: Server, inference_engine: InferenceEngine, 
                discovery: Discovery, shard_downloader: ShardDownloader, 
                partitioning_strategy: PartitioningStrategy = None, ...):
        super().__init__(
            resource_id=f"node-{_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Node {_id}"
        )
        self.id = _id
        # ... existing initialization ...
        
    # Legacy methods for backward compatibility
    async def start(self, wait_for_peers: int = 0) -> None:
        """Legacy method for backward compatibility."""
        await self.initialize()
        if wait_for_peers > 0:
            await self.update_peers(wait_for_peers)
            
    async def stop(self) -> None:
        """Legacy method for backward compatibility."""
        await self.cleanup()
        
    # AsyncResource implementation
    async def _do_initialize(self) -> None:
        """Initialize the node."""
        self.device_capabilities = await device_capabilities()
        await self.server.initialize()
        await self.discovery.initialize()
        await self.update_peers(0)
        await self.collect_topology(set())
        
        # Start the periodic topology collection as a managed task
        self._task_manager.start_task(
            "periodic_topology_collection", 
            lambda: self.periodic_topology_collection(2.0),
            priority=TaskPriority.HIGH
        )
        
    async def _do_cleanup(self) -> None:
        """Clean up node resources."""
        # Cancel all managed tasks
        try:
            if hasattr(self, '_task_manager'):
                num_cancelled = await self._task_manager.cancel_all_tasks(wait=True)
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error cancelling managed tasks: {e}")
                
        # Clean up resources
        cleanup_tasks = []
        
        # Clean up peer handles
        for peer in self.peers:
            cleanup_tasks.append(peer.cleanup())
            
        # Clean up inference engine
        if self.inference_engine:
            cleanup_tasks.append(self.inference_engine.cleanup())
            
        # Clean up discovery and server
        cleanup_tasks.append(self.discovery.cleanup())
        cleanup_tasks.append(self.server.cleanup())
        
        # Run all cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        # Clear buffers
        self.buffered_token_output.clear()
        self.buffered_logits.clear()
        self.buffered_inputs.clear()
        self.buffered_partials.clear()
        self.node_download_progress.clear()
        
    async def _do_health_check(self) -> bool:
        """Check the health of the node."""
        # Check if server and discovery are healthy
        server_healthy = await self.server.check_health() if hasattr(self.server, 'check_health') else True
        discovery_healthy = await self.discovery.check_health() if hasattr(self.discovery, 'check_health') else True
        
        return server_healthy and discovery_healthy
    
    # Public methods should use ensure_ready()
    
    async def process_prompt(self, base_shard: Shard, prompt: str, 
                             request_id: Optional[str] = None, 
                             inference_state: Optional[dict] = {}) -> Optional[np.ndarray]:
        await self.ensure_ready()
        # ... existing implementation ...
```

### 3. Add AsyncResource to Core Components

**File**: `exo/networking/discovery.py` 

**Changes required**:
- Make Discovery base class inherit from AsyncResource
- Add lifecycle methods and state management
- Update UDPDiscovery to use AsyncResource pattern

```python
class Discovery(AsyncResource, ABC):
    RESOURCE_TYPE: ClassVar[str] = "discovery"
    
    def __init__(self, node_id: str):
        super().__init__(
            resource_id=f"discovery-{node_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Discovery for node {node_id}"
        )
        self._node_id = node_id
        
    # Legacy methods for backward compatibility
    async def start(self) -> None:
        """Legacy method for backward compatibility."""
        await self.initialize()
        
    async def stop(self) -> None:
        """Legacy method for backward compatibility."""
        await self.cleanup()
        
    # Public methods should use ensure_ready
    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """Discover peers on the network."""
        await self.ensure_ready()
        return await self._discover_peers(wait_for_peers)
        
    @abstractmethod
    async def _discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """Implementation-specific peer discovery logic."""
        pass
```

**File**: `exo/networking/server.py`

**Changes required**:
- Make Server inherit from AsyncResource
- Implement required lifecycle methods
- Update implementations to follow the pattern

```python
class Server(AsyncResource, ABC):
    RESOURCE_TYPE: ClassVar[str] = "server"
    
    def __init__(self, node_id: str, host: str, port: int):
        super().__init__(
            resource_id=f"server-{node_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Server for node {node_id}"
        )
        self._node_id = node_id
        self._host = host
        self._port = port
        self._real_port = port
        
    # Legacy methods for backward compatibility
    async def start(self) -> None:
        """Legacy method for backward compatibility."""
        await self.initialize()
        
    async def stop(self) -> None:
        """Legacy method for backward compatibility."""
        await self.cleanup()
```

## Medium Priority Tasks

### 1. Update UDPDiscovery Implementation

**File**: `exo/networking/udp/udp_discovery.py`

**Changes required**:
- Update UDPDiscovery to use AsyncResource properly
- Convert context manager to AsyncResource
- Add ensure_ready() checks to operations

```python
class UDPDiscovery(Discovery):
    def __init__(self, node_id: str, port: int = 50000, ...):
        super().__init__(node_id)
        self._port = port
        # ... other initialization ...
        
    # AsyncResource implementation
    async def _do_initialize(self) -> None:
        """Initialize UDP discovery."""
        # ... existing start() code ...
        
    async def _do_cleanup(self) -> None:
        """Clean up UDP discovery."""
        # ... existing stop() code ...
        
    async def _do_health_check(self) -> bool:
        """Check health of UDP discovery."""
        return self._server is not None and not self._server.closed
        
    # Implementation-specific methods
    async def _discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """Discover peers via UDP broadcast."""
        # ... existing implementation ...
```

### 2. Standardize Health Checks

Create a consistent health checking approach across components:

1. Create a utility in `exo/utils/health_check.py` for standardized health monitoring
2. Implement proper health checks in each component
3. Add the ability to monitor health system-wide

### 3. Consistent Error Handling

**Changes required**:
- Create error handling utilities in `exo/utils/error_handling.py`
- Standardize error states and recovery
- Implement graceful degradation when possible

## Low Priority Tasks

### 1. Resource Groups Management

**Changes required**:
- Implement component-specific resource groups
- Add proper dependency tracking between resources
- Ensure ordered initialization and cleanup

### 2. Observability and Metrics

**Changes required**:
- Add detailed resource metrics
- Implement a monitoring dashboard for resources
- Add structured logging for resource lifecycle events

### 3. Testing Improvements

**Changes required**:
- Add specific tests for AsyncResource lifecycle in each component
- Test error handling and recovery
- Add stress tests for resource management

## Implementation Timeline

### Sprint 1 (Week 1-2)
- Update InferenceEngine base class
- Create conversion guide and examples
- Fix Node class implementation

### Sprint 2 (Week 3-4)
- Update Discovery base class
- Update Server base class
- Begin updating UDPDiscovery implementation

### Sprint 3 (Week 5-6)
- Update remaining core components
- Implement consistent health checks
- Add resource groups for related components

### Sprint 4 (Week 7-8)
- Complete observability improvements
- Add comprehensive testing
- Finalize documentation

## Conclusion

By systematically implementing these changes, we'll achieve a consistent AsyncResource pattern across the exo codebase. This will improve resource management, error handling, and system reliability. The phased approach allows us to prioritize critical components while maintaining backward compatibility.

Each component should be updated following the principles in our AsyncResource pattern documentation, with special attention to proper state management, error handling, and ensuring resources are ready before use.
from abc import ABC, abstractmethod
from typing import List, ClassVar, Optional
import uuid
from .peer_handle import PeerHandle
from exo.utils.async_resources import AsyncResource, ResourceState


class Discovery(AsyncResource, ABC):
    """
    Base class for discovery mechanisms with proper AsyncResource lifecycle management.
    
    This class provides:
    1. Standardized lifecycle management (initialize, cleanup)
    2. State tracking and validation
    3. Error handling and health checks
    4. Ensuring the discovery service is ready before use
    """
    RESOURCE_TYPE: ClassVar[str] = "discovery"
    
    def __init__(self, node_id: str, discovery_id: Optional[str] = None):
        """
        Initialize the discovery service with AsyncResource base.
        
        Args:
            node_id: ID of the node this discovery service belongs to
            discovery_id: Optional unique identifier for this discovery instance
        """
        super().__init__(
            resource_id=discovery_id or f"discovery-{node_id}"
        )
        self._node_id = node_id
    
    # No legacy methods for 2.0
    
    # Abstract methods for peer discovery
    
    @abstractmethod
    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """
        Discover peers on the network.
        
        This method should:
        1. Ensure the discovery service is ready
        2. Find peers on the network
        3. Create PeerHandle instances for each peer
        4. Optionally wait for a minimum number of peers
        
        Args:
            wait_for_peers: Minimum number of peers to wait for (0 = don't wait)
            
        Returns:
            List of PeerHandle instances
        """
        # Ensure discovery is ready before use
        await self.ensure_ready()
        pass
    
    # AsyncResource implementation (abstract)
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        Implementation-specific initialization.
        
        Subclasses must implement this to initialize their discovery mechanism.
        """
        pass
    
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """
        Implementation-specific cleanup.
        
        Subclasses must implement this to clean up their discovery mechanism.
        """
        pass
    
    @abstractmethod
    async def _check_health(self) -> bool:
        """
        Implementation-specific health check.
        
        Subclasses must implement this to check if their discovery mechanism is healthy.
        
        Returns:
            True if the discovery mechanism is healthy, False otherwise
        """
        pass
        
    async def ensure_ready(self) -> None:
        """
        Ensure the discovery service is initialized and ready to use.
        
        This method initializes the service if needed and waits for it to be ready.
        """
        if not self.is_initialized:
            await self.initialize()

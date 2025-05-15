from abc import ABC, abstractmethod
from typing import ClassVar, Optional
import uuid
from exo.utils.async_resources import AsyncResource, ResourceState


class Server(AsyncResource, ABC):
    """
    Base class for server implementations with proper AsyncResource lifecycle management.
    
    This class provides:
    1. Standardized lifecycle management (initialize, cleanup)
    2. State tracking and validation
    3. Error handling and health checks
    4. Ensuring the server is ready before use
    """
    RESOURCE_TYPE: ClassVar[str] = "server"
    
    def __init__(self, node_id: str, host: str, port: int, server_id: Optional[str] = None):
        """
        Initialize the server with AsyncResource base.
        
        Args:
            node_id: ID of the node this server belongs to
            host: Hostname or IP address to bind to
            port: Port number to bind to
            server_id: Optional unique identifier for this server instance
        """
        super().__init__(
            resource_id=server_id or f"server-{node_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Server({node_id}@{host}:{port})"
        )
        self._node_id = node_id
        self._host = host
        self._port = port
        self._real_port = port  # May be updated if using port 0 (dynamic)
    
    # No legacy methods for 2.0
    
    # Abstract properties and methods
    
    @property
    def host(self) -> str:
        """Get the hostname or IP address the server is bound to."""
        return self._host
    
    @property
    def port(self) -> int:
        """Get the port number the server is bound to."""
        return self._real_port
    
    # AsyncResource implementation (abstract)
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        Implementation-specific initialization.
        
        Subclasses must implement this to start their server.
        """
        pass
    
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """
        Implementation-specific cleanup.
        
        Subclasses must implement this to stop their server.
        """
        pass
    
    @abstractmethod
    async def _do_health_check(self) -> bool:
        """
        Implementation-specific health check.
        
        Subclasses must implement this to check if their server is healthy.
        
        Returns:
            True if the server is healthy, False otherwise
        """
        pass

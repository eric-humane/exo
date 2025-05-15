from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, ClassVar
import numpy as np
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities
from exo.topology.topology import Topology
from exo.utils.async_resources import AsyncResource, ResourceState


class PeerHandle(AsyncResource, ABC):
  """
  Base class for peer connection handles.
  
  This class inherits from AsyncResource to leverage the state management,
  initialization/cleanup, and health monitoring features of the AsyncResource system.
  """
  # Resource type identifier for the AsyncResource system
  RESOURCE_TYPE: ClassVar[str] = "peer_handle"
  
  @abstractmethod
  def id(self) -> str:
    """Get the peer's unique identifier."""
    pass

  @abstractmethod
  def addr(self) -> str:
    """Get the peer's network address."""
    pass

  @abstractmethod
  def description(self) -> str:
    """Get a human-readable description of the peer."""
    pass

  @abstractmethod
  def device_capabilities(self) -> DeviceCapabilities:
    """Get the peer's device capabilities."""
    pass
  
  async def _do_initialize(self) -> None:
    """
    Initialize the peer connection.
    
    AsyncResource implementation calls this during initialization.
    """
    # This will be called by AsyncResource.initialize()
    await self.connect()
  
  async def _do_cleanup(self) -> None:
    """
    Clean up the peer connection.
    
    AsyncResource implementation calls this during cleanup.
    """
    # This will be called by AsyncResource.cleanup()
    await self.disconnect()
  
  async def _do_health_check(self) -> bool:
    """
    Check the health of the peer connection.
    
    AsyncResource implementation calls this during health checks.
    """
    # This will be called by AsyncResource.check_health()
    return await self.health_check()
  
  @abstractmethod
  async def connect(self) -> None:
    """Establish a connection to the peer."""
    pass

  @abstractmethod
  async def is_connected(self) -> bool:
    """Check if the peer is connected."""
    pass

  @abstractmethod
  async def disconnect(self) -> None:
    """Disconnect from the peer."""
    pass

  @abstractmethod
  async def health_check(self) -> bool:
    """
    Check if the peer is healthy.
    
    Returns:
        True if the peer is healthy, False otherwise.
    """
    pass
    
  async def close(self) -> None:
    """
    Close the peer connection.
    
    This is a convenience method that delegates to the AsyncResource cleanup method.
    """
    await self.cleanup()

  @abstractmethod
  async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_tensor(self, shard: Shard, tensor: np.array, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    pass

  @abstractmethod
  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    pass

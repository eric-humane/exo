import os
import asyncio
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from exo.networking.discovery import Discovery
from exo.topology.device_capabilities import DeviceCapabilities
from exo.networking.manual.network_topology_config import NetworkTopology, PeerConfig
from exo.helpers import DEBUG_DISCOVERY
from exo.networking.peer_handle import PeerHandle
from exo.utils.async_resources import ResourceState


class ManualDiscovery(Discovery):
  def __init__(
    self,
    network_config_path: str,
    node_id: str,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    discovery_id: Optional[str] = None,
  ):
    # Initialize the AsyncResource base class
    super().__init__(node_id=node_id, discovery_id=discovery_id)
    
    self.network_config_path = network_config_path
    self.create_peer_handle = create_peer_handle

    self.listen_task = None
    self.known_peers: Dict[str, PeerHandle] = {}

    self._cached_peers: Dict[str, PeerConfig] = {}
    self._last_modified_time: Optional[float] = None
    self._file_executor = ThreadPoolExecutor(max_workers=1)
    self._is_running = False

  async def start(self) -> None:
    """Start the discovery service, initializing it if necessary."""
    await self.initialize()
    self._is_running = True
    self.listen_task = asyncio.create_task(self.task_find_peers_from_config())

  async def stop(self) -> None:
    """Stop the discovery service and clean up resources."""
    self._is_running = False
    if self.listen_task: self.listen_task.cancel()
    self._file_executor.shutdown(wait=True)
    await self.cleanup()

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    if DEBUG_DISCOVERY >= 2: print(f"Discovered peers: {[peer.id() for peer in self.known_peers.values()]}")
    return list(self.known_peers.values())

  async def task_find_peers_from_config(self):
    if DEBUG_DISCOVERY >= 2: print("Starting task to find peers from config...")
    while True:
      peers_from_config = await self._get_peers()
      new_known_peers = {}
      for peer_id, peer_config in peers_from_config.items():
        try:
          if DEBUG_DISCOVERY >= 2: print(f"Checking peer {peer_id=} at {peer_config.address}:{peer_config.port}")
          peer = self.known_peers.get(peer_id)
          if not peer:
            if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} not found in known peers. Adding.")
            peer = self.create_peer_handle(peer_id, f"{peer_config.address}:{peer_config.port}", "MAN", peer_config.device_capabilities)
          is_healthy = await peer.health_check()
          if is_healthy:
            if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is healthy.")
            new_known_peers[peer_id] = peer
          elif DEBUG_DISCOVERY >= 2:
            print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is not healthy. Removing.")
        except Exception as e:
          if DEBUG_DISCOVERY >= 2: print(f"Exception occurred when attempting to add {peer_id=}: {e}")
      self.known_peers = new_known_peers
      await asyncio.sleep(5.0)

      if DEBUG_DISCOVERY >= 2: print(f"Current known peers: {[peer.id() for peer in self.known_peers.values()]}")

  async def _get_peers(self):
    try:
      loop = asyncio.get_running_loop()
      current_mtime = await loop.run_in_executor(self._file_executor, os.path.getmtime, self.network_config_path)

      if (self._cached_peers is not None and self._last_modified_time is not None and current_mtime <= self._last_modified_time):
        return self._cached_peers

      topology = await loop.run_in_executor(self._file_executor, NetworkTopology.from_path, self.network_config_path)

      if self.node_id not in topology.peers:
        raise ValueError(
          f"Node ID {self.node_id} not found in network config file "
          f"{self.network_config_path}. Please run with `node_id` set to "
          f"one of the keys in the config file: {[k for k, _ in topology.peers]}"
        )

      peers_in_network = topology.peers
      peers_in_network.pop(self.node_id)

      self._cached_peers = peers_in_network
      self._last_modified_time = current_mtime

      return peers_in_network

    except Exception as e:
      if DEBUG_DISCOVERY >= 2:
        print(f"Error when loading network config file from {self.network_config_path}. "
              f"Please update the config file in order to successfully discover peers. "
              f"Exception: {e}")
      return self._cached_peers
      
  async def _do_initialize(self) -> None:
    """Implementation-specific initialization for manual discovery."""
    # Nothing specific to initialize beyond what's done in __init__
    # We'll set the running flag in start() method
    pass
    
  async def _do_cleanup(self) -> None:
    """Implementation-specific cleanup for the manual discovery service."""
    # Cancel listen task if running
    if self.listen_task and not self.listen_task.done():
      self.listen_task.cancel()
      try:
        await self.listen_task
      except asyncio.CancelledError:
        pass
    
    # Close all peer handles
    for peer_id, peer_handle in list(self.known_peers.items()):
      try:
        await peer_handle.close()
      except Exception as e:
        if DEBUG_DISCOVERY >= 1:
          print(f"Error closing peer handle for {peer_id}: {e}")
    
    # Clear the known peers
    self.known_peers.clear()
    
    # Shut down the file executor
    self._file_executor.shutdown(wait=False)
    
  async def _check_health(self) -> bool:
    """Implementation-specific health check for the manual discovery service."""
    # Check if we're running
    if not self._is_running:
      return False
      
    # Check if the listen task is alive
    if not self.listen_task or self.listen_task.done():
      return False
      
    # Check if we can read the config file
    try:
      config_exists = os.path.exists(self.network_config_path)
      if not config_exists:
        return False
    except Exception:
      return False
      
    # All checks passed
    return True

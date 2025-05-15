import asyncio
import json
import random
import socket
import time
import traceback
from typing import List, Dict, Callable, Tuple, Coroutine, Optional
from contextlib import nullcontext
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.networking.connection_pool import pool_manager
from exo.networking.grpc.grpc_connection_pool import initialize_grpc_pool
from exo.networking.auth.peer_auth import initialize_auth, get_auth_instance, shutdown_auth
from exo.networking.dynamic_timeout import get_timeout, add_rtt_sample, add_loss_sample, TimingContext
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses_and_interfaces, get_interface_priority_and_type
from exo.utils import logging
from exo.metrics.metrics import get_metrics_registry, initialize_metrics, shutdown_metrics


class ListenProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable[[
                 bytes, Tuple[str, int]], Coroutine]):
        super().__init__()
        self.on_message = on_message
        self.loop = asyncio.get_event_loop()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        asyncio.create_task(self.on_message(data, addr))


def get_broadcast_address(ip_addr: str) -> Tuple[str, bool]:
    """
    Get the broadcast address for a given IP address.
    For IPv4 addresses, returns a subnet-specific broadcast (x.x.x.255).
    For IPv6 addresses, returns a multicast address.
    Falls back to global broadcast (255.255.255.255) if the IP can't be parsed.

    Args:
      ip_addr: The IP address as a string

    Returns:
      Tuple of (broadcast/multicast address, is_ipv6)
    """
    if not ip_addr or not isinstance(ip_addr, str):
        logging.warning(f"Invalid IP address: {ip_addr}, using global broadcast",
                        component="discovery")
        return "255.255.255.255", False

    # Check if it's an IPv6 address
    if ":" in ip_addr:
        # IPv6 doesn't have broadcast, so we use multicast
        # ff02::1 is the all-nodes link-local multicast address
        return "ff02::1", True

    # It's IPv4
    try:
        # Check if it's a valid IPv4 address
        if ip_addr.count('.') != 3:
            logging.warning(f"IP address {ip_addr} doesn't appear to be valid IPv4, using global broadcast",
                            component="discovery")
            return "255.255.255.255", False

        # Split IP into octets and create broadcast address for the subnet
        ip_parts = ip_addr.split('.')

        # Check if all parts are valid numbers
        for part in ip_parts:
            if not part.isdigit() or int(part) > 255:
                logging.warning(f"Invalid IP octet in {ip_addr}, using global broadcast",
                                component="discovery")
                return "255.255.255.255", False

        # Create subnet broadcast
        return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255", False
    except IndexError as e:
        logging.warning(f"Error parsing IP address {ip_addr}: {e}, using global broadcast",
                        component="discovery",
                        exc_info=e)
        return "255.255.255.255", False
    except Exception as e:
        logging.warning(f"Unexpected error determining broadcast address for {ip_addr}: {e}, using global broadcast",
                        component="discovery",
                        exc_info=e)
        return "255.255.255.255", False


class BroadcastProtocol(asyncio.DatagramProtocol):
    def __init__(self, message: str, broadcast_port: int, source_ip: str):
        self.message = message
        self.broadcast_port = broadcast_port
        self.source_ip = source_ip

    def connection_made(self, transport):
        sock = transport.get_extra_info("socket")

        # Get broadcast/multicast address and IPv6 flag
        broadcast_addr, is_ipv6 = get_broadcast_address(self.source_ip)

        if is_ipv6:
            # IPv6 multicast setup
            # Join the all-nodes multicast group
            # Interface index 0 means "any interface"
            group = socket.inet_pton(socket.AF_INET6, broadcast_addr)
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP,
                            group + socket.inet_pton(socket.AF_INET6, '::', '\0' * 4))
            # Send to the multicast group
            transport.sendto(self.message.encode("utf-8"),
                             (broadcast_addr, self.broadcast_port))
        else:
            # IPv4 broadcast setup
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # Try both subnet-specific and global broadcast
            transport.sendto(self.message.encode("utf-8"),
                             (broadcast_addr, self.broadcast_port))
            if broadcast_addr != "255.255.255.255":
                transport.sendto(
                    self.message.encode("utf-8"),
                    ("255.255.255.255",
                     self.broadcast_port))


class UDPDiscovery(Discovery):
    """
    UDP-based discovery service using the AsyncResource pattern.

    This class handles peer discovery through UDP broadcasts and maintains
    connections to discovered peers.
    """

    def __init__(
        self,
        node_id: str,
        node_port: int,
        listen_port: int,
        broadcast_port: int,
        create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
        broadcast_interval: float = 2.5,
        discovery_timeout: int = 30,
        device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
        allowed_node_ids: Optional[List[str]] = None,
        allowed_interface_types: Optional[List[str]] = None,
        use_dynamic_timeouts: bool = True,
        discovery_id: Optional[str] = None,
    ):
        """
        Initialize UDP discovery with AsyncResource pattern.

        Args:
            node_id: ID of the node this discovery belongs to
            node_port: Port the node's server is listening on
            listen_port: Port to listen for discovery broadcasts
            broadcast_port: Port to send discovery broadcasts to
            create_peer_handle: Factory function to create peer handles
            broadcast_interval: Seconds between broadcasts
            discovery_timeout: Timeout for peer discovery in seconds
            device_capabilities: Capabilities of this node's device
            allowed_node_ids: List of node IDs allowed to connect (None = all)
            allowed_interface_types: List of network interface types to use (None = all)
            use_dynamic_timeouts: Whether to use dynamic timeout adjustment
            discovery_id: Optional unique ID for this discovery instance
        """
        # Initialize the AsyncResource base class
        super().__init__(node_id=node_id, discovery_id=discovery_id)

        # Store configuration
        self.node_port = node_port
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port
        self.create_peer_handle = create_peer_handle
        self.broadcast_interval = broadcast_interval
        self.discovery_timeout = discovery_timeout
        self.device_capabilities = device_capabilities
        self.allowed_node_ids = allowed_node_ids
        self.allowed_interface_types = allowed_interface_types
        self.use_dynamic_timeouts = use_dynamic_timeouts

        # Initialize state
        self.known_peers: Dict[str, Tuple[PeerHandle, float, float, int]] = {}
        self.broadcast_task = None
        self.listen_task = None
        self.cleanup_task = None
        self._listen_transport = None
        self._is_running = False  # Track if the service is running

        # Timeout identifiers
        self.health_check_timeout_id = f"discovery.health_check.{node_id}"
        self.broadcast_timeout_id = f"discovery.broadcast.{node_id}"

        # Metrics to be initialized during _do_initialize
        self.metrics = None

    # AsyncResource implementation

    async def _do_initialize(self) -> None:
        """
        Initialize the UDP discovery service.

        This method:
        1. Initializes device capabilities
        2. Sets up logging context
        3. Initializes connection pools and auth
        4. Sets up metrics
        5. Starts the listen and broadcast tasks
        """
        # Get current device capabilities
        self.device_capabilities = await device_capabilities()

        # Set logging context for this instance
        logging.set_context(
            node_id=self._node_id,
            listen_port=self.listen_port,
            broadcast_port=self.broadcast_port)

        # Initialize connection pools
        await initialize_grpc_pool()
        await pool_manager.start_all()

        # Initialize authentication system
        await initialize_auth(self._node_id)

        # Initialize metrics system
        await initialize_metrics(self._node_id)
        metrics = get_metrics_registry(self._node_id)

        # Create metrics for discovery
        self.metrics = {
            "broadcasts_sent": metrics.counter("discovery.broadcasts_sent",
                                               "Total discovery broadcasts sent",
                                               {"node_id": self._node_id}),
            "broadcasts_received": metrics.counter("discovery.broadcasts_received",
                                                   "Total discovery broadcasts received",
                                                   {"node_id": self._node_id}),
            "auth_failures": metrics.counter("discovery.auth_failures",
                                             "Authentication failures during discovery",
                                             {"node_id": self._node_id}),
            "health_check_failures": metrics.counter("discovery.health_check_failures",
                                                     "Peer health check failures",
                                                     {"node_id": self._node_id}),
            "peers_discovered": metrics.counter("discovery.peers_discovered",
                                                "Total unique peers discovered",
                                                {"node_id": self._node_id}),
            "peers_removed": metrics.counter("discovery.peers_removed",
                                             "Peers removed due to inactivity or failures",
                                             {"node_id": self._node_id}),
            "active_peers": metrics.gauge("discovery.active_peers",
                                          "Currently active peers",
                                          {"node_id": self._node_id}),
            "broadcast_latency": metrics.timer("discovery.broadcast_latency",
                                               "Time taken to broadcast on all interfaces",
                                               {"node_id": self._node_id}),
            "listen_time": metrics.timer("discovery.listen_time",
                                         "Time spent processing incoming broadcasts",
                                         {"node_id": self._node_id}),
        }

        # Mark as running before starting tasks
        self._is_running = True
        
        # Start all tasks
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        self.listen_task = asyncio.create_task(self._listen())
        self.cleanup_task = asyncio.create_task(self._peer_cleanup_loop())

        logging.info(f"UDP Discovery for node {self._node_id} started",
                     component="discovery",
                     ports={"listen": self.listen_port, "broadcast": self.broadcast_port})

    async def discover_peers(
            self,
            wait_for_peers: int = 0) -> List[PeerHandle]:
        """
        Discover peers on the network.

        This method:
        1. Ensures the discovery service is initialized
        2. Optionally waits for a minimum number of peers
        3. Returns the list of peer handles

        Args:
            wait_for_peers: Minimum number of peers to wait for (0 = don't wait)

        Returns:
            List of PeerHandle instances
        """
        # First ensure discovery is ready
        await self.ensure_ready()

        # Wait for peers if requested
        if wait_for_peers > 0:
            start_time = time.time()
            timeout = self.discovery_timeout or 30

            # Wait for peers with timeout
            while len(self.known_peers) < wait_for_peers and time.time(
            ) - start_time < timeout:
                if DEBUG_DISCOVERY > 0:
                    logging.info(f"Waiting for peers: {len(self.known_peers)}/{wait_for_peers}",
                                 component="discovery")
                await asyncio.sleep(0.5)

            # Log result
            if len(self.known_peers) < wait_for_peers:
                logging.warning(f"Timed out waiting for peers: {len(self.known_peers)}/{wait_for_peers}",
                                component="discovery")

        # Return all known peer handles
        return [peer_handle for peer_handle, _,
                _, _ in self.known_peers.values()]

    async def _do_cleanup(self) -> None:
        """
        Implementation-specific cleanup for the UDP discovery service.
        
        This method cleans up all resources used by the discovery service:
        1. Cancels all running tasks
        2. Closes all transports
        3. Closes all peer handles
        4. Clears the internal state
        """
        if not self._is_running:
            logging.debug(f"UDP Discovery for node {self._node_id} not running, ignoring cleanup request",
                          component="discovery")
            return

        logging.info(f"Cleaning up UDP Discovery for node {self._node_id}",
                     component="discovery")

        self._is_running = False
        
        # Rest of the cleanup logic is in the stop method
        await self.stop()
        
    async def _check_health(self) -> bool:
        """
        Implementation-specific health check for the UDP discovery service.
        
        This method checks if the discovery service is healthy:
        1. Verifies that the tasks are running
        2. Checks the connection to known peers
        
        Returns:
            True if the discovery service is healthy, False otherwise
        """
        # Check if the tasks are running
        if not self._is_running:
            return False
            
        if not self.broadcast_task or self.broadcast_task.done():
            return False
            
        if not self.listen_task or self.listen_task.done():
            return False
            
        if not self.cleanup_task or self.cleanup_task.done():
            return False
            
        # If we have peers, check at least one of them is healthy
        if self.known_peers and not any(peer[0].is_usable for peer in self.known_peers.values()):
            return False
            
        return True
        
    @property
    def node_id(self) -> str:
        """Get the node ID."""
        return self._node_id
        
    async def start(self):
        """Start the discovery service, initializing it if necessary."""
        await self.initialize()
        self._is_running = True
        
    async def stop(self):
        """Stop the discovery service and clean up all resources."""
        if not self._is_running:
            logging.debug(f"UDP Discovery for node {self._node_id} not running, ignoring stop request",
                          component="discovery")
            return

        logging.info(f"Stopping UDP Discovery for node {self._node_id}",
                     component="discovery")

        # Cancel all running tasks
        tasks_to_cancel = []
        if self.broadcast_task:
            self.broadcast_task.cancel()
            tasks_to_cancel.append(self.broadcast_task)
        if self.listen_task:
            self.listen_task.cancel()
            tasks_to_cancel.append(self.listen_task)
        if self.cleanup_task:
            self.cleanup_task.cancel()
            tasks_to_cancel.append(self.cleanup_task)

        # Close the listen transports if they exist
        if self._listen_transport:
            if isinstance(self._listen_transport, dict):
                # Close IPv4 transport
                if self._listen_transport.get("ipv4"):
                    try:
                        logging.debug(f"Closing UDP IPv4 listen transport on port {self.listen_port}",
                                      component="discovery")
                        self._listen_transport["ipv4"].close()
                    except Exception as e:
                        logging.error(f"Error closing UDP IPv4 listen transport: {e}",
                                      component="discovery",
                                      exc_info=e)

                # Close IPv6 transport
                if self._listen_transport.get("ipv6"):
                    try:
                        logging.debug(f"Closing UDP IPv6 listen transport on port {self.listen_port}",
                                      component="discovery")
                        self._listen_transport["ipv6"].close()
                    except Exception as e:
                        logging.error(f"Error closing UDP IPv6 listen transport: {e}",
                                      component="discovery",
                                      exc_info=e)
            else:
                # Legacy transport handling (for backward compatibility)
                try:
                    logging.debug(f"Closing UDP listen transport on port {self.listen_port}",
                                  component="discovery")
                    self._listen_transport.close()
                except Exception as e:
                    logging.error(f"Error closing UDP listen transport: {e}",
                                  component="discovery",
                                  exc_info=e)

        # Close all peer handles
        peer_count = len(self.known_peers)
        successful_closes = 0

        for peer_id, (peer_handle, _, _, _) in list(self.known_peers.items()):
            try:
                await peer_handle.close()
                successful_closes += 1
            except Exception as e:
                logging.warning(f"Error closing peer handle for {peer_id}: {e}",
                                component="discovery",
                                peer_id=peer_id,
                                exc_info=e)

        # Wait for tasks to complete
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logging.error(f"Error waiting for tasks to cancel: {e}",
                              component="discovery",
                              exc_info=e)

        # Clear internal state
        self.known_peers.clear()
        self._listen_transport = None

        # Shutdown connection pools
        await pool_manager.stop_all()

        # Shutdown authentication system
        await shutdown_auth()

        # Shutdown metrics system
        await shutdown_metrics()

        # Clear thread-local logging context
        logging.clear_context()

        logging.info(f"UDP Discovery for node {self.node_id} stopped",
                     component="discovery",
                     stats={"peers_closed": successful_closes, "total_peers": peer_count})

    async def _broadcast_loop(self):
        """Task that periodically broadcasts this node's presence on the network."""
        try:
            broadcast_count = 0
            auth = get_auth_instance(self.node_id)

            # Generate a token for our broadcasts
            self_token = auth.generate_token(
                peer_id=self.node_id,
                expiry_seconds=24 * 60 * 60  # 24 hours
            )

            while self._is_running:
                interfaces_tried = 0
                interfaces_succeeded = 0

                # Start timing the broadcast operation
                broadcast_timer = self.metrics["broadcast_latency"].start()

                for addr, interface_name in get_all_ip_addresses_and_interfaces():
                    if not self._is_running:
                        break

                    interfaces_tried += 1
                    interface_priority, interface_type = await get_interface_priority_and_type(interface_name)

                    # Create timestamp for this broadcast message
                    timestamp = time.time()

                    # Create signature for this broadcast
                    signature = auth._sign_data(
                        data=f"{self.node_id}.{timestamp}.{broadcast_count}.{addr}",
                        key=self_token.token_secret
                    )

                    message = json.dumps({
                        "type": "discovery",
                        "node_id": self.node_id,
                        "grpc_port": self.node_port,
                        "device_capabilities": self.device_capabilities.to_dict(),
                        "priority": interface_priority,
                        "interface_name": interface_name,
                        "interface_type": interface_type,
                        "timestamp": timestamp,
                        "broadcast_count": broadcast_count,
                        "auth": {
                            "token_id": self_token.token_id,
                            "signature": signature,
                            "timestamp": timestamp
                        }
                    })

                    transport = None
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                        except AttributeError:
                            pass
                        sock.bind((addr, 0))

                        # Time the datagram endpoint creation with dynamic timeout
                        timeout_id = f"{self.broadcast_timeout_id}.{addr}"
                        start_time = time.time()
                        try:
                            transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                                lambda: BroadcastProtocol(message, self.broadcast_port, addr),
                                sock=sock
                            )
                            # Record success
                            if self.use_dynamic_timeouts:
                                add_rtt_sample(timeout_id, time.time() - start_time)
                                add_loss_sample(timeout_id, True)
                        except Exception as e:
                            # Record failure
                            if self.use_dynamic_timeouts:
                                add_loss_sample(timeout_id, False)
                            raise

                        interfaces_succeeded += 1
                        # Increment broadcast counter
                        self.metrics["broadcasts_sent"].increment()

                        logging.trace(f"Broadcast presence on interface {interface_name} ({addr})",
                                      component="discovery",
                                      interface={"name": interface_name, "address": addr, "type": interface_type, "priority": interface_priority})
                    except Exception as e:
                        logging.warning(f"Error in broadcast presence ({addr} - {interface_name})",
                                        component="discovery",
                                        interface={
                            "name": interface_name,
                            "address": addr,
                            "type": interface_type,
                            "priority": interface_priority},
                            exc_info=e)
                    finally:
                        if transport:
                            try:
                                transport.close()
                            except Exception as e:
                                logging.debug(f"Error closing transport: {e}",
                                              component="discovery",
                                              interface=interface_name,
                                              exc_info=e)

                # Stop timing the broadcast operation
                broadcast_duration = self.metrics["broadcast_latency"].stop(broadcast_timer)

                if interfaces_succeeded > 0:
                    logging.debug(f"Broadcast presence completed in {broadcast_duration:.3f}s",
                                  component="discovery",
                                  stats={"interfaces_tried": interfaces_tried,
                                         "interfaces_succeeded": interfaces_succeeded,
                                         "duration_ms": broadcast_duration * 1000})
                else:
                    logging.warning(f"Failed to broadcast on any interfaces",
                                    component="discovery",
                                    stats={"interfaces_tried": interfaces_tried})

                # Update active peers gauge
                self.metrics["active_peers"].set(len(self.known_peers))

                broadcast_count += 1

                if self._is_running:
                    await asyncio.sleep(self.broadcast_interval)
        except asyncio.CancelledError:
            logging.debug(f"Broadcast task for node {self.node_id} cancelled",
                          component="discovery")
            raise
        except Exception as e:
            logging.error(f"Fatal error in broadcast task",
                          component="discovery",
                          exc_info=e)

    async def on_listen_message(self, data, addr):
        if not data:
            return

        # Start timing message processing
        with self.metrics["listen_time"].time():
            decoded_data = data.decode("utf-8", errors="ignore")

            # Check if the decoded data starts with a valid JSON character
            if not (decoded_data.strip() and decoded_data.strip()[0] in "{["):
                logging.debug(f"Received invalid JSON data from {addr}",
                              component="discovery",
                              data=decoded_data[:100])
                return

            try:
                decoder = json.JSONDecoder(strict=False)
                message = decoder.decode(decoded_data)
            except json.JSONDecodeError as e:
                logging.debug(f"Error decoding JSON data from {addr}: {e}",
                              component="discovery",
                              exc_info=e)
                return

            # Increment received counter
            self.metrics["broadcasts_received"].increment()

            logging.trace(f"Received message from {addr}",
                          component="discovery",
                          message_type=message.get("type"))

            if message.get("type") == "discovery" and message.get("node_id") != self.node_id:
                peer_id = message["node_id"]

                # Verify authentication if auth data is present
                auth = get_auth_instance(self.node_id)
                auth_data = message.get("auth", {})

                if auth_data:
                    token_id = auth_data.get("token_id")
                    signature = auth_data.get("signature")
                    timestamp = auth_data.get("timestamp")

                    # Only verify if we have all auth components
                    if token_id and signature and timestamp:
                        # Verify the message timestamp is recent (within 5 minutes)
                        current_time = time.time()
                        if abs(current_time - timestamp) > 300:  # 5 minutes
                            logging.warning(f"Ignoring peer {peer_id}: Message timestamp too old or in the future",
                                            component="discovery",
                                            peer_id=peer_id,
                                            time_diff=abs(current_time - timestamp))
                            self.metrics["auth_failures"].increment()
                            return

                        # Get the peer's token or create one if we haven't seen this peer before
                        if not auth.is_peer_authorized(peer_id):
                            # This is a new peer we haven't authenticated before
                            # We'll provisionally accept it but create a token for it that we'll use
                            # for future authentication
                            auth.generate_token(peer_id=peer_id)
                            logging.info(f"Provisionally accepted new peer {peer_id}",
                                         component="discovery",
                                         peer_id=peer_id)
                else:
                    # No auth data, check if we require authentication
                    if auth.auth_enabled and not auth.is_peer_authorized(peer_id):
                        logging.warning(f"Ignoring unauthenticated peer {peer_id}",
                                        component="discovery",
                                        peer_id=peer_id)
                        self.metrics["auth_failures"].increment()
                        return

                # Skip if peer_id is not in allowed list
                if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
                    logging.debug(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list",
                                  component="discovery",
                                  peer_id=peer_id)
                    return

                peer_host = addr[0]
                peer_port = message["grpc_port"]
                peer_prio = message["priority"]
                peer_interface_name = message["interface_name"]
                peer_interface_type = message["interface_type"]

                # Skip if interface type is not in allowed list
                if self.allowed_interface_types and peer_interface_type not in self.allowed_interface_types:
                    logging.debug(f"Ignoring peer {peer_id} with interface type {peer_interface_type}",
                                  component="discovery",
                                  peer_id=peer_id,
                                  interface_type=peer_interface_type)
                    return

                device_capabilities = DeviceCapabilities(**message["device_capabilities"])

                if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
                    if peer_id in self.known_peers:
                        existing_peer_prio = self.known_peers[peer_id][3]
                        if existing_peer_prio >= peer_prio:
                            if DEBUG >= 1:
                                print(
                                    f"Ignoring peer {peer_id} at {peer_host}:{peer_port} with priority {peer_prio} because we already know about a peer with higher or equal priority: {existing_peer_prio}")
                            return

                    # Create a new peer handle
                    new_peer_handle = self.create_peer_handle(
                        peer_id,
                        f"{peer_host}:{peer_port}",
                        f"{peer_interface_type} ({peer_interface_name})",
                        device_capabilities)
                    if not await new_peer_handle.health_check():
                        self.metrics["health_check_failures"].increment()
                        logging.info(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.",
                                     component="discovery",
                                     peer_id=peer_id,
                                     address=f"{peer_host}:{peer_port}")
                        return

                    logging.info(f"Adding new peer {peer_id} at {peer_host}:{peer_port}",
                                 component="discovery",
                                 peer_id=peer_id,
                                 address=f"{peer_host}:{peer_port}")

                    self.metrics["peers_discovered"].increment()
                    self.known_peers[peer_id] = (new_peer_handle, time.time(), time.time(), peer_prio)
                else:
                    # Existing peer, check health
                    if not await self.known_peers[peer_id][0].health_check():
                        self.metrics["health_check_failures"].increment()
                        logging.info(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.",
                                     component="discovery",
                                     peer_id=peer_id,
                                     address=f"{peer_host}:{peer_port}")
                        if peer_id in self.known_peers:
                            del self.known_peers[peer_id]
                            self.metrics["peers_removed"].increment()
                        return

                    # Update last seen timestamp
                    if peer_id in self.known_peers:
                        self.known_peers[peer_id] = (self.known_peers[peer_id][0],
                                                     self.known_peers[peer_id][1],
                                                     time.time(),
                                                     peer_prio)

    async def _listen(self):
        """Task that listens for discovery broadcasts from other peers."""
        await self.task_listen_for_peers()
        
    async def task_listen_for_peers(self):
        """Task that listens for discovery broadcasts from other peers."""
        max_retry_count = 5  # Increased from 3 to 5 for better resilience
        retry_count = 0
        retry_delay = 1.5  # seconds, slightly reduced initial delay

        # List of ports to try in case the primary port is unavailable
        fallback_ports = [self.listen_port + i for i in range(1, 6)]
        current_port = self.listen_port

        # Set up both IPv4 and IPv6 listeners
        ipv4_transport = None
        ipv6_transport = None

        while retry_count < max_retry_count:
            try:
                # First, set up IPv4 listener
                logging.debug(f"Setting up IPv4 listener on port {current_port}",
                              component="discovery")

                # Create an IPv4 socket with options set before binding
                ipv4_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ipv4_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    # SO_REUSEPORT may not be available on all platforms
                    ipv4_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except (AttributeError, OSError):
                    pass

                # Set timeout to avoid blocking indefinitely
                ipv4_sock.settimeout(10)

                # Try to bind to the current port
                ipv4_sock.bind(("0.0.0.0", current_port))

                # Create datagram endpoint using the configured socket
                ipv4_transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                    lambda: ListenProtocol(self.on_listen_message),
                    sock=ipv4_sock
                )

                # Now, try to set up IPv6 listener
                try:
                    logging.debug(f"Setting up IPv6 listener on port {current_port}",
                                  component="discovery")

                    # Create an IPv6 socket with options set before binding
                    ipv6_sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                    ipv6_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                    # Set IPV6_V6ONLY to false to allow dual-stack socket (both IPv4 and IPv6)
                    # But this can fail on some platforms, so we catch and ignore failures
                    try:
                        ipv6_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                    except (AttributeError, OSError) as e:
                        logging.debug(f"Failed to set IPV6_V6ONLY=0: {e}",
                                      component="discovery")

                    try:
                        ipv6_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except (AttributeError, OSError):
                        pass

                    # Set timeout to avoid blocking indefinitely
                    ipv6_sock.settimeout(10)

                    # Try to bind to the current port
                    ipv6_sock.bind(("::", current_port))

                    # Create datagram endpoint using the configured socket
                    ipv6_transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                        lambda: ListenProtocol(self.on_listen_message),
                        sock=ipv6_sock
                    )

                    # Join IPv6 multicast group for discovery
                    group_bin = socket.inet_pton(socket.AF_INET6, "ff02::1")
                    if_index = 0  # 0 means all interfaces
                    ipv6_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP,
                                         group_bin + socket.inet_pton(socket.AF_INET6, "::", b"\x00" * 4))

                    logging.info(f"Successfully set up IPv6 listener on port {current_port}",
                                 component="discovery")
                except Exception as e:
                    logging.warning(f"Failed to set up IPv6 listener: {e}",
                                    component="discovery",
                                    exc_info=e)
                    ipv6_transport = None

                # Store the transports so we can close them properly on shutdown
                self._listen_transport = {
                    "ipv4": ipv4_transport,
                    "ipv6": ipv6_transport
                }

                logging.info(f"Started listen task on port {current_port}",
                             component="discovery",
                             ipv6_supported=ipv6_transport is not None)

                # If we're using a fallback port, log this at a higher debug level
                if current_port != self.listen_port:
                    logging.warning(f"Using fallback port {current_port} instead of configured port {self.listen_port}",
                                    component="discovery")

                return
            except OSError as e:
                retry_count += 1

                # Try fallback ports if the primary one fails
                if current_port == self.listen_port and fallback_ports:
                    current_port = fallback_ports.pop(0)
                    print(
                        f"Warning: Failed to bind UDP listener to primary port {self.listen_port}, trying fallback port {current_port}: {e}")
                    # Don't count this as a retry, just try the next port
                    retry_count -= 1
                elif retry_count >= max_retry_count:
                    print(
                        f"ERROR: Failed to bind UDP listener to port {current_port} after {max_retry_count} attempts: {e}")
                    print(
                        f"Please check if another process is using ports {self.listen_port}-{self.listen_port + 5} or try a different port range.")
                    # Re-raise the exception after logging, as this is a critical error
                    raise
                else:
                    print(
                        f"Warning: Failed to bind UDP listener to port {current_port}, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                    # Add jitter to backoff to prevent reconnection storms
                    jitter = 0.2 * retry_delay * (0.5 + 0.5 * random.random())
                    # Gentler exponential backoff with jitter
                    retry_delay = min(10, retry_delay * 1.5 + jitter)
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                if DEBUG_DISCOVERY >= 2:
                    print("Listen task cancelled")
                raise
            except Exception as e:
                print(f"ERROR: Unexpected error starting UDP listener on port {current_port}: {e}")
                import traceback
                traceback.print_exc()

                # For non-OSError exceptions, try a different port before giving up
                if current_port == self.listen_port and fallback_ports:
                    current_port = fallback_ports.pop(0)
                    print(f"Trying fallback port {current_port} after unexpected error")
                    # Don't increment retry count when switching ports
                    continue
                raise

    async def _peer_cleanup_loop(self):
        """Task that periodically checks the health of peers and removes stale ones."""
        await self.task_cleanup_peers()

    async def task_cleanup_peers(self):
        """Task that periodically checks the health of known peers and removes stale ones."""
        try:
            while self._is_running:
                try:
                    current_time = time.time()
                    peers_to_remove = []

                    peer_ids = list(self.known_peers.keys())
                    if peer_ids:  # Only attempt to gather if we have peers
                        results = await asyncio.gather(*[self.check_peer(peer_id, current_time) for peer_id in peer_ids], return_exceptions=True)

                        for peer_id, should_remove in zip(peer_ids, results):
                            if isinstance(should_remove, Exception):
                                # Handle exceptions in gather results
                                print(f"Error checking peer {peer_id}: {should_remove}")
                                # Consider the peer unhealthy if we couldn't check it
                                peers_to_remove.append(peer_id)
                            elif should_remove:
                                peers_to_remove.append(peer_id)

                    if DEBUG_DISCOVERY >= 2 and self.known_peers:
                        try:
                            statuses = {}
                            for peer_id, (peer_handle, connected_at, last_seen,
                                          prio) in self.known_peers.items():
                                try:
                                    is_connected = await peer_handle.is_connected()
                                    health_check = await peer_handle.health_check()
                                    statuses[peer_handle.id(
                                    )] = f"is_connected={is_connected}, health_check={health_check}, connected_at={connected_at:.2f}, last_seen={last_seen:.2f}, prio={prio}"
                                except Exception as e:
                                    statuses[peer_handle.id()] = f"Error getting status: {e}"
                            print("Peer statuses:", statuses)
                        except Exception as e:
                            print(f"Error getting peer statuses: {e}")

                    for peer_id in peers_to_remove:
                        if peer_id in self.known_peers:
                            try:
                                # Try to close the peer handle properly
                                peer_handle = self.known_peers[peer_id][0]
                                await peer_handle.close()
                            except Exception as e:
                                if DEBUG_DISCOVERY >= 2:
                                    print(
                                        f"Error closing peer handle for {peer_id} during cleanup: {e}")
                            finally:
                                del self.known_peers[peer_id]
                                if DEBUG_DISCOVERY >= 2:
                                    print(
                                        f"Removed peer {peer_id} due to inactivity or failed health check.")
                except Exception as e:
                    print(f"Error in cleanup peers: {e}")
                    print(traceback.format_exc())
                finally:
                    if self._is_running:
                        await asyncio.sleep(self.broadcast_interval)
        except asyncio.CancelledError:
            if DEBUG_DISCOVERY >= 2:
                print(f"Cleanup task for node {self.node_id} cancelled")
            raise
        except Exception as e:
            print(f"Fatal error in cleanup task: {e}")
            traceback.print_exc()

    async def check_peer(self, peer_id: str, current_time: float) -> bool:
        """Check if a peer is healthy and should remain connected."""
        peer_handle, connected_at, last_seen, prio = self.known_peers.get(
            peer_id, (None, None, None, None))
        if peer_handle is None:
            return False

        # Track health check failures for this peer
        if not hasattr(peer_handle, '_health_check_failures'):
            setattr(peer_handle, '_health_check_failures', 0)
        if not hasattr(peer_handle, '_connection_failures'):
            setattr(peer_handle, '_connection_failures', 0)

        # Set thresholds for failures before removing peer
        max_health_failures = 3  # Allow up to 3 consecutive health check failures
        max_connection_failures = 2  # Allow up to 2 consecutive connection failures

        try:
            # Generate timeout ID for this peer
            timeout_id = f"{self.health_check_timeout_id}.{peer_id}"

            # Check connection state with dynamic timeout
            start_time = time.time()
            try:
                # Use timing context to record timing and success/failure
                with TimingContext(timeout_id) if self.use_dynamic_timeouts else nullcontext():
                    is_connected = await peer_handle.is_connected()

                if not is_connected:
                    peer_handle._connection_failures += 1
                    logging.debug(f"Peer {peer_id} connection check failed ({peer_handle._connection_failures}/{max_connection_failures})",
                                  component="discovery",
                                  peer_id=peer_id)
                else:
                    peer_handle._connection_failures = 0  # Reset counter on success
            except Exception as e:
                # Record connection failure
                if self.use_dynamic_timeouts:
                    add_loss_sample(timeout_id, False)
                peer_handle._connection_failures += 1
                logging.debug(f"Peer {peer_id} connection check error: {e}",
                              component="discovery",
                              peer_id=peer_id)
                raise

            # Only run health check if still connected or failures below threshold
            health_ok = False
            if is_connected or peer_handle._connection_failures < max_connection_failures:
                try:
                    # Use timing context to record health check timing
                    with TimingContext(timeout_id) if self.use_dynamic_timeouts else nullcontext():
                        health_ok = await peer_handle.health_check()
                except Exception as e:
                    # Record health check failure
                    if self.use_dynamic_timeouts:
                        add_loss_sample(timeout_id, False)
                    raise

                if not health_ok:
                    peer_handle._health_check_failures += 1
                    logging.debug(f"Peer {peer_id} health check failed ({peer_handle._health_check_failures}/{max_health_failures})",
                                  component="discovery",
                                  peer_id=peer_id)
                    self.metrics["health_check_failures"].increment()
                else:
                    peer_handle._health_check_failures = 0  # Reset counter on success

        except Exception as e:
            logging.warning(f"Error checking peer {peer_id}",
                            component="discovery",
                            peer_id=peer_id,
                            exc_info=e)
            peer_handle._health_check_failures += 1
            return peer_handle._health_check_failures >= max_health_failures

        # Determine if peer should be removed:
        # 1. Based on timeout (hasn't been seen in a while)
        timeout_exceeded = current_time - last_seen > self.discovery_timeout

        # 2. Based on consecutive health check failures
        health_failed = peer_handle._health_check_failures >= max_health_failures

        # 3. Based on consecutive connection failures when we previously connected successfully
        conn_failed = not is_connected and peer_handle._connection_failures >= max_connection_failures and current_time - connected_at > 10

        should_remove = timeout_exceeded or health_failed or conn_failed

        if should_remove:
            reason = "timeout exceeded" if timeout_exceeded else "health checks failed" if health_failed else "connection failed"
            logging.info(f"Removing peer {peer_id} due to: {reason}",
                         component="discovery",
                         peer_id=peer_id,
                         reason=reason)

            # Increment the peer removal metric
            self.metrics["peers_removed"].increment()

            # Update active peers gauge
            # -1 since this peer will be removed
            self.metrics["active_peers"].set(len(self.known_peers) - 1)

        return should_remove
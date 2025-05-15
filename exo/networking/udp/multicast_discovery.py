"""
Multicast-based peer discovery for exo.

This module provides a specialized version of UDP discovery that uses IP multicast
for more efficient peer discovery on local networks.
"""

import asyncio
import json
import socket
import time
import traceback
import struct
import platform
import re
from typing import List, Dict, Callable, Tuple, Coroutine, Optional, Set
from contextlib import nullcontext

from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.networking.auth.peer_auth import get_auth_instance
from exo.networking.dynamic_timeout import TimingContext
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses_and_interfaces, get_interface_priority_and_type
from exo.utils import logging


# Default multicast group addresses
IPV4_MULTICAST_GROUP = "224.0.0.251"  # Standard mDNS multicast group
IPV6_MULTICAST_GROUP = "ff02::fb"     # IPv6 equivalent of mDNS multicast group
MULTICAST_PORT = 5353                 # Standard mDNS port


class MulticastListenProtocol(asyncio.DatagramProtocol):
    """Protocol for receiving multicast messages."""
    
    def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
        super().__init__()
        self.on_message = on_message
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport
    
    def datagram_received(self, data, addr):
        asyncio.create_task(self.on_message(data, addr))
    
    def error_received(self, exc):
        logging.error(f"Multicast listener protocol error: {exc}",
                     component="multicast_discovery",
                     exc_info=exc)
    
    def connection_lost(self, exc):
        if exc:
            logging.warning(f"Multicast listener connection lost: {exc}",
                          component="multicast_discovery",
                          exc_info=exc)


class MulticastSendProtocol(asyncio.DatagramProtocol):
    """Protocol for sending multicast messages."""
    
    def __init__(self, message: str, multicast_addr: str, multicast_port: int, ttl: int = 1):
        self.message = message
        self.multicast_addr = multicast_addr
        self.multicast_port = multicast_port
        self.ttl = ttl
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport
        
        # Get the socket
        sock = transport.get_extra_info('socket')
        
        # Set TTL (Time to Live) for multicast packets
        # This controls how many network hops the packet will traverse
        if self.multicast_addr.count('.') == 3:  # IPv4
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        else:  # IPv6
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, self.ttl)
        
        # Send the message to the multicast group
        transport.sendto(self.message.encode('utf-8'), (self.multicast_addr, self.multicast_port))
        
        # Close the transport right away - we're done sending
        transport.close()
    
    def error_received(self, exc):
        logging.error(f"Multicast send protocol error: {exc}",
                     component="multicast_discovery",
                     exc_info=exc)
    
    def connection_lost(self, exc):
        if exc:
            logging.warning(f"Multicast send protocol connection lost: {exc}",
                          component="multicast_discovery",
                          exc_info=exc)


class MulticastDiscovery(Discovery):
    """
    Discovery implementation using IP multicast.
    
    This class implements peer discovery using IP multicast, which is more
    efficient than UDP broadcasts when multiple nodes are on the same network.
    """
    
    def __init__(
        self,
        node_id: str,
        node_port: int,
        multicast_port: int = MULTICAST_PORT,
        ipv4_multicast_group: str = IPV4_MULTICAST_GROUP,
        ipv6_multicast_group: str = IPV6_MULTICAST_GROUP,
        create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle] = None,
        announcement_interval: int = 5,
        discovery_timeout: int = 30,
        device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
        allowed_node_ids: Optional[List[str]] = None,
        ttl: int = 1,  # TTL of 1 restricts to local network
        use_ipv6: bool = True,
        use_dynamic_timeouts: bool = True,
    ):
        self.node_id = node_id
        self.node_port = node_port
        self.multicast_port = multicast_port
        self.ipv4_multicast_group = ipv4_multicast_group
        self.ipv6_multicast_group = ipv6_multicast_group
        self.create_peer_handle = create_peer_handle
        self.announcement_interval = announcement_interval
        self.discovery_timeout = discovery_timeout
        self.device_capabilities = device_capabilities
        self.allowed_node_ids = allowed_node_ids
        self.ttl = ttl
        self.use_ipv6 = use_ipv6
        self.use_dynamic_timeouts = use_dynamic_timeouts
        
        # Internal state
        self.known_peers: Dict[str, Tuple[PeerHandle, float, float, int]] = {}
        self.pending_msgs: Dict[str, Tuple[str, float]] = {}  # node_id -> (message, timestamp)
        self.seen_message_ids: Set[str] = set()
        self.interfaces: List[Tuple[str, str]] = []  # (ip, interface_name)
        
        # Tasks
        self.announce_task = None
        self.ipv4_listen_task = None
        self.ipv6_listen_task = None
        self.cleanup_task = None
        
        # Transports
        self._ipv4_transport = None
        self._ipv6_transport = None
        
        # State flag
        self._is_running = False
        
        # Timeout identifiers
        self.health_check_timeout_id = f"multicast.health_check.{node_id}"
        self.announcement_timeout_id = f"multicast.announcement.{node_id}"
    
    async def __aenter__(self):
        """Async context manager entry - starts the discovery service."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - stops the discovery service."""
        await self.stop()
        return False  # Don't suppress exceptions
    
    async def start(self):
        """Start the discovery service and all related tasks."""
        if self._is_running:
            logging.debug(f"Multicast discovery for node {self.node_id} already running, ignoring start request",
                         component="multicast_discovery")
            return
        
        self._is_running = True
        
        # Get device capabilities
        self.device_capabilities = await device_capabilities()
        
        # Set logging context for this instance
        logging.set_context(node_id=self.node_id, 
                           port=self.multicast_port, 
                           ipv4_group=self.ipv4_multicast_group,
                           ipv6_group=self.ipv6_multicast_group if self.use_ipv6 else None)
        
        # Get all local interfaces for multicast
        self.interfaces = get_all_ip_addresses_and_interfaces(include_ipv6=self.use_ipv6)
        
        # Set up IPv4 multicast listener
        self.ipv4_listen_task = asyncio.create_task(self._setup_ipv4_listener())
        
        # Set up IPv6 multicast listener if requested
        if self.use_ipv6:
            self.ipv6_listen_task = asyncio.create_task(self._setup_ipv6_listener())
        
        # Start the announcement task
        self.announce_task = asyncio.create_task(self._task_announce_presence())
        
        # Start the cleanup task
        self.cleanup_task = asyncio.create_task(self._task_cleanup_peers())
        
        logging.info(f"Multicast discovery for node {self.node_id} started",
                    component="multicast_discovery",
                    config={
                        "port": self.multicast_port,
                        "ipv4_group": self.ipv4_multicast_group,
                        "ipv6_group": self.ipv6_multicast_group if self.use_ipv6 else None,
                        "announcement_interval": self.announcement_interval,
                        "ttl": self.ttl
                    })
    
    async def stop(self):
        """Stop the discovery service and clean up all resources."""
        if not self._is_running:
            logging.debug(f"Multicast discovery for node {self.node_id} not running, ignoring stop request",
                         component="multicast_discovery")
            return
        
        logging.info(f"Stopping multicast discovery for node {self.node_id}",
                    component="multicast_discovery")
        
        self._is_running = False
        
        # Cancel all running tasks
        tasks_to_cancel = []
        if self.announce_task:
            self.announce_task.cancel()
            tasks_to_cancel.append(self.announce_task)
        if self.ipv4_listen_task:
            self.ipv4_listen_task.cancel()
            tasks_to_cancel.append(self.ipv4_listen_task)
        if self.ipv6_listen_task:
            self.ipv6_listen_task.cancel()
            tasks_to_cancel.append(self.ipv6_listen_task)
        if self.cleanup_task:
            self.cleanup_task.cancel()
            tasks_to_cancel.append(self.cleanup_task)
        
        # Close the transports
        if self._ipv4_transport:
            try:
                logging.debug(f"Closing IPv4 multicast transport on port {self.multicast_port}",
                             component="multicast_discovery")
                self._ipv4_transport.close()
            except Exception as e:
                logging.error(f"Error closing IPv4 multicast transport: {e}",
                             component="multicast_discovery",
                             exc_info=e)
        
        if self._ipv6_transport:
            try:
                logging.debug(f"Closing IPv6 multicast transport on port {self.multicast_port}",
                             component="multicast_discovery")
                self._ipv6_transport.close()
            except Exception as e:
                logging.error(f"Error closing IPv6 multicast transport: {e}",
                             component="multicast_discovery",
                             exc_info=e)
        
        # Close all peer handles
        for peer_id, (peer_handle, _, _, _) in list(self.known_peers.items()):
            try:
                await peer_handle.close()
            except Exception as e:
                logging.warning(f"Error closing peer handle for {peer_id}",
                              component="multicast_discovery",
                              peer_id=peer_id,
                              exc_info=e)
        
        # Wait for tasks to complete
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logging.error(f"Error waiting for tasks to cancel: {e}",
                             component="multicast_discovery",
                             exc_info=e)
        
        # Clear internal state
        self.known_peers.clear()
        self.pending_msgs.clear()
        self.seen_message_ids.clear()
        self._ipv4_transport = None
        self._ipv6_transport = None
        
        # Clear thread-local logging context
        logging.clear_context()
        
        logging.info(f"Multicast discovery for node {self.node_id} stopped",
                    component="multicast_discovery")
    
    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """
        Discover peers on the network.
        
        Args:
            wait_for_peers: Number of peers to wait for
            
        Returns:
            List of discovered peer handles
        """
        if wait_for_peers > 0:
            timeout = time.time() + 30  # 30 second timeout
            while len(self.known_peers) < wait_for_peers and time.time() < timeout:
                logging.debug(f"Waiting for more peers: {len(self.known_peers)}/{wait_for_peers}",
                             component="multicast_discovery")
                await asyncio.sleep(0.5)
        
        return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]
    
    async def _setup_ipv4_listener(self):
        """Set up IPv4 multicast listener."""
        try:
            # Create an IPv4 socket with multicast options
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Allow reusing the address (important for multicast)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Attempt to set SO_REUSEPORT if available
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            
            # Bind to the multicast port on all interfaces
            sock.bind(('0.0.0.0', self.multicast_port))
            
            # Join the multicast group on all interfaces
            group = socket.inet_aton(self.ipv4_multicast_group)
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Create the datagram endpoint
            transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                lambda: MulticastListenProtocol(self._on_message_received),
                sock=sock
            )
            
            # Store the transport
            self._ipv4_transport = transport
            
            logging.info(f"IPv4 multicast listener started on {self.ipv4_multicast_group}:{self.multicast_port}",
                        component="multicast_discovery")
                        
        except Exception as e:
            logging.error(f"Failed to set up IPv4 multicast listener: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _setup_ipv6_listener(self):
        """Set up IPv6 multicast listener."""
        try:
            # Create an IPv6 socket with multicast options
            sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            
            # Allow reusing the address (important for multicast)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Attempt to set SO_REUSEPORT if available
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            
            # Bind to the multicast port on all interfaces
            sock.bind(('::', self.multicast_port))
            
            # Join the multicast group on all interfaces
            group_bin = socket.inet_pton(socket.AF_INET6, self.ipv6_multicast_group)
            
            # On some platforms, IPV6_JOIN_GROUP expects different parameters
            if platform.system() == 'Windows':
                # Windows uses a different structure: group address and interface index
                mreq = group_bin + struct.pack('@I', 0)  # Interface index 0 = any
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
            else:
                # Most Unix systems use a sockaddr_in6 structure
                mreq = group_bin + struct.pack('@I', 0)  # Interface index 0 = any
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
            
            # Create the datagram endpoint
            transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                lambda: MulticastListenProtocol(self._on_message_received),
                sock=sock
            )
            
            # Store the transport
            self._ipv6_transport = transport
            
            logging.info(f"IPv6 multicast listener started on [{self.ipv6_multicast_group}]:{self.multicast_port}",
                        component="multicast_discovery")
                        
        except Exception as e:
            logging.error(f"Failed to set up IPv6 multicast listener: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _task_announce_presence(self):
        """Task that periodically announces this node's presence on the network."""
        try:
            while self._is_running:
                await self._announce_presence()
                await asyncio.sleep(self.announcement_interval)
                
        except asyncio.CancelledError:
            logging.debug(f"Announce task for node {self.node_id} cancelled",
                         component="multicast_discovery")
            raise
            
        except Exception as e:
            logging.error(f"Error in announce task: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _announce_presence(self):
        """Announce this node's presence on the network."""
        try:
            # Get authentication instance
            auth = get_auth_instance(self.node_id)
            
            # Create a token for authentication
            node_token = auth.generate_token(
                peer_id=self.node_id,
                expiry_seconds=24 * 60 * 60  # 24 hours
            )
            
            # Create the announcement message
            timestamp = time.time()
            message_id = f"{self.node_id}:{timestamp}"
            
            # Sign the message
            signature = auth._sign_data(
                data=f"{self.node_id}.{timestamp}",
                key=node_token.token_secret
            )
            
            # Create the message JSON
            message = json.dumps({
                "type": "multicast_discovery",
                "message_id": message_id,
                "node_id": self.node_id,
                "grpc_port": self.node_port,
                "device_capabilities": self.device_capabilities.to_dict(),
                "timestamp": timestamp,
                "auth": {
                    "token_id": node_token.token_id,
                    "signature": signature,
                    "timestamp": timestamp
                }
            })
            
            # Send to IPv4 multicast group
            await self._send_multicast(message, self.ipv4_multicast_group, is_ipv6=False)
            
            # Send to IPv6 multicast group if enabled
            if self.use_ipv6:
                await self._send_multicast(message, self.ipv6_multicast_group, is_ipv6=True)
                
            logging.debug(f"Multicast announcement sent for node {self.node_id}",
                         component="multicast_discovery")
                         
        except Exception as e:
            logging.error(f"Error announcing presence: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _send_multicast(self, message: str, group: str, is_ipv6: bool = False):
        """
        Send a message to a multicast group.
        
        Args:
            message: Message to send
            group: Multicast group address
            is_ipv6: Whether the group is an IPv6 address
        """
        try:
            # Create a socket for the appropriate address family
            family = socket.AF_INET6 if is_ipv6 else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_DGRAM)
            
            # Allow reusing the address
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set TTL (or hop limit for IPv6)
            if is_ipv6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, self.ttl)
            else:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
            
            # Create the datagram endpoint and send the message
            transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                lambda: MulticastSendProtocol(message, group, self.multicast_port, self.ttl),
                sock=sock
            )
            
            # Note: The transport is closed by the protocol after sending
            
        except Exception as e:
            logging.error(f"Error sending multicast message to {group}: {e}",
                         component="multicast_discovery",
                         multicast_group=group,
                         is_ipv6=is_ipv6,
                         exc_info=e)
    
    async def _on_message_received(self, data: bytes, addr: Tuple[str, int]):
        """
        Handle a received multicast message.
        
        Args:
            data: Raw message data
            addr: Sender address as (host, port) tuple
        """
        if not data:
            return
        
        try:
            # Decode the message
            decoded_data = data.decode('utf-8', errors='ignore')
            
            # Check if it's valid JSON
            if not (decoded_data.strip() and decoded_data.strip()[0] in "{["):
                logging.debug(f"Received invalid JSON data from {addr}",
                             component="multicast_discovery",
                             data=decoded_data[:100])
                return
            
            # Parse the JSON
            message = json.loads(decoded_data)
            
            # Check if it's a discovery message and not from ourselves
            if message.get("type") == "multicast_discovery" and message.get("node_id") != self.node_id:
                # Extract message ID and check if we've seen it before
                message_id = message.get("message_id")
                if message_id in self.seen_message_ids:
                    return  # Already processed this message
                
                # Add to seen messages
                self.seen_message_ids.add(message_id)
                
                # Process the discovery message
                await self._process_discovery_message(message, addr)
                
                # Clean up old message IDs (keep only recent ones)
                self._cleanup_message_ids()
                
        except json.JSONDecodeError as e:
            logging.debug(f"Error decoding JSON data from {addr}: {e}",
                         component="multicast_discovery",
                         exc_info=e)
        except Exception as e:
            logging.error(f"Error processing multicast message: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _process_discovery_message(self, message: dict, addr: Tuple[str, int]):
        """
        Process a discovery message from another node.
        
        Args:
            message: Parsed message data
            addr: Sender address as (host, port) tuple
        """
        try:
            peer_id = message["node_id"]
            
            # Verify authentication if auth data is present
            auth = get_auth_instance(self.node_id)
            auth_data = message.get("auth", {})
            
            if auth_data:
                token_id = auth_data.get("token_id")
                signature = auth_data.get("signature")
                timestamp = auth_data.get("timestamp")
                
                # Verify authentication data if available
                if token_id and signature and timestamp:
                    # Verify timestamp is recent
                    current_time = time.time()
                    if abs(current_time - timestamp) > 300:  # 5 minutes
                        logging.warning(f"Ignoring peer {peer_id}: Message timestamp too old or in future",
                                      component="multicast_discovery",
                                      peer_id=peer_id,
                                      time_diff=abs(current_time - timestamp))
                        return
                    
                    # If peer is not authorized yet, generate a token for it
                    if not auth.is_peer_authorized(peer_id):
                        auth.generate_token(peer_id=peer_id)
                        logging.info(f"Provisionally accepted new peer {peer_id}",
                                    component="multicast_discovery",
                                    peer_id=peer_id)
            elif auth.auth_enabled and not auth.is_peer_authorized(peer_id):
                # Require authentication if enabled
                logging.warning(f"Ignoring unauthenticated peer {peer_id}",
                              component="multicast_discovery",
                              peer_id=peer_id)
                return
            
            # Check if peer is in allowed list
            if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
                logging.debug(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list",
                             component="multicast_discovery",
                             peer_id=peer_id)
                return
            
            # Extract peer information
            peer_host = addr[0]
            peer_port = message.get("grpc_port")
            device_capabilities = DeviceCapabilities(**message.get("device_capabilities", {}))
            
            # Clean IPv6 address format if needed
            if peer_host.startswith("[") and peer_host.endswith("]"):
                peer_host = peer_host[1:-1]
            
            # Create peer handle if needed
            peer_addr = f"{peer_host}:{peer_port}"
            interface_info = "Multicast"
            
            if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != peer_addr:
                if peer_id in self.known_peers:
                    try:
                        # Close old peer handle
                        await self.known_peers[peer_id][0].close()
                    except Exception as e:
                        logging.warning(f"Error closing old peer handle for {peer_id}",
                                      component="multicast_discovery",
                                      peer_id=peer_id,
                                      exc_info=e)
                
                # Create new peer handle
                new_peer_handle = self.create_peer_handle(peer_id, peer_addr, interface_info, device_capabilities)
                
                # Check if peer is healthy
                timeout_id = f"{self.health_check_timeout_id}.{peer_id}"
                
                try:
                    # Use timing context for health check if dynamic timeouts enabled
                    with TimingContext(timeout_id) if self.use_dynamic_timeouts else nullcontext():
                        is_healthy = await new_peer_handle.health_check()
                except Exception as e:
                    logging.warning(f"Error checking health of peer {peer_id}: {e}",
                                  component="multicast_discovery",
                                  peer_id=peer_id,
                                  exc_info=e)
                    is_healthy = False
                
                if not is_healthy:
                    logging.info(f"Peer {peer_id} at {peer_addr} is not healthy, skipping",
                                component="multicast_discovery",
                                peer_id=peer_id,
                                addr=peer_addr)
                    return
                
                # Add to known peers
                self.known_peers[peer_id] = (new_peer_handle, time.time(), time.time(), 0)
                logging.info(f"Added new peer {peer_id} at {peer_addr}",
                            component="multicast_discovery",
                            peer_id=peer_id,
                            addr=peer_addr)
            else:
                # Update last seen time for existing peer
                self.known_peers[peer_id] = (
                    self.known_peers[peer_id][0],  # peer handle
                    self.known_peers[peer_id][1],  # first seen
                    time.time(),                   # last seen
                    self.known_peers[peer_id][3]   # priority
                )
                
                logging.debug(f"Updated existing peer {peer_id}",
                            component="multicast_discovery",
                            peer_id=peer_id)
                
        except KeyError as e:
            logging.warning(f"Missing required field in discovery message: {e}",
                          component="multicast_discovery",
                          exc_info=e)
        except Exception as e:
            logging.error(f"Error processing discovery message: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    async def _task_cleanup_peers(self):
        """Task that periodically cleans up inactive peers."""
        try:
            while self._is_running:
                current_time = time.time()
                peers_to_remove = []
                
                # Check all known peers
                for peer_id, (peer_handle, _, last_seen, _) in list(self.known_peers.items()):
                    # Check if peer has expired based on timeout
                    if current_time - last_seen > self.discovery_timeout:
                        peers_to_remove.append(peer_id)
                        continue
                    
                    # Check if peer is still healthy
                    timeout_id = f"{self.health_check_timeout_id}.{peer_id}"
                    try:
                        with TimingContext(timeout_id) if self.use_dynamic_timeouts else nullcontext():
                            is_healthy = await peer_handle.health_check()
                        
                        if not is_healthy:
                            peers_to_remove.append(peer_id)
                    except Exception as e:
                        logging.warning(f"Error checking health of peer {peer_id}: {e}",
                                      component="multicast_discovery",
                                      peer_id=peer_id,
                                      exc_info=e)
                        peers_to_remove.append(peer_id)
                
                # Remove expired peers
                for peer_id in peers_to_remove:
                    if peer_id in self.known_peers:
                        try:
                            # Close the peer handle
                            await self.known_peers[peer_id][0].close()
                        except Exception as e:
                            logging.warning(f"Error closing peer handle for {peer_id}: {e}",
                                          component="multicast_discovery",
                                          peer_id=peer_id,
                                          exc_info=e)
                        finally:
                            del self.known_peers[peer_id]
                            logging.info(f"Removed peer {peer_id} due to inactivity or failed health check",
                                        component="multicast_discovery",
                                        peer_id=peer_id)
                
                # Clean up old message IDs
                self._cleanup_message_ids()
                
                # Sleep until next cleanup
                await asyncio.sleep(self.announcement_interval)
                
        except asyncio.CancelledError:
            logging.debug(f"Cleanup task for node {self.node_id} cancelled",
                         component="multicast_discovery")
            raise
        except Exception as e:
            logging.error(f"Error in cleanup task: {e}",
                         component="multicast_discovery",
                         exc_info=e)
    
    def _cleanup_message_ids(self):
        """Clean up old message IDs to prevent memory growth."""
        try:
            # Keep only the most recent 1000 message IDs
            if len(self.seen_message_ids) > 1000:
                # Sort by timestamp (part after colon)
                sorted_ids = sorted(
                    self.seen_message_ids,
                    key=lambda x: float(x.split(":", 1)[1]) if ":" in x else 0,
                    reverse=True
                )
                # Keep only the most recent
                self.seen_message_ids = set(sorted_ids[:1000])
        except Exception as e:
            logging.warning(f"Error cleaning up message IDs: {e}",
                          component="multicast_discovery",
                          exc_info=e)
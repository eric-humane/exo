"""
Discovery system with network partition handling.

This module provides a wrapper around discovery implementations that adds
network partition detection, recovery, and resilience features.
"""

import asyncio
import time
import json
from typing import List, Dict, Set, Optional, Any, Callable, Awaitable

from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.networking.partition_recovery import (
    NetworkPartitionManager,
    PartitionEvent,
    create_partition_multicast_sender,
    create_partition_multicast_receiver
)
from exo.utils import logging


class PartitionAwareDiscovery(Discovery):
    """
    Wrapper for Discovery implementations that adds network partition handling.
    
    This class wraps any Discovery implementation and adds:
    1. Network partition detection
    2. Partition recovery
    3. Special recovery messages through multicast
    """
    
    def __init__(
        self,
        base_discovery: Discovery,
        node_id: str,
        partition_detection_enabled: bool = True,
        suspicious_threshold: int = 3,      # How many peer failures before becoming suspicious
        partition_timeout: float = 10.0,    # Seconds before declaring partitioned after becoming suspicious
        recovery_timeout: float = 30.0,     # Seconds to wait for recovery before giving up
        heartbeat_interval: float = 2.0,    # Seconds between heartbeats
        leader_timeout: float = 5.0,        # Seconds before considering leader as failed
        multicast_group: str = "224.0.0.252",
        multicast_port: int = 5354
    ):
        """
        Initialize the partition-aware discovery wrapper.
        
        Args:
            base_discovery: The underlying Discovery implementation to wrap
            node_id: Unique identifier for this node
            partition_detection_enabled: Whether to enable partition detection
            suspicious_threshold: Number of peer failures to enter suspicious state
            partition_timeout: Time to wait in suspicious state before declaring partitioned
            recovery_timeout: Maximum time to wait for recovery
            heartbeat_interval: Time between leader heartbeats
            leader_timeout: Time after which a leader is considered failed if no heartbeat
            multicast_group: Multicast group for partition recovery messages
            multicast_port: Multicast port for partition recovery messages
        """
        self.base_discovery = base_discovery
        self.node_id = node_id
        self.partition_detection_enabled = partition_detection_enabled
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        
        # Track connected peers
        self.connected_peers: Set[str] = set()
        self.peer_health_status: Dict[str, bool] = {}
        
        # Cached peer handles
        self._cached_peers: Dict[str, PeerHandle] = {}
        
        # Tasks
        self._monitor_task = None
        self._receiver_task = None
        
        # Partition manager
        self._partition_manager = None
        if partition_detection_enabled:
            self._partition_manager = NetworkPartitionManager(
                node_id=node_id,
                broadcast_fn=self._send_partition_message,
                rejoin_network_fn=self._rejoin_network,
                suspicious_threshold=suspicious_threshold,
                partition_timeout=partition_timeout,
                recovery_timeout=recovery_timeout,
                heartbeat_interval=heartbeat_interval,
                leader_timeout=leader_timeout
            )
        
        # State flag
        self._is_running = False
    
    async def __aenter__(self):
        """Async context manager entry - starts the discovery service."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - stops the discovery service."""
        await self.stop()
        return False  # Don't suppress exceptions
    
    async def start(self):
        """Start the discovery service."""
        if self._is_running:
            logging.debug(f"Partition-aware discovery for node {self.node_id} already running",
                         component="partition_aware_discovery")
            return
            
        self._is_running = True
        
        # Start base discovery
        logging.info(f"Starting partition-aware discovery for node {self.node_id}",
                    component="partition_aware_discovery")
        await self.base_discovery.start()
        
        # Set up and start partition manager if enabled
        if self.partition_detection_enabled and self._partition_manager:
            # Create multicast sender
            self._multicast_sender = await create_partition_multicast_sender(
                node_id=self.node_id,
                multicast_group=self.multicast_group,
                multicast_port=self.multicast_port
            )
            
            # Create and start multicast receiver
            self._receiver_task = await create_partition_multicast_receiver(
                node_id=self.node_id,
                message_handler=self._handle_partition_message,
                multicast_group=self.multicast_group,
                multicast_port=self.multicast_port
            )
            
            # Start partition manager
            await self._partition_manager.start()
            
            # Add event handlers for partition events
            await self._partition_manager.add_event_handler(
                PartitionEvent.CONNECTIVITY_LOST,
                self._on_connectivity_lost
            )
            await self._partition_manager.add_event_handler(
                PartitionEvent.CONNECTIVITY_RESTORED,
                self._on_connectivity_restored
            )
            
            # Start peer monitor task
            self._monitor_task = asyncio.create_task(self._monitor_peers())
            
            logging.info(f"Partition detection enabled for node {self.node_id}",
                        component="partition_aware_discovery")
                        
    async def stop(self):
        """Stop the discovery service."""
        if not self._is_running:
            return
            
        self._is_running = False
        
        # Stop partition manager if enabled
        if self.partition_detection_enabled and self._partition_manager:
            await self._partition_manager.stop()
            
            # Cancel receiver task
            if self._receiver_task:
                self._receiver_task.cancel()
                try:
                    await self._receiver_task
                except asyncio.CancelledError:
                    pass
                    
            # Cancel monitor task
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
        
        # Stop base discovery
        await self.base_discovery.stop()
        
        # Clear state
        self.connected_peers.clear()
        self.peer_health_status.clear()
        self._cached_peers.clear()
        
        logging.info(f"Stopped partition-aware discovery for node {self.node_id}",
                    component="partition_aware_discovery")
    
    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """
        Discover peers on the network.
        
        Args:
            wait_for_peers: Number of peers to wait for
            
        Returns:
            List of discovered peer handles
        """
        # Use base discovery to find peers
        peers = await self.base_discovery.discover_peers(wait_for_peers)
        
        # Update connected peers
        for peer in peers:
            peer_id = peer.id()
            self.connected_peers.add(peer_id)
            self._cached_peers[peer_id] = peer
            
            # Notify partition manager if enabled
            if self.partition_detection_enabled and self._partition_manager:
                await self._partition_manager.node_connected(peer_id)
        
        return peers
    
    async def _monitor_peers(self):
        """Task that monitors peer health and detects disconnections."""
        try:
            while self._is_running:
                # Get current peers
                peers = await self.base_discovery.discover_peers()
                
                # Get current peer IDs
                current_peer_ids = {peer.id() for peer in peers}
                
                # Find disconnected peers
                for peer_id in list(self.connected_peers):
                    if peer_id not in current_peer_ids:
                        # Consider the peer disconnected
                        self.connected_peers.discard(peer_id)
                        self.peer_health_status.pop(peer_id, None)
                        self._cached_peers.pop(peer_id, None)
                        
                        # Notify partition manager
                        if self.partition_detection_enabled and self._partition_manager:
                            await self._partition_manager.node_disconnected(peer_id)
                            
                        logging.info(f"Peer {peer_id} disconnected",
                                    component="partition_aware_discovery")
                
                # Check health of current peers
                for peer in peers:
                    peer_id = peer.id()
                    
                    # Add to connected peers if new
                    if peer_id not in self.connected_peers:
                        self.connected_peers.add(peer_id)
                        self._cached_peers[peer_id] = peer
                        
                        # Notify partition manager
                        if self.partition_detection_enabled and self._partition_manager:
                            await self._partition_manager.node_connected(peer_id)
                            
                        logging.info(f"Peer {peer_id} connected",
                                    component="partition_aware_discovery")
                    
                    # Check health
                    try:
                        is_healthy = await peer.health_check()
                        
                        # Update health status
                        prev_status = self.peer_health_status.get(peer_id)
                        self.peer_health_status[peer_id] = is_healthy
                        
                        # Log health changes
                        if prev_status is not None and prev_status != is_healthy:
                            if is_healthy:
                                logging.info(f"Peer {peer_id} health restored",
                                           component="partition_aware_discovery")
                            else:
                                logging.warning(f"Peer {peer_id} health degraded",
                                              component="partition_aware_discovery")
                                              
                    except Exception as e:
                        logging.warning(f"Error checking health of peer {peer_id}: {e}",
                                      component="partition_aware_discovery",
                                      exc_info=e)
                        
                        # Consider the peer unhealthy
                        self.peer_health_status[peer_id] = False
                
                # Sleep before next check
                await asyncio.sleep(5.0)
                
        except asyncio.CancelledError:
            logging.debug("Peer monitor task cancelled",
                         component="partition_aware_discovery")
            raise
        except Exception as e:
            logging.error(f"Fatal error in peer monitor task: {e}",
                         component="partition_aware_discovery",
                         exc_info=e)
    
    async def _send_partition_message(self, message_type: str, message_data: bytes):
        """
        Send a partition-related message to other nodes.
        
        Args:
            message_type: Type of message
            message_data: Message content
        """
        try:
            # Send via multicast
            await self._multicast_sender(message_type, message_data)
        except Exception as e:
            logging.error(f"Error sending partition message: {e}",
                         component="partition_aware_discovery",
                         message_type=message_type,
                         exc_info=e)
    
    async def _handle_partition_message(self, source_node_id: str, message_type: str, message_data: dict):
        """
        Handle a partition-related message from another node.
        
        Args:
            source_node_id: ID of the node that sent the message
            message_type: Type of message
            message_data: Message data
        """
        # Forward to partition manager
        if self.partition_detection_enabled and self._partition_manager:
            await self._partition_manager.receive_message(source_node_id, message_type, message_data)
    
    async def _rejoin_network(self):
        """Attempt to rejoin the network after a partition."""
        try:
            logging.info(f"Attempting to rejoin network after partition",
                        component="partition_aware_discovery")
            
            # Restart the base discovery
            await self.base_discovery.stop()
            await asyncio.sleep(1.0)  # Brief pause
            await self.base_discovery.start()
            
            # Rediscover peers
            await self.discover_peers()
            
            logging.info(f"Successfully rejoined network",
                        component="partition_aware_discovery")
                        
            return True
        except Exception as e:
            logging.error(f"Error rejoining network: {e}",
                         component="partition_aware_discovery",
                         exc_info=e)
            return False
    
    async def _on_connectivity_lost(self, data: Any):
        """
        Handle connectivity lost event.
        
        Args:
            data: Event data
        """
        logging.warning(f"Network connectivity lost - partition detected",
                      component="partition_aware_discovery",
                      partition_id=data.get("partition_id"),
                      disconnected_nodes=data.get("disconnected_nodes"))
    
    async def _on_connectivity_restored(self, data: Any):
        """
        Handle connectivity restored event.
        
        Args:
            data: Event data
        """
        logging.info(f"Network connectivity restored",
                    component="partition_aware_discovery",
                    connected_nodes=data.get("connected_nodes"))
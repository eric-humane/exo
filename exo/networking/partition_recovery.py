"""
Network partition detection and recovery mechanisms.

This module provides utilities for detecting and recovering from network
partitions in distributed systems.
"""

import asyncio
import time
import random
import json
import socket
import struct
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, Awaitable
from enum import Enum, auto
import threading
from datetime import datetime, timezone

from exo.utils import logging


class PartitionState(Enum):
    """States of a network partition manager."""
    NORMAL = auto()        # Normal operation
    SUSPICIOUS = auto()    # May be partitioned
    PARTITIONED = auto()   # Confirmed partitioned
    RECOVERING = auto()    # Recovering from partition


class PartitionEvent(Enum):
    """Events that can occur in a partition detection system."""
    CONNECTIVITY_LOST = auto()      # Lost connection to multiple peers
    CONNECTIVITY_RESTORED = auto()  # Restored connection to peers
    TIMEOUT = auto()                # Timeout while waiting for recovery
    RECOVERY_STARTED = auto()       # Started recovery process
    RECOVERY_COMPLETED = auto()     # Completed recovery process
    HEARTBEAT_MISSING = auto()      # Missing heartbeat from partition leader
    HEARTBEAT_RESTORED = auto()     # Restored heartbeat from partition leader


class NetworkPartitionManager:
    """
    Manager for detecting and recovering from network partitions.
    
    This class provides:
    1. Detection of network partitions through peer connectivity monitoring
    2. Leader election within partitions
    3. Partition recovery through special recovery messages
    4. Automatic re-synchronization of state after recovery
    """
    
    def __init__(
        self,
        node_id: str,
        broadcast_fn: Callable[[str, bytes], Awaitable[None]],
        rejoin_network_fn: Callable[[], Awaitable[None]],
        suspicious_threshold: int = 3,      # How many peer failures before becoming suspicious
        partition_timeout: float = 10.0,    # Seconds before declaring partitioned after becoming suspicious
        recovery_timeout: float = 30.0,     # Seconds to wait for recovery before giving up
        heartbeat_interval: float = 2.0,    # Seconds between heartbeats
        leader_timeout: float = 5.0,        # Seconds before considering leader as failed
    ):
        """
        Initialize the partition manager.
        
        Args:
            node_id: Unique identifier for this node
            broadcast_fn: Function to broadcast a message to all directly connected peers
            rejoin_network_fn: Function to attempt to rejoin the network (typically restarts discovery)
            suspicious_threshold: Number of peer failures to enter suspicious state
            partition_timeout: Time to wait in suspicious state before declaring partitioned
            recovery_timeout: Maximum time to wait for recovery
            heartbeat_interval: Time between leader heartbeats
            leader_timeout: Time after which a leader is considered failed if no heartbeat
        """
        self.node_id = node_id
        self.broadcast_fn = broadcast_fn
        self.rejoin_network_fn = rejoin_network_fn
        self.suspicious_threshold = suspicious_threshold
        self.partition_timeout = partition_timeout
        self.recovery_timeout = recovery_timeout
        self.heartbeat_interval = heartbeat_interval
        self.leader_timeout = leader_timeout
        
        # State
        self.state = PartitionState.NORMAL
        self.connected_nodes: Set[str] = set()
        self.disconnected_nodes: Dict[str, float] = {}  # node_id -> time_disconnected
        self.partition_leaders: Dict[str, float] = {}   # leader_id -> last_heartbeat_time
        self.my_partition_id: Optional[str] = None      # ID of the partition this node is in
        self.leader_node_id: Optional[str] = None       # Node ID of the current leader
        self.last_leader_heartbeat: float = 0           # Time of last leader heartbeat
        self.recovery_start_time: Optional[float] = None
        
        # Component-specific state
        self.recovery_attempts: int = 0
        self.recovery_backoff: float = 1.0  # Initial backoff time
        
        # Tasks
        self._monitor_task = None
        self._heartbeat_task = None
        self._recovery_task = None
        
        # Locks
        self._state_lock = asyncio.Lock()
        
        # Event handlers
        self._event_handlers: Dict[PartitionEvent, List[Callable[[Any], Awaitable[None]]]] = {
            event: [] for event in PartitionEvent
        }
        
        # Assign a random weight for leader election
        # Use some deterministic node properties as part of the weight to ensure consistent election
        self._node_weight = random.Random(node_id).random()
    
    async def start(self):
        """Start the partition manager."""
        logging.info(f"Starting network partition manager for node {self.node_id}",
                    component="partition_recovery")
        
        # Start the monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_partition_state())
        
        # Initialize as not partitioned
        await self._set_state(PartitionState.NORMAL)
    
    async def stop(self):
        """Stop the partition manager."""
        logging.info(f"Stopping network partition manager for node {self.node_id}",
                    component="partition_recovery")
        
        # Cancel all tasks
        tasks = []
        if self._monitor_task:
            self._monitor_task.cancel()
            tasks.append(self._monitor_task)
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            tasks.append(self._heartbeat_task)
        
        if self._recovery_task:
            self._recovery_task.cancel()
            tasks.append(self._recovery_task)
        
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Reset state
        self.state = PartitionState.NORMAL
        self.connected_nodes.clear()
        self.disconnected_nodes.clear()
        self.partition_leaders.clear()
        self.my_partition_id = None
        self.leader_node_id = None
        self.recovery_start_time = None
    
    async def node_connected(self, node_id: str):
        """
        Called when a node is connected.
        
        Args:
            node_id: ID of the node that connected
        """
        async with self._state_lock:
            self.connected_nodes.add(node_id)
            if node_id in self.disconnected_nodes:
                del self.disconnected_nodes[node_id]
            
            # If we're in a partition, try to recover
            if self.state == PartitionState.PARTITIONED:
                await self._attempt_recovery()
    
    async def node_disconnected(self, node_id: str):
        """
        Called when a node is disconnected.
        
        Args:
            node_id: ID of the node that disconnected
        """
        async with self._state_lock:
            if node_id in self.connected_nodes:
                self.connected_nodes.remove(node_id)
            
            # Record disconnection time
            self.disconnected_nodes[node_id] = time.time()
            
            # Check if we need to become suspicious
            if len(self.disconnected_nodes) >= self.suspicious_threshold and self.state == PartitionState.NORMAL:
                await self._set_state(PartitionState.SUSPICIOUS)
    
    async def receive_message(self, source_node_id: str, message_type: str, message_data: dict):
        """
        Process a partition-related message.
        
        Args:
            source_node_id: ID of the node that sent the message
            message_type: Type of message
            message_data: Message data
        """
        try:
            if message_type == "partition_heartbeat":
                await self._handle_heartbeat(source_node_id, message_data)
            elif message_type == "partition_recovery":
                await self._handle_recovery(source_node_id, message_data)
            elif message_type == "partition_leader_election":
                await self._handle_leader_election(source_node_id, message_data)
            else:
                logging.warning(f"Received unknown partition message type: {message_type}",
                              component="partition_recovery",
                              source_node_id=source_node_id)
        except Exception as e:
            logging.error(f"Error processing partition message: {e}",
                         component="partition_recovery",
                         message_type=message_type,
                         source_node_id=source_node_id,
                         exc_info=e)
    
    async def add_event_handler(self, event: PartitionEvent, handler: Callable[[Any], Awaitable[None]]):
        """
        Add an event handler for a specific event.
        
        Args:
            event: The event to handle
            handler: Async function to call when the event occurs
        """
        self._event_handlers[event].append(handler)
    
    async def _set_state(self, new_state: PartitionState):
        """
        Set the partition state and perform state-specific actions.
        
        Args:
            new_state: The new state to set
        """
        if new_state == self.state:
            return
        
        old_state = self.state
        self.state = new_state
        
        logging.info(f"Network partition state changed: {old_state.name} -> {new_state.name}",
                    component="partition_recovery",
                    node_id=self.node_id)
        
        # Perform state-specific actions
        if new_state == PartitionState.SUSPICIOUS:
            # Start a timer to go to PARTITIONED if not resolved
            async def become_partitioned():
                await asyncio.sleep(self.partition_timeout)
                async with self._state_lock:
                    if self.state == PartitionState.SUSPICIOUS:
                        await self._set_state(PartitionState.PARTITIONED)
            
            asyncio.create_task(become_partitioned())
            
        elif new_state == PartitionState.PARTITIONED:
            # We are officially partitioned, start partition recovery
            self.my_partition_id = f"partition-{self.node_id}-{int(time.time())}"
            await self._start_leader_election()
            await self._trigger_event(PartitionEvent.CONNECTIVITY_LOST, {
                "disconnected_nodes": list(self.disconnected_nodes.keys()),
                "partition_id": self.my_partition_id
            })
            
        elif new_state == PartitionState.RECOVERING:
            # Start recovery process
            self.recovery_start_time = time.time()
            await self._trigger_event(PartitionEvent.RECOVERY_STARTED, {
                "partition_id": self.my_partition_id,
                "recovery_attempt": self.recovery_attempts
            })
            
            # Start recovery timeout
            async def recovery_timeout():
                await asyncio.sleep(self.recovery_timeout)
                async with self._state_lock:
                    if self.state == PartitionState.RECOVERING:
                        # Recovery failed, go back to PARTITIONED
                        await self._set_state(PartitionState.PARTITIONED)
                        await self._trigger_event(PartitionEvent.TIMEOUT, {
                            "recovery_time": time.time() - self.recovery_start_time
                        })
            
            self._recovery_task = asyncio.create_task(recovery_timeout())
            
        elif new_state == PartitionState.NORMAL:
            # Clear partition-related state
            self.my_partition_id = None
            self.leader_node_id = None
            self.recovery_start_time = None
            self.recovery_attempts = 0
            self.recovery_backoff = 1.0
            
            # Cancel recovery task if it exists
            if self._recovery_task:
                self._recovery_task.cancel()
                self._recovery_task = None
            
            # Cancel heartbeat task if it exists
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                self._heartbeat_task = None
            
            # Trigger recovery completed event if we were recovering
            if old_state == PartitionState.RECOVERING:
                await self._trigger_event(PartitionEvent.RECOVERY_COMPLETED, {
                    "recovery_time": time.time() - (self.recovery_start_time or time.time())
                })
            elif old_state == PartitionState.PARTITIONED:
                await self._trigger_event(PartitionEvent.CONNECTIVITY_RESTORED, {
                    "connected_nodes": list(self.connected_nodes)
                })
    
    async def _monitor_partition_state(self):
        """Task that monitors the partition state."""
        try:
            while True:
                try:
                    current_time = time.time()
                    
                    async with self._state_lock:
                        # Monitor leader heartbeats if we have a leader
                        if (self.state in (PartitionState.PARTITIONED, PartitionState.RECOVERING) and
                                self.leader_node_id and self.leader_node_id != self.node_id):
                            if current_time - self.last_leader_heartbeat > self.leader_timeout:
                                # Leader timeout, start a new election
                                logging.warning(f"Leader {self.leader_node_id} timed out, starting election",
                                              component="partition_recovery")
                                await self._trigger_event(PartitionEvent.HEARTBEAT_MISSING, {
                                    "leader_id": self.leader_node_id,
                                    "last_heartbeat": self.last_leader_heartbeat,
                                    "current_time": current_time
                                })
                                await self._start_leader_election()
                        
                        # If normal state but many disconnected nodes, become suspicious
                        if (self.state == PartitionState.NORMAL and 
                                len(self.disconnected_nodes) >= self.suspicious_threshold):
                            await self._set_state(PartitionState.SUSPICIOUS)
                        
                        # If suspicious state but no longer many disconnected nodes, return to normal
                        elif (self.state == PartitionState.SUSPICIOUS and 
                                len(self.disconnected_nodes) < self.suspicious_threshold):
                            await self._set_state(PartitionState.NORMAL)
                
                except Exception as e:
                    logging.error(f"Error in partition state monitor: {e}",
                                 component="partition_recovery",
                                 exc_info=e)
                
                await asyncio.sleep(1.0)  # Check state every second
                
        except asyncio.CancelledError:
            logging.debug("Partition state monitor task cancelled",
                         component="partition_recovery")
            raise
        except Exception as e:
            logging.error(f"Fatal error in partition state monitor: {e}",
                         component="partition_recovery",
                         exc_info=e)
    
    async def _start_leader_election(self):
        """Start a leader election in the current partition."""
        # Cancel any existing heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        
        # Send leader election message
        election_message = {
            "type": "partition_leader_election",
            "partition_id": self.my_partition_id,
            "node_id": self.node_id,
            "weight": self._node_weight,
            "timestamp": time.time()
        }
        
        try:
            await self.broadcast_fn("partition_leader_election", json.dumps(election_message).encode("utf-8"))
            
            # Initially set ourselves as leader
            self.leader_node_id = self.node_id
            
            # Start sending heartbeats if we are the leader
            self._heartbeat_task = asyncio.create_task(self._leader_heartbeat_task())
            
            logging.info(f"Started leader election for partition {self.my_partition_id}",
                        component="partition_recovery",
                        weight=self._node_weight)
                        
        except Exception as e:
            logging.error(f"Error starting leader election: {e}",
                         component="partition_recovery",
                         exc_info=e)
    
    async def _leader_heartbeat_task(self):
        """Task that sends leader heartbeats if this node is the leader."""
        try:
            while self.leader_node_id == self.node_id:
                try:
                    # Prepare heartbeat message
                    heartbeat_message = {
                        "type": "partition_heartbeat",
                        "partition_id": self.my_partition_id,
                        "leader_id": self.node_id,
                        "timestamp": time.time(),
                        "state": self.state.name,
                        "connected_nodes": list(self.connected_nodes),
                        "recovery_attempts": self.recovery_attempts
                    }
                    
                    # Send heartbeat
                    await self.broadcast_fn("partition_heartbeat", json.dumps(heartbeat_message).encode("utf-8"))
                    
                    # Record our own heartbeat
                    self.last_leader_heartbeat = time.time()
                    
                except Exception as e:
                    logging.error(f"Error sending leader heartbeat: {e}",
                                 component="partition_recovery",
                                 exc_info=e)
                
                # Wait for next heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
        except asyncio.CancelledError:
            logging.debug("Leader heartbeat task cancelled",
                         component="partition_recovery")
            raise
        except Exception as e:
            logging.error(f"Fatal error in leader heartbeat task: {e}",
                         component="partition_recovery",
                         exc_info=e)
    
    async def _attempt_recovery(self):
        """Attempt to recover from a partition."""
        if self.state != PartitionState.PARTITIONED or self.leader_node_id != self.node_id:
            return  # Only the leader should initiate recovery
        
        # Increment attempt counter
        self.recovery_attempts += 1
        
        # Set state to recovering
        await self._set_state(PartitionState.RECOVERING)
        
        try:
            # Prepare recovery message
            recovery_message = {
                "type": "partition_recovery",
                "partition_id": self.my_partition_id,
                "leader_id": self.node_id,
                "timestamp": time.time(),
                "attempt": self.recovery_attempts,
                "connected_nodes": list(self.connected_nodes)
            }
            
            # Broadcast recovery message
            await self.broadcast_fn("partition_recovery", json.dumps(recovery_message).encode("utf-8"))
            
            # If we have enough connected nodes, consider recovery successful
            if len(self.connected_nodes) >= self.suspicious_threshold:
                # Give a little time for messages to propagate
                await asyncio.sleep(2.0)
                
                # Try to rejoin the network
                await self.rejoin_network_fn()
                
                # Return to normal state
                await self._set_state(PartitionState.NORMAL)
                
            else:
                # Not enough nodes, backoff and try again later
                backoff = self.recovery_backoff * (1.0 + 0.2 * random.random())  # Add jitter
                self.recovery_backoff = min(backoff, 60.0)  # Cap at 60 seconds
                
                async def delayed_retry():
                    await asyncio.sleep(backoff)
                    async with self._state_lock:
                        if self.state == PartitionState.RECOVERING:
                            # Go back to partitioned to try again
                            await self._set_state(PartitionState.PARTITIONED)
                
                asyncio.create_task(delayed_retry())
            
        except Exception as e:
            logging.error(f"Error attempting partition recovery: {e}",
                         component="partition_recovery",
                         exc_info=e)
    
    async def _handle_heartbeat(self, source_node_id: str, message_data: dict):
        """
        Handle a heartbeat message from a partition leader.
        
        Args:
            source_node_id: ID of the node that sent the message
            message_data: Heartbeat message data
        """
        async with self._state_lock:
            partition_id = message_data.get("partition_id")
            leader_id = message_data.get("leader_id")
            timestamp = message_data.get("timestamp", 0)
            
            # Validate message
            if not partition_id or not leader_id:
                logging.warning(f"Received invalid heartbeat message",
                              component="partition_recovery",
                              source_node_id=source_node_id)
                return
            
            # Only process heartbeats from our partition or if we're in NORMAL state
            if self.state == PartitionState.NORMAL or partition_id == self.my_partition_id:
                # Update leader info
                self.partition_leaders[leader_id] = timestamp
                
                # If this is our leader, update last heartbeat time
                if leader_id == self.leader_node_id:
                    self.last_leader_heartbeat = time.time()
                    
                    # Trigger event if heartbeat was missing
                    if self.state in (PartitionState.PARTITIONED, PartitionState.RECOVERING):
                        await self._trigger_event(PartitionEvent.HEARTBEAT_RESTORED, {
                            "leader_id": leader_id,
                            "partition_id": partition_id
                        })
                
                # If no leader yet, accept this one
                elif not self.leader_node_id and self.state != PartitionState.NORMAL:
                    self.leader_node_id = leader_id
                    self.last_leader_heartbeat = time.time()
                    self.my_partition_id = partition_id
                    
                    # Cancel our heartbeat task if we have one
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()
                        self._heartbeat_task = None
                        
                    logging.info(f"Accepted leader {leader_id} for partition {partition_id}",
                                component="partition_recovery")
    
    async def _handle_recovery(self, source_node_id: str, message_data: dict):
        """
        Handle a recovery message.
        
        Args:
            source_node_id: ID of the node that sent the message
            message_data: Recovery message data
        """
        async with self._state_lock:
            partition_id = message_data.get("partition_id")
            leader_id = message_data.get("leader_id")
            
            # Validate message
            if not partition_id or not leader_id:
                logging.warning(f"Received invalid recovery message",
                              component="partition_recovery",
                              source_node_id=source_node_id)
                return
            
            # If we're in a partition and this is our leader, follow their lead
            if (self.state in (PartitionState.PARTITIONED, PartitionState.RECOVERING) and
                    (self.my_partition_id == partition_id or self.leader_node_id == leader_id)):
                
                # Update leader
                self.leader_node_id = leader_id
                self.my_partition_id = partition_id
                self.last_leader_heartbeat = time.time()
                
                # Set to recovering state if we're not already
                if self.state != PartitionState.RECOVERING:
                    await self._set_state(PartitionState.RECOVERING)
                
                # Try to rejoin the network if there are enough connected nodes
                connected_nodes = set(message_data.get("connected_nodes", []))
                if len(connected_nodes.union(self.connected_nodes)) >= self.suspicious_threshold:
                    # Try to rejoin the network
                    await self.rejoin_network_fn()
                    
                    # Return to normal state
                    await self._set_state(PartitionState.NORMAL)
    
    async def _handle_leader_election(self, source_node_id: str, message_data: dict):
        """
        Handle a leader election message.
        
        Args:
            source_node_id: ID of the node that sent the message
            message_data: Leader election message data
        """
        async with self._state_lock:
            partition_id = message_data.get("partition_id")
            node_id = message_data.get("node_id")
            weight = message_data.get("weight", 0.0)
            timestamp = message_data.get("timestamp", 0)
            
            # Validate message
            if not partition_id or not node_id:
                logging.warning(f"Received invalid leader election message",
                              component="partition_recovery",
                              source_node_id=source_node_id)
                return
            
            # Only process elections for our partition
            if self.my_partition_id == partition_id:
                # Determine if the sender should be leader
                sender_should_be_leader = False
                
                # If we're not a leader yet, accept any leader
                if not self.leader_node_id:
                    sender_should_be_leader = True
                
                # If the sender has higher weight, they should be leader
                elif weight > self._node_weight:
                    sender_should_be_leader = True
                
                # If equal weight, use node ID as tiebreaker
                elif weight == self._node_weight and node_id > self.node_id:
                    sender_should_be_leader = True
                
                # Update leader if necessary
                if sender_should_be_leader:
                    if self.leader_node_id == self.node_id:
                        # We were the leader, but now we're not
                        logging.info(f"Yielding leadership to {node_id} with weight {weight}",
                                   component="partition_recovery",
                                   our_weight=self._node_weight)
                        
                        # Cancel our heartbeat task
                        if self._heartbeat_task:
                            self._heartbeat_task.cancel()
                            self._heartbeat_task = None
                    
                    # Update leader
                    self.leader_node_id = node_id
                    self.last_leader_heartbeat = time.time()
                    
                # If we have higher weight, assert our leadership
                elif self.leader_node_id == self.node_id and weight < self._node_weight:
                    # Send a new election message to assert our leadership
                    election_message = {
                        "type": "partition_leader_election",
                        "partition_id": self.my_partition_id,
                        "node_id": self.node_id,
                        "weight": self._node_weight,
                        "timestamp": time.time()
                    }
                    
                    await self.broadcast_fn("partition_leader_election", json.dumps(election_message).encode("utf-8"))
                    
                    logging.info(f"Asserting leadership over {node_id} with weight {weight}",
                               component="partition_recovery",
                               our_weight=self._node_weight)
    
    async def _trigger_event(self, event: PartitionEvent, data: Any):
        """
        Trigger an event with the given data.
        
        Args:
            event: The event to trigger
            data: Data to pass to event handlers
        """
        for handler in self._event_handlers[event]:
            try:
                await handler(data)
            except Exception as e:
                logging.error(f"Error in event handler for {event.name}: {e}",
                             component="partition_recovery",
                             exc_info=e)


# Function to create a multicast message sender for partition recovery
async def create_partition_multicast_sender(
    node_id: str,
    multicast_group: str = "224.0.0.252",
    multicast_port: int = 5354,
    ttl: int = 32
) -> Callable[[str, bytes], Awaitable[None]]:
    """
    Create a function that sends multicast messages for partition recovery.
    
    This creates a multicast sender that uses a different multicast group/port
    than the main discovery system to avoid interference.
    
    Args:
        node_id: ID of this node
        multicast_group: Multicast group address
        multicast_port: Multicast port
        ttl: TTL (time to live) for multicast packets
        
    Returns:
        Function that sends a multicast message
    """
    
    async def send_multicast(message_type: str, message_data: bytes) -> None:
        """
        Send a multicast message.
        
        Args:
            message_type: Type of message
            message_data: Message content
        """
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set TTL
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            
            # Create message with header
            header = f"{node_id}:{message_type}:".encode("utf-8")
            full_message = header + message_data
            
            # Send the message
            sock.sendto(full_message, (multicast_group, multicast_port))
            
            # Close the socket
            sock.close()
            
        except Exception as e:
            logging.error(f"Error sending multicast message: {e}",
                         component="partition_recovery",
                         message_type=message_type,
                         exc_info=e)
    
    return send_multicast


# Function to create a multicast message receiver for partition recovery
async def create_partition_multicast_receiver(
    node_id: str,
    message_handler: Callable[[str, str, dict], Awaitable[None]],
    multicast_group: str = "224.0.0.252",
    multicast_port: int = 5354
) -> asyncio.Task:
    """
    Create a task that receives multicast messages for partition recovery.
    
    Args:
        node_id: ID of this node
        message_handler: Function to call with received messages
        multicast_group: Multicast group address
        multicast_port: Multicast port
        
    Returns:
        Task that receives multicast messages
    """
    
    async def receive_task():
        """Task that receives multicast messages."""
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to port
            sock.bind(('', multicast_port))
            
            # Join multicast group
            group = socket.inet_aton(multicast_group)
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Make socket non-blocking
            sock.setblocking(False)
            
            logging.info(f"Partition recovery multicast receiver started on {multicast_group}:{multicast_port}",
                        component="partition_recovery")
            
            # Receive loop
            while True:
                # Wait for data with asyncio
                data, addr = await asyncio.get_event_loop().sock_recvfrom(sock, 4096)
                
                # Process the message
                try:
                    # Decode header
                    header, data = data.split(b':', 2)
                    source_node_id, message_type = header.decode("utf-8").split(':', 1)
                    
                    # Ignore our own messages
                    if source_node_id == node_id:
                        continue
                    
                    # Parse message data as JSON
                    message_data = json.loads(data.decode("utf-8"))
                    
                    # Pass to handler
                    await message_handler(source_node_id, message_type, message_data)
                    
                except ValueError:
                    logging.warning(f"Received invalid partition message from {addr}",
                                  component="partition_recovery")
                except json.JSONDecodeError:
                    logging.warning(f"Received invalid JSON in partition message from {addr}",
                                  component="partition_recovery")
                except Exception as e:
                    logging.error(f"Error processing partition message: {e}",
                                 component="partition_recovery",
                                 exc_info=e)
                    
        except asyncio.CancelledError:
            logging.debug("Partition multicast receiver task cancelled",
                         component="partition_recovery")
            sock.close()
            raise
        except Exception as e:
            logging.error(f"Fatal error in partition multicast receiver: {e}",
                         component="partition_recovery",
                         exc_info=e)
            sock.close()
    
    # Create and return the task
    return asyncio.create_task(receive_task())
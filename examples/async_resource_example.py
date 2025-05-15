#!/usr/bin/env python3
"""
Example showing how to convert an existing class to use the AsyncResource pattern.

This file demonstrates:
1. How to convert a class with start/stop methods to AsyncResource
2. How to maintain backward compatibility
3. Proper error handling and state management
4. Implementing health checks
5. Using ensure_ready() for operations
"""

import asyncio
import uuid
import sys
import os
import time
import logging
from typing import List, Optional, Dict, Any, ClassVar

# Add the parent directory to sys.path to import exo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo.utils.async_resources import AsyncResource, ResourceState


# Original class without AsyncResource pattern
class OriginalDiscovery:
    """Original version of a discovery service without AsyncResource pattern."""
    
    def __init__(self, node_id: str, port: int = 50000):
        self.node_id = node_id
        self.port = port
        self.running = False
        self.server = None
        self.transport = None
        self.peers = {}
        
    async def start(self):
        """Start the discovery service."""
        if self.running:
            return
            
        try:
            # Create UDP server
            loop = asyncio.get_running_loop()
            self.transport, self.server = await loop.create_datagram_endpoint(
                lambda: DiscoveryProtocol(self.on_peer_discovered),
                local_addr=('0.0.0.0', self.port)
            )
            self.running = True
            
            # Start broadcasting
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            
            logging.info(f"Discovery started on port {self.port}")
        except Exception as e:
            logging.error(f"Failed to start discovery: {e}")
            self.running = False
            if self.transport:
                self.transport.close()
                self.transport = None
            self.server = None
            raise
            
    async def stop(self):
        """Stop the discovery service."""
        if not self.running:
            return
            
        # Cancel broadcast task
        if hasattr(self, 'broadcast_task') and self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
                
        # Close transport
        if self.transport:
            self.transport.close()
            self.transport = None
            
        self.server = None
        self.running = False
        logging.info("Discovery stopped")
        
    async def _broadcast_loop(self):
        """Broadcast presence periodically."""
        while True:
            try:
                # Send broadcast
                message = f"DISCOVERY:{self.node_id}:{time.time()}"
                self.transport.sendto(message.encode(), ('255.255.255.255', self.port))
            except Exception as e:
                logging.error(f"Error in broadcast: {e}")
                
            await asyncio.sleep(5)  # Broadcast every 5 seconds
            
    def on_peer_discovered(self, peer_id: str, addr: str):
        """Called when a peer is discovered."""
        if peer_id != self.node_id and peer_id not in self.peers:
            logging.info(f"Discovered peer: {peer_id} at {addr}")
            self.peers[peer_id] = addr
            
    async def discover_peers(self, wait_for_peers: int = 0) -> Dict[str, str]:
        """Discover peers on the network."""
        if not self.running:
            await self.start()
            
        # Wait for the specified number of peers if requested
        if wait_for_peers > 0:
            start_time = time.time()
            while len(self.peers) < wait_for_peers and time.time() - start_time < 30:
                await asyncio.sleep(1)
                
        return self.peers.copy()


# Mock protocol for the example
class DiscoveryProtocol:
    def __init__(self, callback):
        self.callback = callback
        
    def connection_made(self, transport):
        pass
        
    def datagram_received(self, data, addr):
        try:
            message = data.decode()
            if message.startswith("DISCOVERY:"):
                parts = message.split(":")
                if len(parts) >= 3:
                    peer_id = parts[1]
                    self.callback(peer_id, f"{addr[0]}:{addr[1]}")
        except Exception as e:
            logging.error(f"Error processing datagram: {e}")
            
    def error_received(self, exc):
        logging.error(f"Protocol error: {exc}")


# Converted class using AsyncResource pattern
class ImprovedDiscovery(AsyncResource):
    """Improved version of discovery service using AsyncResource pattern."""
    
    RESOURCE_TYPE: ClassVar[str] = "discovery"
    
    def __init__(self, node_id: str, port: int = 50000):
        # Initialize the AsyncResource base class
        super().__init__(
            resource_id=f"discovery-{node_id}",
            resource_type=self.RESOURCE_TYPE,
            display_name=f"Discovery for node {node_id}"
        )
        
        # Initialize discovery-specific attributes
        self.node_id = node_id
        self.port = port
        self.server = None
        self.transport = None
        self.peers = {}
        self.broadcast_task = None
        
    # Legacy methods for backward compatibility
    
    async def start(self):
        """Legacy method for backward compatibility."""
        await self.initialize()
        
    async def stop(self):
        """Legacy method for backward compatibility."""
        await self.cleanup()
        
    # AsyncResource implementation
    
    async def _do_initialize(self) -> None:
        """Initialize the discovery service."""
        try:
            # Create UDP server
            loop = asyncio.get_running_loop()
            self.transport, self.server = await loop.create_datagram_endpoint(
                lambda: DiscoveryProtocol(self.on_peer_discovered),
                local_addr=('0.0.0.0', self.port)
            )
            
            # Start broadcasting
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            
            logging.info(f"Discovery started on port {self.port}")
        except Exception as e:
            logging.error(f"Failed to start discovery: {e}")
            if self.transport:
                self.transport.close()
                self.transport = None
            self.server = None
            raise
            
    async def _do_cleanup(self) -> None:
        """Clean up the discovery service."""
        # Cancel broadcast task
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
                
        # Close transport
        if self.transport:
            self.transport.close()
            self.transport = None
            
        self.server = None
        logging.info("Discovery stopped")
        
    async def _do_health_check(self) -> bool:
        """Check if the discovery service is healthy."""
        # Check if the transport is active
        return self.transport is not None and not self.transport._closing
        
    # Implementation-specific methods
    
    async def _broadcast_loop(self):
        """Broadcast presence periodically."""
        while True:
            try:
                # Check if we're still healthy
                if not await self.check_health():
                    logging.warning("Discovery service is unhealthy, attempting to recover")
                    # Try to recover by reinitializing 
                    await self.cleanup()
                    await self.initialize()
                    continue
                    
                # Send broadcast
                message = f"DISCOVERY:{self.node_id}:{time.time()}"
                self.transport.sendto(message.encode(), ('255.255.255.255', self.port))
            except Exception as e:
                logging.error(f"Error in broadcast: {e}")
                
            await asyncio.sleep(5)  # Broadcast every 5 seconds
            
    def on_peer_discovered(self, peer_id: str, addr: str):
        """Called when a peer is discovered."""
        if peer_id != self.node_id and peer_id not in self.peers:
            logging.info(f"Discovered peer: {peer_id} at {addr}")
            self.peers[peer_id] = addr
            
    # Public methods ensure the resource is ready
    
    async def discover_peers(self, wait_for_peers: int = 0) -> Dict[str, str]:
        """Discover peers on the network."""
        # Ensure the discovery service is initialized and ready
        await self.ensure_ready()
        
        # Wait for the specified number of peers if requested
        if wait_for_peers > 0:
            start_time = time.time()
            while len(self.peers) < wait_for_peers and time.time() - start_time < 30:
                await asyncio.sleep(1)
                
        return self.peers.copy()
        
    async def broadcast_message(self, message: str) -> None:
        """Broadcast a custom message to all peers."""
        # Ensure the discovery service is initialized and ready
        await self.ensure_ready()
        
        # Send the message
        full_message = f"MESSAGE:{self.node_id}:{message}"
        self.transport.sendto(full_message.encode(), ('255.255.255.255', self.port))


# Example usage
async def run_example():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the improved discovery service
    discovery = ImprovedDiscovery(f"node-{uuid.uuid4().hex[:8]}")
    
    try:
        # Using initialize() directly
        logging.info("Initializing discovery service...")
        await discovery.initialize()
        
        # Discover peers using the ensure_ready pattern internally
        logging.info("Discovering peers...")
        peers = await discovery.discover_peers()
        logging.info(f"Found {len(peers)} peers: {peers}")
        
        # Do some work
        for i in range(3):
            await discovery.broadcast_message(f"Hello from example {i}")
            await asyncio.sleep(2)
            
        # Check health
        is_healthy = await discovery.check_health()
        logging.info(f"Discovery service health: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
        
        # Display state
        logging.info(f"Current state: {discovery.state.name}")
        
    except Exception as e:
        logging.error(f"Error in example: {e}")
    finally:
        # Always clean up
        logging.info("Cleaning up discovery service...")
        await discovery.cleanup()
        
    logging.info("Example completed")


# Run the example
if __name__ == "__main__":
    asyncio.run(run_example())
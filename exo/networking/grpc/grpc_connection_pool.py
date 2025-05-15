"""
gRPC-specific connection pool implementation.

This module provides a specialized connection pool for gRPC channels and stubs.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable
import grpc
from grpc.aio import Channel
from exo.utils import logging
from exo.networking.connection_pool import ConnectionPool, pool_manager


class GRPCChannelWrapper:
    """
    Wrapper for a gRPC channel and associated stubs.
    
    This class manages a gRPC channel and any service stubs created from it.
    """
    
    def __init__(self, channel: grpc.aio.Channel, address: str, options: List = None):
        self.channel = channel
        self.address = address
        self.options = options or []
        self.created_at = time.time()
        self.stubs: Dict[str, Any] = {}  # Service stubs by service name
        self.last_health_check = 0
        self.health_status = False
    
    def get_stub(self, service_name: str, stub_factory: Callable[[Channel], Any]) -> Any:
        """
        Get or create a service stub for this channel.
        
        Args:
            service_name: Service identifier (e.g., "NodeService")
            stub_factory: Function to create a new stub (e.g., node_service_pb2_grpc.NodeServiceStub)
            
        Returns:
            Service stub
        """
        if service_name not in self.stubs:
            self.stubs[service_name] = stub_factory(self.channel)
        return self.stubs[service_name]


# Factory function to create a new gRPC channel
async def create_grpc_channel(address: str, options: List = None, compression=grpc.Compression.NoCompression) -> GRPCChannelWrapper:
    """
    Create a new gRPC channel for the given address.
    
    Args:
        address: Server address in the format "host:port"
        options: gRPC channel options
        compression: Compression method
        
    Returns:
        GRPCChannelWrapper instance
    """
    options = options or []
    try:
        channel = grpc.aio.insecure_channel(
            address,
            options=options,
            compression=compression
        )
        
        wrapper = GRPCChannelWrapper(channel, address, options)
        
        # Perform initial connectivity check
        await channel.channel_ready()
        
        logging.debug(f"Created new gRPC channel to {address}",
                     component="grpc_pool",
                     address=address)
        
        return wrapper
    except Exception as e:
        logging.error(f"Failed to create gRPC channel to {address}",
                     component="grpc_pool",
                     address=address,
                     exc_info=e)
        raise


# Function to clean up a gRPC channel
async def cleanup_grpc_channel(channel_wrapper: GRPCChannelWrapper) -> None:
    """
    Close a gRPC channel and clean up resources.
    
    Args:
        channel_wrapper: GRPCChannelWrapper to close
    """
    try:
        await channel_wrapper.channel.close()
        logging.debug(f"Closed gRPC channel to {channel_wrapper.address}",
                     component="grpc_pool",
                     address=channel_wrapper.address)
    except Exception as e:
        logging.warning(f"Error closing gRPC channel to {channel_wrapper.address}",
                       component="grpc_pool",
                       address=channel_wrapper.address,
                       exc_info=e)


# Health check function for gRPC channels
async def check_grpc_channel_health(channel_wrapper: GRPCChannelWrapper) -> bool:
    """
    Check if a gRPC channel is healthy and connected.
    
    Args:
        channel_wrapper: GRPCChannelWrapper to check
        
    Returns:
        True if the channel is healthy, False otherwise
    """
    current_time = time.time()
    
    # Rate limit health checks to avoid excessive network traffic
    if current_time - channel_wrapper.last_health_check < 10.0 and channel_wrapper.health_status:
        return channel_wrapper.health_status
    
    channel_wrapper.last_health_check = current_time
    
    try:
        # Get the connectivity state
        state = channel_wrapper.channel.get_state()
        
        # Consider READY and IDLE as healthy states
        is_healthy = state in (
            grpc.ChannelConnectivity.READY,
            grpc.ChannelConnectivity.IDLE
        )
        
        # If not healthy but in CONNECTING state, give it a chance to connect
        if not is_healthy and state == grpc.ChannelConnectivity.CONNECTING:
            try:
                # Try to wait for the channel to become ready with a short timeout
                await asyncio.wait_for(channel_wrapper.channel.channel_ready(), timeout=2.0)
                is_healthy = True
            except asyncio.TimeoutError:
                is_healthy = False
            except Exception:
                is_healthy = False
                
        channel_wrapper.health_status = is_healthy
        return is_healthy
    except Exception as e:
        logging.warning(f"Health check failed for gRPC channel to {channel_wrapper.address}",
                      component="grpc_pool",
                      address=channel_wrapper.address,
                      exc_info=e)
        channel_wrapper.health_status = False
        return False


# Create a specialized gRPC connection pool
grpc_pool = ConnectionPool(
    factory=create_grpc_channel,
    cleanup=cleanup_grpc_channel,
    health_check=check_grpc_channel_health,
    max_size=20,              # Allow up to 20 connections per address
    max_idle_time=600.0,      # 10 minutes idle timeout
    max_lifetime=3600.0 * 6,  # 6 hours max lifetime
    cleanup_interval=120.0    # Clean up every 2 minutes
)

# Register the gRPC pool with the global manager
pool_manager.register_pool("grpc", grpc_pool)


class PooledGRPCClient:
    """
    Client class that uses the connection pool for gRPC connections.
    
    This class makes it easy to get and use pooled gRPC connections.
    """
    
    def __init__(self, address: str, options: List = None, compression=grpc.Compression.NoCompression):
        self.address = address
        self.options = options or []
        self.compression = compression
        self._channel_wrapper = None
    
    async def __aenter__(self) -> 'PooledGRPCClient':
        """Async context manager entry - gets a connection from the pool."""
        self._channel_wrapper = await grpc_pool.get_connection((self.address, tuple(self.options), self.compression))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - releases the connection back to the pool."""
        if self._channel_wrapper:
            await grpc_pool.release_connection(
                (self.address, tuple(self.options), self.compression),
                self._channel_wrapper
            )
            self._channel_wrapper = None
    
    def get_stub(self, service_name: str, stub_factory: Callable[[Channel], Any]) -> Any:
        """Get a service stub for the current channel."""
        if not self._channel_wrapper:
            raise RuntimeError("Not inside a connection context. Use 'async with' to acquire a connection first.")
        return self._channel_wrapper.get_stub(service_name, stub_factory)
    
    @property
    def channel(self) -> grpc.aio.Channel:
        """Get the underlying gRPC channel."""
        if not self._channel_wrapper:
            raise RuntimeError("Not inside a connection context. Use 'async with' to acquire a connection first.")
        return self._channel_wrapper.channel


# Initialize the pool
async def initialize_grpc_pool():
    """Initialize and start the gRPC connection pool."""
    await grpc_pool.start()
    logging.info("gRPC connection pool initialized", component="grpc_pool")


# Shutdown the pool
async def shutdown_grpc_pool():
    """Shut down the gRPC connection pool."""
    await grpc_pool.stop()
    logging.info("gRPC connection pool shut down", component="grpc_pool")
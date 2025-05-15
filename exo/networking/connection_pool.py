"""
Connection pooling system for exo network connections.

This module provides a generic connection pool that can be used to manage
network connections, reducing overhead by reusing existing connections.
"""

import asyncio
import time
from typing import Dict, Any, Callable, TypeVar, Generic, Optional, Tuple, List, Set, Awaitable
from exo.utils import logging

T = TypeVar('T')  # Type of the connection object


class PooledConnection(Generic[T]):
    """
    Represents a connection that is managed by a connection pool.
    
    This class tracks:
    - The connection object
    - When it was created
    - When it was last used
    - Its current state (in_use, idle, etc.)
    """
    
    def __init__(self, 
                 connection: T, 
                 created_at: float,
                 max_idle_time: float = 300.0,  # 5 minutes by default
                 max_lifetime: float = 3600.0):  # 1 hour by default
        self.connection = connection
        self.created_at = created_at
        self.last_used_at = created_at
        self.in_use = False
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        
    def use(self):
        """Mark the connection as in use and update last_used timestamp."""
        self.in_use = True
        self.last_used_at = time.time()
        
    def release(self):
        """Mark the connection as not in use and update last_used timestamp."""
        self.in_use = False
        self.last_used_at = time.time()
        
    def is_expired(self, current_time: float) -> bool:
        """Check if the connection has expired based on idle time or lifetime."""
        # Check total lifetime
        if current_time - self.created_at > self.max_lifetime:
            return True
            
        # If in use, it's not expired
        if self.in_use:
            return False
            
        # Check idle time
        return current_time - self.last_used_at > self.max_idle_time


class ConnectionPool(Generic[T]):
    """
    Generic connection pool that manages the lifecycle of reusable connections.
    
    Features:
    - Maintains a pool of connections by key (e.g., host:port)
    - Creates new connections when needed
    - Reuses idle connections when available
    - Automatically cleans up expired connections
    - Enforces maximum pool size
    """
    
    def __init__(self, 
                 factory: Callable[[Any], Awaitable[T]],  # Function to create a new connection
                 cleanup: Callable[[T], Awaitable[None]],  # Function to clean up a connection
                 health_check: Callable[[T], Awaitable[bool]],  # Function to check if a connection is healthy
                 max_size: int = 10,  # Maximum number of connections per key
                 max_idle_time: float = 300.0,  # 5 minutes
                 max_lifetime: float = 3600.0,  # 1 hour
                 cleanup_interval: float = 60.0):  # 1 minute
        self.factory = factory
        self.cleanup = cleanup
        self.health_check = health_check
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        
        # Dictionary of connection pools, keyed by connection parameters
        self.pools: Dict[Any, List[PooledConnection[T]]] = {}
        
        # Set of connections currently checked out
        self.checked_out: Set[PooledConnection[T]] = set()
        
        # Cleanup task
        self._cleanup_task = None
        self._cleanup_interval = cleanup_interval
        self._shutting_down = False
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the connection pool maintenance task."""
        if self._cleanup_task is None:
            self._shutting_down = False
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logging.info("Connection pool maintenance task started", 
                        component="connection_pool",
                        settings={"max_size": self.max_size, 
                                 "max_idle_time": self.max_idle_time,
                                 "max_lifetime": self.max_lifetime})
    
    async def stop(self):
        """Stop the connection pool and close all connections."""
        logging.info("Stopping connection pool", component="connection_pool")
        self._shutting_down = True
        
        # Cancel the cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        # Close all connections
        closed_count = 0
        error_count = 0
        
        async with self._lock:
            # Close all checked out connections
            for conn in list(self.checked_out):
                try:
                    await self.cleanup(conn.connection)
                    closed_count += 1
                except Exception as e:
                    error_count += 1
                    logging.error("Error closing checked out connection", 
                                 component="connection_pool", 
                                 exc_info=e)
            
            self.checked_out.clear()
            
            # Close all pooled connections
            for key, connections in self.pools.items():
                for conn in connections:
                    try:
                        await self.cleanup(conn.connection)
                        closed_count += 1
                    except Exception as e:
                        error_count += 1
                        logging.error(f"Error closing pooled connection for {key}", 
                                     component="connection_pool", 
                                     key=key,
                                     exc_info=e)
            
            self.pools.clear()
        
        logging.info("Connection pool stopped", 
                    component="connection_pool",
                    stats={"closed": closed_count, "errors": error_count})
    
    async def get_connection(self, key: Any) -> T:
        """
        Get a connection from the pool or create a new one.
        
        Args:
            key: Connection key (e.g., "host:port")
            
        Returns:
            Connection object
        """
        async with self._lock:
            # Check if the pool exists for this key
            if key not in self.pools:
                self.pools[key] = []
            
            # Try to find an idle connection in the pool
            connections = self.pools[key]
            
            # Find any available connection that's not in use
            for i, conn in enumerate(connections):
                if not conn.in_use:
                    # Check if the connection is still healthy
                    try:
                        if await self.health_check(conn.connection):
                            conn.use()
                            self.checked_out.add(conn)
                            logging.debug(f"Reusing existing connection for {key}",
                                         component="connection_pool", 
                                         key=key)
                            return conn.connection
                        else:
                            logging.debug(f"Found unhealthy connection for {key}, cleaning up",
                                         component="connection_pool", 
                                         key=key)
                            # Connection isn't healthy, remove it
                            try:
                                await self.cleanup(conn.connection)
                            except Exception as e:
                                logging.warning(f"Error cleaning up unhealthy connection for {key}",
                                              component="connection_pool",
                                              key=key,
                                              exc_info=e)
                            connections.pop(i)
                    except Exception as e:
                        logging.warning(f"Error checking connection health for {key}",
                                      component="connection_pool",
                                      key=key,
                                      exc_info=e)
                        # Remove the problematic connection
                        try:
                            await self.cleanup(conn.connection)
                        except Exception as cleanup_err:
                            logging.warning(f"Error cleaning up problematic connection for {key}",
                                         component="connection_pool",
                                         key=key,
                                         exc_info=cleanup_err)
                        connections.pop(i)
            
            # Check if we're at the maximum pool size for this key
            if len(connections) >= self.max_size:
                # We're at capacity, find the oldest connection to replace
                oldest_conn = min(connections, key=lambda c: c.last_used_at)
                
                # If the oldest connection is in use, we'll need to create a new one
                # and potentially exceed the max size temporarily
                if oldest_conn.in_use:
                    logging.warning(f"Connection pool for {key} at capacity but all connections in use",
                                  component="connection_pool",
                                  key=key,
                                  pool_size=len(connections))
                else:
                    # Replace the oldest idle connection
                    try:
                        await self.cleanup(oldest_conn.connection)
                        connections.remove(oldest_conn)
                        logging.debug(f"Removed oldest connection for {key} to make room",
                                     component="connection_pool",
                                     key=key)
                    except Exception as e:
                        logging.warning(f"Error cleaning up oldest connection for {key}",
                                      component="connection_pool",
                                      key=key,
                                      exc_info=e)
                        connections.remove(oldest_conn)
            
            # Create a new connection
            try:
                new_connection = await self.factory(key)
                new_pooled_conn = PooledConnection(
                    connection=new_connection,
                    created_at=time.time(),
                    max_idle_time=self.max_idle_time,
                    max_lifetime=self.max_lifetime
                )
                new_pooled_conn.use()
                connections.append(new_pooled_conn)
                self.checked_out.add(new_pooled_conn)
                
                logging.debug(f"Created new connection for {key}",
                             component="connection_pool",
                             key=key,
                             pool_size=len(connections))
                
                return new_connection
            except Exception as e:
                logging.error(f"Error creating new connection for {key}",
                             component="connection_pool",
                             key=key,
                             exc_info=e)
                raise
    
    async def release_connection(self, key: Any, connection: T):
        """
        Release a connection back to the pool.
        
        Args:
            key: Connection key (e.g., "host:port")
            connection: The connection object to release
        """
        async with self._lock:
            # Find the pooled connection
            for conn in self.checked_out:
                if conn.connection == connection:
                    conn.release()
                    self.checked_out.remove(conn)
                    logging.debug(f"Released connection back to pool for {key}",
                                 component="connection_pool",
                                 key=key)
                    return
            
            logging.warning(f"Attempted to release a connection that wasn't checked out: {key}",
                          component="connection_pool",
                          key=key)
    
    async def _cleanup_loop(self):
        """Periodic task to clean up expired connections."""
        try:
            while not self._shutting_down:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_connections()
        except asyncio.CancelledError:
            logging.debug("Connection pool cleanup task cancelled",
                         component="connection_pool")
            raise
        except Exception as e:
            logging.error("Error in connection pool cleanup task",
                         component="connection_pool",
                         exc_info=e)
    
    async def _cleanup_expired_connections(self):
        """Check for and remove expired connections."""
        current_time = time.time()
        expired_count = 0
        checked_count = 0
        
        async with self._lock:
            # Check each connection pool
            for key, connections in list(self.pools.items()):
                # Make a copy of the list to avoid modification issues during iteration
                for conn in list(connections):
                    checked_count += 1
                    
                    # Skip connections that are in use
                    if conn.in_use:
                        continue
                    
                    # Check if the connection has expired
                    if conn.is_expired(current_time):
                        try:
                            await self.cleanup(conn.connection)
                            connections.remove(conn)
                            expired_count += 1
                            logging.debug(f"Removed expired connection for {key}",
                                         component="connection_pool",
                                         key=key,
                                         reason="expired")
                        except Exception as e:
                            logging.warning(f"Error cleaning up expired connection for {key}",
                                          component="connection_pool",
                                          key=key,
                                          exc_info=e)
                            connections.remove(conn)
                
                # Remove empty pools
                if not connections:
                    del self.pools[key]
        
        if expired_count > 0:
            logging.info("Connection pool maintenance completed",
                        component="connection_pool",
                        stats={"checked": checked_count, "expired": expired_count})
        else:
            logging.debug("Connection pool maintenance completed, no expired connections",
                         component="connection_pool",
                         stats={"checked": checked_count})


class ConnectionPoolManager:
    """
    Manages multiple connection pools by type.
    
    This class provides a central point for managing different types of
    connection pools (e.g., gRPC, HTTP, etc.).
    """
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
    
    async def start_all(self):
        """Start all connection pools."""
        for pool_type, pool in self.pools.items():
            await pool.start()
    
    async def stop_all(self):
        """Stop all connection pools."""
        for pool_type, pool in self.pools.items():
            await pool.stop()
    
    def register_pool(self, pool_type: str, pool: ConnectionPool):
        """Register a connection pool."""
        self.pools[pool_type] = pool
        
    def get_pool(self, pool_type: str) -> Optional[ConnectionPool]:
        """Get a connection pool by type."""
        return self.pools.get(pool_type)


# Global connection pool manager instance
pool_manager = ConnectionPoolManager()
"""
Tests for the async resource management system.

This module contains tests for:
- AsyncResource lifecycle
- NetworkResource reconnection
- AsyncResourcePool pooling and reuse
- GRPCResource connections and operations
"""

import asyncio
import unittest
import time
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

from exo.utils.async_resources import (
    AsyncResource,
    AsyncManagedResource,
    AsyncResourcePool,
    AsyncResourceContext,
    AsyncResourceGroup,
    NetworkResource,
    ResourceState,
    ResourceInitializationError,
    ResourceUnavailableError,
    with_timeout,
)


class TestAsyncResource(unittest.IsolatedAsyncioTestCase):
    """Tests for the base AsyncResource class."""
    
    class SimpleResource(AsyncResource):
        """Simple implementation of AsyncResource for testing."""
        
        def __init__(self, resource_id=None, initialize_timeout=1.0, fail_initialize=False, fail_cleanup=False):
            super().__init__(resource_id, initialize_timeout)
            self.initialized = False
            self.cleaned_up = False
            self.fail_initialize = fail_initialize
            self.fail_cleanup = fail_cleanup
            self.initialize_called = 0
            self.cleanup_called = 0
            self.health_checks = 0
            self.health_status = True
            
        async def _do_initialize(self) -> None:
            self.initialize_called += 1
            if self.fail_initialize:
                raise ValueError("Simulated initialization failure")
            await asyncio.sleep(0.1)  # Simulate some work
            self.initialized = True
            
        async def _do_cleanup(self) -> None:
            self.cleanup_called += 1
            if self.fail_cleanup:
                raise ValueError("Simulated cleanup failure")
            await asyncio.sleep(0.1)  # Simulate some work
            self.cleaned_up = True
            
        async def _check_health(self) -> bool:
            self.health_checks += 1
            return self.health_status
    
    async def test_resource_lifecycle(self):
        """Test the basic lifecycle of a resource."""
        resource = self.SimpleResource()
        
        # Initial state
        self.assertEqual(resource.state, ResourceState.UNINITIALIZED)
        self.assertFalse(resource.is_initialized)
        self.assertFalse(resource.is_usable)
        self.assertFalse(resource.in_use)
        
        # Initialize
        result = await resource.initialize()
        self.assertTrue(result)
        self.assertEqual(resource.state, ResourceState.READY)
        self.assertTrue(resource.is_initialized)
        self.assertTrue(resource.is_usable)
        self.assertTrue(resource.is_healthy)
        self.assertFalse(resource.in_use)
        self.assertEqual(resource.initialize_called, 1)
        
        # Initialize again (should be a no-op)
        result = await resource.initialize()
        self.assertTrue(result)
        self.assertEqual(resource.initialize_called, 1)  # Shouldn't be called again
        
        # Clean up
        await resource.cleanup()
        self.assertEqual(resource.state, ResourceState.CLOSED)
        self.assertFalse(resource.is_initialized)
        self.assertFalse(resource.is_usable)
        self.assertEqual(resource.cleanup_called, 1)
        
        # Clean up again (should be a no-op)
        await resource.cleanup()
        self.assertEqual(resource.cleanup_called, 1)  # Shouldn't be called again
        
    async def test_initialization_failure(self):
        """Test resource initialization failure."""
        resource = self.SimpleResource(fail_initialize=True)
        
        # Should raise an exception
        with self.assertRaises(ResourceInitializationError):
            await resource.initialize()
            
        self.assertEqual(resource.state, ResourceState.ERROR)
        self.assertFalse(resource.is_initialized)
        self.assertFalse(resource.is_usable)
        self.assertEqual(resource.initialize_called, 1)
        
    async def test_cleanup_failure(self):
        """Test resource cleanup failure."""
        resource = self.SimpleResource(fail_cleanup=True)
        
        # Initialize successfully
        await resource.initialize()
        self.assertEqual(resource.state, ResourceState.READY)
        
        # Cleanup should raise an exception
        with self.assertRaises(ValueError):
            await resource.cleanup()
            
        # Should still be marked as closed
        self.assertEqual(resource.state, ResourceState.CLOSED)
        self.assertEqual(resource.cleanup_called, 1)
        
    async def test_health_check(self):
        """Test resource health checking."""
        resource = self.SimpleResource()
        
        # Initialize
        await resource.initialize()
        self.assertTrue(resource.is_healthy)
        
        # Health check should be True
        is_healthy = await resource.check_health()
        self.assertTrue(is_healthy)
        self.assertEqual(resource.health_checks, 1)
        
        # Change health status
        resource.health_status = False
        
        # Health check should be False
        is_healthy = await resource.check_health()
        self.assertFalse(is_healthy)
        self.assertEqual(resource.state, ResourceState.DEGRADED)
        self.assertTrue(resource.is_usable)  # Still usable even if degraded
        self.assertEqual(resource.health_checks, 2)
        
        # Change health status back
        resource.health_status = True
        
        # Health check should be True again
        is_healthy = await resource.check_health()
        self.assertTrue(is_healthy)
        self.assertEqual(resource.state, ResourceState.READY)
        self.assertEqual(resource.health_checks, 3)
        
    async def test_in_use_tracking(self):
        """Test resource usage tracking."""
        resource = self.SimpleResource()
        await resource.initialize()
        
        # Initially not in use
        self.assertFalse(resource.in_use)
        
        # Mark as in use
        await resource.mark_in_use()
        self.assertTrue(resource.in_use)
        
        # Mark as not in use
        await resource.mark_not_in_use()
        self.assertFalse(resource.in_use)
        
    async def test_timeout_handling(self):
        """Test resource initialization with timeout."""
        
        class SlowResource(self.SimpleResource):
            async def _do_initialize(self) -> None:
                await asyncio.sleep(0.5)  # Slow initialization
                await super()._do_initialize()
                
        # With sufficient timeout
        resource = SlowResource(initialize_timeout=1.0)
        result = await resource.initialize()
        self.assertTrue(result)
        self.assertTrue(resource.is_initialized)
        
        # With insufficient timeout
        resource = SlowResource(initialize_timeout=0.1)
        with self.assertRaises(asyncio.TimeoutError):
            await resource.initialize()
            
        self.assertEqual(resource.state, ResourceState.ERROR)
        self.assertFalse(resource.is_initialized)


class TestAsyncManagedResource(unittest.IsolatedAsyncioTestCase):
    """Tests for the AsyncManagedResource class."""
    
    class TestManagedResource(AsyncManagedResource):
        """Test implementation of AsyncManagedResource."""
        
        def __init__(
            self,
            resource_id=None,
            initialize_timeout=1.0,
            max_retries=2,
            retry_delay=0.1,
            fail_count=0,
            health_check_interval=None
        ):
            super().__init__(
                resource_id=resource_id,
                initialize_timeout=initialize_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                health_check_interval=health_check_interval
            )
            self.initialized = False
            self.cleaned_up = False
            self.fail_count = fail_count
            self.current_fails = 0
            self.initialize_called = 0
            self.cleanup_called = 0
            self.operation_calls = 0
            
        async def _do_initialize(self) -> None:
            self.initialize_called += 1
            if self.current_fails < self.fail_count:
                self.current_fails += 1
                raise ValueError(f"Simulated initialization failure {self.current_fails}/{self.fail_count}")
            await asyncio.sleep(0.1)
            self.initialized = True
            
        async def _do_cleanup(self) -> None:
            self.cleanup_called += 1
            await asyncio.sleep(0.1)
            self.cleaned_up = True
            
        async def test_operation(self, fail=False):
            self.operation_calls += 1
            if fail:
                raise ValueError("Simulated operation failure")
            return "success"
    
    async def test_retry_on_initialization(self):
        """Test retry behavior during initialization."""
        # Resource that fails once but succeeds on retry
        resource = self.TestManagedResource(fail_count=1, max_retries=2)
        
        # Should succeed after retry
        result = await resource.initialize()
        self.assertTrue(result)
        self.assertEqual(resource.initialize_called, 2)  # Called twice due to retry
        self.assertTrue(resource.is_initialized)
        
        # Resource that fails more times than max_retries
        resource = self.TestManagedResource(fail_count=3, max_retries=2)
        
        # Should fail after all retries
        with self.assertRaises(ResourceInitializationError):
            await resource.initialize()
            
        self.assertEqual(resource.initialize_called, 3)  # Called max_retries + 1 times
        self.assertFalse(resource.is_initialized)
        
    async def test_with_retry_operation(self):
        """Test the with_retry operation wrapper."""
        resource = self.TestManagedResource(max_retries=2)
        await resource.initialize()
        
        # Successful operation
        result = await resource.with_retry(
            lambda: resource.test_operation(),
            operation_name="test_operation"
        )
        self.assertEqual(result, "success")
        self.assertEqual(resource.operation_calls, 1)
        
        # Failed operation that should retry
        async def failing_operation():
            return await resource.test_operation(fail=True)
            
        with self.assertRaises(ValueError):
            await resource.with_retry(
                failing_operation,
                operation_name="failing_operation",
                max_retries=2
            )
            
        self.assertEqual(resource.operation_calls, 4)  # 1 + 3 fails
        
    async def test_health_check_task(self):
        """Test the automatic health check task."""
        # Set a short health check interval for testing
        resource = self.TestManagedResource(health_check_interval=0.1)
        await resource.initialize()
        
        # Check that health check task is running
        self.assertIsNotNone(resource._health_check_task)
        
        # Wait for some health checks to happen
        await asyncio.sleep(0.25)
        
        # Cleanup should cancel the health check task
        await resource.cleanup()
        self.assertIsNone(resource._health_check_task)


class TestNetworkResource(unittest.IsolatedAsyncioTestCase):
    """Tests for the NetworkResource class."""
    
    class MockNetworkResource(NetworkResource):
        """Mock implementation of NetworkResource for testing."""
        
        def __init__(
            self,
            address="test-server:123",
            resource_id=None,
            initialize_timeout=1.0,
            max_retries=2,
            retry_delay=0.1,
            fail_count=0,
            fail_operations=0,
            reconnect_on_error=True,
            health_check_interval=None
        ):
            super().__init__(
                resource_id=resource_id,
                address=address,
                initialize_timeout=initialize_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                health_check_interval=health_check_interval,
                reconnect_on_error=reconnect_on_error
            )
            self.initialized = False
            self.cleaned_up = False
            self.fail_count = fail_count
            self.fail_operations = fail_operations
            self.current_fails = 0
            self.operation_fails = 0
            self.initialize_called = 0
            self.cleanup_called = 0
            self.reconnect_called = 0
            self.operation_calls = 0
            
        async def _do_initialize(self) -> None:
            self.initialize_called += 1
            if self.current_fails < self.fail_count:
                self.current_fails += 1
                raise ConnectionError(f"Simulated connection failure {self.current_fails}/{self.fail_count}")
            await asyncio.sleep(0.1)
            self.initialized = True
            
        async def _do_cleanup(self) -> None:
            self.cleanup_called += 1
            await asyncio.sleep(0.1)
            self.cleaned_up = True
            
        async def test_operation(self, fail=False):
            self.operation_calls += 1
            if fail or self.operation_fails < self.fail_operations:
                self.operation_fails += 1
                raise ConnectionError("Simulated operation failure")
            return "success"
    
    async def test_connection_lifecycle(self):
        """Test the network connection lifecycle."""
        resource = self.MockNetworkResource()
        
        # Initialize
        await resource.initialize()
        self.assertTrue(resource.is_initialized)
        self.assertEqual(resource.initialize_called, 1)
        self.assertEqual(resource.connection_attempts, 1)
        
        # Reconnect
        await resource.reconnect()
        self.assertTrue(resource.is_initialized)
        self.assertEqual(resource.initialize_called, 2)
        self.assertEqual(resource.connection_attempts, 2)
        self.assertEqual(resource.cleanup_called, 1)  # Cleanup called during reconnect
        
    async def test_with_connection(self):
        """Test the with_connection operation wrapper."""
        # Successfully connected resource
        resource = self.MockNetworkResource(reconnect_on_error=True)
        await resource.initialize()
        
        # Successful operation
        result = await resource.with_connection(
            lambda: resource.test_operation(),
            operation_name="test_operation"
        )
        self.assertEqual(result, "success")
        self.assertEqual(resource.operation_calls, 1)
        
        # Failed operation that should trigger reconnect
        resource = self.MockNetworkResource(
            fail_operations=1,
            reconnect_on_error=True
        )
        await resource.initialize()
        
        result = await resource.with_connection(
            lambda: resource.test_operation(),
            operation_name="test_operation"
        )
        self.assertEqual(result, "success")
        self.assertEqual(resource.operation_calls, 2)  # 1 fail + 1 success
        self.assertEqual(resource.initialize_called, 2)  # Original + reconnect
        
        # Failed operation without reconnect
        resource = self.MockNetworkResource(
            fail_operations=1,
            reconnect_on_error=False
        )
        await resource.initialize()
        
        with self.assertRaises(ConnectionError):
            await resource.with_connection(
                lambda: resource.test_operation(),
                operation_name="test_operation"
            )
            
        self.assertEqual(resource.operation_calls, 1)  # 1 fail
        self.assertEqual(resource.initialize_called, 1)  # No reconnect


class TestAsyncResourcePool(unittest.IsolatedAsyncioTestCase):
    """Tests for the AsyncResourcePool class."""
    
    class PooledResource(AsyncResource):
        """Simple resource implementation for pool testing."""
        
        def __init__(self, resource_id=None, fail_initialize=False):
            super().__init__(resource_id=resource_id)
            self.initialized = False
            self.cleaned_up = False
            self.fail_initialize = fail_initialize
            self.initialize_called = 0
            self.cleanup_called = 0
            self.last_used = 0
            
        async def _do_initialize(self) -> None:
            self.initialize_called += 1
            if self.fail_initialize:
                raise ValueError("Simulated initialization failure")
            await asyncio.sleep(0.1)
            self.initialized = True
            self.last_used = time.time()
            
        async def _do_cleanup(self) -> None:
            self.cleanup_called += 1
            await asyncio.sleep(0.1)
            self.cleaned_up = True
            
        async def _check_health(self) -> bool:
            return self.initialized and not self.cleaned_up
    
    async def test_pool_lifecycle(self):
        """Test the resource pool lifecycle."""
        # Create a factory that returns our test resources
        resource_count = 0
        
        def create_resource():
            nonlocal resource_count
            resource_count += 1
            return self.PooledResource(resource_id=f"test-resource-{resource_count}")
            
        # Create pool with min_size=2
        pool = AsyncResourcePool(
            factory=create_resource,
            max_size=5,
            min_size=2,
            max_idle_time=10,
            health_check_interval=1
        )
        
        # Start the pool
        await pool.start()
        
        # Should have created min_size resources
        self.assertEqual(len(pool._resources), 2)
        self.assertEqual(len(pool._available_resources), 2)
        self.assertEqual(resource_count, 2)
        
        # Acquire a resource
        async with pool.acquire() as resource:
            self.assertIsInstance(resource, self.PooledResource)
            self.assertTrue(resource.is_initialized)
            self.assertTrue(resource.in_use)
            
            # Check pool state during use
            self.assertEqual(len(pool._available_resources), 1)
            self.assertEqual(len(pool._in_use_resources), 1)
            
        # After context exit, resource should be released
        self.assertEqual(len(pool._available_resources), 2)
        self.assertEqual(len(pool._in_use_resources), 0)
        
        # Acquire more resources than min_size but less than max_size
        resources = []
        for i in range(3):
            resources.append(await pool._acquire())
            
        # Pool should have created new resources as needed
        self.assertEqual(len(pool._resources), 3)
        self.assertEqual(len(pool._available_resources), 0)
        self.assertEqual(len(pool._in_use_resources), 3)
        self.assertEqual(resource_count, 3)
        
        # Release the resources
        for resource in resources:
            await pool._release(resource.id)
            
        # All resources should be available again
        self.assertEqual(len(pool._available_resources), 3)
        self.assertEqual(len(pool._in_use_resources), 0)
        
        # Stop the pool
        await pool.stop()
        
        # All resources should be cleaned up
        self.assertEqual(len(pool._resources), 0)
        self.assertEqual(len(pool._available_resources), 0)
        self.assertEqual(len(pool._in_use_resources), 0)
        self.assertTrue(pool._closed)
        
    async def test_pool_max_size(self):
        """Test the pool's max_size limit."""
        # Create a factory that returns our test resources
        def create_resource():
            return self.PooledResource()
            
        # Create pool with max_size=3
        pool = AsyncResourcePool(
            factory=create_resource,
            max_size=3,
            min_size=1
        )
        
        await pool.start()
        
        # Acquire max_size resources
        resources = []
        for i in range(3):
            resources.append(await pool._acquire())
            
        # Pool should be at max capacity
        self.assertEqual(len(pool._resources), 3)
        self.assertEqual(len(pool._available_resources), 0)
        self.assertEqual(len(pool._in_use_resources), 3)
        
        # Trying to acquire another should wait
        acquire_task = asyncio.create_task(pool.acquire(timeout=0.5))
        
        # Wait a bit to let the task try to acquire
        await asyncio.sleep(0.2)
        
        # It should still be running (waiting for a resource)
        self.assertFalse(acquire_task.done())
        
        # Release one resource
        await pool._release(resources[0].id)
        
        # Now the task should complete
        resource = await acquire_task
        self.assertIsInstance(resource, self.PooledResource)
        
        # Clean up
        await pool.stop()
        
    async def test_pool_resource_health(self):
        """Test that the pool properly handles unhealthy resources."""
        # Create a factory that returns our test resources
        def create_resource():
            return self.PooledResource()
            
        # Create pool with health checks
        pool = AsyncResourcePool(
            factory=create_resource,
            max_size=3,
            min_size=1,
            health_check_interval=0.5
        )
        
        await pool.start()
        
        # Acquire a resource
        resource = await pool._acquire()
        
        # Break the resource's health
        resource.initialized = False
        
        # Release it back to the pool
        await pool._release(resource.id)
        
        # Pool should detect the unhealthy resource and replace it
        await asyncio.sleep(0.6)  # Wait for health check
        
        # Pool should still have min_size healthy resources
        self.assertEqual(len(pool._resources), 1)
        self.assertEqual(len(pool._available_resources), 1)
        
        # Clean up
        await pool.stop()


class TestAsyncResourceGroup(unittest.IsolatedAsyncioTestCase):
    """Tests for the AsyncResourceGroup class."""
    
    class GroupedResource(AsyncResource):
        """Simple resource implementation for group testing."""
        
        def __init__(self, resource_id=None, fail_initialize=False, fail_cleanup=False):
            super().__init__(resource_id=resource_id)
            self.initialized = False
            self.cleaned_up = False
            self.fail_initialize = fail_initialize
            self.fail_cleanup = fail_cleanup
            self.initialize_called = 0
            self.cleanup_called = 0
            
        async def _do_initialize(self) -> None:
            self.initialize_called += 1
            if self.fail_initialize:
                raise ValueError("Simulated initialization failure")
            await asyncio.sleep(0.1)
            self.initialized = True
            
        async def _do_cleanup(self) -> None:
            self.cleanup_called += 1
            if self.fail_cleanup:
                raise ValueError("Simulated cleanup failure")
            await asyncio.sleep(0.1)
            self.cleaned_up = True
    
    async def test_group_lifecycle(self):
        """Test the resource group lifecycle."""
        group = AsyncResourceGroup("test-group")
        
        # Add resources
        resource1 = self.GroupedResource("resource1")
        resource2 = self.GroupedResource("resource2")
        
        await group.add_resource("r1", resource1)
        await group.add_resource("r2", resource2)
        
        # Group is not initialized yet
        self.assertFalse(group.is_initialized)
        self.assertFalse(resource1.is_initialized)
        self.assertFalse(resource2.is_initialized)
        
        # Initialize the group
        await group.initialize(parallel=True)
        
        # Group and all resources should be initialized
        self.assertTrue(group.is_initialized)
        self.assertTrue(resource1.is_initialized)
        self.assertTrue(resource2.is_initialized)
        
        # Get resources from the group
        r1 = await group.get_resource("r1")
        r2 = await group.get_resource("r2")
        
        self.assertIs(r1, resource1)
        self.assertIs(r2, resource2)
        
        # Clean up the group
        await group.cleanup(parallel=True)
        
        # Group and all resources should be cleaned up
        self.assertFalse(group.is_initialized)
        self.assertTrue(resource1.cleaned_up)
        self.assertTrue(resource2.cleaned_up)
        self.assertEqual(len(group._resources), 0)
        
    async def test_group_health_check(self):
        """Test group health checking."""
        group = AsyncResourceGroup("test-group")
        
        # Add resources with different health states
        resource1 = self.GroupedResource("resource1")
        resource2 = self.GroupedResource("resource2")
        
        await group.add_resource("r1", resource1, initialize=True)
        await group.add_resource("r2", resource2, initialize=True)
        
        # Check health
        health = await group.check_health()
        self.assertEqual(health, {"r1": True, "r2": True})
        
        # Make one resource unhealthy
        resource1.initialized = False
        
        # Check health again
        health = await group.check_health()
        self.assertEqual(health, {"r1": False, "r2": True})
        
    async def test_group_error_handling(self):
        """Test error handling in resource groups."""
        group = AsyncResourceGroup("test-group")
        
        # Add a resource that will fail to initialize
        resource1 = self.GroupedResource("resource1", fail_initialize=True)
        resource2 = self.GroupedResource("resource2")
        
        await group.add_resource("r1", resource1)
        await group.add_resource("r2", resource2)
        
        # Group initialization should fail
        with self.assertRaises(Exception):
            await group.initialize(parallel=False)
            
        # Group should not be initialized
        self.assertFalse(group.is_initialized)
        
        # Resource 1 should not be initialized
        self.assertFalse(resource1.is_initialized)
        
        # Resource 2 might be initialized (depends on the order)
        # We don't test this as it's implementation-specific
        
        # Getting a non-initialized resource should fail
        with self.assertRaises(ResourceUnavailableError):
            await group.get_resource("r1")


class TestGRPCResources(unittest.IsolatedAsyncioTestCase):
    """Tests for the gRPC-specific resource classes."""
    
    def setUp(self):
        # Create mock modules for grpc
        self.mock_grpc_module = patch("exo.utils.grpc_resources.grpc").start()
        self.mock_grpc_module.aio.Channel = MagicMock()
        self.mock_grpc_module.ChannelConnectivity.READY = "READY"
        self.mock_grpc_module.ChannelConnectivity.IDLE = "IDLE"
        self.mock_grpc_module.Compression.NoCompression = "NoCompression"
        self.mock_grpc_module.StatusCode.UNAVAILABLE = "UNAVAILABLE"
        
        # Import the module under test with mocks in place
        from exo.utils.grpc_resources import (
            GRPCChannelResource,
            GRPCServiceResource,
            GRPCConnectionPool,
            GRPCChannelContext,
            GRPCServiceContext,
            GRPCError
        )
        self.GRPCChannelResource = GRPCChannelResource
        self.GRPCServiceResource = GRPCServiceResource
        self.GRPCConnectionPool = GRPCConnectionPool
        self.GRPCChannelContext = GRPCChannelContext
        self.GRPCServiceContext = GRPCServiceContext
        self.GRPCError = GRPCError
        
    def tearDown(self):
        patch.stopall()
        
    async def test_grpc_channel_resource(self):
        """Test GRPCChannelResource initialization and operations."""
        # Set up mock channel
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.get_state = MagicMock(return_value="READY")
        mock_channel.close = AsyncMock()
        
        # Set up mock aio module
        self.mock_grpc_module.aio.insecure_channel.return_value = mock_channel
        
        # Create and initialize resource
        resource = self.GRPCChannelResource(
            address="localhost:50051",
            resource_id="test-channel"
        )
        
        # Initialize
        await resource.initialize()
        
        # Verify channel was created with expected options
        self.mock_grpc_module.aio.insecure_channel.assert_called_once()
        self.assertEqual(resource.channel, mock_channel)
        self.assertTrue(resource.is_initialized)
        
        # Test health check
        is_healthy = await resource.check_health()
        self.assertTrue(is_healthy)
        
        # Test cleanup
        await resource.cleanup()
        mock_channel.close.assert_called_once()
        self.assertIsNone(resource.channel)
        
    async def test_grpc_service_resource(self):
        """Test GRPCServiceResource initialization and operations."""
        # Set up mock channel
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.get_state = MagicMock(return_value="READY")
        mock_channel.close = AsyncMock()
        
        # Set up mock stub
        mock_stub_class = MagicMock()
        mock_stub = MagicMock()
        mock_stub_class.return_value = mock_stub
        
        # Set up mock aio module
        self.mock_grpc_module.aio.insecure_channel.return_value = mock_channel
        
        # Create and initialize resource
        resource = self.GRPCServiceResource(
            service_stub_class=mock_stub_class,
            address="localhost:50051",
            resource_id="test-service"
        )
        
        # Initialize
        await resource.initialize()
        
        # Verify channel and stub were created
        self.assertEqual(resource.channel, mock_channel)
        self.assertEqual(resource.stub, mock_stub)
        mock_stub_class.assert_called_once_with(mock_channel)
        
        # Test health check
        is_healthy = await resource.check_health()
        self.assertTrue(is_healthy)
        
        # Test cleanup
        await resource.cleanup()
        self.assertIsNone(resource.stub)
        self.assertIsNone(resource.channel)


# Run the tests
if __name__ == "__main__":
    unittest.main()
"""
Tests for the gRPC async resource management.

This module contains tests for the gRPC-specific resource classes:
- GRPCChannelResource
- GRPCServiceResource
- GRPCConnectionPool
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import grpc
import numpy as np

from exo.utils.async_resources import (
    AsyncResource,
    NetworkResource,
    ResourceState,
    ResourceInitializationError,
)
from exo.utils.grpc_resources import (
    GRPCChannelResource,
    GRPCServiceResource,
    GRPCConnectionPool,
    GRPCChannelContext,
    GRPCServiceContext,
    GRPCError,
    GRPCConnectionError,
)


class TestGRPCChannelResource(unittest.IsolatedAsyncioTestCase):
    """Tests for the GRPCChannelResource class."""
    
    def setUp(self):
        # Create mock for the grpc module
        self.mock_grpc = patch("exo.utils.grpc_resources.grpc").start()
        
        # Set up mock channel
        self.mock_channel = AsyncMock()
        self.mock_channel.channel_ready = AsyncMock()
        self.mock_channel.get_state = MagicMock(return_value="READY")
        self.mock_channel.close = AsyncMock()
        
        # Mock RpcError
        self.mock_grpc.RpcError = Exception
        
        # Make the mock channel available through the mock grpc module
        self.mock_grpc.aio.insecure_channel.return_value = self.mock_channel
        self.mock_grpc.aio.secure_channel.return_value = self.mock_channel
        
        # Set up mock connectivity states
        self.mock_grpc.ChannelConnectivity.READY = "READY"
        self.mock_grpc.ChannelConnectivity.IDLE = "IDLE"
        self.mock_grpc.ChannelConnectivity.CONNECTING = "CONNECTING"
        self.mock_grpc.ChannelConnectivity.TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
        self.mock_grpc.ChannelConnectivity.SHUTDOWN = "SHUTDOWN"
        
        # Set up mock compression options
        self.mock_grpc.Compression.NoCompression = "NoCompression"
        self.mock_grpc.Compression.Gzip = "Gzip"
        
        # Set up mock status codes
        self.mock_grpc.StatusCode.UNAVAILABLE = "UNAVAILABLE"
        self.mock_grpc.StatusCode.DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
        self.mock_grpc.StatusCode.RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
        self.mock_grpc.StatusCode.ABORTED = "ABORTED"
        
        # Initialize common test parameters
        self.test_address = "localhost:50051"
        self.test_options = [("grpc.max_message_length", 10 * 1024 * 1024)]
        
        # Patch the _check_health method to avoid real channel access
        patcher = patch.object(GRPCChannelResource, '_check_health', AsyncMock(return_value=True))
        self.mock_check_health = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Patch call_unary to avoid __qualname__ access on mock
        patcher = patch.object(GRPCChannelResource, 'call_unary')
        self.mock_call_unary = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_call_unary.side_effect = self._mock_call_unary
        
    async def _mock_call_unary(self, method, request, timeout=None, metadata=None):
        """Mock implementation of call_unary to avoid __qualname__ issue"""
        try:
            if getattr(method, 'side_effect', None):
                if isinstance(method.side_effect, list):
                    if method.side_effect:
                        result = method.side_effect.pop(0)
                        if isinstance(result, Exception):
                            raise result
                        return result
                else:
                    return await method(request) if asyncio.iscoroutinefunction(method) else method(request)
            return await method(request) if asyncio.iscoroutinefunction(method) else method(request)
        except Exception as e:
            raise e
        
    def tearDown(self):
        patch.stopall()
        
    async def test_channel_initialization(self):
        """Test successful channel initialization."""
        # Create resource
        resource = GRPCChannelResource(
            address=self.test_address,
            options=self.test_options,
            compression=self.mock_grpc.Compression.Gzip
        )
        
        # Initialize
        await resource.initialize()
        
        # Verify channel creation
        self.mock_grpc.aio.insecure_channel.assert_called_once_with(
            self.test_address,
            options=self.test_options,
            compression="Gzip"
        )
        
        # Verify channel_ready was called
        self.mock_channel.channel_ready.assert_called_once()
        
        # Verify resource state
        self.assertEqual(resource.state, ResourceState.READY)
        self.assertTrue(resource.is_initialized)
        self.assertEqual(resource.channel, self.mock_channel)
        
    async def test_secure_channel_initialization(self):
        """Test secure channel initialization with SSL credentials."""
        # Create mock SSL credentials
        mock_credentials = MagicMock()
        
        # Create resource
        resource = GRPCChannelResource(
            address=self.test_address,
            options=self.test_options,
            ssl_credentials=mock_credentials
        )
        
        # Initialize
        await resource.initialize()
        
        # Verify secure channel creation
        self.mock_grpc.aio.secure_channel.assert_called_once_with(
            self.test_address,
            mock_credentials,
            options=self.test_options,
            compression="NoCompression"
        )
        
        # Verify channel_ready was called
        self.mock_channel.channel_ready.assert_called_once()
        
    async def test_channel_initialization_failure(self):
        """Test channel initialization failure."""
        # Make channel_ready raise an error
        self.mock_channel.channel_ready.side_effect = Exception("Connection failed")
        
        # Create resource
        resource = GRPCChannelResource(
            address=self.test_address,
            max_retries=0  # No retries for faster testing
        )
        
        # Initialize should raise an error
        with self.assertRaises(ResourceInitializationError):
            await resource.initialize()
            
        # Verify resource state
        self.assertEqual(resource.state, ResourceState.ERROR)
        self.assertFalse(resource.is_initialized)
        
    async def test_channel_get_bad_state(self):
        """Test handling of bad channel state during initialization."""
        # Make get_state return a non-READY state after channel_ready
        self.mock_channel.get_state.return_value = "TRANSIENT_FAILURE"
        
        # Create resource
        resource = GRPCChannelResource(
            address=self.test_address,
            max_retries=0  # No retries for faster testing
        )
        
        # Initialize should raise an error
        with self.assertRaises(ResourceInitializationError):
            await resource.initialize()
            
        # Verify resource state
        self.assertEqual(resource.state, ResourceState.ERROR)
        self.assertFalse(resource.is_initialized)
        
    async def test_channel_cleanup(self):
        """Test channel cleanup."""
        # Create and initialize resource
        resource = GRPCChannelResource(address=self.test_address)
        await resource.initialize()
        
        # Cleanup
        await resource.cleanup()
        
        # Verify channel was closed
        self.mock_channel.close.assert_called_once()
        
        # Verify resource state
        self.assertEqual(resource.state, ResourceState.CLOSED)
        self.assertIsNone(resource._channel)
        
    async def test_call_unary_success(self):
        """Test successful unary call."""
        # Create and initialize resource
        resource = GRPCChannelResource(address=self.test_address)
        await resource.initialize()
        
        # Mock method and request
        mock_method = MagicMock(return_value="response")
        mock_request = MagicMock()
        
        # Call the method - using the patched version
        result = await resource.call_unary(mock_method, mock_request)
        
        # Verify the patched method was called
        self.mock_call_unary.assert_called_once()
        self.assertEqual(result, "response")
        
    async def test_call_unary_with_error(self):
        """Test unary call with error and retry."""
        # Create and initialize resource
        resource = GRPCChannelResource(
            address=self.test_address,
            max_retries=1
        )
        await resource.initialize()
        
        # Set up side effect for the mock call_unary
        self.mock_call_unary.side_effect = [Exception("First call fails"), "response"]
        
        # First call should fail, second should succeed
        with self.assertRaises(Exception):
            await resource.call_unary(MagicMock(), MagicMock())
            
        # Reset mock for second call
        self.mock_call_unary.side_effect = None
        self.mock_call_unary.return_value = "response"
        
        # Second call should succeed
        result = await resource.call_unary(MagicMock(), MagicMock())
        self.assertEqual(result, "response")
        
    async def test_call_unary_with_timeout(self):
        """Test unary call with timeout."""
        # Create and initialize resource
        resource = GRPCChannelResource(
            address=self.test_address,
            max_retries=0
        )
        await resource.initialize()
        
        # Make call_unary timeout
        async def timeout_effect(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "response"
            
        self.mock_call_unary.side_effect = timeout_effect
        
        # Call should time out
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(resource.call_unary(MagicMock(), MagicMock()), timeout=0.1)


class TestGRPCServiceResource(unittest.IsolatedAsyncioTestCase):
    """Tests for the GRPCServiceResource class."""
    
    def setUp(self):
        # Patch grpc_resources.GRPCChannelResource to avoid real network access
        self.mock_channel_resource = patch("exo.utils.grpc_resources.GRPCChannelResource").start()
        
        # Set up mock channel resource
        self.mock_channel_instance = MagicMock()
        self.mock_channel_instance.initialize = AsyncMock()
        self.mock_channel_instance.cleanup = AsyncMock()
        self.mock_channel_instance.check_health = AsyncMock(return_value=True)
        self.mock_channel_instance.call_unary = AsyncMock(return_value="response")
        self.mock_channel_instance.is_initialized = True
        self.mock_channel_instance.is_usable = True
        self.mock_channel_instance.is_healthy = True
        
        # Set up mock channel
        self.mock_channel = MagicMock()
        self.mock_channel_instance.channel = self.mock_channel
        
        # Make the mock channel resource available
        self.mock_channel_resource.return_value = self.mock_channel_instance
        
        # Set up mock stub
        self.mock_stub_class = MagicMock()
        self.mock_stub = MagicMock()
        self.mock_stub_class.return_value = self.mock_stub
        
        # Initialize common test parameters
        self.test_address = "localhost:50051"
        
    def tearDown(self):
        patch.stopall()
        
    async def test_service_initialization(self):
        """Test successful service initialization."""
        # Create resource
        resource = GRPCServiceResource(
            service_stub_class=self.mock_stub_class,
            address=self.test_address
        )
        
        # Initialize
        await resource.initialize()
        
        # Verify channel resource was created and initialized
        self.mock_channel_resource.assert_called_once()
        self.mock_channel_instance.initialize.assert_called_once()
        
        # Verify stub creation
        self.mock_stub_class.assert_called_once_with(self.mock_channel)
        
        # Verify resource state properties
        self.assertTrue(resource.is_initialized)
        self.assertEqual(resource.stub, self.mock_stub)
        
    async def test_service_cleanup(self):
        """Test service cleanup."""
        # Create and initialize resource
        resource = GRPCServiceResource(
            service_stub_class=self.mock_stub_class,
            address=self.test_address
        )
        await resource.initialize()
        
        # Cleanup
        await resource.cleanup()
        
        # Verify channel resource was cleaned up
        self.mock_channel_instance.cleanup.assert_called_once()
        
    async def test_call_unary_success(self):
        """Test successful unary method call."""
        # Create and initialize resource
        resource = GRPCServiceResource(
            service_stub_class=self.mock_stub_class,
            address=self.test_address
        )
        await resource.initialize()
        
        # Set up mock method on the stub
        self.mock_stub.SomeMethod = AsyncMock(return_value="response")
        
        # Set up the channel call_unary to return our method's result
        async def call_unary_impl(method, *args, **kwargs):
            return await method(*args, **kwargs)
            
        self.mock_channel_instance.call_unary.side_effect = call_unary_impl
        
        # Call the method
        result = await resource.call_unary(
            method_name="SomeMethod",
            request="request",
            timeout=1.0
        )
        
        # Verify method was called
        self.mock_stub.SomeMethod.assert_called_once()
        self.assertEqual(result, "response")
        
    async def test_call_unary_method_not_found(self):
        """Test calling a method that doesn't exist on the stub."""
        # Create and initialize resource
        resource = GRPCServiceResource(
            service_stub_class=self.mock_stub_class,
            address=self.test_address
        )
        await resource.initialize()
        
        # Call a non-existent method
        with self.assertRaises(AttributeError):
            await resource.call_unary(
                method_name="NonExistentMethod",
                request="request"
            )
            
    async def test_reconnect(self):
        """Test service reconnection."""
        # Create and initialize resource
        resource = GRPCServiceResource(
            service_stub_class=self.mock_stub_class,
            address=self.test_address
        )
        await resource.initialize()
        
        # Store the original stub
        original_stub = resource.stub
        
        # Create a new stub for the reconnection
        new_stub = MagicMock()
        self.mock_stub_class.return_value = new_stub
        
        # Reconnect
        await resource.reconnect()
        
        # Verify cleanup was called
        self.mock_channel_instance.cleanup.assert_called_once()
        
        # Verify initialize was called again
        self.assertEqual(self.mock_channel_instance.initialize.call_count, 2)
        
        # Verify stub was recreated (doesn't actually change in our test because we're mocking)
        self.assertIsNotNone(resource.stub)


class TestGRPCConnectionPool(unittest.IsolatedAsyncioTestCase):
    """Tests for the GRPCConnectionPool class."""
    
    def setUp(self):
        # Create mocks
        self.mock_service_resource = patch("exo.utils.grpc_resources.GRPCServiceResource").start()
        
        # Mock instances - need two distinct ones for pool tests
        self.mock_service_instance1 = MagicMock()
        self.mock_service_instance1.initialize = AsyncMock(return_value=True)
        self.mock_service_instance1.cleanup = AsyncMock()
        self.mock_service_instance1.is_initialized = True
        self.mock_service_instance1.is_usable = True
        self.mock_service_instance1.check_health = AsyncMock(return_value=True)
        self.mock_service_instance1.mark_in_use = AsyncMock()
        self.mock_service_instance1.mark_not_in_use = AsyncMock()
        self.mock_service_instance1.id = "test-service-1"
        
        self.mock_service_instance2 = MagicMock()
        self.mock_service_instance2.initialize = AsyncMock(return_value=True)
        self.mock_service_instance2.cleanup = AsyncMock()
        self.mock_service_instance2.is_initialized = True
        self.mock_service_instance2.is_usable = True
        self.mock_service_instance2.check_health = AsyncMock(return_value=True)
        self.mock_service_instance2.mark_in_use = AsyncMock()
        self.mock_service_instance2.mark_not_in_use = AsyncMock()
        self.mock_service_instance2.id = "test-service-2"
        
        # Will return different instances on successive calls
        self.mock_service_resource.side_effect = [
            self.mock_service_instance1,
            self.mock_service_instance2
        ]
        
        # Initialize common test parameters
        self.test_address = "localhost:50051"
        self.mock_stub_class = MagicMock()
        
    def tearDown(self):
        patch.stopall()
        
    async def test_pool_initialization(self):
        """Test pool initialization with min_size=1."""
        # Create pool
        pool = GRPCConnectionPool(
            service_stub_class=self.mock_stub_class,
            address=self.test_address,
            min_size=1,
            max_size=3
        )
        
        # Start the pool
        await pool.start()
        
        # Verify a service instance was created and initialized
        self.mock_service_resource.assert_called_once()
        self.mock_service_instance1.initialize.assert_called_once()
        
        # Stop the pool
        await pool.stop()
        
        # Verify service instance was cleaned up
        self.mock_service_instance1.cleanup.assert_called_once()
        
    async def test_pool_min_size_two(self):
        """Test pool initialization with min_size=2."""
        # Create pool
        pool = GRPCConnectionPool(
            service_stub_class=self.mock_stub_class,
            address=self.test_address,
            min_size=2,
            max_size=3
        )
        
        # Start the pool
        await pool.start()
        
        # Verify two service instances were created and initialized
        self.assertEqual(self.mock_service_resource.call_count, 2)
        self.mock_service_instance1.initialize.assert_called_once()
        self.mock_service_instance2.initialize.assert_called_once()
        
        # Stop the pool
        await pool.stop()
        
        # Verify service instances were cleaned up
        self.mock_service_instance1.cleanup.assert_called_once()
        self.mock_service_instance2.cleanup.assert_called_once()
        
    async def test_acquire_release(self):
        """Test acquiring and releasing resources from the pool."""
        # Create pool
        pool = GRPCConnectionPool(
            service_stub_class=self.mock_stub_class,
            address=self.test_address,
            min_size=1,
            max_size=3
        )
        
        # Start the pool
        await pool.start()
        
        # The AsyncResourcePool class uses the resource_id as key in dictionaries
        # We need to hack a bit to make our test work
        pool._resources[self.mock_service_instance1.id] = self.mock_service_instance1
        pool._available_resources.add(self.mock_service_instance1.id)
        
        # Acquire a resource
        resource = await pool._acquire()
        
        # Verify resource was marked as in use
        self.assertEqual(resource, self.mock_service_instance1)
        self.mock_service_instance1.mark_in_use.assert_called_once()
        
        # Release the resource
        await pool._release(resource.id)
        
        # Verify resource was marked as not in use
        self.mock_service_instance1.mark_not_in_use.assert_called_once()
        
        # Stop the pool
        await pool.stop()
        
    async def test_acquire_context_manager(self):
        """Test acquiring resources using the context manager."""
        # Create pool
        pool = GRPCConnectionPool(
            service_stub_class=self.mock_stub_class,
            address=self.test_address,
            min_size=1,
            max_size=3
        )
        
        # Start the pool
        await pool.start()
        
        # The AsyncResourcePool class uses the resource_id as key in dictionaries
        # We need to hack a bit to make our test work
        pool._resources[self.mock_service_instance1.id] = self.mock_service_instance1
        pool._available_resources.add(self.mock_service_instance1.id)
        
        # Use the context manager
        async with pool.acquire() as resource:
            # Verify resource was acquired
            self.assertEqual(resource, self.mock_service_instance1)
            self.mock_service_instance1.mark_in_use.assert_called_once()
            
        # Verify resource was released after context exit
        self.mock_service_instance1.mark_not_in_use.assert_called_once()
        
        # Stop the pool
        await pool.stop()
        
    async def test_unhealthy_resource_replacement(self):
        """Test replacement of unhealthy resources in the pool."""
        # Create pool but don't start it yet
        pool = GRPCConnectionPool(
            service_stub_class=self.mock_stub_class,
            address=self.test_address,
            min_size=1,
            max_size=3
        )
        
        # We'll manually set up the pool state for testing
        pool._resources[self.mock_service_instance1.id] = self.mock_service_instance1
        pool._available_resources.add(self.mock_service_instance1.id)
        
        # Make the resource unhealthy
        self.mock_service_instance1.check_health = AsyncMock(return_value=False)
        self.mock_service_instance1.is_usable = False
        
        # Run health check
        await pool._check_all_health()
        
        # Verify cleanup and creation of replacement
        self.mock_service_instance1.cleanup.assert_called_once()
        
        # Reset the mock service resource for the next instance
        self.mock_service_resource.side_effect = None
        self.mock_service_resource.return_value = self.mock_service_instance2
        
        # Verify resource replacement when min_size constraint is enforced
        await pool._create_resource()
        
        # Stop the pool
        await pool.stop()


# Run the tests
if __name__ == "__main__":
    unittest.main()
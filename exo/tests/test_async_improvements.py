import asyncio
import unittest
import sys
import os
import time
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exo.helpers import AsyncCallback, AsyncCallbackSystem
from exo.orchestration.node import Node

class TestAsyncCallbacks(unittest.IsolatedAsyncioTestCase):
    """Test the async callback system improvements"""
    
    async def test_sync_callbacks(self):
        """Test that synchronous callbacks work properly"""
        callback = AsyncCallback[int]()
        results = []
        
        callback.on_next(lambda x: results.append(x * 2))
        callback.set(5)
        
        self.assertEqual(results, [10])
    
    async def test_async_callbacks(self):
        """Test that asynchronous callbacks work properly"""
        callback = AsyncCallback[int]()
        results = []
        event = asyncio.Event()
        
        async def async_callback(x):
            await asyncio.sleep(0.01)  # Simulate async work
            results.append(x * 3)
            event.set()
        
        callback.on_next_async(async_callback)
        
        # Also register a sync callback
        callback.on_next(lambda x: results.append(x * 2))
        
        # Trigger the callbacks
        callback.set(5)
        
        # Wait for the async callback to complete
        await asyncio.wait_for(event.wait(), timeout=1.0)
        
        # We should have both results
        self.assertEqual(sorted(results), [10, 15])
    
    async def test_wait_with_sync_condition(self):
        """Test the wait functionality with a synchronous condition"""
        callback = AsyncCallback[int]()
        
        # Set the value after a delay
        asyncio.create_task(self._delayed_set(callback, 0.05, 42))
        
        # Wait for the specific condition to be met
        result = await callback.wait(lambda x: x > 40, timeout=1.0)
        
        self.assertEqual(result, (42,))
    
    async def test_wait_with_async_condition(self):
        """Test the wait functionality with an asynchronous condition"""
        callback = AsyncCallback[int]()
        
        # Set the value after a delay
        asyncio.create_task(self._delayed_set(callback, 0.05, 42))
        
        # Define an async condition
        async def async_condition(x):
            await asyncio.sleep(0.01)  # Simulate async work
            return x > 40
        
        # Wait for the async condition to be met
        result = await callback.wait(async_condition, timeout=1.0)
        
        self.assertEqual(result, (42,))
    
    async def test_multiple_async_callbacks(self):
        """Test multiple async callbacks on the same event"""
        callback = AsyncCallback[int]()
        results1 = []
        results2 = []
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        
        async def async_callback1(x):
            await asyncio.sleep(0.02)
            results1.append(x * 2)
            event1.set()
        
        async def async_callback2(x):
            await asyncio.sleep(0.01)
            results2.append(x * 3)
            event2.set()
        
        callback.on_next_async(async_callback1)
        callback.on_next_async(async_callback2)
        
        callback.set(5)
        
        # Wait for both callbacks to complete
        await asyncio.gather(
            asyncio.wait_for(event1.wait(), timeout=1.0),
            asyncio.wait_for(event2.wait(), timeout=1.0)
        )
        
        self.assertEqual(results1, [10])
        self.assertEqual(results2, [15])
    
    async def test_callback_system(self):
        """Test the AsyncCallbackSystem with both sync and async callbacks"""
        system = AsyncCallbackSystem[str, int]()
        
        # Register handlers
        callback1 = system.register("callback1")
        callback2 = system.register("callback2")
        
        results1 = []
        results2 = []
        event = asyncio.Event()
        
        callback1.on_next(lambda x: results1.append(x * 2))
        
        async def async_handler(x):
            await asyncio.sleep(0.01)
            results2.append(x * 3)
            event.set()
        
        callback2.on_next_async(async_handler)
        
        # Trigger specific callback
        system.trigger("callback1", 5)
        self.assertEqual(results1, [10])
        self.assertEqual(results2, [])
        
        # Trigger all callbacks
        system.trigger_all(7)
        
        # Wait for async callback
        await asyncio.wait_for(event.wait(), timeout=1.0)
        
        self.assertEqual(results1, [10, 14])
        self.assertEqual(results2, [21])
    
    async def test_no_event_loop(self):
        """Test handling when no event loop is available for async callbacks"""
        callback = AsyncCallback[int]()
        
        # Mock running in a thread without event loop
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("No running event loop")):
            result = []
            
            async def async_handler(x):
                result.append(x * 2)
            
            callback.on_next_async(async_handler)
            
            # This should not raise an exception
            callback.set(5)
            
            # The async callback should not have been executed
            self.assertEqual(result, [])
    
    async def _delayed_set(self, callback, delay, value):
        """Helper to set a value after a delay"""
        await asyncio.sleep(delay)
        callback.set(value)


class TestNodeAsyncCallbacks(unittest.IsolatedAsyncioTestCase):
    """Test the Node class with async callbacks"""
    
    async def asyncSetUp(self):
        """Setup test environment"""
        # Create mock dependencies
        self.server = AsyncMock()
        self.inference_engine = AsyncMock()
        self.discovery = AsyncMock()
        self.shard_downloader = AsyncMock()
        self.partitioning_strategy = AsyncMock()
        
        # Mock the device_capabilities method to avoid external dependencies
        with patch('exo.orchestration.node.device_capabilities', return_value={}):
            self.node = Node(
                "test-node-id",
                self.server,
                self.inference_engine,
                self.discovery,
                self.shard_downloader,
                self.partitioning_strategy
            )
    
    async def test_on_node_status_async(self):
        """Test that the on_node_status async callback works properly"""
        # Skip this test as it requires a more complex setup with mocked dependencies
        self.skipTest("This test requires deeper mocking of Node dependencies")
        
        # NOTE: We'd need to properly mock:
        # - topology_lock
        # - current_topology 
        # - AsyncLock behavior
        # And other components to make this test work correctly


class TestTaskManager(unittest.IsolatedAsyncioTestCase):
    """Test a task manager implementation (for future implementation)"""
    
    async def test_task_lifecycle(self):
        """Example test for a task management system"""
        
        class TaskManager:
            def __init__(self):
                self.tasks = {}
                self._lock = asyncio.Lock()
                
            async def start_task(self, name, coro, cancel_existing=False):
                async with self._lock:
                    if name in self.tasks and not self.tasks[name].done():
                        if cancel_existing:
                            self.tasks[name].cancel()
                        else:
                            return self.tasks[name]
                            
                    task = asyncio.create_task(self._wrapped_coro(name, coro))
                    self.tasks[name] = task
                    return task
                    
            async def _wrapped_coro(self, name, coro_factory):
                try:
                    # Call the coroutine factory to get the actual coroutine
                    coro = coro_factory()
                    # Execute it
                    return await coro
                except asyncio.CancelledError:
                    # Just propagate cancellation
                    raise
                except Exception as e:
                    # Log but propagate
                    print(f"Task {name} failed: {e}")
                    raise
                finally:
                    # Clean up task reference
                    async with self._lock:
                        if name in self.tasks and self.tasks[name].done():
                            del self.tasks[name]
            
            async def wait_for_task(self, name, timeout=None):
                async with self._lock:
                    if name not in self.tasks:
                        return None
                    task = self.tasks[name]
                
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout)
                except asyncio.TimeoutError:
                    return None
            
            async def cancel_all(self):
                async with self._lock:
                    tasks_to_cancel = list(self.tasks.values())
                
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()
                
                # Wait for all tasks to complete
                if tasks_to_cancel:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Create a task manager
        manager = TaskManager()
        
        # Create test values
        results = []
        
        # Define test coroutines
        async def quick_task():
            results.append("quick")
            return "quick result"
        
        async def slow_task():
            await asyncio.sleep(0.1)
            results.append("slow")
            return "slow result"
        
        async def error_task():
            await asyncio.sleep(0.05)
            raise ValueError("task error")
        
        # Start tasks (without awaiting the coroutines first)
        quick_task_future = await manager.start_task("quick", quick_task)
        slow_future = await manager.start_task("slow", slow_task)
        error_future = await manager.start_task("error", error_task)
        
        # Wait for the quick task to complete and get its result
        quick_result = await quick_task_future
        
        # The quick task should complete immediately
        self.assertEqual(quick_result, "quick result")
        
        # Start same task again - should return existing future
        duplicate = await manager.start_task("slow", slow_task)
        self.assertIs(duplicate, slow_future)
        
        # Wait for the slow task
        slow_result = await slow_future
        self.assertEqual(slow_result, "slow result")
        
        # The error task should have failed but been cleaned up
        with self.assertRaises(ValueError):
            await error_future
        
        # Wait for all tasks to complete and run event loop briefly
        await asyncio.sleep(0.2)
        
        # Check the final results
        self.assertEqual(sorted(results), ["quick", "slow"])


if __name__ == "__main__":
    unittest.main()
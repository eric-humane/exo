"""
Tests for the TaskManager class
"""
import asyncio
import unittest
import sys
import os
import time
from unittest.mock import Mock, patch

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exo.utils.task_manager import (
    TaskManager, TaskPriority, TaskStatus, TaskInfo, 
    global_task_manager, managed_task
)

class TestTaskManager(unittest.IsolatedAsyncioTestCase):
    """Tests for TaskManager functionality"""
    
    async def test_basic_task_lifecycle(self):
        """Test the basic lifecycle of a task"""
        manager = TaskManager()
        
        # Define a test task
        async def test_task():
            await asyncio.sleep(0.1)
            return 42
        
        # Start the task
        task = await manager.start_task("test_task", lambda: test_task())
        
        # Wait a moment for the task to start
        await asyncio.sleep(0.05)
        
        # Check the task is registered
        self.assertIn("test_task", manager.tasks)
        self.assertEqual(manager.tasks["test_task"].status, TaskStatus.RUNNING)
        
        # Wait for the task to complete
        result = await task
        
        # Check the result and status
        self.assertEqual(result, 42)
        self.assertEqual(manager.tasks["test_task"].status, TaskStatus.COMPLETED)
        self.assertEqual(manager.tasks["test_task"].result, 42)
    
    async def test_task_exception_handling(self):
        """Test that exceptions in tasks are properly handled"""
        manager = TaskManager()
        
        # Define a task that raises an exception
        async def failing_task():
            await asyncio.sleep(0.1)
            raise ValueError("Test error")
        
        # Start the task
        task = await manager.start_task("failing_task", failing_task)
        
        # Wait for the task to complete (with exception)
        with self.assertRaises(ValueError):
            await task
        
        # Check the task status and exception
        self.assertEqual(manager.tasks["failing_task"].status, TaskStatus.FAILED)
        self.assertIsInstance(manager.tasks["failing_task"].exception, ValueError)
        self.assertEqual(str(manager.tasks["failing_task"].exception), "Test error")
    
    async def test_task_cancellation(self):
        """Test that tasks can be cancelled"""
        manager = TaskManager()
        
        # Define a long-running task
        async def long_task():
            await asyncio.sleep(10.0)
            return "completed"
        
        # Start the task
        task = await manager.start_task("long_task", lambda: long_task())
        
        # Wait a moment to ensure the task is running
        await asyncio.sleep(0.1)
        
        # Cancel the task
        await manager.cancel_task("long_task")
        
        # Wait for the task to be cancelled
        with self.assertRaises(asyncio.CancelledError):
            await task
        
        # Wait for status to update
        await asyncio.sleep(0.1)
        
        # Check the task status
        self.assertEqual(manager.tasks["long_task"].status, TaskStatus.CANCELLED)
    
    async def test_wait_for_task(self):
        """Test waiting for a task to complete"""
        manager = TaskManager()
        
        # Define a test task
        async def test_task():
            await asyncio.sleep(0.1)
            return "done"
        
        # Start the task
        await manager.start_task("test_task", test_task)
        
        # Wait for the task to complete
        result = await manager.wait_for_task("test_task")
        
        # Check the result
        self.assertEqual(result, "done")
    
    async def test_wait_for_task_timeout(self):
        """Test that waiting for a task can time out"""
        manager = TaskManager()
        
        # Define a long-running task
        async def long_task():
            await asyncio.sleep(10.0)
            return "completed"
        
        # Start the task
        await manager.start_task("long_task", long_task)
        
        # Wait for the task with a timeout
        with self.assertRaises(asyncio.TimeoutError):
            await manager.wait_for_task("long_task", timeout=0.1)
        
        # Cancel the task for cleanup
        await manager.cancel_task("long_task")
    
    async def test_task_groups(self):
        """Test managing tasks in groups"""
        manager = TaskManager()
        
        # Define some tasks
        async def group_task_1():
            await asyncio.sleep(0.5)
            return "task1"
        
        async def group_task_2():
            await asyncio.sleep(0.5)
            return "task2"
        
        # Start tasks in a group
        await manager.start_task("task1", lambda: group_task_1(), group="test_group")
        await manager.start_task("task2", lambda: group_task_2(), group="test_group")
        
        # Wait for tasks to be running
        await asyncio.sleep(0.1)
        
        # Check group tasks
        group_tasks = manager.get_group_tasks("test_group")
        self.assertEqual(len(group_tasks), 2)
        self.assertIn("task1", group_tasks)
        self.assertIn("task2", group_tasks)
        
        # Cancel the group
        cancelled = await manager.cancel_group("test_group")
        self.assertEqual(cancelled, 2)
    
    async def test_task_priorities(self):
        """Test task priorities"""
        manager = TaskManager()
        
        # Define test tasks
        async def task_func():
            await asyncio.sleep(0.1)
            return "done"
        
        # Start tasks with different priorities
        await manager.start_task("high_task", task_func, priority=TaskPriority.HIGH)
        await manager.start_task("low_task", task_func, priority=TaskPriority.LOW)
        
        # Check the priorities
        self.assertEqual(manager.tasks["high_task"].priority, TaskPriority.HIGH)
        self.assertEqual(manager.tasks["low_task"].priority, TaskPriority.LOW)
        
        # Check stats
        stats = manager.task_stats()
        self.assertEqual(stats["by_priority"][TaskPriority.HIGH.value], 1)
        self.assertEqual(stats["by_priority"][TaskPriority.LOW.value], 1)
    
    async def test_completion_callbacks(self):
        """Test that completion callbacks are called"""
        manager = TaskManager()
        callback_called = asyncio.Event()
        
        # Define a callback
        async def on_task_complete(name, info):
            self.assertEqual(name, "callback_test")
            self.assertEqual(info.result, "success")
            callback_called.set()
        
        # Register the callback
        manager.register_completion_callback(on_task_complete)
        
        # Define and start a task
        async def test_task():
            return "success"
        
        await manager.start_task("callback_test", test_task)
        
        # Wait for the callback to be called
        await asyncio.wait_for(callback_called.wait(), timeout=1.0)
        self.assertTrue(callback_called.is_set())
    
    async def test_cleanup_completed_tasks(self):
        """Test cleanup of completed tasks"""
        manager = TaskManager()
        
        # Define and start some tasks
        async def quick_task():
            return "done"
        
        await manager.start_task("task1", quick_task)
        await manager.start_task("task2", quick_task)
        
        # Wait for both tasks to complete
        await asyncio.sleep(0.1)
        
        # Clean up tasks
        removed = await manager.cleanup_completed_tasks()
        self.assertEqual(removed, 2)
        self.assertEqual(len(manager.tasks), 0)
    
    async def test_managed_task_decorator(self):
        """Test the managed_task decorator"""
        test_mgr = TaskManager()
        
        # Define a task with the decorator
        @managed_task(name="decorated_task", group="test_group", manager=test_mgr)
        async def decorated_task(param):
            await asyncio.sleep(0.1)
            return f"result: {param}"
        
        # Call the decorated function
        task = await decorated_task("test")
        
        # Wait for the task to complete
        result = await task
        
        # Check the result
        self.assertEqual(result, "result: test")
        
        # Check the task was properly managed
        self.assertIn("decorated_task", test_mgr.tasks)
        self.assertEqual(test_mgr.tasks["decorated_task"].status, TaskStatus.COMPLETED)
        self.assertEqual(test_mgr.tasks["decorated_task"].group, "test_group")
    
    async def test_duplicate_task_handling(self):
        """Test handling of duplicate task names"""
        manager = TaskManager()
        
        # Define test tasks
        async def task_impl():
            await asyncio.sleep(0.5)
            return "original"
        
        async def new_task_impl():
            return "replacement"
        
        # Start the original task
        original_task = await manager.start_task("duplicate", lambda: task_impl())
        
        # Wait a bit for the task to start
        await asyncio.sleep(0.1)
        
        # Try to start another with the same name (should return the existing one)
        duplicate_task = await manager.start_task("duplicate", lambda: new_task_impl())
        
        # Should be the same task object
        self.assertIs(duplicate_task, original_task)
        
        # Now try with cancel_existing=True
        replacement_task = await manager.start_task("duplicate", lambda: new_task_impl(), cancel_existing=True)
        
        # Should be a different task
        self.assertIsNot(replacement_task, original_task)
        
        # Wait for it to complete
        result = await replacement_task
        
        # Should have the result from the replacement task
        self.assertEqual(result, "replacement")
    
    async def test_cancel_all_tasks(self):
        """Test cancelling all tasks"""
        manager = TaskManager()
        
        # Define a long-running task
        async def long_task():
            await asyncio.sleep(10.0)
            return "done"
        
        # Start several long-running tasks
        for i in range(5):
            # Use a fresh lambda for each task to avoid capturing the loop variable
            await manager.start_task(f"task_{i}", lambda: long_task())
        
        # Wait for tasks to be running
        await asyncio.sleep(0.1)
        
        # Make sure they're all registered
        self.assertEqual(len(manager.tasks), 5)
        
        # Cancel all tasks
        cancelled = await manager.cancel_all_tasks()
        
        # All tasks should have been cancelled
        self.assertEqual(cancelled, 5)
        
        # Wait for cancellation to complete
        await asyncio.sleep(0.1)
        
        # All tasks should be in cancelled state
        for name, info in manager.tasks.items():
            self.assertEqual(info.status, TaskStatus.CANCELLED)

    async def test_task_duration(self):
        """Test task duration tracking"""
        manager = TaskManager()
        
        # Define a task that takes a specific time
        async def timed_task():
            await asyncio.sleep(0.1)
            return "done"
        
        # Start the task
        await manager.start_task("timed_task", timed_task)
        
        # Wait for it to complete
        await asyncio.sleep(0.2)
        
        # Check the duration
        duration = manager.tasks["timed_task"].duration
        self.assertIsNotNone(duration)
        self.assertGreaterEqual(duration, 0.1)
        self.assertLess(duration, 0.5)  # Allow some tolerance


if __name__ == "__main__":
    unittest.main()
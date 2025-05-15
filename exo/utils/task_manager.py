"""
Task Manager for exo

A centralized system for managing async tasks to:
- Track all running tasks
- Provide graceful cancellation
- Prevent resource leaks
- Properly handle exceptions
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple
import functools
import uuid
from enum import Enum
from exo import DEBUG

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Status of a managed task"""
    PENDING = "pending"        # Task is registered but not yet running
    RUNNING = "running"        # Task is currently running
    COMPLETED = "completed"    # Task completed successfully
    FAILED = "failed"          # Task failed with an exception
    CANCELLED = "cancelled"    # Task was cancelled

class TaskInfo:
    """Information about a managed task"""
    def __init__(self, name: str, task: asyncio.Task, priority: TaskPriority, 
                 group: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.task = task
        self.priority = priority
        self.group = group
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.status = TaskStatus.PENDING
        self.exception: Optional[Exception] = None
        self.result: Any = None

    @property
    def duration(self) -> Optional[float]:
        """Return the duration of the task in seconds, or None if not completed"""
        if self.started_at is None:
            return None
        if self.completed_at is None:
            return time.time() - self.started_at
        return self.completed_at - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation"""
        return {
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "group": self.group,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "exception": str(self.exception) if self.exception else None,
            "metadata": self.metadata
        }

class TaskManager:
    """
    Centralized manager for async tasks.
    
    Features:
    - Tracks all running tasks with detailed metadata
    - Provides graceful cancellation and cleanup
    - Prevents resource leaks
    - Handles exceptions properly
    - Groups related tasks for management
    - Supports task priorities
    
    Usage:
    ```python
    # Create a task manager
    manager = TaskManager()
    
    # Start a task with a name
    task = await manager.start_task("my_task", my_coroutine())
    
    # Wait for a task to complete
    result = await manager.wait_for_task("my_task")
    
    # Cancel a group of tasks
    await manager.cancel_group("background_tasks")
    
    # Get information about all tasks
    task_info = manager.get_all_task_info()
    
    # Graceful shutdown
    await manager.cancel_all_tasks()
    ```
    """
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self._task_lock = asyncio.Lock()
        self._shutdown_requested = False
        self._task_completion_callbacks: List[Callable[[str, TaskInfo], Awaitable[None]]] = []
        
    async def start_task(self, name: str, coro_factory: Callable[[], Awaitable[Any]], 
                    priority: TaskPriority = TaskPriority.NORMAL, 
                    group: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    cancel_existing: bool = False) -> asyncio.Task:
        """
        Start a new task with the given name.
        
        Args:
            name: Unique identifier for the task
            coro_factory: Factory function that returns the coroutine to run
            priority: Task priority level
            group: Optional group name for organizing related tasks
            metadata: Optional dictionary of metadata for the task
            cancel_existing: If True, cancel any existing task with the same name
            
        Returns:
            The created asyncio.Task object
        """
        # Generate a unique task ID if not provided
        task_id = name if name else f"task_{uuid.uuid4()}"
        
        async with self._task_lock:
            # Handle existing task with the same name
            if task_id in self.tasks:
                existing_task = self.tasks[task_id].task
                if not existing_task.done():
                    if cancel_existing:
                        if DEBUG >= 2: 
                            print(f"Cancelling existing task: {task_id}")
                        existing_task.cancel()
                    else:
                        if DEBUG >= 2: 
                            print(f"Returning existing task: {task_id}")
                        return existing_task
            
            # Create and wrap the task
            coro = coro_factory()
            task = asyncio.create_task(
                self._wrapped_coro(task_id, coro, priority, group, metadata)
            )
            
            # Store task info
            task_info = TaskInfo(task_id, task, priority, group, metadata)
            self.tasks[task_id] = task_info
            
            if DEBUG >= 2: 
                print(f"Started task: {task_id} (priority={priority.name}, group={group})")
            
            return task
            
    async def _wrapped_coro(self, name: str, coro: Awaitable[Any], 
                           priority: TaskPriority, group: Optional[str], 
                           metadata: Optional[Dict[str, Any]]) -> Any:
        """Internal wrapper for task coroutines to handle lifecycle events"""
        # Update task status to running
        async with self._task_lock:
            if name in self.tasks:
                self.tasks[name].status = TaskStatus.RUNNING
                self.tasks[name].started_at = time.time()
        
        try:
            # Execute the coroutine
            result = await coro
            
            # Update task info on successful completion
            async with self._task_lock:
                if name in self.tasks:
                    self.tasks[name].status = TaskStatus.COMPLETED
                    self.tasks[name].completed_at = time.time()
                    self.tasks[name].result = result
            
            return result
            
        except asyncio.CancelledError:
            # Handle cancellation
            async with self._task_lock:
                if name in self.tasks:
                    self.tasks[name].status = TaskStatus.CANCELLED
                    self.tasks[name].completed_at = time.time()
            
            # Re-raise to propagate the cancellation
            raise
            
        except Exception as e:
            # Handle exceptions
            if DEBUG >= 1: 
                print(f"Task {name} failed: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
            
            # Update task info
            async with self._task_lock:
                if name in self.tasks:
                    self.tasks[name].status = TaskStatus.FAILED
                    self.tasks[name].completed_at = time.time()
                    self.tasks[name].exception = e
            
            # Re-raise the exception
            raise
            
        finally:
            # Notify completion callbacks
            if name in self.tasks:
                task_info = self.tasks[name]
                for callback in self._task_completion_callbacks:
                    try:
                        asyncio.create_task(callback(name, task_info))
                    except Exception as callback_error:
                        if DEBUG >= 1:
                            print(f"Error in task completion callback: {callback_error}")
            
            # Perform cleanup for completed tasks if appropriate
            if not self._shutdown_requested:
                async with self._task_lock:
                    if name in self.tasks and (
                        self.tasks[name].status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                    ):
                        # Keep completed tasks in the registry for history/debugging
                        # But we could add a cleanup policy here
                        pass
    
    async def wait_for_task(self, name: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a task to complete and return its result.
        
        Args:
            name: The name of the task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The result of the task
            
        Raises:
            KeyError: If the task doesn't exist
            asyncio.TimeoutError: If the wait times out
            Exception: Any exception raised by the task
        """
        async with self._task_lock:
            if name not in self.tasks:
                raise KeyError(f"Task {name} not found")
            task = self.tasks[name].task
        
        try:
            # Wait for the task to complete
            return await asyncio.wait_for(task, timeout)
        except asyncio.TimeoutError:
            # Just re-raise timeout errors
            raise
        except Exception:
            # Re-raise any other exceptions
            raise
    
    def get_task_info(self, name: str) -> Optional[TaskInfo]:
        """Get information about a specific task"""
        return self.tasks.get(name)
    
    def get_all_task_info(self) -> Dict[str, TaskInfo]:
        """Get information about all tasks"""
        return dict(self.tasks)
    
    def get_group_tasks(self, group: str) -> Dict[str, TaskInfo]:
        """Get all tasks in a specific group"""
        return {name: info for name, info in self.tasks.items() 
                if info.group == group}
    
    async def cancel_task(self, name: str) -> bool:
        """
        Cancel a specific task.
        
        Returns:
            True if the task was cancelled, False if it wasn't found or already done
        """
        async with self._task_lock:
            if name in self.tasks and not self.tasks[name].task.done():
                self.tasks[name].task.cancel()
                return True
        return False
    
    async def cancel_group(self, group: str) -> int:
        """
        Cancel all tasks in a specific group.
        
        Returns:
            The number of tasks that were cancelled
        """
        count = 0
        async with self._task_lock:
            for name, info in list(self.tasks.items()):
                if info.group == group and not info.task.done():
                    info.task.cancel()
                    count += 1
        return count
    
    async def cancel_all_tasks(self, wait: bool = True) -> int:
        """
        Cancel all tasks managed by this TaskManager.
        
        Args:
            wait: If True, wait for all tasks to complete after cancellation
            
        Returns:
            The number of tasks that were cancelled
        """
        self._shutdown_requested = True
        
        # Get tasks to cancel
        async with self._task_lock:
            tasks_to_cancel = [
                (name, info.task) 
                for name, info in self.tasks.items() 
                if not info.task.done()
            ]
        
        # Cancel all pending tasks
        for name, task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete if requested
        if wait and tasks_to_cancel:
            # Wait for all tasks to complete, ignoring any exceptions
            await asyncio.gather(
                *[task for _, task in tasks_to_cancel],
                return_exceptions=True
            )
        
        return len(tasks_to_cancel)
    
    def register_completion_callback(self, callback: Callable[[str, TaskInfo], Awaitable[None]]) -> None:
        """
        Register a callback to be called when a task completes.
        
        Args:
            callback: An async function that takes task name and info parameters
        """
        self._task_completion_callbacks.append(callback)
    
    async def cleanup_completed_tasks(self, max_age: Optional[float] = None) -> int:
        """
        Remove completed tasks from the registry.
        
        Args:
            max_age: Optional maximum age in seconds for tasks to be kept
            
        Returns:
            The number of tasks that were removed
        """
        now = time.time()
        to_remove = []
        
        async with self._task_lock:
            for name, info in self.tasks.items():
                if info.task.done():
                    if max_age is None or (info.completed_at and now - info.completed_at > max_age):
                        to_remove.append(name)
            
            for name in to_remove:
                del self.tasks[name]
        
        return len(to_remove)

    def task_stats(self) -> Dict[str, Any]:
        """Get statistics about tasks"""
        stats = {
            "total": len(self.tasks),
            "by_status": {status.value: 0 for status in TaskStatus},
            "by_priority": {priority.value: 0 for priority in TaskPriority},
            "by_group": {},
            "average_duration": 0.0,
        }
        
        durations = []
        
        for info in self.tasks.values():
            # Count by status
            stats["by_status"][info.status.value] += 1
            
            # Count by priority
            stats["by_priority"][info.priority.value] += 1
            
            # Count by group
            if info.group:
                stats["by_group"].setdefault(info.group, 0)
                stats["by_group"][info.group] += 1
            
            # Calculate durations for completed tasks
            if info.duration is not None:
                durations.append(info.duration)
        
        # Calculate average duration
        if durations:
            stats["average_duration"] = sum(durations) / len(durations)
        
        return stats


# Create a global task manager instance
global_task_manager = TaskManager()

# Decorator for managing tasks
def managed_task(name: Optional[str] = None, 
                priority: TaskPriority = TaskPriority.NORMAL,
                group: Optional[str] = None,
                manager: Optional[TaskManager] = None):
    """
    Decorator to run a coroutine function as a managed task.
    
    Example:
    ```python
    @managed_task(group="background")
    async def my_task(param1, param2):
        # Task implementation
        return result
    ```
    """
    def decorator(coro_func):
        @functools.wraps(coro_func)
        async def wrapper(*args, **kwargs):
            # Get the task manager to use
            task_mgr = manager or global_task_manager
            
            # Generate a name if not provided
            task_name = name or f"{coro_func.__name__}_{uuid.uuid4()}"
            
            # Start the task
            return await task_mgr.start_task(
                task_name,
                lambda: coro_func(*args, **kwargs),
                priority=priority,
                group=group,
                metadata={"args": str(args), "kwargs": str(kwargs)}
            )
        
        return wrapper
    
    return decorator
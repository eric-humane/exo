# Exo Utilities

This directory contains utility modules used across the exo codebase.

## TaskManager

The `TaskManager` provides a centralized system for managing asynchronous tasks in exo. It offers several advantages over using raw `asyncio.create_task()` calls:

### Features

- **Tracking and Monitoring**: Keeps track of all running tasks with detailed metadata
- **Graceful Cancellation**: Provides utilities for gracefully cancelling tasks
- **Resource Management**: Prevents resource leaks by properly handling task lifecycle
- **Exception Handling**: Properly handles and records exceptions from tasks
- **Task Grouping**: Allows organizing related tasks into logical groups
- **Priority Support**: Supports different priority levels for tasks

### Usage

```python
from exo.utils import TaskManager, TaskPriority, managed_task

# Create a TaskManager instance 
manager = TaskManager()

# Start a task with a name
task = await manager.start_task("my_task", lambda: my_coroutine())

# Wait for a task to complete
result = await manager.wait_for_task("my_task")

# Cancel a group of tasks
await manager.cancel_group("background_tasks")

# Get information about all tasks
task_info = manager.get_all_task_info()

# Graceful shutdown
await manager.cancel_all_tasks()
```

### Decorator Usage

Tasks can also be managed using the decorator syntax:

```python
from exo.utils import managed_task, TaskPriority

@managed_task(group="background_tasks", priority=TaskPriority.LOW)
async def my_background_task():
    # Task implementation
    pass

# Call the function - it will be managed automatically
await my_background_task()
```

### Benefits in Exo

The `TaskManager` helps improve exo's stability and performance by:

1. **Resource Leak Prevention**: Tasks always properly cleaned up
2. **System Visibility**: Allows inspecting all running tasks at any time
3. **Debugging Improvements**: Tracks task status and exceptions
4. **Clean Shutdowns**: Ensures graceful shutdown by managing task cancellation

## Global Task Manager

A global task manager instance (`global_task_manager`) is available for components that don't need their own isolated task management:

```python
from exo.utils import global_task_manager

# Use the global task manager
await global_task_manager.start_task("global_task", my_coroutine)
```

## Integration with Node

The TaskManager has been integrated into the Node class for managing all asynchronous operations in the orchestration layer.
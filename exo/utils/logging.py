"""
Structured logging module for exo.

This module provides a consistent logging API for the exo project, with support for:
- Different log levels
- Structured log formats (JSON for machines, pretty for humans)
- Log filtering by component
- Environment variable control of log levels
"""

import os
import sys
import json
import time
import traceback
from typing import Any, Dict, Optional, Union, List
from enum import IntEnum
import threading

# Thread-local storage for context data
_thread_local = threading.local()


class LogLevel(IntEnum):
    """Log level enum with integer values matching the DEBUG env var convention."""
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 5
    TRACE = 9


# Default log level from environment
DEFAULT_LOG_LEVEL = LogLevel(int(os.getenv("DEBUG", "0")))
# Component-specific log levels
_COMPONENT_LOG_LEVELS = {}

# Fill component log levels from environment variables
for env_var, value in os.environ.items():
    if env_var.startswith("DEBUG_") and value.isdigit():
        component = env_var[6:].lower()  # Strip 'DEBUG_' prefix
        _COMPONENT_LOG_LEVELS[component] = LogLevel(int(value))

# Log format: 'text' for human-readable, 'json' for machine-readable
LOG_FORMAT = os.getenv("EXO_LOG_FORMAT", "text").lower()


def get_log_level(component: Optional[str] = None) -> LogLevel:
    """Get the log level for a component."""
    if component is not None:
        component = component.lower()
        return _COMPONENT_LOG_LEVELS.get(component, DEFAULT_LOG_LEVEL)
    return DEFAULT_LOG_LEVEL


def set_context(**kwargs) -> None:
    """Set thread-local context values that will be included in all logs."""
    if not hasattr(_thread_local, "context"):
        _thread_local.context = {}
    _thread_local.context.update(kwargs)


def get_context() -> Dict[str, Any]:
    """Get the current thread-local context."""
    if not hasattr(_thread_local, "context"):
        _thread_local.context = {}
    return _thread_local.context


def clear_context() -> None:
    """Clear the thread-local context."""
    if hasattr(_thread_local, "context"):
        _thread_local.context.clear()


def _build_log_record(
    level: LogLevel,
    message: str,
    component: Optional[str] = None,
    exc_info: Optional[Exception] = None,
    **kwargs
) -> Dict[str, Any]:
    """Build a structured log record."""
    record = {
        "timestamp": time.time(),
        "level": level.name,
        "message": message,
    }
    
    # Add component if specified
    if component:
        record["component"] = component
    
    # Add exception info if provided
    if exc_info:
        record["exception"] = {
            "type": exc_info.__class__.__name__,
            "message": str(exc_info),
            "traceback": traceback.format_exception(
                type(exc_info), exc_info, exc_info.__traceback__
            )
        }
    
    # Add thread-local context
    context = get_context()
    if context:
        record["context"] = context
    
    # Add other keyword arguments
    if kwargs:
        record["data"] = kwargs
    
    return record


def _format_text_log(record: Dict[str, Any]) -> str:
    """Format a log record as human-readable text."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["timestamp"]))
    level_str = f"{record['level']:8}"
    
    # Base message with timestamp and level
    parts = [f"{timestamp} [{level_str}]"]
    
    # Add component if present
    if "component" in record:
        parts.append(f"[{record['component']}]")
    
    # Add the message
    parts.append(record["message"])
    
    # Format the basic log line
    log_line = " ".join(parts)
    
    # Add context if present
    if "context" in record:
        context_items = [f"{k}={v}" for k, v in record["context"].items()]
        log_line += f" [context: {', '.join(context_items)}]"
    
    # Add data if present
    if "data" in record:
        data_items = [f"{k}={v}" for k, v in record["data"].items()]
        log_line += f" [data: {', '.join(data_items)}]"
    
    # Add exception if present
    if "exception" in record:
        exc = record["exception"]
        log_line += f"\nException: {exc['type']}: {exc['message']}"
        log_line += "\nTraceback:\n" + "".join(exc["traceback"])
    
    return log_line


def log(
    level: Union[int, LogLevel],
    message: str,
    component: Optional[str] = None,
    exc_info: Optional[Exception] = None,
    **kwargs
) -> None:
    """
    Log a message with the specified level.
    
    Args:
        level: The log level as an int or LogLevel enum
        message: The log message
        component: The component name (optional)
        exc_info: Exception information (optional)
        **kwargs: Additional key-value pairs to include in the log
    """
    # Convert level to enum if it's an int
    if isinstance(level, int):
        level = LogLevel(level)
    
    # Check if this log should be emitted based on level
    component_level = get_log_level(component)
    if level > component_level:
        return
    
    # Build the log record
    record = _build_log_record(level, message, component, exc_info, **kwargs)
    
    # Format and output the log
    if LOG_FORMAT == "json":
        log_line = json.dumps(record)
    else:
        log_line = _format_text_log(record)
    
    # Determine output stream based on level
    if level <= LogLevel.ERROR:
        print(log_line, file=sys.stderr)
    else:
        print(log_line, file=sys.stdout)


# Convenience methods for different log levels
def critical(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log a critical message."""
    log(LogLevel.CRITICAL, message, component, exc_info, **kwargs)


def error(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log an error message."""
    log(LogLevel.ERROR, message, component, exc_info, **kwargs)


def warning(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log a warning message."""
    log(LogLevel.WARNING, message, component, exc_info, **kwargs)


def info(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log an info message."""
    log(LogLevel.INFO, message, component, exc_info, **kwargs)


def debug(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log a debug message."""
    log(LogLevel.DEBUG, message, component, exc_info, **kwargs)


def trace(message: str, component: Optional[str] = None, exc_info: Optional[Exception] = None, **kwargs) -> None:
    """Log a trace message."""
    log(LogLevel.TRACE, message, component, exc_info, **kwargs)
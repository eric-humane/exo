"""
Dynamic timeout calculation based on network conditions.

This module provides utilities for dynamically adjusting timeouts based on
observed network performance and conditions.
"""

import time
import math
import statistics
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import threading

# Thread-local storage for samples and stats
_tls = threading.local()


class TimeoutStats:
    """Statistics for timeout calculations."""
    
    def __init__(
        self,
        window_size: int = 100,
        min_timeout: float = 0.5,
        max_timeout: float = 30.0,
        default_timeout: float = 5.0,
        multiplier: float = 2.0
    ):
        self.window_size = window_size
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.default_timeout = default_timeout
        self.multiplier = multiplier
        
        # Samples for RTT (round-trip time)
        self.rtt_samples = deque(maxlen=window_size)
        
        # Samples for packet/response loss
        self.loss_samples = deque(maxlen=window_size)
        
        # Last calculated timeout
        self.last_timeout = default_timeout
        
        # Network condition score (0-1, higher is worse)
        self.condition_score = 0.0
        
        # Timestamp of last update
        self.last_update = time.time()
    
    def add_rtt_sample(self, rtt: float) -> None:
        """
        Add a round-trip time sample.
        
        Args:
            rtt: Round-trip time in seconds
        """
        self.rtt_samples.append(rtt)
        self.last_update = time.time()
    
    def add_loss_sample(self, success: bool) -> None:
        """
        Add a loss sample.
        
        Args:
            success: True if the request succeeded, False if it failed/timed out
        """
        self.loss_samples.append(1.0 if success else 0.0)
        self.last_update = time.time()
    
    def get_timeout(self) -> float:
        """
        Calculate a timeout based on current network conditions.
        
        Returns:
            Timeout in seconds
        """
        self._update_condition_score()
        
        # If we don't have enough samples, use the default
        if len(self.rtt_samples) < 3:
            return self.default_timeout
        
        # Calculate timeout based on RTT stats
        try:
            # Use mean + 2*std_dev as a baseline
            mean_rtt = statistics.mean(self.rtt_samples)
            std_dev = statistics.stdev(self.rtt_samples)
            
            # Start with a statistical timeout
            timeout = mean_rtt + 2 * std_dev
            
            # Apply multiplier based on condition score
            # Higher condition score = more buffer in the timeout
            timeout *= (1.0 + self.condition_score * (self.multiplier - 1.0))
            
            # Ensure timeout is within bounds
            timeout = max(self.min_timeout, min(self.max_timeout, timeout))
            
            self.last_timeout = timeout
            return timeout
        except (statistics.StatisticsError, ValueError, ZeroDivisionError):
            # In case of any error, use previous timeout
            return self.last_timeout
    
    def get_condition_score(self) -> float:
        """
        Get the network condition score.
        
        Returns:
            Score from 0.0 (excellent) to 1.0 (terrible)
        """
        self._update_condition_score()
        return self.condition_score
    
    def _update_condition_score(self) -> None:
        """Update the network condition score based on samples."""
        # Calculate loss rate
        loss_rate = 0.0
        if self.loss_samples:
            loss_rate = 1.0 - (sum(self.loss_samples) / len(self.loss_samples))
        
        # Calculate RTT score
        rtt_score = 0.0
        if self.rtt_samples:
            try:
                mean_rtt = statistics.mean(self.rtt_samples)
                # Normalize RTT score: 0.0 for RTT <= 0.1s, 1.0 for RTT >= 1.0s
                rtt_score = min(1.0, max(0.0, (mean_rtt - 0.1) / 0.9))
            except (statistics.StatisticsError, ValueError):
                pass
        
        # Combined score (loss has higher weight than RTT)
        self.condition_score = 0.7 * loss_rate + 0.3 * rtt_score
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.rtt_samples.clear()
        self.loss_samples.clear()
        self.last_timeout = self.default_timeout
        self.condition_score = 0.0
        self.last_update = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics as a dictionary.
        
        Returns:
            Dictionary with statistics
        """
        rtt_stats = {}
        if self.rtt_samples:
            try:
                rtt_stats = {
                    "mean": statistics.mean(self.rtt_samples),
                    "median": statistics.median(self.rtt_samples),
                    "stdev": statistics.stdev(self.rtt_samples) if len(self.rtt_samples) > 1 else 0.0,
                    "min": min(self.rtt_samples),
                    "max": max(self.rtt_samples),
                    "samples": len(self.rtt_samples)
                }
            except (statistics.StatisticsError, ValueError):
                pass
        
        loss_rate = 0.0
        if self.loss_samples:
            loss_rate = 1.0 - (sum(self.loss_samples) / len(self.loss_samples))
        
        return {
            "rtt": rtt_stats,
            "loss_rate": loss_rate,
            "samples": len(self.loss_samples),
            "condition_score": self.condition_score,
            "current_timeout": self.last_timeout,
            "last_update": self.last_update
        }


# Dictionary of timeout stats by name
_timeout_stats: Dict[str, TimeoutStats] = {}
_timeout_stats_lock = threading.RLock()


def get_timeout_stats(name: str) -> TimeoutStats:
    """
    Get timeout statistics for a named endpoint/service.
    
    Args:
        name: Identifier for the endpoint/service
        
    Returns:
        TimeoutStats instance
    """
    with _timeout_stats_lock:
        if name not in _timeout_stats:
            _timeout_stats[name] = TimeoutStats()
        return _timeout_stats[name]


def get_timeout(name: str) -> float:
    """
    Get a timeout for a named endpoint/service.
    
    Args:
        name: Identifier for the endpoint/service
        
    Returns:
        Timeout in seconds
    """
    return get_timeout_stats(name).get_timeout()


def add_rtt_sample(name: str, rtt: float) -> None:
    """
    Add a round-trip time sample for a named endpoint/service.
    
    Args:
        name: Identifier for the endpoint/service
        rtt: Round-trip time in seconds
    """
    get_timeout_stats(name).add_rtt_sample(rtt)


def add_loss_sample(name: str, success: bool) -> None:
    """
    Add a loss sample for a named endpoint/service.
    
    Args:
        name: Identifier for the endpoint/service
        success: True if the request succeeded, False if it failed/timed out
    """
    get_timeout_stats(name).add_loss_sample(success)


def reset_stats(name: str) -> None:
    """
    Reset statistics for a named endpoint/service.
    
    Args:
        name: Identifier for the endpoint/service
    """
    if name in _timeout_stats:
        _timeout_stats[name].reset()


def get_all_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all endpoints/services.
    
    Returns:
        Dictionary mapping endpoint/service names to their statistics
    """
    with _timeout_stats_lock:
        return {name: stats.get_stats() for name, stats in _timeout_stats.items()}


# Context manager for timing operations and recording results
class TimingContext:
    """
    Context manager for timing operations and recording results.
    
    This can be used to time operations and automatically add RTT and
    loss samples.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            rtt = time.time() - self.start_time
            add_rtt_sample(self.name, rtt)
            add_loss_sample(self.name, exc_type is None)
        return False  # Don't suppress exceptions


# Thread-local timing context
def start_timing(name: str) -> None:
    """
    Start timing an operation.
    
    This should be matched with a corresponding call to end_timing().
    
    Args:
        name: Identifier for the endpoint/service
    """
    if not hasattr(_tls, "timing_contexts"):
        _tls.timing_contexts = {}
    
    _tls.timing_contexts[name] = time.time()


def end_timing(name: str, success: bool = True) -> None:
    """
    End timing an operation and record the results.
    
    Args:
        name: Identifier for the endpoint/service
        success: True if the operation succeeded, False otherwise
    """
    if not hasattr(_tls, "timing_contexts") or name not in _tls.timing_contexts:
        # If we didn't start timing, we can't end it
        return
    
    start_time = _tls.timing_contexts.pop(name)
    rtt = time.time() - start_time
    
    add_rtt_sample(name, rtt)
    add_loss_sample(name, success)


# Convenience function for timing a function call
def timed_call(name: str, func, *args, **kwargs):
    """
    Call a function and time its execution.
    
    Args:
        name: Identifier for the endpoint/service
        func: Function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    with TimingContext(name):
        return func(*args, **kwargs)


# Decorator for timing function calls
def timed(name: str = None):
    """
    Decorator to time function calls.
    
    Args:
        name: Optional identifier for the endpoint/service. If not provided,
              the function's fully qualified name will be used.
              
    Returns:
        Decorated function
    """
    def decorator(func):
        nonlocal name
        if name is None:
            # Use the function's fully qualified name
            name = f"{func.__module__}.{func.__qualname__}"
        
        def wrapper(*args, **kwargs):
            with TimingContext(name):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
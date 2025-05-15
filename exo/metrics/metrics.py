"""
Metrics collection and reporting system for exo.

This module provides a simple, lightweight metrics system for tracking
performance, resource usage, and operational statistics.
"""

import os
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, deque
from exo.utils import logging


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricValue:
    """Base class for metric values."""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = "", tags: Dict[str, str] = None):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.tags = tags or {}
        self.created_at = time.time()
        self.last_updated = self.created_at
    
    def get_value(self) -> Any:
        """Get the current value of the metric."""
        raise NotImplementedError()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "tags": self.tags,
            "value": self.get_value(),
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }


class Counter(MetricValue):
    """
    Counter metric that can only increase.
    
    Used for tracking events, requests, errors, etc.
    """
    
    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None):
        super().__init__(name, MetricType.COUNTER, description, tags)
        self.value = 0
    
    def increment(self, value: int = 1):
        """Increment the counter by the given value."""
        if value <= 0:
            raise ValueError("Counter can only be incremented by positive values")
        self.value += value
        self.last_updated = time.time()
    
    def get_value(self) -> int:
        """Get the current counter value."""
        return self.value


class Gauge(MetricValue):
    """
    Gauge metric that can increase or decrease.
    
    Used for measuring values that can go up or down, like memory usage,
    queue sizes, etc.
    """
    
    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None):
        super().__init__(name, MetricType.GAUGE, description, tags)
        self.value = 0
    
    def set(self, value: float):
        """Set the gauge to a specific value."""
        self.value = value
        self.last_updated = time.time()
    
    def increment(self, value: float = 1.0):
        """Increment the gauge by the given value."""
        self.value += value
        self.last_updated = time.time()
    
    def decrement(self, value: float = 1.0):
        """Decrement the gauge by the given value."""
        self.value -= value
        self.last_updated = time.time()
    
    def get_value(self) -> float:
        """Get the current gauge value."""
        return self.value


class Histogram(MetricValue):
    """
    Histogram metric for tracking distributions of values.
    
    Used for measuring value distributions like request durations,
    sizes, etc.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        tags: Dict[str, str] = None,
        max_samples: int = 1000,
        buckets: List[float] = None
    ):
        super().__init__(name, MetricType.HISTOGRAM, description, tags)
        self.samples = deque(maxlen=max_samples)
        self.count = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.buckets = buckets or [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        self.bucket_counts = defaultdict(int)
    
    def record(self, value: float):
        """Record a value in the histogram."""
        self.samples.append(value)
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        
        # Update buckets
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
        
        self.last_updated = time.time()
    
    def get_value(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        if self.count == 0:
            return {
                "count": 0,
                "sum": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "buckets": {str(bucket): 0 for bucket in self.buckets}
            }
        
        avg = self.sum / self.count if self.count > 0 else 0
        
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else 0,
            "max": self.max if self.max != float('-inf') else 0,
            "avg": avg,
            "buckets": {str(bucket): self.bucket_counts[bucket] for bucket in self.buckets}
        }


class Timer(MetricValue):
    """
    Timer metric for measuring durations.
    
    Provides convenience methods for timing code execution.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        tags: Dict[str, str] = None,
        max_samples: int = 1000
    ):
        super().__init__(name, MetricType.TIMER, description, tags)
        self.histogram = Histogram(name, description, tags, max_samples)
        self._start_times = {}  # Thread-local storage for start times
    
    def start(self) -> int:
        """
        Start a timer and return a timer ID.
        
        Returns:
            Timer ID for later stopping
        """
        timer_id = hash(f"{threading.get_ident()}:{time.time()}")
        self._start_times[timer_id] = time.time()
        return timer_id
    
    def stop(self, timer_id: int) -> float:
        """
        Stop a timer and record its duration.
        
        Args:
            timer_id: The timer ID from start()
            
        Returns:
            Duration in seconds
        """
        if timer_id not in self._start_times:
            raise ValueError(f"Timer ID {timer_id} not found")
        
        start_time = self._start_times.pop(timer_id)
        duration = time.time() - start_time
        self.histogram.record(duration)
        self.last_updated = time.time()
        
        return duration
    
    def record(self, duration: float):
        """
        Record a duration directly.
        
        Args:
            duration: Time in seconds
        """
        self.histogram.record(duration)
        self.last_updated = time.time()
    
    def time(self):
        """
        Context manager for timing a block of code.
        
        Example:
            with timer.time():
                # Code to time
        """
        return TimerContext(self)
    
    def get_value(self) -> Dict[str, Any]:
        """Get timer statistics."""
        return self.histogram.get_value()


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, timer: Timer):
        self.timer = timer
        self.timer_id = None
    
    def __enter__(self):
        self.timer_id = self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.stop(self.timer_id)
        return False  # Don't suppress exceptions


class MetricsRegistry:
    """
    Registry for managing metrics.
    
    This class provides a central place to register, retrieve, and report metrics.
    """
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id
        self.metrics: Dict[str, MetricValue] = {}
        self.lock = threading.RLock()
        self._export_task = None
        self._shutting_down = False
        
        # Export configuration
        self.export_enabled = os.environ.get("EXO_METRICS_EXPORT", "1").lower() in ("1", "true", "yes")
        self.export_interval = float(os.environ.get("EXO_METRICS_INTERVAL", "60"))
        self.export_path = Path(os.environ.get("EXO_METRICS_PATH", "/tmp/exo_metrics"))
        
        # Ensure export directory exists if export is enabled
        if self.export_enabled and not self.export_path.exists():
            try:
                self.export_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create metrics export directory: {self.export_path}",
                             component="metrics",
                             exc_info=e)
                self.export_enabled = False
    
    def _get_metric_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Get a unique key for a metric based on name and tags."""
        if not tags:
            return name
            
        tag_str = ";".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def counter(self, name: str, description: str = "", tags: Dict[str, str] = None) -> Counter:
        """
        Get or create a counter metric.
        
        Args:
            name: Metric name
            description: Description of the metric
            tags: Metric tags (dimensions)
            
        Returns:
            Counter instance
        """
        with self.lock:
            key = self._get_metric_key(name, tags)
            
            if key not in self.metrics:
                self.metrics[key] = Counter(name, description, tags)
            elif not isinstance(self.metrics[key], Counter):
                raise TypeError(f"Metric {key} already exists with type {type(self.metrics[key]).__name__}")
                
            return self.metrics[key]
    
    def gauge(self, name: str, description: str = "", tags: Dict[str, str] = None) -> Gauge:
        """
        Get or create a gauge metric.
        
        Args:
            name: Metric name
            description: Description of the metric
            tags: Metric tags (dimensions)
            
        Returns:
            Gauge instance
        """
        with self.lock:
            key = self._get_metric_key(name, tags)
            
            if key not in self.metrics:
                self.metrics[key] = Gauge(name, description, tags)
            elif not isinstance(self.metrics[key], Gauge):
                raise TypeError(f"Metric {key} already exists with type {type(self.metrics[key]).__name__}")
                
            return self.metrics[key]
    
    def histogram(
        self, 
        name: str, 
        description: str = "", 
        tags: Dict[str, str] = None,
        max_samples: int = 1000,
        buckets: List[float] = None
    ) -> Histogram:
        """
        Get or create a histogram metric.
        
        Args:
            name: Metric name
            description: Description of the metric
            tags: Metric tags (dimensions)
            max_samples: Maximum number of samples to keep
            buckets: Histogram buckets (upper bounds)
            
        Returns:
            Histogram instance
        """
        with self.lock:
            key = self._get_metric_key(name, tags)
            
            if key not in self.metrics:
                self.metrics[key] = Histogram(name, description, tags, max_samples, buckets)
            elif not isinstance(self.metrics[key], Histogram):
                raise TypeError(f"Metric {key} already exists with type {type(self.metrics[key]).__name__}")
                
            return self.metrics[key]
    
    def timer(
        self, 
        name: str, 
        description: str = "", 
        tags: Dict[str, str] = None,
        max_samples: int = 1000
    ) -> Timer:
        """
        Get or create a timer metric.
        
        Args:
            name: Metric name
            description: Description of the metric
            tags: Metric tags (dimensions)
            max_samples: Maximum number of samples to keep
            
        Returns:
            Timer instance
        """
        with self.lock:
            key = self._get_metric_key(name, tags)
            
            if key not in self.metrics:
                self.metrics[key] = Timer(name, description, tags, max_samples)
            elif not isinstance(self.metrics[key], Timer):
                raise TypeError(f"Metric {key} already exists with type {type(self.metrics[key]).__name__}")
                
            return self.metrics[key]
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as a dictionary."""
        with self.lock:
            return {key: metric.to_dict() for key, metric in self.metrics.items()}
    
    def get_metrics_by_prefix(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        """Get metrics that match a name prefix."""
        with self.lock:
            return {
                key: metric.to_dict() 
                for key, metric in self.metrics.items() 
                if metric.name.startswith(prefix)
            }
    
    def export_metrics(self) -> str:
        """Export metrics as a JSON string."""
        metrics_data = {
            "timestamp": time.time(),
            "node_id": self.node_id,
            "metrics": self.get_metrics()
        }
        
        return json.dumps(metrics_data, indent=2)
    
    async def start_export_task(self):
        """Start the metrics export task."""
        if not self.export_enabled:
            logging.info("Metrics export is disabled", component="metrics")
            return
            
        self._shutting_down = False
        self._export_task = asyncio.create_task(self._export_loop())
        
        logging.info(f"Metrics export task started with interval {self.export_interval}s",
                    component="metrics",
                    export_path=str(self.export_path))
    
    async def stop_export_task(self):
        """Stop the metrics export task."""
        if not self.export_enabled or not self._export_task:
            return
            
        self._shutting_down = True
        
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            
        logging.info("Metrics export task stopped", component="metrics")
    
    async def _export_loop(self):
        """Periodically export metrics to a file."""
        try:
            while not self._shutting_down:
                try:
                    # Wait for the export interval
                    await asyncio.sleep(self.export_interval)
                    
                    # Export metrics
                    await self._export_metrics_to_file()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.error("Error exporting metrics",
                                 component="metrics",
                                 exc_info=e)
        except asyncio.CancelledError:
            logging.debug("Metrics export task cancelled", component="metrics")
    
    async def _export_metrics_to_file(self):
        """Export metrics to a file."""
        if not self.export_enabled:
            return
            
        try:
            metrics_json = self.export_metrics()
            file_path = self.export_path / f"metrics_{self.node_id}_{int(time.time())}.json"
            
            # Write to a temporary file and then rename to avoid partial writes
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                f.write(metrics_json)
                
            # Rename to final path
            temp_path.rename(file_path)
            
            # Clean up old files (keep only the latest 10)
            self._cleanup_old_metrics_files()
            
            logging.debug("Exported metrics to file", 
                         component="metrics", 
                         file_path=str(file_path),
                         metrics_count=len(self.metrics))
        except Exception as e:
            logging.error("Failed to export metrics",
                         component="metrics",
                         exc_info=e)
    
    def _cleanup_old_metrics_files(self, max_files: int = 10):
        """Clean up old metrics files, keeping only the latest ones."""
        try:
            if not self.export_path.exists():
                return
                
            # Get metrics files for this node
            files = list(self.export_path.glob(f"metrics_{self.node_id}_*.json"))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Delete older files
            for file in files[max_files:]:
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete old metrics file: {file}",
                                  component="metrics",
                                  exc_info=e)
        except Exception as e:
            logging.error("Error cleaning up old metrics files",
                         component="metrics",
                         exc_info=e)


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry(node_id: str = None) -> MetricsRegistry:
    """
    Get the global metrics registry.
    
    Args:
        node_id: Node ID (required when creating the registry)
        
    Returns:
        MetricsRegistry instance
    """
    global _metrics_registry
    
    if _metrics_registry is None:
        if node_id is None:
            raise ValueError("node_id is required when creating the metrics registry")
            
        _metrics_registry = MetricsRegistry(node_id)
        
    return _metrics_registry


async def initialize_metrics(node_id: str) -> MetricsRegistry:
    """
    Initialize the metrics registry.
    
    Args:
        node_id: Node ID
        
    Returns:
        MetricsRegistry instance
    """
    registry = get_metrics_registry(node_id)
    await registry.start_export_task()
    return registry


async def shutdown_metrics() -> None:
    """Shutdown the metrics system."""
    global _metrics_registry
    
    if _metrics_registry:
        await _metrics_registry.stop_export_task()
        _metrics_registry = None
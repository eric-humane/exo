"""
Performance metrics component for displaying LLM performance data.
"""

from typing import Dict, Any, Optional
import time
import gradio as gr


class PerformanceMetrics:
    """
    Component for tracking and displaying LLM performance metrics.
    """
    
    def __init__(self):
        """Initialize performance metrics tracking."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = 0
        self.first_token_time = 0
        self.time_till_first = 0
        self.tokens_generated = 0
        self.tokens_per_second = 0
        self.total_tokens = 0
        self.model_name = ""
    
    def start_generation(self, model_name: str):
        """
        Start tracking a new generation.
        
        Args:
            model_name: Name of the model being used
        """
        self.reset()
        self.start_time = time.time() * 1000  # milliseconds
        self.model_name = model_name
    
    def token_received(self):
        """Register that a token was received."""
        self.tokens_generated += 1
        self.total_tokens += 1
        
        # Calculate first token time
        if self.tokens_generated == 1:
            self.first_token_time = time.time() * 1000
            self.time_till_first = self.first_token_time - self.start_time
        
        # Calculate tokens per second
        if self.first_token_time > 0:
            elapsed_time = (time.time() * 1000 - self.first_token_time) / 1000  # seconds
            if elapsed_time > 0:
                self.tokens_per_second = (self.tokens_generated - 1) / elapsed_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "model_name": self.model_name,
            "time_till_first": self.time_till_first / 1000,  # Convert to seconds
            "tokens_per_second": self.tokens_per_second,
            "total_tokens": self.total_tokens
        }


def create_performance_panel() -> tuple[gr.Row, PerformanceMetrics]:
    """
    Create a performance metrics display panel.
    
    Returns:
        Tuple of (row component, performance metrics instance)
    """
    metrics = PerformanceMetrics()
    
    with gr.Row(equal_height=True) as performance_row:
        model_name = gr.Markdown("", elem_classes=["perf-model-name"])
        first_token = gr.Markdown("", elem_classes=["perf-stat"])
        tokens_per_sec = gr.Markdown("", elem_classes=["perf-stat"])
        total_tokens = gr.Markdown("", elem_classes=["perf-stat"])
    
    def update_metrics_display(metrics_data: Optional[Dict[str, Any]] = None):
        """
        Update the performance metrics display.
        
        Args:
            metrics_data: Optional metrics data, if None, use current metrics
            
        Returns:
            Updated Markdown components
        """
        data = metrics_data or metrics.get_metrics()
        
        model_text = f"**Model:** {data['model_name']}"
        first_text = f"**First Token:** {data['time_till_first']:.2f}s"
        tps_text = f"**Tokens/sec:** {data['tokens_per_second']:.1f}"
        total_text = f"**Total Tokens:** {data['total_tokens']}"
        
        return model_text, first_text, tps_text, total_text
    
    # Initialize display
    model_name.update, first_token.update, tokens_per_sec.update, total_tokens.update = update_metrics_display()
    
    # Add method to the row for updating
    performance_row.update_display = lambda data=None: (
        model_name.update(value=update_metrics_display(data)[0]),
        first_token.update(value=update_metrics_display(data)[1]),
        tokens_per_sec.update(value=update_metrics_display(data)[2]),
        total_tokens.update(value=update_metrics_display(data)[3])
    )
    
    return performance_row, metrics
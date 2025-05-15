"""
Utility functions for the chat API.

This module contains various utility functions used by the chat API.
"""

import base64
import platform
from io import BytesIO
from typing import Any, Dict, Optional
from PIL import Image

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
    import mlx.core as mx
else:
    import numpy as mx


def base64_decode(base64_string):
    """
    Decode a base64 image and reshape it.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Processed image array
    """
    # Decode and reshape image
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data))
    
    # Ensure dimensions are divisible by 64
    W, H = (dim - dim % 64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
        img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter
    
    # Convert to array and normalize
    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1
    img = img[None]
    
    return img


def get_progress_bar(current_step, total_steps, bar_length=50):
    """
    Generate an ASCII progress bar.
    
    Args:
        current_step: Current step
        total_steps: Total number of steps
        bar_length: Length of the progress bar in characters
        
    Returns:
        Progress bar string
    """
    # Calculate the percentage of completion
    percent = float(current_step) / total_steps
    
    # Calculate the number of hashes to display
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    # Create the progress bar string
    progress_bar = f'Progress: [{arrow}{spaces}] {int(percent * 100)}% ({current_step}/{total_steps})'
    
    return progress_bar


def extract_provider_info(model_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract provider information from a model ID.
    
    Args:
        model_id: The model ID
        metadata: Optional metadata
        
    Returns:
        Dictionary with provider information
    """
    provider = "unknown"
    is_cloud_model = False
    
    # Check metadata first if available
    if metadata:
        if "provider" in metadata:
            provider = metadata["provider"]
        
        # Use explicit cloud model flag if present
        if "isCloudModel" in metadata:
            is_cloud_model = metadata["isCloudModel"]
        elif "local" in metadata:
            is_cloud_model = not metadata["local"]
    
    # If no provider in metadata, infer from model ID
    if provider == "unknown":
        if model_id.startswith("gpt-"):
            provider = "openai"
            is_cloud_model = True
        elif model_id.startswith("claude-"):
            provider = "anthropic"
            is_cloud_model = True
        elif model_id.startswith("ollama-"):
            provider = "ollama"
            is_cloud_model = False
        elif model_id.startswith("llamacpp-"):
            provider = "llamacpp"
            is_cloud_model = False
    
    return {
        "provider": provider,
        "isCloudModel": is_cloud_model,
        "canDelete": not is_cloud_model
    }
import numpy as np
import os
import uuid
import logging
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional, ClassVar, Dict, Any
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.utils.async_resources import AsyncResource, ResourceState


class InferenceEngine(AsyncResource, ABC):
    """
    Base class for inference engines with proper AsyncResource lifecycle management.
    
    This class provides:
    1. Standardized lifecycle management (initialize, cleanup)
    2. State tracking and validation 
    3. Error handling and health checks
    4. Ensuring resources are ready before use
    """
    RESOURCE_TYPE: ClassVar[str] = "inference_engine"
    
    def __init__(self, engine_id: Optional[str] = None):
        """
        Initialize the inference engine with AsyncResource base.
        
        Args:
            engine_id: Unique identifier for this engine instance.
                      If not provided, a UUID will be generated.
        """
        super().__init__(
            resource_id=engine_id or f"engine-{str(uuid.uuid4())[:8]}"
        )
        self.session: Dict[str, Any] = {}
    
    # No legacy methods needed for 2.0
    
    # AsyncResource implementation
    
    async def _do_initialize(self) -> None:
        """
        Initialize the inference engine.
        
        Default implementation is a no-op for backward compatibility.
        Subclasses should override this with their specific initialization.
        """
        pass
    
    async def _do_cleanup(self) -> None:
        """
        Clean up resources used by the inference engine.
        
        Default implementation clears the session.
        Subclasses should override this with their specific cleanup logic.
        """
        self.session.clear()
    
    async def _do_health_check(self) -> bool:
        """
        Check if the inference engine is healthy.
        
        Default implementation returns True.
        Subclasses should override this with their specific health check logic.
        """
        return True
    
    # Session management
    
    async def save_session(self, key, value):
        """Save a value to the session dictionary."""
        self.session[key] = value

    async def clear_session(self):
        """
        Clears the current session dictionary.
        This method should be called to free up memory when a session is no longer needed.
        """
        self.session.clear()
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """
        Encode a prompt string to tokens.
        
        Args:
            shard: Model shard to use
            prompt: Text prompt to encode
            
        Returns:
            Token array
        """
        await self.ensure_ready()
        pass

    @abstractmethod
    async def sample(self, x: np.ndarray) -> np.ndarray:
        """
        Sample from output logits.
        
        Args:
            x: Logit output from model
            
        Returns:
            Sampled tokens
        """
        await self.ensure_ready()
        pass

    @abstractmethod
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """
        Decode tokens to a string.
        
        Args:
            shard: Model shard to use
            tokens: Token array to decode
            
        Returns:
            Decoded text
        """
        await self.ensure_ready()
        pass

    @abstractmethod
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """
        Run inference on a tensor input.
        
        Args:
            request_id: Unique identifier for this request
            shard: Model shard to use 
            input_data: Input tensor for inference
            inference_state: Optional state for stateful inference
            
        Returns:
            Tuple of (output data, updated inference state)
        """
        await self.ensure_ready()
        pass

    @abstractmethod
    async def load_checkpoint(self, shard: Shard, path: str):
        """
        Load model weights from a checkpoint.
        
        Args:
            shard: Model shard to load into
            path: Path to the checkpoint file
        """
        await self.ensure_ready()
        pass

    async def save_checkpoint(self, shard: Shard, path: str):
        """
        Save model weights to a checkpoint.
        
        Args:
            shard: Model shard to save
            path: Path to save the checkpoint file
        """
        await self.ensure_ready()
        pass
    
    # Public inference methods with ensure_ready
    
    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """
        Run inference on a text prompt.
        
        This method:
        1. Ensures the engine is ready
        2. Encodes the prompt to tokens
        3. Reshapes as needed for the model
        4. Runs inference on the token tensor
        5. Returns the output data and updated state
        
        Args:
            request_id: Unique identifier for this request
            shard: Model shard to use
            prompt: Text prompt for inference
            inference_state: Optional state for stateful inference
            
        Returns:
            Tuple of (output data, updated inference state)
        """
        # Ensure the engine is ready before use
        await self.ensure_ready()
        
        # Encode the prompt
        tokens = await self.encode(shard, prompt)
        
        # Reshape based on model type
        if shard.model_id != 'stable-diffusion-2-1-base':
            x = tokens.reshape(1, -1)
        else:
            x = tokens
            
        # Run inference on the tensor
        output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

        return output_data, inference_state


inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
}


def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
    """
    Factory function to create and initialize the appropriate inference engine.
    
    Args:
        inference_engine_name: Name of the inference engine to create ("mlx", "tinygrad", or "dummy")
        shard_downloader: Downloader instance for model shards
        
    Returns:
        Initialized InferenceEngine instance
        
    Raises:
        ValueError: If the specified engine is not supported
    """
    if DEBUG >= 2:
        print(f"get_inference_engine called with: {inference_engine_name}")
        
    if inference_engine_name == "mlx":
        from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
        
        # Create MLX engine with a descriptive ID
        return MLXDynamicShardInferenceEngine(
            shard_downloader=shard_downloader, 
            engine_id=f"mlx-engine-{str(uuid.uuid4())[:8]}"
        )
        
    elif inference_engine_name == "tinygrad":
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
        import tinygrad.helpers
        
        # Set tinygrad debug level from environment
        tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

        # Create tinygrad engine
        return TinygradDynamicShardInferenceEngine(
            shard_downloader=shard_downloader,
            engine_id=f"tinygrad-engine-{str(uuid.uuid4())[:8]}"
        )
        
    elif inference_engine_name == "dummy":
        from exo.inference.dummy_inference_engine import DummyInferenceEngine
        
        # Create dummy engine
        return DummyInferenceEngine(
            engine_id=f"dummy-engine-{str(uuid.uuid4())[:8]}"
        )
        
    # Not a supported engine
    raise ValueError(f"Unsupported inference engine: {inference_engine_name}")

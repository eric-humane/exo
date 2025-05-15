from typing import Optional, Tuple, ClassVar
import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import DummyTokenizer
import asyncio

class DummyInferenceEngine(InferenceEngine):
    """
    Dummy inference engine for testing and development.
    
    This engine provides minimal implementations of all required methods
    without actually performing any real inference.
    """
    
    def __init__(self, engine_id: Optional[str] = None):
        """
        Initialize the dummy inference engine.
        
        Args:
            engine_id: Optional unique identifier for this engine
        """
        # Initialize the AsyncResource base
        super().__init__(engine_id=engine_id or "dummy-engine")
        
        # Initialize dummy-specific attributes
        self.shard = None
        self.vocab_size = 1000
        self.hidden_size = 256
        self.eos_token_id = 0
        self.latency_mean = 0.1
        self.latency_stddev = 0.02
        self.num_generate_dummy_tokens = 10
        self.tokenizer = DummyTokenizer()
        
    # AsyncResource implementation
    
    async def _do_initialize(self) -> None:
        """
        Initialize the dummy engine.
        
        No real initialization needed for the dummy engine.
        """
        pass
    
    async def _do_cleanup(self) -> None:
        """
        Clean up the dummy engine.
        
        No real cleanup needed for the dummy engine.
        """
        self.session.clear()
    
    async def _do_health_check(self) -> bool:
        """
        Check if the dummy engine is healthy.
        
        Always returns True for the dummy engine.
        """
        return True
    
    # Implementation-specific methods
    
    async def ensure_shard(self, shard: Shard):
        """Ensure the specified shard is loaded."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Simple shard update
        if self.shard == shard: 
            return
        self.shard = shard
    
    # Abstract method implementations
    
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode a prompt to tokens."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Simple encoding using the tokenizer
        return np.array(self.tokenizer.encode(prompt))
    
    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
        """Sample from logits."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Simple dummy sampling - just return EOS if beyond token limit
        if x[0] > self.num_generate_dummy_tokens: 
            return np.array([self.tokenizer.eos_token_id])
        return x
    
    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """Decode tokens to a string."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Simple decoding using the tokenizer
        return self.tokenizer.decode(tokens)
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """Run inference on a tensor."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Ensure the correct shard is loaded
        await self.ensure_shard(shard)
        
        # Simulate a short processing delay
        await asyncio.sleep(0.01)
        
        # Return a simple transformation of the input tensor
        return input_data + 1 if self.shard.is_last_layer() else input_data, None
    
    async def load_checkpoint(self, shard: Shard, path: str):
        """Load model weights from a checkpoint."""
        # Ensure the engine is ready
        await self.ensure_ready()
        
        # Ensure the correct shard is loaded
        await self.ensure_shard(shard)
        
        # No-op for dummy engine

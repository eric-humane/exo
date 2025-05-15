import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
import mlx.optimizers as optim
import uuid
from ..inference_engine import InferenceEngine
from .sharded_utils import load_model_shard, resolve_tokenizer
from .losses import loss_fns
from ..shard import Shard
from typing import Dict, Optional, Tuple, ClassVar
from exo.download.shard_download import ShardDownloader
from exo.utils.async_resources import ResourceState
import asyncio
from collections import OrderedDict
from mlx_lm.models.cache import make_prompt_cache
from concurrent.futures import ThreadPoolExecutor

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader, engine_id: Optional[str] = None):
    """
    Initialize MLX-based inference engine with proper AsyncResource management.
    
    Args:
        shard_downloader: Downloader for model shards
        engine_id: Optional unique ID for this engine instance
    """
    # Initialize the AsyncResource base class with proper ID
    super().__init__(engine_id=engine_id or f"mlx-{str(uuid.uuid4())[:8]}")
    
    # MLX-specific initialization
    self.shard = None
    self.shard_downloader = shard_downloader
    self.caches = OrderedDict()
    self.sampler_params: tuple[float, float] = (0.0, 0.0, 0.0, 1)
    self.sampler = make_sampler(*self.sampler_params)
    self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    self._tokenizer_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")
    self._shard_lock = asyncio.Lock()
    
    # Model and tokenizer will be initialized later
    self.model = None
    self.tokenizer = None

  async def _eval_mlx(self, *args):
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, mx.eval, *args)

  async def poll_state(self, request_id: str, max_caches=2):
    """
    Get or create model state for a specific request.
    
    This method:
    1. Ensures the engine is ready
    2. Fetches existing cache or creates a new one
    3. Updates LRU order to keep the most recently used caches
    4. Returns the cache in a format ready for the model
    
    Args:
        request_id: Unique identifier for the request
        max_caches: Maximum number of caches to keep in memory
        
    Returns:
        Dictionary with model state
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Move this cache to the end of LRU order if it exists
    if request_id in self.caches:
      self.caches.move_to_end(request_id)
    else:
      # Create a new cache if needed
      newcache = make_prompt_cache(self.model)
      # Evict oldest cache if we're at capacity
      if len(self.caches) > max_caches:
        self.caches.popitem(last=False)
      # Add the new cache
      self.caches[request_id] = newcache
      
    # Return state dictionary with cache
    return {"cache": self.caches[request_id]}

  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    """
    Sample from output logits.
    
    Args:
        x: Logit output from model
        temp: Temperature for sampling
        top_p: Top-p value for nucleus sampling
        
    Returns:
        Sampled tokens
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Update sampler if parameters have changed
    if (temp, top_p, 0.0, 1) != self.sampler_params:
      self.sampler_params = (temp, top_p, 0.0, 1)
      self.sampler = make_sampler(*self.sampler_params)
      
    # Convert to MLX array and prepare for sampling
    logits = mx.array(x)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    
    # Sample from logits
    result = self.sampler(logprobs)
    await self._eval_mlx(result)
    return np.asarray(result, dtype=int)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """
    Encode a prompt string to tokens.
    
    Args:
        shard: Model shard to use
        prompt: Text prompt to encode
        
    Returns:
        Token array
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Encode the prompt
    return np.asarray(
      await asyncio.get_running_loop().run_in_executor(
        self._tokenizer_thread,
        self.tokenizer.encode,
        prompt
      )
    )

  async def decode(self, shard: Shard, tokens) -> str:
    """
    Decode tokens to a string.
    
    Args:
        shard: Model shard to use
        tokens: Token array to decode
        
    Returns:
        Decoded text
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Decode the tokens
    return await asyncio.get_running_loop().run_in_executor(
      self._tokenizer_thread,
      self.tokenizer.decode,
      tokens
    )

  async def save_checkpoint(self, shard: Shard, path: str):
    """
    Save model weights to a checkpoint file.
    
    Args:
        shard: Model shard to save
        path: Path to save the checkpoint file
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Save the weights
    await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread, 
        lambda: self.model.save_weights(path)
    )

  async def load_checkpoint(self, shard: Shard, path: str):
    """
    Load model weights from a checkpoint file.
    
    Args:
        shard: Model shard to load weights into
        path: Path to the checkpoint file
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Load the weights
    await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread, 
        lambda: self.model.load_weights(path)
    )

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
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Get state for this request
    state = await self.poll_state(request_id) if self.model.model_type != 'StableDiffusionPipeline' else {}
    
    # Convert to MLX array
    x = mx.array(input_data)

    # Run inference based on model type
    if self.model.model_type != 'StableDiffusionPipeline':
      # Standard model inference
      output_data = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      inference_state = None
    else:
      # Stable Diffusion pipeline
      result = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      output_data, inference_state = result

    # Ensure MLX evaluation is complete
    await self._eval_mlx(output_data)
    
    # Convert to numpy array
    output_data = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: np.array(output_data, copy=False)
    )
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce"):
    """
    Evaluate the model on inputs and targets using the specified loss function.
    
    Args:
        request_id: Unique identifier for this request
        shard: Model shard to use
        inputs: Input tensor data
        targets: Target tensor data
        lengths: Sequence length tensor
        loss: Name of the loss function to use
        
    Returns:
        Evaluation score
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)
    
    # Set up loss function in session
    await self.save_session('loss', loss_fns[loss])
    
    # Convert to MLX arrays
    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)

    # Calculate loss
    score = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: self.session['loss'](self.model, x, y, l)
    )
    return score

  async def ensure_train(self, shard: Shard, loss: str, opt=optim.SGD, lr=1e-5, trainable_layers=['input_layernorm', 'gate_proj']):
    """
    Ensure the model is set up for training.
    
    Args:
        shard: Model shard to use
        loss: Name of the loss function to use
        opt: Optimizer class
        lr: Learning rate
        trainable_layers: List of layer names to unfreeze for training
        
    Returns:
        True if set up successfully
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the correct shard is loaded
    await self.ensure_shard(shard)

    # Configure trainable layers if needed
    if 'train_layers' not in self.session or self.session['train_layers'] != trainable_layers:
      await self.save_session('train_layers', trainable_layers)
      def freeze_unfreeze():
        self.model.freeze()
        self.model.apply_to_modules(
          lambda k, v: v.unfreeze() if any(k.endswith(layer_name) for layer_name in trainable_layers) else None
        )
      await asyncio.get_running_loop().run_in_executor(self._mlx_thread, freeze_unfreeze)

    # Set up loss and gradient functions if needed
    if 'lossname' not in self.session or 'LVaG' not in self.session or self.session['lossname'] != loss:
      await self.save_session('lossname', loss)
      await self.save_session('LVaG', nn.value_and_grad(self.model, loss_fns[loss]))

    # Create optimizer if needed
    if 'opt' not in self.session:
      await self.save_session('opt', opt(lr))
    return True

  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce", opt=optim.SGD, lr=1e-5):
    """
    Train the model on inputs and targets.
    
    Args:
        request_id: Unique identifier for this request
        shard: Model shard to use
        inputs: Input tensor data
        targets: Target tensor data
        lengths: Sequence length tensor
        loss: Name of the loss function to use
        opt: Optimizer class
        lr: Learning rate
        
    Returns:
        Tuple of (loss score, gradients)
    """
    # Ensure the engine is ready
    await self.ensure_ready()
    
    # Ensure the model is set up for training
    await self.ensure_train(shard, loss, opt, lr)

    # Define the training step function
    def train_step(inp, tar, lng):
      lval, grad = self.session['LVaG'](self.model, inp, tar, lng)
      gradlayers = grad['model']['layers']
      self.session['opt'].update(self.model, grad)
      return lval, gradlayers, (self.model.parameters(), self.session['opt'].state, lval)

    # Convert to MLX arrays
    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)
    
    # Perform training step
    score, gradients, eval_args = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: train_step(x, y, l)
    )
    
    # Ensure MLX evaluation is complete
    await self._eval_mlx(*eval_args)

    # Extract gradients for return
    layers = [{k: v["weight"] for k, v in layer.items() if 'weight' in v} for layer in gradients if layer]
    first_layer = np.array(layers[0]['input_layernorm'], copy=False)
    await self._eval_mlx(first_layer)
    return score, first_layer

  # AsyncResource implementation
  
  async def _do_initialize(self) -> None:
    """
    Initialize the MLX engine.
    
    This is called by AsyncResource's initialize() method.
    No specific initialization needed beyond what's in __init__.
    """
    pass
    
  async def _do_cleanup(self) -> None:
    """
    Clean up resources used by the MLX engine.
    
    This is called by AsyncResource's cleanup() method.
    """
    # Shutdown thread pools
    self._mlx_thread.shutdown(wait=True)
    self._tokenizer_thread.shutdown(wait=True)
    
    # Clear references and caches
    self.model = None
    self.tokenizer = None
    self.caches.clear()
    self.session.clear()
    
  async def _do_health_check(self) -> bool:
    """
    Check if the MLX engine is healthy.
    
    Returns:
        True if the engine is operational, False otherwise.
    """
    # Check thread pools are operational
    if self._mlx_thread._shutdown or self._tokenizer_thread._shutdown:
        return False
    
    # Basic check - more comprehensive checks could be added
    return True
    
  # Implementation-specific methods
    
  async def ensure_shard(self, shard: Shard):
    """
    Ensure the specified model shard is loaded.
    
    This method:
    1. Ensures the engine is initialized
    2. Acquires a lock to prevent concurrent shard loading
    3. Checks if the requested shard is already loaded
    4. Downloads and loads the shard if needed
    5. Updates model, tokenizer, and clears caches
    
    Args:
        shard: The model shard to ensure is loaded
    """
    # First ensure the engine itself is ready
    await self.ensure_ready()
    
    async with self._shard_lock:
      # Skip if shard is already loaded
      if self.shard == shard: 
          return
          
      # Download the shard
      model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
      
      # Load the model shard
      if self.shard != shard:
        model_shard = await asyncio.get_running_loop().run_in_executor(
          self._mlx_thread,
          lambda: load_model_shard(model_path, shard, lazy=False)
        )
        
        # Set up tokenizer
        if hasattr(model_shard, "tokenizer"):
          self.tokenizer = model_shard.tokenizer
        else:
          self.tokenizer = await resolve_tokenizer(model_path)
          
        # Update instance state
        self.shard = shard
        self.model = model_shard
        self.caches = OrderedDict()
        self.session = {}
  
  def __del__(self):
    """
    Destructor to ensure thread pools are properly shut down when the object is garbage collected.
    
    This is a safety mechanism - explicit cleanup using the async cleanup() method is preferred.
    """
    try:
      # Check if thread pools exist and haven't been shut down already
      if hasattr(self, '_mlx_thread') and not self._mlx_thread._shutdown:
        print(f"Warning: {self.__class__.__name__} being garbage collected without proper cleanup")
        # Use non-waiting shutdown in __del__ to avoid blocking
        self._mlx_thread.shutdown(wait=False)

      if hasattr(self, '_tokenizer_thread') and not self._tokenizer_thread._shutdown:
        self._tokenizer_thread.shutdown(wait=False)

    except Exception as e:
      # Never let exceptions escape from __del__
      print(f"Error in {self.__class__.__name__}.__del__: {e}")

    # Clear references to help with garbage collection
    if hasattr(self, 'model'):
      self.model = None
    if hasattr(self, 'tokenizer'):
      self.tokenizer = None
    if hasattr(self, 'caches'):
      self.caches.clear()

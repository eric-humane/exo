import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.download_progress import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.utils.task_manager import TaskManager, TaskPriority

class Node:
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    shard_downloader: ShardDownloader,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.0,
    topology_viz: Optional[TopologyViz] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.shard_downloader = shard_downloader
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}

    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    # Register async callback for handling node status updates
    self._on_opaque_status.register("node_status").on_next_async(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}

    # Add locks for thread safety
    self._topology_lock = asyncio.Lock()  # Lock for updating topology
    self._node_status_lock = asyncio.Lock()  # Lock for updating node status
    
    # Initialize task manager for this node
    from exo.utils.task_manager import TaskManager, TaskPriority
    self._task_manager = TaskManager()

  async def start(self, wait_for_peers: int = 0) -> None:
    self.device_capabilities = await device_capabilities()
    await self.server.start()
    await self.discovery.start()
    await self.update_peers(wait_for_peers)
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    
    # Start the periodic topology collection as a managed task
    asyncio.create_task(self._task_manager.start_task(
      "periodic_topology_collection", 
      lambda: self.periodic_topology_collection(2.0),
      priority=TaskPriority.HIGH
    ))

  async def stop(self) -> None:
    # Cancel all managed tasks
    try:
      if hasattr(self, '_task_manager'):
        if DEBUG >= 1:
          print("Cancelling all managed tasks...")
        num_cancelled = await self._task_manager.cancel_all_tasks(wait=True)
        if DEBUG >= 1:
          print(f"Cancelled {num_cancelled} managed tasks")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error cancelling managed tasks: {e}")
        if DEBUG >= 2:
          traceback.print_exc()

    # Clean up peer handles using AsyncResource cleanup
    try:
      if self.peers:
        if DEBUG >= 1:
          print(f"Cleaning up {len(self.peers)} peer handles...")
        cleanup_results = await asyncio.gather(*[peer.cleanup() for peer in self.peers], return_exceptions=True)
        if DEBUG >= 2:
          for peer, result in zip(self.peers, cleanup_results):
            if isinstance(result, Exception):
              print(f"Error cleaning up peer {peer.id()}: {result}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error cleaning up peers: {e}")
        traceback.print_exc()

    # Clean up resources
    try:
      # Clean up inference engine resources
      if self.inference_engine:
        await self.inference_engine.cleanup()
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error cleaning up inference engine: {e}")
        traceback.print_exc()

    # Clean up buffered data
    self.buffered_token_output.clear()
    self.buffered_logits.clear()
    self.buffered_inputs.clear()
    self.buffered_partials.clear()
    self.node_download_progress.clear()

    # Stop discovery and server
    try:
      await self.discovery.stop()
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error stopping discovery: {e}")
        traceback.print_exc()

    try:
      await self.server.stop()
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error stopping server: {e}")
        traceback.print_exc()

    if DEBUG >= 1:
      print("Node stopped successfully")

  async def on_node_status(self, request_id, opaque_status):
    # Ensure this method is async for proper lock acquisition
    try:
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")

      if status_type == "supported_inference_engines":
        # Use the lock when updating the engines pool
        async with self._node_status_lock:
          node_id = status_data.get("node_id")
          engines = status_data.get("engines", [])
          self.topology_inference_engines_pool.append(engines)

      elif status_type == "node_status":
        # Use the topology lock when updating the topology's active node
        async with self._topology_lock:
          if status_data.get("status", "").startswith("start_"):
            self.current_topology.active_node_id = status_data.get("node_id")
            if DEBUG >= 2:
              print(f"Setting active node: {self.current_topology.active_node_id}")
          elif status_data.get("status", "").startswith("end_"):
            # Only clear if the active node matches the current node
            if status_data.get("node_id") == self.current_topology.active_node_id:
              if DEBUG >= 2:
                print(f"Clearing active node: {self.current_topology.active_node_id}")
              self.current_topology.active_node_id = None

      elif status_type == "download_progress":
        # Use the lock when updating download progress
        async with self._node_status_lock:
          if DEBUG >= 8:
            print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
          download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
          self.node_download_progress[status_data.get('node_id')] = download_progress

      # Update the visualization if needed - use locks to ensure consistency
      if self.topology_viz:
        # Create a snapshot of the current state under the lock to prevent data races
        async with self._topology_lock, self._node_status_lock:
          topology_copy = self.topology
          node_download_progress_copy = self.node_download_progress.copy()
          partitions = self.partitioning_strategy.partition(topology_copy) if self.partitioning_strategy else []

        # Then update visualization with the copied/snapshot data
        self.topology_viz.update_visualization(
          topology_copy,
          partitions,
          self.id,
          node_download_progress_copy
        )

    except json.JSONDecodeError as e:
      if DEBUG >= 1:
        print(f"Error decoding JSON in on_node_status: {e}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error in on_node_status: {e}")
        traceback.print_exc()

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
      supported_engine_names.append('tinygrad')
    else:
      supported_engine_names.append('tinygrad')
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  token_count = 0
  first_token_time = 0
  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ):
    from exo.utils.task_manager import TaskPriority
    
    if shard.model_id != 'stable-diffusion-2-1-base':
      if request_id not in self.buffered_token_output:
        self.buffered_token_output[request_id] = ([], False)
      is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
      if shard.is_last_layer() and not is_finished:
        token = await self.inference_engine.sample(result, temp=self.default_sample_temperature)
        await self.inference_engine.ensure_shard(shard)
        self.buffered_token_output[request_id][0].append(token.item())
        is_finished = token.item() == self.inference_engine.tokenizer.eos_token_id or is_finished or len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
        if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")
        forward = token.reshape(1, -1)
        intermediate_result = [self.buffered_token_output[request_id][0][-1]]
      else:
        forward = result
    else:
      await self.inference_engine.ensure_shard(shard)
      is_finished = inference_state.get("is_finished", False)
      intermediate_result, inference_state = self.handle_stable_diffusion(inference_state, result)
      forward = result
    if shard.is_last_layer():
      self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
      # Use TaskManager for broadcasting results
      await self._task_manager.start_task(
        f"broadcast_result_{request_id}",
        lambda: self.broadcast_result(request_id, intermediate_result, is_finished),
        priority=TaskPriority.NORMAL,
        group="token_broadcast"
      )

    if is_finished:
      if shard.model_id != 'stable-diffusion-2-1-base':
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
      self.outstanding_requests.pop(request_id)
    else:
      self.outstanding_requests[request_id] = "waiting"
      # Use TaskManager for forwarding tensors
      await self._task_manager.start_task(
        f"forward_tensor_{request_id}",
        lambda: self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1), inference_state),
        priority=TaskPriority.HIGH,
        group="inference"
      )

    return  np.array(self.buffered_token_output[request_id][0]) if shard.model_id != 'stable-diffusion-2-1-base' else intermediate_result


  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = {},
  ) -> Optional[np.ndarray]:
    from exo.utils.task_manager import TaskPriority
    
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    
    # Use TaskManager for status broadcasting
    await self._task_manager.start_task(
      f"broadcast_start_prompt_{request_id}",
      lambda: self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      ),
      priority=TaskPriority.NORMAL,
      group="status_broadcast"
    )
    
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    
    # Use TaskManager for status broadcasting
    await self._task_manager.start_task(
      f"broadcast_end_prompt_{request_id}",
      lambda: self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      ),
      priority=TaskPriority.NORMAL,
      group="status_broadcast"
    )
    
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=} {elapsed_time_ns=}")

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None, inference_state: Optional[dict] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")

    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
      self.outstanding_requests[request_id] = "waiting"
      resp = await self.forward_prompt(shard, prompt, request_id, 0, inference_state)
      return None
    else:
      self.outstanding_requests[request_id] = "processing"
      result, inference_state = await self.inference_engine.infer_prompt(request_id, shard, prompt, inference_state)
      ret = await self.process_inference_result(shard, result, request_id, inference_state)
      return result

  async def enqueue_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    request_id: Optional[str] = None,
    train: bool = False,
  ):
    shard = self.get_current_shard(base_shard)
    if shard.is_first_layer():
      loss = await self.process_example(shard, example, target, length, train, request_id)
      return loss
    else:
      if request_id is None:
        request_id = str(uuid.uuid4())
      self.outstanding_requests[request_id] = "waiting"
      loss = await self.forward_example(shard, example, target, length, train, request_id, 0) 
    return loss

  async def coordinate_save(
    self,
    base_shard: Shard,
    iteration: int,
    destination: str,
  ):
    shard = self.get_current_shard(base_shard)
    model = shard.model_id
    sid = shard.__hash__()
    path = f"{destination}/{model}/{sid}-{iteration}.safetensors"
    self.outstanding_requests[f"{sid}::{iteration}"] = "Checking"
    if model not in self.checkpoints:
      self.checkpoints[model] = {}
    if sid not in self.checkpoints[model]:
      self.checkpoints[model][sid] = []
    if len(self.checkpoints[model][sid]) < 1 or self.checkpoints[model][sid][-1] < iteration:
      print(f"Saving checkpoint to {path}")
      self.outstanding_requests[f"{sid}::{iteration}"] = "Saving"
      import os
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      await self.inference_engine.save_checkpoint(shard, path)
      self.checkpoints[model][sid] = sorted(self.checkpoints[model][sid] + [iteration])
    self.outstanding_requests.pop(f"{sid}::{iteration}")

  async def process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ):
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"start_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "example_size": example.size,
          "example_shape": example.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_example(shard, example, target, length, train, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"end_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    return resp

  async def _process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 1: print(f"[{request_id}] process_example: {example.shape=}")
    try:
      target = target.astype(int)
      if train:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "training"
          loss, grad = await self.inference_engine.train(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss, backgrad = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
          self.outstanding_requests[request_id] = "training"
          partial_loss, grad = await self.inference_engine.train(request_id, shard, example, backgrad, length, loss="back_gradient")
        self.outstanding_requests.pop(request_id)
        if shard.is_first_layer():
          return loss
        else:
          return loss, grad
      else:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "evaluating"
          loss = await self.inference_engine.evaluate(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
        self.outstanding_requests.pop(request_id)
        return loss
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing example for shard {shard}: {e}")
      traceback.print_exc()
      return None
        
  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    if DEBUG >= 2: print(f"[{request_id}] process_tensor: {base_shard=} {shard=} {tensor.size=} {tensor.shape=} {elapsed_time_ns=}")
    return resp  # Return the response from _process_tensor

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    """
    Process a tensor through the inference engine.

    Args:
        base_shard: The base shard information
        tensor: Input tensor to process
        request_id: Optional request identifier
        inference_state: Optional state for the inference

    Returns:
        Optional[np.ndarray]: The processed tensor result, or None if an error occurred
    """
    if request_id is None:
      request_id = str(uuid.uuid4())

    # Get the current shard configuration
    shard = self.get_current_shard(base_shard)
    ret = None  # Initialize return value to ensure consistent return

    try:
      # Mark request as being processed
      self.outstanding_requests[request_id] = "processing"

      # Run the tensor through the inference engine
      result, inference_state = await self.inference_engine.infer_tensor(
        request_id, shard, tensor, inference_state
      )

      # Process the result
      ret = await self.process_inference_result(shard, result, request_id, inference_state)

      # Return the processed result
      return ret
    except Exception as e:
      # Clean up the outstanding request on error
      if request_id in self.outstanding_requests:
        self.outstanding_requests.pop(request_id)

      # Log the error
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()

      # Return None explicitly to indicate failure
      return None
  
  async def forward_example(
    self,
    base_shard: Shard,
    step: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")
    # Ensure the peer is ready before sending the example
    await target_peer.ensure_ready()
    resp = await target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train)
    return resp

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      # Ensure the peer is ready before sending the prompt
      await target_peer.ensure_ready()
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id, inference_state=inference_state)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    # Validate the tensor size and shape
    MAX_TENSOR_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB maximum tensor size (configurable)

    # Check if tensor is None
    if tensor is None:
      raise ValueError("Cannot forward None tensor")

    # Check tensor data type
    if not isinstance(tensor, np.ndarray):
      raise TypeError(f"Expected tensor to be a numpy ndarray, got {type(tensor)}")

    # Calculate tensor size in bytes
    tensor_size_bytes = tensor.size * tensor.itemsize

    # Log tensor info at debug level
    if DEBUG >= 2:
      print(f"Tensor validation: {tensor.shape=}, {tensor.dtype=}, {tensor_size_bytes=} bytes")

    # Check for unreasonably large tensors
    if tensor_size_bytes > MAX_TENSOR_SIZE_BYTES:
      error_msg = f"Tensor too large to forward: {tensor_size_bytes} bytes exceeds limit of {MAX_TENSOR_SIZE_BYTES} bytes"
      print(f"ERROR: {error_msg}")
      raise ValueError(error_msg)

    # Check for NaN or Inf values that could cause problems
    if np.issubdtype(tensor.dtype, np.floating) and (np.isnan(tensor).any() or np.isinf(tensor).any()):
      if DEBUG >= 1:
        nan_count = np.isnan(tensor).sum()
        inf_count = np.isinf(tensor).sum()
        print(f"WARNING: Tensor contains {nan_count} NaN values and {inf_count} Inf values")

    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")

    if target_id == self.id:
      await self.process_tensor(next_shard, tensor, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")

      # Check for device capability match before sending large tensors
      peer_capabilities = target_peer.device_capabilities()
      if tensor_size_bytes > 100 * 1024 * 1024:  # For tensors > 100MB
        if peer_capabilities and hasattr(peer_capabilities, 'available_memory'):
          # Make sure peer has at least 2x the tensor size in available memory
          if peer_capabilities.available_memory < tensor_size_bytes * 2:
            warning_msg = (f"WARNING: Peer {target_id} may not have enough memory for tensor: "
                          f"{tensor_size_bytes} bytes vs {peer_capabilities.available_memory} available bytes")
            print(warning_msg)

      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: size={tensor_size_bytes} bytes")
      # Ensure the peer is ready before sending the tensor
      await target_peer.ensure_ready()
      await target_peer.send_tensor(next_shard, tensor, request_id=request_id, inference_state=inference_state)

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    
    # Use ResourceState check instead of direct is_connected check
    peers_to_cleanup = peers_removed
    # All new or updated peers need to be initialized
    peers_to_initialize = peers_added + peers_updated

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_cleanup={peers_to_cleanup} to_initialize={peers_to_initialize}")

    async def cleanup_with_timeout(peer, timeout=5):
      try:
        # Use AsyncResource.cleanup instead of disconnect
        await asyncio.wait_for(peer.cleanup(), timeout)
        return True
      except Exception as e:
        print(f"Error cleaning up peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def initialize_with_timeout(peer, timeout=5):
      try:
        # Use AsyncResource.initialize instead of connect
        await asyncio.wait_for(peer.initialize(), timeout)
        return True
      except Exception as e:
        print(f"Error initializing peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    cleanup_results = await asyncio.gather(*(cleanup_with_timeout(peer) for peer in peers_to_cleanup), return_exceptions=True)
    initialize_results = await asyncio.gather(*(initialize_with_timeout(peer) for peer in peers_to_initialize), return_exceptions=True)

    successful_cleanups = [peer for peer, result in zip(peers_to_cleanup, cleanup_results) if result is True]
    failed_cleanups = [peer for peer, result in zip(peers_to_cleanup, cleanup_results) if result is False]
    successful_initializes = [peer for peer, result in zip(peers_to_initialize, initialize_results) if result is True]
    failed_initializes = [peer for peer, result in zip(peers_to_initialize, initialize_results) if result is False]
    if DEBUG >= 1:
      if successful_cleanups: print(f"Successfully cleaned up peers: {_pretty(successful_cleanups)}")
      if failed_cleanups: print(f"Failed to clean up peers: {_pretty(failed_cleanups)}")
      if successful_initializes: print(f"Successfully initialized peers: {_pretty(successful_initializes)}")
      if failed_initializes: print(f"Failed to initialize peers: {_pretty(failed_initializes)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    if self.inference_engine.__class__.__name__ == 'DummyInferenceEngine': return
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 2: print(f"{did_peers_change=}")
        await self.collect_topology(set())
        if did_peers_change:
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    # Create a new topology to accumulate results
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    # Make a copy to avoid modifying the original set during iteration
    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    # Collect topology data from peers
    peer_topologies = []
    for peer in self.peers:
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id(), peer.description())

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(
          peer.collect_topology(visited, max_depth=max_depth - 1),
          timeout=5.0
        )
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
        peer_topologies.append((peer.id(), other_topology))
      except Exception as e:
        print(f"Error collecting topology from {peer.id()}: {e}")
        traceback.print_exc()

    # Now update our topology safely with the lock
    async with self._topology_lock:
      # Copy the active node ID from current topology
      next_topology.active_node_id = self.topology.active_node_id

      # Merge all peer topologies
      for peer_id, other_topology in peer_topologies:
        next_topology.merge(peer_id, other_topology)

      # Update the current topology
      self.topology = next_topology

    # Create a snapshot for visualization
    if self.topology_viz:
      async with self._topology_lock:
        topology_copy = self.topology
        partitions = self.partitioning_strategy.partition(topology_copy) if self.partitioning_strategy else []

      # Update visualization with the copied data
      self.topology_viz.update_visualization(topology_copy, partitions, self.id)

    return next_topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} {tokens=} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)
  
  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Broadcasting result: {request_id=} {result=} {is_finished=}")
    
    # Track successful broadcasts
    success_count = 0
    failed_peers = []
    max_retries = 2
    
    for peer in self.peers:
      retry_count = 0
      success = False
      
      while retry_count <= max_retries and not success:
        try:
          # Exponential backoff with jitter for retries
          if retry_count > 0:
            jitter = 0.2 * (0.5 + 0.5 * np.random.random())
            backoff_time = 1.0 * (2 ** (retry_count - 1)) + jitter
            await asyncio.sleep(backoff_time)
            if DEBUG >= 2: 
              print(f"Retry {retry_count}/{max_retries} sending result to {peer.id()} after {backoff_time:.2f}s")
          
          # Use ensure_ready instead of manual connection check
          await peer.ensure_ready()
              
          # Send the result with timeout
          await asyncio.wait_for(peer.send_result(request_id, result, is_finished), 
                                timeout=15.0 if retry_count == 0 else 20.0)
          success = True
          success_count += 1
          
        except asyncio.TimeoutError:
          retry_count += 1
          if retry_count > max_retries:
            print(f"Timeout broadcasting result to {peer.id()} after {max_retries} retries")
            failed_peers.append(peer.id())
            
        except Exception as e:
          retry_count += 1
          if retry_count > max_retries:
            print(f"Error broadcasting result to {peer.id()}: {e}")
            if DEBUG >= 2:
              traceback.print_exc()
            failed_peers.append(peer.id())
    
    # Log summary if we had failures
    if failed_peers and DEBUG >= 1:
      peer_count = len(self.peers)
      print(f"Result broadcast summary: {success_count}/{peer_count} successful. Failed peers: {failed_peers}")

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")
    
    # Track successful broadcasts
    success_count = 0
    failed_peers = []
    max_retries = 2
    
    # For certain critical status types, use more retries
    status_obj = None
    try:
      status_obj = json.loads(status)
      status_type = status_obj.get("type", "")
      if status_type in ["node_status", "download_progress"]:
        max_retries = 3  # More retries for critical status updates
    except (json.JSONDecodeError, TypeError, KeyError):
      pass
    
    for peer in self.peers:
      retry_count = 0
      success = False
      
      while retry_count <= max_retries and not success:
        try:
          # Exponential backoff with jitter for retries
          if retry_count > 0:
            jitter = 0.2 * (0.5 + 0.5 * np.random.random())
            backoff_time = 0.5 * (2 ** (retry_count - 1)) + jitter
            await asyncio.sleep(backoff_time)
            if DEBUG >= 3:  # Higher debug level for status retries
              print(f"Retry {retry_count}/{max_retries} sending status to {peer.id()} after {backoff_time:.2f}s")
          
          # Use ensure_ready instead of manual connection check
          await peer.ensure_ready()
              
          # Send the status with potentially increased timeout for retries
          await asyncio.wait_for(
            peer.send_opaque_status(request_id, status), 
            timeout=10.0 if retry_count == 0 else 15.0
          )
          success = True
          success_count += 1
          
        except asyncio.TimeoutError:
          retry_count += 1
          if retry_count > max_retries:
            if DEBUG >= 2:  # Only log timeouts at debug level 2+
              print(f"Timeout sending status to {peer.id()} after {max_retries} retries")
            failed_peers.append(peer.id())
            
        except Exception as e:
          retry_count += 1
          if retry_count > max_retries:
            if DEBUG >= 2:
              print(f"Error sending status to {peer.id()}: {e}")
              traceback.print_exc()
            failed_peers.append(peer.id())
    
    # Log summary if we had failures and this was a critical status
    if failed_peers and status_obj and status_obj.get("type", "") in ["node_status", "download_progress"]:
      peer_count = len(self.peers)
      print(f"Status broadcast summary: {success_count}/{peer_count} successful. Failed: {failed_peers}")
    
    # In the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    return self.topology

  def handle_stable_diffusion(self, inference_state, result):
    if inference_state['is_step_finished']:
      inference_state['step']+=1
    progress = [inference_state['step'],inference_state['total_steps']]
    intermediate_result = result
    if progress[0] == progress[1]:
      intermediate_result = result
    return intermediate_result, inference_state

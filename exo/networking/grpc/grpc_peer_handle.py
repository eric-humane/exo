import grpc
import numpy as np
import asyncio
from typing import Optional, Tuple, List

from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.helpers import DEBUG
from exo.utils.grpc_resources import GRPCChannelResource
from exo.utils.async_resources import ResourceState
import json
import platform

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class GRPCPeerHandle(PeerHandle):
  def __init__(self, _id: str, address: str, desc: str, device_capabilities: DeviceCapabilities):
    # Initialize the AsyncResource base class
    super().__init__(
      resource_id=f"peer-{_id}",
      resource_type=self.RESOURCE_TYPE,
      display_name=f"Peer {_id} ({address})"
    )
    
    # Initialize peer-specific attributes
    self._id = _id
    self.address = address
    self.desc = desc
    self._device_capabilities = device_capabilities
    
    # Create the channel resource
    self._channel_resource = GRPCChannelResource(
      address=address,
      resource_id=f"peer-{_id}",
      options=[
        ("grpc.max_metadata_size", 32 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_concurrent_streams", 100),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.keepalive_time_ms", 10000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ("grpc.tcp_nodelay", 1),
        ("grpc.optimization_target", "throughput"),
      ],
      compression=grpc.Compression.Gzip,
      max_retries=5,
      retry_delay=1.0,
      max_retry_delay=30.0,
      health_check_interval=60.0,
    )
    self.stub = None

  def id(self) -> str:
    return self._id

  def addr(self) -> str:
    return self.address

  def description(self) -> str:
    return self.desc

  def device_capabilities(self) -> DeviceCapabilities:
    return self._device_capabilities

  async def connect(self):
    """
    Initialize the channel resource and create the service stub.
    
    This method is now a wrapper around the channel resource's initialize
    method, which handles all the connection complexity including retries
    and proper error handling.
    """
    await self._channel_resource.initialize()
    if self._channel_resource.channel:
      self.stub = node_service_pb2_grpc.NodeServiceStub(self._channel_resource.channel)
    else:
      raise ConnectionError(f"Failed to create channel for {self._id}@{self.address}")

  async def is_connected(self) -> bool:
    """
    Check if the channel is connected and ready.
    
    Uses the channel resource's health check to determine connection status.
    """
    if not self._channel_resource.is_initialized:
      return False
      
    channel = self._channel_resource.channel
    if not channel:
      return False
      
    return channel.get_state() == grpc.ChannelConnectivity.READY

  async def disconnect(self):
    """
    Clean up the channel resource.
    
    This method is now a wrapper around the channel resource's cleanup
    method, which handles all the cleanup complexity.
    """
    self.stub = None
    await self._channel_resource.cleanup()

  async def _ensure_connected(self):
    """
    Ensures that the gRPC channel is connected.
    
    This is a legacy method that now delegates to ensure_ready().
    Use ensure_ready() instead of calling this method directly.
    """
    # This now just delegates to the standard AsyncResource pattern
    await self.ensure_ready()

  async def health_check(self) -> bool:
    """
    Perform a health check on the remote peer.
    
    Uses the channel resource's health check and also tests the stub
    with a direct health check request.
    
    Returns:
        bool: True if the peer is healthy and responsive, False otherwise.
    """
    # First check the channel health using the resource
    if not await self._channel_resource.check_health():
      return False
      
    # Then check the application-level health through the stub
    try:
      if not self.stub:
        await self.connect()
        
      request = node_service_pb2.HealthCheckRequest()
      response = await self._channel_resource.call_unary(
        lambda: self.stub.HealthCheck(request),
        timeout=5
      )
      return response.is_healthy
    except Exception as e:
      if DEBUG >= 3:
        print(f"Health check failed for {self._id}@{self.address}: {e}")
      return False

  async def send_prompt(self, shard: Shard, prompt: str, inference_state: Optional[dict] = None, request_id: Optional[str] = None) -> Optional[np.array]:
    """Send a prompt to the peer for processing."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.PromptRequest(
      prompt=prompt,
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      request_id=request_id,
      inference_state=None if inference_state is None else self.serialize_inference_state(inference_state)
    )

    # Use the channel resource to handle the connection and retries
    await self._channel_resource.call_unary(
      lambda: self.stub.SendPrompt(request),
      timeout=30
    )

  async def send_tensor(self, shard: Shard, tensor: np.ndarray, inference_state: Optional[dict] = None, request_id: Optional[str] = None) -> Optional[np.array]:
    """Send a tensor to the peer for processing."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
      request_id=request_id,
      inference_state=None if inference_state is None else self.serialize_inference_state(inference_state)
    )

    # Use the channel resource to handle the connection and retries
    response = await self._channel_resource.call_unary(
      lambda: self.stub.SendTensor(request),
      timeout=30
    )

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def send_example(self, shard: Shard, example: np.ndarray, target: np.ndarray, length: np.ndarray, train: bool, request_id: Optional[str] = None) -> Optional[np.array]:
    """Send an example to the peer for training."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.ExampleRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      example=node_service_pb2.Tensor(tensor_data=example.tobytes(), shape=example.shape, dtype=str(example.dtype)),
      target=node_service_pb2.Tensor(tensor_data=target.tobytes(), shape=target.shape, dtype=str(target.dtype)),
      length=node_service_pb2.Tensor(tensor_data=length.tobytes(), shape=length.shape, dtype=str(length.dtype)),
      train=train,
      request_id=request_id,
    )
    
    # Use the channel resource to handle the connection and retries
    response = await self._channel_resource.call_unary(
      lambda: self.stub.SendExample(request),
      timeout=30
    )
    
    loss = response.loss
    if train and not shard.is_first_layer():
      grads = np.frombuffer(response.grads.tensor_data, dtype=np.dtype(response.grads.dtype)).reshape(response.grads.shape)
      return loss, grads
    else:
      return loss

  async def send_loss(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.array]:
    """Send loss gradients to the peer for backpropagation."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
      request_id=request_id,
    )
    
    # Use the channel resource to handle the connection and retries
    response = await self._channel_resource.call_unary(
      lambda: self.stub.SendLoss(request),
      timeout=30
    )

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    """Collect topology information from the peer."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.CollectTopologyRequest(visited=visited, max_depth=max_depth)
    
    # Use the channel resource to handle the connection and retries
    response = await self._channel_resource.call_unary(
      lambda: self.stub.CollectTopology(request),
      timeout=30
    )
    
    topology = Topology()
    for node_id, capabilities in response.nodes.items():
      device_capabilities = DeviceCapabilities(
        model=capabilities.model, chip=capabilities.chip, memory=capabilities.memory, flops=DeviceFlops(fp16=capabilities.flops.fp16, fp32=capabilities.flops.fp32, int8=capabilities.flops.int8)
      )
      topology.update_node(node_id, device_capabilities)
    for node_id, peer_connections in response.peer_graph.items():
      for conn in peer_connections.connections:
        topology.add_edge(node_id, conn.to_id, conn.description)
    return topology

  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    """Send a result back to the peer."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    tensor = None
    if isinstance(result, np.ndarray):
      tensor = node_service_pb2.Tensor(tensor_data=result.tobytes(), shape=result.shape, dtype=str(result.dtype))
      result = []
      
    request = node_service_pb2.SendResultRequest(request_id=request_id, result=result, tensor=tensor, is_finished=is_finished)
    
    # Use the channel resource to handle the connection and retries
    await self._channel_resource.call_unary(
      lambda: self.stub.SendResult(request),
      timeout=10
    )

  async def send_opaque_status(self, request_id: str, status: str) -> None:
    """Send opaque status information to the peer."""
    # Ensure we're initialized
    await self.ensure_ready()
    
    request = node_service_pb2.SendOpaqueStatusRequest(request_id=request_id, status=status)
    
    # Use the channel resource to handle the connection and retries
    await self._channel_resource.call_unary(
      lambda: self.stub.SendOpaqueStatus(request),
      timeout=10
    )

  def serialize_inference_state(self, inference_state: dict) -> node_service_pb2.InferenceState:
    """Serialize inference state to protobuf format."""
    proto_inference_state = node_service_pb2.InferenceState()
    other_data = {}
    for k, v in inference_state.items():
      if isinstance(v, mx.array):
        np_array = np.array(v)
        tensor_data = node_service_pb2.Tensor(tensor_data=np_array.tobytes(), shape=list(np_array.shape), dtype=str(np_array.dtype))
        proto_inference_state.tensor_data[k].CopyFrom(tensor_data)
      elif isinstance(v, list) and all(isinstance(item, mx.array) for item in v):
        tensor_list = node_service_pb2.TensorList()
        for tensor in v:
          np_array = np.array(tensor)
          tensor_data = node_service_pb2.Tensor(tensor_data=np_array.tobytes(), shape=list(np_array.shape), dtype=str(np_array.dtype))
          tensor_list.tensors.append(tensor_data)
        proto_inference_state.tensor_list_data[k].CopyFrom(tensor_list)
      else:
        # For non-tensor data, we'll still use JSON
        other_data[k] = v
    if other_data:
      proto_inference_state.other_data_json = json.dumps(other_data)
    return proto_inference_state
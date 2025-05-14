"""
Adapter for integrating LiteLLM with exo's Node system.

This module bridges the gap between exo's internal Node-based completion
mechanism and the LiteLLM service, allowing for seamless integration of
external LLM providers with exo's distributed architecture.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Union, Any, ClassVar, TypedDict, Callable
from exo.utils.async_resources import AsyncResource
from exo.orchestration.node import Node
from exo.inference.shard import Shard
from exo.models import MODEL_CARDS
from exo import DEBUG
from .litellm_service import LiteLLMService


class ModelMapping(TypedDict, total=False):
    """Type definition for model mappings."""
    local_model_id: str  # exo model ID
    external_model_id: str  # LiteLLM/provider model ID
    max_tokens: int  # Maximum tokens the model can generate
    context_window: int  # Total context window size
    supports_tools: bool  # Whether the model supports tools/functions
    priority: int  # Priority (lower number = higher priority)


class LiteLLMAdapter(AsyncResource):
    """
    Adapter for using LiteLLM with exo's Node system.
    
    This class acts as a bridge between exo's Node architecture and LiteLLM,
    routing certain prompts to external LLM providers while keeping others
    on local exo nodes.
    """
    
    RESOURCE_TYPE: ClassVar[str] = "litellm_adapter"
    
    def __init__(
        self,
        node: Node,
        litellm_service: Optional[LiteLLMService] = None,
        config_path: Optional[str] = None,
        resource_id: Optional[str] = None,
        model_mappings: Optional[List[ModelMapping]] = None,
        local_model_priority: int = 10,
        external_model_priority: int = 20
    ):
        """
        Initialize the LiteLLM adapter.
        
        Args:
            node: The exo Node to connect with
            litellm_service: Existing LiteLLMService instance (optional)
            config_path: Path to a LiteLLM config file (if no service provided)
            resource_id: Unique identifier for this adapter
            model_mappings: Custom mappings between exo models and external provider models
            local_model_priority: Default priority for local models (lower = higher priority)
            external_model_priority: Default priority for external models
        """
        super().__init__(
            resource_id=resource_id or f"litellm-adapter-{str(uuid.uuid4())[:8]}",
            resource_type=self.RESOURCE_TYPE,
            display_name="LiteLLM Adapter"
        )
        
        self.node = node
        self.litellm_service = litellm_service
        self.config_path = config_path
        self.created_service = False
        self.token_queues: Dict[str, asyncio.Queue] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self.token_callback_id = None
        
        # Initialize model mappings
        self.model_mappings = model_mappings or []
        self.local_model_priority = local_model_priority
        self.external_model_priority = external_model_priority
        
    async def _do_initialize(self) -> None:
        """
        Initialize the LiteLLM adapter.
        
        Creates the LiteLLM service if not provided and registers
        callbacks with the node.
        """
        # Create LiteLLMService if not provided
        if not self.litellm_service:
            self.litellm_service = LiteLLMService(config_path=self.config_path)
            await self.litellm_service.initialize()
            self.created_service = True
            
        # Register callback for token handling
        self.token_callback_id = f"litellm-adapter-{self.id}"
        self.node.on_token.register(self.token_callback_id).on_next_async(self._handle_tokens)
        
        # Extend model mappings with defaults based on available models
        await self._setup_model_mappings()
        
    async def _do_cleanup(self) -> None:
        """
        Clean up resources used by the adapter.
        """
        # Deregister callback
        if self.token_callback_id:
            self.node.on_token.deregister(self.token_callback_id)
            self.token_callback_id = None
            
        # Clean up LiteLLM service if we created it
        if self.created_service and self.litellm_service:
            await self.litellm_service.cleanup()
            self.litellm_service = None
            self.created_service = False
            
        # Clear queues and cancel tasks
        for request_id, task in self.stream_tasks.items():
            task.cancel()
        self.stream_tasks.clear()
        self.token_queues.clear()
        
    async def _check_health(self) -> bool:
        """
        Check if the adapter is healthy.
        
        Returns:
            True if the adapter is operational, False otherwise.
        """
        # Check if our node is still operational
        if not self.node or self.node.outstanding_requests is None:
            return False
            
        # Check if LiteLLM service is healthy (if we're managing it)
        if self.created_service and self.litellm_service:
            return await self.litellm_service.check_health()
            
        return True
        
    async def _setup_model_mappings(self) -> None:
        """
        Set up model mappings between exo models and external provider models.
        """
        # If no litellm service, we can't get mappings
        if not self.litellm_service:
            return
            
        # Get supported models from LiteLLM
        litellm_models = self.litellm_service.get_supported_models()
        
        # For each exo model, create a mapping
        for model_id, info in MODEL_CARDS.items():
            # Skip models already in mappings
            if any(m.get("local_model_id") == model_id for m in self.model_mappings):
                continue
                
            repo_info = info.get("repo", {})
            
            # Create a mapping with our local model as default
            mapping = ModelMapping(
                local_model_id=model_id,
                max_tokens=info.get("max_tokens", 4096),
                context_window=info.get("context_length", 8192),
                supports_tools=info.get("supports_tools", False),
                priority=self.local_model_priority
            )
            
            self.model_mappings.append(mapping)
            
        # Add external-only models from LiteLLM
        for model in litellm_models:
            model_id = model["id"]
            
            # Skip models already in mappings
            if any(m.get("external_model_id") == model_id for m in self.model_mappings):
                continue
                
            # Add external-only model
            mapping = ModelMapping(
                local_model_id=f"external-{model_id}",
                external_model_id=model_id,
                max_tokens=model.get("max_tokens", 4096),
                context_window=model.get("context_length", 8192),
                supports_tools=True,  # Assume external models support tools
                priority=self.external_model_priority
            )
            
            self.model_mappings.append(mapping)
            
    def get_model_mapping_by_id(self, model_id: str) -> Optional[ModelMapping]:
        """
        Get model mapping by local model ID.
        
        Args:
            model_id: The local model ID to lookup
            
        Returns:
            The model mapping or None if not found
        """
        for mapping in self.model_mappings:
            if mapping.get("local_model_id") == model_id:
                return mapping
        return None
        
    def should_use_external_provider(self, shard: Shard) -> bool:
        """
        Determine if a request should use an external provider.
        
        Args:
            shard: The shard from the request
            
        Returns:
            True if the request should use an external provider, False otherwise
        """
        # Check if we have an external mapping for this model
        mapping = self.get_model_mapping_by_id(shard.model_id)
        if not mapping:
            return False
            
        # If the mapping has an external model ID, use external provider
        return "external_model_id" in mapping
        
    async def ensure_ready(self) -> None:
        """
        Ensure the adapter is initialized and ready to use.
        
        Raises:
            RuntimeError: If the adapter is not in a usable state
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.is_usable:
            raise RuntimeError(f"LiteLLM adapter is not in a usable state: {self.state}")
        
    async def process_prompt(
        self,
        shard: Shard,
        prompt: str,
        request_id: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        inference_state: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Process a prompt using either the local node or an external provider.
        
        Args:
            shard: The shard to use for processing
            prompt: The prompt to process
            request_id: Unique identifier for this request
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            inference_state: Additional state for inference
            tools: Function calling tools
            
        Raises:
            RuntimeError: If the adapter is not initialized
            ValueError: If arguments are invalid
        """
        await self.ensure_ready()
        
        # Determine whether to use external provider
        use_external = self.should_use_external_provider(shard)
        
        if use_external:
            # Process with external provider
            await self._process_with_external(
                shard, prompt, request_id, temperature, max_tokens, stream, tools
            )
        else:
            # Process with local node
            await self.node.process_prompt(
                shard, prompt, request_id, temperature, max_tokens, inference_state
            )
            
    async def _process_with_external(
        self,
        shard: Shard,
        prompt: str,
        request_id: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Process a prompt using an external provider via LiteLLM.
        
        Args:
            shard: The shard to use for processing
            prompt: The prompt to process
            request_id: Unique identifier for this request
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            tools: Function calling tools
            
        Raises:
            RuntimeError: If LiteLLM service is not available
            ValueError: If mappings are invalid
        """
        if not self.litellm_service:
            raise RuntimeError("LiteLLM service is not available")
            
        # Get model mapping
        mapping = self.get_model_mapping_by_id(shard.model_id)
        if not mapping or "external_model_id" not in mapping:
            raise ValueError(f"No external model mapping for {shard.model_id}")
            
        external_model = mapping["external_model_id"]
        
        # Create token queue for this request
        self.token_queues[request_id] = asyncio.Queue()
        
        # Convert prompt to messages format
        # This is a simple conversion - more sophisticated parsing would be needed
        # for a production implementation
        messages = [{"role": "user", "content": prompt}]
        
        # Set up parameters for completion
        params = {
            "model": external_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if tools:
            params["tools"] = tools
            
        try:
            if stream:
                # Create streaming task
                stream_task = asyncio.create_task(
                    self._handle_streaming_response(request_id, params)
                )
                self.stream_tasks[request_id] = stream_task
            else:
                # Get completion directly
                response = await self.litellm_service.get_completions(**params)
                
                # Extract tokens (this would need proper tokenization in production)
                content = response["choices"][0]["message"]["content"]
                
                # Simulate tokens for now - in production you'd need to tokenize properly
                tokens = [ord(c) for c in content]
                
                # Send tokens to callback
                await self._handle_tokens(request_id, tokens, True)
                
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error in LiteLLM processing: {str(e)}")
            
            # Notify of error through token callback
            # In a production implementation, you'd want proper error handling
            await self._handle_tokens(request_id, [], True)
            
            # Clean up
            if request_id in self.token_queues:
                del self.token_queues[request_id]
            if request_id in self.stream_tasks:
                del self.stream_tasks[request_id]
                
            raise
            
    async def _handle_streaming_response(self, request_id: str, params: Dict[str, Any]) -> None:
        """
        Handle streaming response from LiteLLM.
        
        Args:
            request_id: The request ID
            params: Parameters for the completion call
        """
        try:
            all_tokens = []
            async for chunk in await self.litellm_service.get_completions(**params):
                if not chunk["choices"]:
                    continue
                    
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    # In production, you'd need to properly tokenize this content
                    # This is a simplified version for demonstration
                    new_tokens = [ord(c) for c in content]
                    all_tokens.extend(new_tokens)
                    
                    # Send tokens to callback
                    is_finished = False
                    await self._handle_tokens(request_id, new_tokens, is_finished)
                    
                if "finish_reason" in chunk["choices"][0]:
                    # Final chunk
                    await self._handle_tokens(request_id, all_tokens, True)
                    break
                    
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error in streaming response: {str(e)}")
                
            # Notify of error
            await self._handle_tokens(request_id, [], True)
            
        finally:
            # Clean up
            if request_id in self.stream_tasks:
                del self.stream_tasks[request_id]
                
    async def _handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
        """
        Handle tokens for a request, either from local node or external provider.
        
        This method makes the external provider responses appear the same as
        local node responses to the rest of the system.
        
        Args:
            request_id: The request ID
            tokens: List of token IDs
            is_finished: Whether this is the final set of tokens
        """
        try:
            # Put the tokens in the queue for this request
            if request_id in self.token_queues:
                await self.token_queues[request_id].put((tokens, is_finished))
                
            # Notify the node's token callback system to maintain compatibility
            # with the rest of exo
            if hasattr(self.node, "on_token") and self.node.on_token:
                # Keep the same interface as node.on_token
                await self.node.on_token.notify(request_id, tokens, is_finished)
                
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error handling tokens: {str(e)}")
                
    def get_supported_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all supported models (both local and external).
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Add local models from mappings
        for mapping in self.model_mappings:
            local_id = mapping.get("local_model_id")
            external_id = mapping.get("external_model_id")
            
            if local_id:
                model_info = {
                    "id": local_id,
                    "object": "model",
                    "owned_by": "exo"
                }
                
                if external_id:
                    model_info["external_id"] = external_id
                    model_info["owned_by"] = "litellm"
                    
                models.append(model_info)
                
        return models
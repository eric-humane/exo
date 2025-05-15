"""
Adapter for integrating LiteLLM with exo's Node system.

This module bridges the gap between exo's internal Node-based completion
mechanism and the LiteLLM service, allowing for seamless integration of
external LLM providers with exo's distributed architecture.

It includes intelligent routing to the fastest client with valid API keys.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Union, Any, ClassVar, TypedDict, Callable, Tuple
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
    provider: str  # Provider name (openai, anthropic, etc.)
    model_type: str  # Model category/type (gpt-4, llama, etc.)


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
            resource_id=resource_id or f"litellm-adapter-{str(uuid.uuid4())[:8]}"
        )
        # Store these for reference but don't pass to parent
        self.resource_type = self.RESOURCE_TYPE
        self.display_name = "LiteLLM Adapter"
        
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
            print(f"[LiteLLMAdapter] Creating LiteLLMService with config_path: {self.config_path}")
            self.litellm_service = LiteLLMService(config_path=self.config_path)
            try:
                await self.litellm_service.initialize()
                print("[LiteLLMAdapter] LiteLLMService initialized successfully")
                self.created_service = True
            except Exception as e:
                print(f"[LiteLLMAdapter] ERROR initializing LiteLLMService: {e}")
                import traceback
                traceback.print_exc()
                raise
            
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
        
        This creates a unified mapping system where:
        1. All cloud models (OpenAI, Anthropic, etc.) are handled through LiteLLM
        2. Local models can be optionally routed through LiteLLM for consistent handling
        """
        # If no litellm service, we can't get mappings
        if not self.litellm_service:
            return
            
        try:
            # First, get supported models from LiteLLM service which will include
            # both cloud models and any local models that LiteLLM can handle
            litellm_models = self.litellm_service.get_supported_models()
            
            # Track processed models to avoid duplicates
            processed_models = set()
            
            # Process all models from LiteLLM first
            for model in litellm_models:
                model_id = model["id"]
                
                # Skip if already processed
                if model_id in processed_models or any(m.get("local_model_id") == model_id for m in self.model_mappings):
                    continue
                    
                processed_models.add(model_id)
                
                # Get provider and determine if this is a cloud model
                provider = model.get("owned_by", "").lower() or model.get("metadata", {}).get("provider", "litellm")
                is_cloud_model = provider in ("openai", "anthropic", "azure", "cohere", "gemini", "mistral") or model.get("metadata", {}).get("isCloudModel", False)
                
                # Determine model type from the model id
                model_type = "unknown"
                if "gpt-4" in model_id.lower():
                    model_type = "gpt-4"
                elif "gpt-3" in model_id.lower():
                    model_type = "gpt-3"
                elif "claude" in model_id.lower():
                    model_type = "claude"
                elif "llama" in model_id.lower():
                    model_type = "llama"
                elif "mistral" in model_id.lower():
                    model_type = "mistral"
                elif "phi" in model_id.lower():
                    model_type = "phi"
                elif "qwen" in model_id.lower():
                    model_type = "qwen"
                
                # Get priority based on provider type
                priority = self.external_model_priority if is_cloud_model else self.local_model_priority
                
                # Check if this model is ready
                is_ready = model.get("ready", False)
                
                # Only add if ready
                if is_ready:
                    # Create a unified mapping that works for both local and cloud models
                    # For cloud models, both local_model_id and external_model_id will be the same
                    mapping = ModelMapping(
                        local_model_id=model_id,
                        external_model_id=model_id if is_cloud_model else None,  # Only set for cloud models
                        max_tokens=model.get("metadata", {}).get("max_tokens", 4096),
                        context_window=model.get("metadata", {}).get("context_length", 8192),
                        supports_tools=model.get("metadata", {}).get("supports_tools", False),
                        priority=priority,
                        provider=provider,
                        model_type=model_type
                    )
                    
                    self.model_mappings.append(mapping)
                    
                    if DEBUG >= 2:
                        logging.info(f"Added model mapping: {model_id} (provider: {provider}, type: {model_type}, is_cloud: {is_cloud_model})")
            
            # Now add any local exo models that weren't already handled by LiteLLM
            for model_id, info in MODEL_CARDS.items():
                # Skip if already processed
                if model_id in processed_models or any(m.get("local_model_id") == model_id for m in self.model_mappings):
                    continue
                    
                processed_models.add(model_id)
                
                # Determine model type from the model id
                model_type = "unknown"
                if "llama" in model_id.lower():
                    model_type = "llama"
                elif "phi" in model_id.lower():
                    model_type = "phi"
                elif "mistral" in model_id.lower():
                    model_type = "mistral"
                elif "qwen" in model_id.lower():
                    model_type = "qwen"
                
                # Create a mapping for the local model
                # Note: We don't set external_model_id for pure local models
                mapping = ModelMapping(
                    local_model_id=model_id,
                    max_tokens=info.get("max_tokens", 4096),
                    context_window=info.get("context_length", 8192),
                    supports_tools=info.get("supports_tools", False),
                    priority=self.local_model_priority,
                    provider="exo",
                    model_type=model_type
                )
                
                self.model_mappings.append(mapping)
                
                if DEBUG >= 2:
                    logging.info(f"Added local model mapping: {model_id} (type: {model_type})")
                    
        except Exception as e:
            # Log error but continue - we can still use local models
            if DEBUG >= 1:
                logging.error(f"Error setting up model mappings: {str(e)}")
                if DEBUG >= 2:
                    import traceback
                    traceback.print_exc()
            
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
        # Check if we have a mapping for this model
        mapping = self.get_model_mapping_by_id(shard.model_id)
        if not mapping:
            return False
            
        # If the mapping has a valid external model ID, use external provider
        # A valid external_model_id must exist and not be None
        return "external_model_id" in mapping and mapping["external_model_id"] is not None
        
    def get_alternative_models(self, model_id: str, prefer_external: bool = False) -> List[str]:
        """
        Find alternative models with similar capabilities that could be used for routing.
        
        Args:
            model_id: The model ID to find alternatives for
            prefer_external: Whether to prefer external models over local ones
            
        Returns:
            List of alternative model IDs ordered by priority
        """
        # Get the original model mapping
        original_mapping = self.get_model_mapping_by_id(model_id)
        if not original_mapping:
            return []
            
        # Get the model type
        model_type = original_mapping.get("model_type", "unknown")
        if model_type == "unknown":
            # Try to infer model type from the model ID
            if "gpt-4" in model_id.lower():
                model_type = "gpt-4"
            elif "gpt-3" in model_id.lower():
                model_type = "gpt-3"
            elif "claude" in model_id.lower():
                model_type = "claude"
            elif "llama" in model_id.lower():
                model_type = "llama"
            elif "mistral" in model_id.lower():
                model_type = "mistral"
                
        # Find alternative models of the same type
        alternatives = []
        external_alternatives = []
        local_alternatives = []
        
        for mapping in self.model_mappings:
            alt_id = mapping.get("local_model_id")
            if alt_id == model_id:
                continue  # Skip the original model
                
            # Check if the model type matches
            alt_type = mapping.get("model_type", "unknown")
            if alt_type == model_type or model_type == "unknown":
                if "external_model_id" in mapping:
                    external_alternatives.append((alt_id, mapping.get("priority", 999)))
                else:
                    local_alternatives.append((alt_id, mapping.get("priority", 999)))
        
        # Sort by priority (lower is better)
        external_alternatives.sort(key=lambda x: x[1])
        local_alternatives.sort(key=lambda x: x[1])
        
        # Order based on preference
        if prefer_external:
            alternatives = [alt[0] for alt in external_alternatives] + [alt[0] for alt in local_alternatives]
        else:
            alternatives = [alt[0] for alt in local_alternatives] + [alt[0] for alt in external_alternatives]
            
        return alternatives
        
    async def find_fastest_model(self, model_id: str) -> Tuple[Optional[str], bool]:
        """
        Find the fastest model that can handle a request, considering both local and external options.
        
        Args:
            model_id: The model ID requested
            
        Returns:
            Tuple of (selected_model_id, use_external) where:
            - selected_model_id: The model to use, or None if no suitable model was found
            - use_external: Whether to use an external provider
        """
        # Check if we have a valid mapping for this model
        original_mapping = self.get_model_mapping_by_id(model_id)
        if not original_mapping:
            return None, False
            
        # Fast path: if this is a local-only model, use it directly
        is_external = "external_model_id" in original_mapping and original_mapping["external_model_id"] is not None
        if not is_external:
            return model_id, False
            
        # If we don't have a LiteLLM service with latency tracking, use the original model
        if not self.litellm_service or not hasattr(self.litellm_service, "latency_tracker") or not self.litellm_service.latency_tracker:
            return model_id, is_external
                
        # Get the model type
        model_type = original_mapping.get("model_type", "unknown")
        
        # Find the fastest provider for this model type
        fastest_provider = await self.litellm_service.get_fastest_provider_for_model_type(model_type)
        
        # If we don't have a fastest provider but we have an external mapping, use it
        if not fastest_provider and is_external:
            return model_id, True
            
        # If we have a fastest provider, try to find a model from that provider
        if fastest_provider:
            # Look for models from the fastest provider with the same model type
            for mapping in self.model_mappings:
                if mapping.get("provider") == fastest_provider and mapping.get("model_type") == model_type:
                    model_id = mapping.get("local_model_id")
                    use_external = "external_model_id" in mapping and mapping["external_model_id"] is not None
                    return model_id, use_external
                    
        # If we couldn't find a matching model from the fastest provider,
        # get alternatives and check if any of them are from the fastest provider
        alternatives = self.get_alternative_models(model_id, prefer_external=True if fastest_provider else False)
        
        for alt_id in alternatives:
            alt_mapping = self.get_model_mapping_by_id(alt_id)
            if alt_mapping and alt_mapping.get("provider") == fastest_provider:
                use_external = "external_model_id" in alt_mapping and alt_mapping["external_model_id"] is not None
                return alt_id, use_external
                
        # If no alternatives match the fastest provider, use the original model
        return model_id, is_external
        
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
        
        The adapter will intelligently route to the fastest client with valid API keys
        when multiple options are available.
        
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
        
        # Find the fastest model that can handle this request
        model_id, use_external = await self.find_fastest_model(shard.model_id)
        
        if model_id is None:
            # If no suitable model found, try to use the original shard directly
            model_id = shard.model_id
            use_external = self.should_use_external_provider(shard)
            
            if DEBUG >= 1:
                logging.warning(f"Could not find a suitable model for {shard.model_id}, using original model")
        
        if DEBUG >= 2:
            source = "external provider" if use_external else "local node"
            if model_id != shard.model_id:
                logging.info(f"Routing request for {shard.model_id} to {model_id} via {source}")
            else:
                logging.info(f"Processing {model_id} via {source}")
        
        # If the selected model is different from the requested one, create a new shard
        actual_shard = shard
        if model_id != shard.model_id:
            # Create a new shard with the selected model ID
            actual_shard = Shard(
                shard_id=shard.shard_id,
                model_id=model_id,
                layers=shard.layers,
                dtype=shard.dtype,
                device=shard.device,
                filename=shard.filename,
                vocab_size=shard.vocab_size
            )
        
        if use_external:
            # Process with external provider
            try:
                await self._process_with_external(
                    actual_shard, prompt, request_id, temperature, max_tokens, stream, tools
                )
            except Exception as e:
                # If external processing fails, try to fall back to local processing if possible
                if DEBUG >= 1:
                    logging.error(f"External processing failed: {str(e)}. Attempting fallback to local processing.")
                
                # Check if we have a local alternative
                local_alternatives = self.get_alternative_models(model_id, prefer_external=False)
                if not local_alternatives:
                    # No local alternatives, re-raise the error
                    raise
                
                # Try local alternatives
                for alt_id in local_alternatives:
                    alt_mapping = self.get_model_mapping_by_id(alt_id)
                    # For local models, external_model_id is either not present or is None
                    if alt_mapping and ("external_model_id" not in alt_mapping or alt_mapping["external_model_id"] is None):
                        # Found a local alternative, try to use it
                        if DEBUG >= 1:
                            logging.info(f"Falling back to local model {alt_id}")
                        
                        fallback_shard = Shard(
                            shard_id=shard.shard_id,
                            model_id=alt_id,
                            layers=shard.layers,
                            dtype=shard.dtype,
                            device=shard.device,
                            filename=shard.filename,
                            vocab_size=shard.vocab_size
                        )
                        
                        await self.node.process_prompt(
                            fallback_shard, prompt, request_id, temperature, max_tokens, inference_state
                        )
                        return
                
                # If we get here, no local fallback was possible
                raise
        else:
            # Process with local node
            await self.node.process_prompt(
                actual_shard, prompt, request_id, temperature, max_tokens, inference_state
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
        if not mapping or "external_model_id" not in mapping or mapping["external_model_id"] is None:
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
            "stream": stream,
            "track_latency": True  # Enable latency tracking
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if tools:
            params["tools"] = tools
            
        # Get provider information
        provider = mapping.get("provider", "unknown")
        if DEBUG >= 2:
            logging.info(f"Using external provider {provider} for model {external_model}")
        
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
            
            # Mark API key as potentially invalid if authentication error
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                if DEBUG >= 1:
                    logging.warning(f"Authentication error with provider {provider}, may have invalid API key")
                
                # Record the error in the latency tracker if available
                if self.litellm_service and hasattr(self.litellm_service, "latency_tracker") and self.litellm_service.latency_tracker:
                    if provider:
                        try:
                            api_key = None
                            # Extract API key from params if available
                            if "api_key" in params:
                                api_key = params["api_key"]
                            
                            await self.litellm_service.latency_tracker.record_latency(
                                provider=provider,
                                model=external_model,
                                latency=0.0,  # No latency, just recording failure
                                api_key=api_key,
                                success=False,
                                error=e
                            )
                        except Exception as tracker_error:
                            if DEBUG >= 1:
                                logging.error(f"Error recording API key failure: {str(tracker_error)}")
            
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
            List of model information dictionaries with enhanced metadata
        """
        models = []
        
        # Create a mapping to get display names for providers
        provider_display_names = {
            "openai": "OpenAI",
            "anthropic": "Anthropic", 
            "ollama": "Ollama",
            "llamacpp": "LlamaCPP",
            "cohere": "Cohere",
            "gemini": "Google Gemini",
            "mistral": "Mistral AI",
            "azure": "Azure OpenAI"
        }
        
        # Add local models from mappings
        for mapping in self.model_mappings:
            local_id = mapping.get("local_model_id")
            external_id = mapping.get("external_model_id") 
            provider = mapping.get("provider", "")
            
            if local_id:
                # Initialize metadata
                metadata = {}
                
                # Determine if this is a cloud model or local model
                is_cloud_model = external_id is not None and provider in ("openai", "anthropic", "cohere", "mistral", "gemini", "azure") 
                is_local = not is_cloud_model
                
                # Format display name
                display_name = local_id
                if provider == "openai" and "-" in local_id:
                    # Extract parts and capitalize (e.g., "gpt-4" -> "GPT-4")
                    parts = local_id.split("-")
                    display_name = parts[0].upper()
                    for part in parts[1:]:
                        if part.isdigit():
                            display_name += "-" + part
                        else:
                            display_name += "-" + part.capitalize()
                elif provider == "anthropic" and "-" in local_id:
                    # Format Claude models (e.g., "claude-3-opus" -> "Claude-3 Opus")
                    parts = local_id.split("-")
                    display_name = parts[0].capitalize()
                    for part in parts[1:]:
                        display_name += "-" + part.capitalize()
                
                # Create the model info
                model_info = {
                    "id": local_id,
                    "object": "model",
                    "owned_by": provider if provider else "exo",
                    "ready": True,  # Mark all models as ready for the UI
                    "metadata": {
                        "provider": provider if provider else "exo",
                        "providerDisplayName": provider_display_names.get(provider, provider.capitalize()) if provider else "Exo",
                        "displayName": display_name,
                        "isCloudModel": is_cloud_model,
                        "canDelete": not is_cloud_model,  # Can't delete cloud models
                        "local": is_local,
                        "isExternal": is_cloud_model,  # For compatibility with some checks
                        "context_length": mapping.get("context_window", 8192),
                        "supports_tools": mapping.get("supports_tools", False),
                        "model_type": mapping.get("model_type", "unknown")
                    }
                }
                
                models.append(model_info)
        
        # If we have a LiteLLM service, also fetch models from there to ensure
        # we have the most complete and up-to-date list
        if self.litellm_service:
            try:
                litellm_models = self.litellm_service.get_supported_models()
                
                # Create a set of existing model IDs to avoid duplicates
                existing_ids = {model["id"] for model in models}
                
                # Add models that aren't already in our list
                for model in litellm_models:
                    model_id = model["id"]
                    if model_id not in existing_ids:
                        # Only add ready models
                        if model.get("ready", False):
                            models.append(model)
                            existing_ids.add(model_id)
                
            except Exception as e:
                if DEBUG >= 1:
                    logging.error(f"Error getting models from LiteLLM service: {str(e)}")
        
        return models
"""
LiteLLM integration for exo.

This module provides a service for interacting with LiteLLM, allowing exo to use
various LLM providers through a unified interface, while maintaining proper
resource lifecycle management using the AsyncResource pattern.

It also provides intelligent routing to the fastest client with valid API keys.
"""

import os
import json
import asyncio
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Union, ClassVar, AsyncIterator, Tuple
from pathlib import Path

# Import AsyncResource base class
from exo.utils.async_resources import AsyncResource

# Import LiteLLM
import litellm
from litellm import completion, acompletion
from litellm.router import Router
from litellm.utils import ModelResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.exceptions import AuthenticationError, ServiceUnavailableError, BadRequestError

from exo import DEBUG, get_exo_config_dir
from .provider_latency_tracker import ModelProviderLatencyTracker


class LiteLLMService(AsyncResource):
    """
    LiteLLM service with proper AsyncResource lifecycle management.
    
    This class provides:
    1. Structured lifecycle management for LiteLLM connections
    2. Configuration loading from various sources
    3. Connection to cloud providers and local model servers
    4. Routing and load balancing across multiple providers
    5. Proper error handling and retries
    """
    
    RESOURCE_TYPE: ClassVar[str] = "litellm_service"
    
    def __init__(
        self,
        resource_id: Optional[str] = None,
        config_path: Optional[str] = None,
        load_env_vars: bool = True,
        display_name: str = "LiteLLM Service",
        enable_latency_tracking: bool = True
    ):
        """
        Initialize the LiteLLM service.
        
        Args:
            resource_id: Unique identifier for this service. If not provided, a UUID will be generated.
            config_path: Path to a LiteLLM config file. If not provided, will attempt to load from 
                         default locations.
            load_env_vars: Whether to load provider API keys from environment variables.
            display_name: Display name for this service in logs and UI.
            enable_latency_tracking: Whether to enable latency tracking for providers.
        """
        # Check if litellm is available
        try:
            import litellm
            LITELLM_AVAILABLE = True
        except ImportError:
            LITELLM_AVAILABLE = False
            
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install it with: pip install litellm"
            )
            
        super().__init__(
            resource_id=resource_id or f"litellm-{str(uuid.uuid4())[:8]}"
        )
        # Store these for reference but don't pass to parent
        self.resource_type = self.RESOURCE_TYPE
        self.display_name = display_name
        
        self.config_path = config_path
        self.load_env_vars = load_env_vars
        self.router: Optional[Router] = None
        self.default_model: Optional[str] = None
        
        # Latency tracking
        self.enable_latency_tracking = enable_latency_tracking
        self.latency_tracker = ModelProviderLatencyTracker() if enable_latency_tracking else None
        
        # API key tracking
        self.provider_api_keys: Dict[str, List[str]] = {}
        
        # Provider to model mapping (needed for tracking)
        self.model_to_provider: Dict[str, str] = {}
        
        # Initialize model_deployments to avoid 'no attribute' error
        self.model_deployments = {}
    
    async def _do_initialize(self) -> None:
        """
        Initialize the LiteLLM service.
        
        This method:
        1. Locates and loads configuration from the specified path or default locations
        2. Sets up the LiteLLM router with provider configurations
        3. Validates connections to providers
        
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file is invalid
            ConnectionError: If unable to connect to any LLM provider
        """
        try:
            # Load configuration
            config = await self._load_config()
            
            # Configure litellm global settings
            if DEBUG >= 2:
                os.environ["LITELLM_LOG"] = "DEBUG"
                
            # Disable cache to avoid issues
            litellm.cache = None
                
            if "litellm_settings" in config:
                for key, value in config["litellm_settings"].items():
                    if key == "set_verbose" and value:
                        os.environ["LITELLM_LOG"] = "DEBUG"
                    elif key == "cache":
                        # Ignore cache setting, already set to None
                        pass
                    else:
                        setattr(litellm, key, value)
            
            # Skip router - use direct litellm instead
            if DEBUG >= 1:
                logging.info("Using direct LiteLLM completion instead of router")
                
            # Set default model
            self.default_model = config.get("default_model")
            
            # Initialize router as None - we'll use direct litellm calls
            self.router = None
            
            # Store model deployments for later use
            self.model_deployments = {}
            model_deployments = config.get("model_deployments", [])
            
            # Process model configurations
            for deployment in model_deployments:
                model_name = deployment.get("model_name")
                model_id = deployment.get("model")
                
                if not model_name or not model_id:
                    continue
                    
                # Store deployment config by model name
                self.model_deployments[model_name] = deployment
                
                if DEBUG >= 2:
                    logging.info(f"Added model: {model_name} -> {model_id}")
                
            # Log available models
            if DEBUG >= 1:
                logging.info(f"Available models: {list(self.model_deployments.keys())}")
                
            # Skip test connection during initialization
            # We'll validate connections on-demand when actually making API calls
            if DEBUG >= 1:
                logging.info("Skipping connection test - will validate on first request")
                
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Failed to initialize LiteLLM service: {str(e)}")
            raise
    
    async def _do_cleanup(self) -> None:
        """
        Cleanup resources used by the LiteLLM service.
        """
        if self.router:
            # Close connections in the router
            await self.router.reset()
            self.router = None
            
    async def _check_health(self) -> bool:
        """
        Check if the LiteLLM service is healthy.
        
        Returns:
            True if the service is operational, False otherwise.
        """
        try:
            if not self.router:
                return False
                
            # Quick ping to a provider to make sure it's responding
            await self._test_connection(timeout=5.0)
            return True
            
        except Exception:
            return False
    
    async def _load_config(self) -> Dict[str, Any]:
        """
        Load LiteLLM configuration from file, environment variables, and dynamic discovery.
        
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file is invalid
        """
        config: Dict[str, Any] = {
            "litellm_settings": {},
            "router": {},
            "model_deployments": []
        }
        
        # Try to load from specified config path
        if self.config_path:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                
        # Otherwise try default locations
        else:
            # Try ~/.exo/litellm_config.json
            config_dir = get_exo_config_dir()
            default_config_path = config_dir / "litellm_config.json"
            
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
        
        # Check and collect available API keys from environment
        self._collect_api_keys_from_env()
        
        # Dynamically discover models
        try:
            from exo.api.model_discovery import ModelDiscovery
            
            if DEBUG >= 1:
                logging.info("Discovering available LLM models...")
                
            discovered_models = ModelDiscovery.discover_all_models()
            
            # Add models to deployments but verify API key validity first
            for model in discovered_models:
                # Mark model as ready only if API key is valid
                model_provider = model.get("metadata", {}).get("provider", "").lower()
                has_valid_keys = self._has_valid_keys_for_provider(model_provider)
                
                # Only consider a model available if both:
                # 1. It's marked available by the discovery module
                # 2. We have valid keys for its provider
                model_available = model.get("available", False) and has_valid_keys
                
                # For local models (Ollama, LlamaCPP), don't require API keys
                if model.get("metadata", {}).get("local", False):
                    model_available = model.get("available", False)
                
                if DEBUG >= 2:
                    model_name = model.get("model_name", "unknown")
                    logging.info(f"Model {model_name} availability: discovery={model.get('available', False)}, " + 
                               f"valid_keys={has_valid_keys}, final={model_available}")
                
                # Only add models that are actually available
                if model_available:
                    # Skip models that are already configured
                    if any(d.get("model_name") == model["model_name"] for d in config["model_deployments"]):
                        continue
                        
                    # Add the model deployment
                    deployment = {
                        "model_name": model["model_name"],
                        "litellm_params": model["litellm_params"]
                    }
                    config["model_deployments"].append(deployment)
                    
                    # Track provider for this model
                    self.model_to_provider[model["model_name"]] = model_provider
                    
                    if DEBUG >= 2:
                        logging.info(f"Added model: {model['model_name']} with provider {model_provider}")
            
            # If no models found and no default set, use llama-3.2-1b
            if not config["model_deployments"] and not config.get("default_model"):
                config["default_model"] = "llama-3.2-1b"
            # If we have models but no default, set a sensible default
            elif config["model_deployments"] and not config.get("default_model"):
                # Prefer certain models as defaults
                preferred_defaults = ["gpt-3.5-turbo", "claude-3-haiku", "ollama-llama3", "llamacpp-llama"]
                for model_name in preferred_defaults:
                    if any(d["model_name"] == model_name for d in config["model_deployments"]):
                        config["default_model"] = model_name
                        break
                        
                # If no preferred default found, use the first model
                if not config.get("default_model") and config["model_deployments"]:
                    config["default_model"] = config["model_deployments"][0]["model_name"]
                    
        except ImportError:
            if DEBUG >= 1:
                logging.warning("Model discovery module not available. Using static configuration.")
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error during model discovery: {str(e)}")
                    
        return config
        
    def _collect_api_keys_from_env(self) -> None:
        """
        Collect API keys from environment variables and validate them.
        """
        # Check for OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.provider_api_keys["openai"] = [openai_key]
            if DEBUG >= 2:
                logging.info(f"Found OpenAI API key: {openai_key[:4]}{'*' * 20}")
        
        # Check for Anthropic API key
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.provider_api_keys["anthropic"] = [anthropic_key]
            if DEBUG >= 2:
                logging.info(f"Found Anthropic API key: {anthropic_key[:4]}{'*' * 20}")
                
        # Check for other provider keys (can be extended as needed)
        for env_var, provider in [
            ("COHERE_API_KEY", "cohere"),
            ("AZURE_API_KEY", "azure"),
            ("GEMINI_API_KEY", "gemini"),
            ("MISTRAL_API_KEY", "mistral")
        ]:
            key = os.environ.get(env_var)
            if key:
                self.provider_api_keys[provider] = [key]
                if DEBUG >= 2:
                    logging.info(f"Found {provider} API key: {key[:4]}{'*' * 20}")
    
    def _has_valid_keys_for_provider(self, provider: str) -> bool:
        """
        Check if we have valid API keys for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            True if we have at least one valid key, False otherwise
        """
        # Skip key validation for local providers
        if provider.lower() in ("ollama", "llamacpp", "local"):
            return True
            
        # Check if we have any keys for this provider
        return provider.lower() in self.provider_api_keys and len(self.provider_api_keys[provider.lower()]) > 0
    
    async def _test_connection(self, timeout: float = 10.0) -> None:
        """
        Test connection to LLM providers.
        
        Args:
            timeout: Maximum time to wait for a response in seconds.
            
        Raises:
            ConnectionError: If unable to connect to any provider
        """
        if not self.model_deployments:
            raise ConnectionError("No LLM providers configured")
            
        model = self.default_model or next(iter(self.model_deployments.keys()))
        
        try:
            # Send a minimal request to test connectivity
            response = await asyncio.wait_for(
                self.get_completions(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                ),
                timeout=timeout
            )
            
            if DEBUG >= 2:
                logging.info(f"LiteLLM connection test succeeded: {response}")
                
        except asyncio.TimeoutError:
            raise ConnectionError(f"Timeout connecting to LLM provider for model {model}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to LLM provider: {str(e)}")
    
    async def ensure_ready(self) -> None:
        """
        Ensure the service is initialized and ready to use.
        
        Raises:
            RuntimeError: If the service is not in a usable state
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.is_usable:
            raise RuntimeError(f"LiteLLM service is not in a usable state: {self.state}")
    
    async def get_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        track_latency: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Get completions from the LLM provider.
        
        Args:
            model: The model to use for completion
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            track_latency: Whether to track latency for this request
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Completion response or async iterator for streaming
            
        Raises:
            ValueError: If arguments are invalid
            ConnectionError: If unable to connect to the provider
            RuntimeError: If the service is not initialized
        """
        await self.ensure_ready()
        
        # Get the actual model to use
        actual_model = model or self.default_model
        if not actual_model:
            raise ValueError("No model specified and no default model configured")
            
        # Look up model info
        model_config = self.model_deployments.get(actual_model)
        if not model_config:
            raise ValueError(f"Model {actual_model} not found in config")
            
        # Extract model ID and other params
        model_id = model_config.get("model")
        if not model_id:
            raise ValueError(f"Model ID not specified for {actual_model}")
            
        # Get the provider for this model
        provider = "litellm"
        if model_id.startswith("openai/"):
            provider = "openai"
        elif model_id.startswith("anthropic/"):
            provider = "anthropic"
        elif model_id.startswith("ollama/"):
            provider = "ollama"
        elif model_id.startswith("llamacpp/"):
            provider = "llamacpp"
        
        # Set up params for completion
        completion_kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs  # Add any other params
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
            
        # Add API key if specified in config
        api_key = None
        if "api_key" in model_config:
            # Handle ${ENV_VAR} syntax in the config
            api_key = model_config["api_key"]
            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                # Extract environment variable name
                env_var = api_key[2:-1]
                # Get value from environment
                api_key = os.environ.get(env_var, "")
                if not api_key:
                    logging.warning(f"Environment variable {env_var} not set for API key")
            
            completion_kwargs["api_key"] = api_key
            
        # Add API base if specified in config
        if "api_base" in model_config:
            completion_kwargs["api_base"] = model_config["api_base"]
            
        if DEBUG >= 2:
            logging.info(f"Sending completion request for model: {model_id}")
        
        # Timing variables for latency tracking
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Use direct litellm completion
            if stream:
                # For streaming, create a wrapper that tracks latency at the end
                return self._streaming_wrapper(
                    acompletion(**completion_kwargs),
                    provider,
                    actual_model,
                    start_time,
                    api_key,
                    track_latency
                )
            else:
                # For non-streaming, await the result and track latency
                result = await acompletion(**completion_kwargs)
                success = True
                return result
                
        except Exception as e:
            success = False
            error = e
            if DEBUG >= 1:
                logging.error(f"Error in LiteLLM completion: {str(e)}")
            raise
        finally:
            # Track latency for non-streaming requests
            if track_latency and self.latency_tracker and not stream:
                latency = time.time() - start_time
                await self.latency_tracker.record_latency(
                    provider=provider,
                    model=actual_model,
                    latency=latency,
                    api_key=api_key,
                    success=success,
                    error=error
                )
    
    async def _streaming_wrapper(
        self, 
        stream_iterator: AsyncIterator[Dict[str, Any]],
        provider: str,
        model: str,
        start_time: float,
        api_key: Optional[str],
        track_latency: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Wrapper around streaming response to track latency at the end.
        
        Args:
            stream_iterator: Original stream iterator
            provider: Provider name
            model: Model name
            start_time: Start time of the request
            api_key: API key used
            track_latency: Whether to track latency
            
        Yields:
            Chunks from the original stream
        """
        success = False
        error = None
        
        try:
            # Stream the chunks
            async for chunk in stream_iterator:
                yield chunk
                
            # If we get here without an exception, the stream was successful
            success = True
            
        except Exception as e:
            error = e
            raise
            
        finally:
            # Record latency at the end of the stream
            if track_latency and self.latency_tracker:
                latency = time.time() - start_time
                await self.latency_tracker.record_latency(
                    provider=provider,
                    model=model,
                    latency=latency,
                    api_key=api_key,
                    success=success,
                    error=error
                )
                
    async def get_fastest_provider_for_model_type(self, model_type: str) -> Optional[str]:
        """
        Get the fastest provider for a given model type.
        
        Args:
            model_type: Type of model (e.g., 'gpt-4', 'claude')
            
        Returns:
            The fastest provider name or None if no providers available
        """
        if self.latency_tracker:
            return await self.latency_tracker.get_fastest_provider(model_type)
        return None
        
    async def is_key_valid(self, provider: str, api_key: str) -> bool:
        """
        Check if a provider's API key is valid.
        
        Args:
            provider: Provider name
            api_key: API key to check
            
        Returns:
            True if the key is valid, False otherwise
        """
        if self.latency_tracker:
            return await self.latency_tracker.is_key_valid(provider, api_key)
        
        # If we don't have a latency tracker, assume the key is valid
        # for local providers or if the key exists in our provider_api_keys
        if provider.lower() in ("ollama", "llamacpp", "local"):
            return True
            
        return provider.lower() in self.provider_api_keys and api_key in self.provider_api_keys[provider.lower()]
    
    def get_supported_models(self) -> List[Dict[str, Any]]:
        """
        Get list of supported models.
        
        Returns:
            List of model information dictionaries, including both available
            and unavailable models with their metadata.
            
        Note:
            A model is marked as "ready" only if:
            1. It's configured in deployments OR it's available via discovery
            2. The host has valid API keys for the model's provider (for cloud models)
        """
        try:
            # Get dynamically discovered models (both available and unavailable)
            from exo.api.model_discovery import ModelDiscovery
            all_models = ModelDiscovery.discover_all_models()
            
            # Get the set of models that are actually configured
            available_models = set(self.model_deployments.keys()) if self.model_deployments else set()
            
            # Create models list
            models_list = []
            
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
            
            # First add models from our configured deployments
            for model_name, config in self.model_deployments.items():
                model_id = config.get("model", "")
                
                # Determine provider
                provider = "litellm"
                if model_id.startswith("openai/"):
                    provider = "openai"
                elif model_id.startswith("anthropic/"):
                    provider = "anthropic"
                elif model_id.startswith("ollama/"):
                    provider = "ollama"
                elif model_id.startswith("llamacpp/"):
                    provider = "llamacpp"
                
                # Track provider for this model if not already tracked
                if model_name not in self.model_to_provider:
                    self.model_to_provider[model_name] = provider
                    
                # Check if we have valid keys for this provider (local models always pass)
                is_local = provider in ("ollama", "llamacpp", "local")
                has_valid_keys = is_local or self._has_valid_keys_for_provider(provider)
                
                # Format display name
                display_name = model_name
                if provider == "openai" and "-" in model_name:
                    # Extract parts and capitalize
                    parts = model_name.split("-")
                    display_name = parts[0].upper()
                    for part in parts[1:]:
                        if part.isdigit():
                            display_name += "-" + part
                        else:
                            display_name += "-" + part.capitalize()
                elif provider == "anthropic" and "-" in model_name:
                    parts = model_name.split("-")
                    display_name = parts[0].capitalize()
                    for part in parts[1:]:
                        display_name += "-" + part.capitalize()
                    
                # Get context window and tools support from model discovery
                metadata = {"provider": provider, "local": is_local}
                for discovered_model in all_models:
                    if discovered_model["model_name"] == model_name:
                        metadata.update(discovered_model.get("metadata", {}))
                        break
                        
                # Add display info
                metadata["displayName"] = display_name
                metadata["providerDisplayName"] = provider_display_names.get(provider, provider.capitalize())
                metadata["isCloudModel"] = not is_local
                metadata["canDelete"] = False  # Cloud models can't be deleted
                
                # Mark as ready only if valid keys available (local models are always ready)
                models_list.append({
                    "id": model_name,
                    "object": "model",
                    "owned_by": provider,
                    "ready": has_valid_keys,
                    "metadata": metadata
                })
                
            # Then add dynamically discovered models that aren't already in the list
            for model in all_models:
                model_name = model["model_name"]
                if model_name not in available_models:
                    provider = model.get("metadata", {}).get("provider", "litellm").lower()
                    is_local = model.get("metadata", {}).get("local", False)
                    
                    # Check if we have valid keys for this provider
                    has_valid_keys = is_local or self._has_valid_keys_for_provider(provider)
                    
                    # Mark as ready only if:
                    # 1. Model is available according to discovery, AND
                    # 2. We have valid keys for its provider (unless it's local)
                    is_ready = model.get("available", False) and has_valid_keys
                    
                    # Format metadata with display info
                    metadata = model.get("metadata", {}).copy()
                    
                    # Add display name
                    display_name = model_name
                    if provider == "openai" and "-" in model_name:
                        # Extract parts and capitalize
                        parts = model_name.split("-")
                        display_name = parts[0].upper()
                        for part in parts[1:]:
                            if part.isdigit():
                                display_name += "-" + part
                            else:
                                display_name += "-" + part.capitalize()
                    elif provider == "anthropic" and "-" in model_name:
                        parts = model_name.split("-")
                        display_name = parts[0].capitalize()
                        for part in parts[1:]:
                            display_name += "-" + part.capitalize()
                    
                    # Add display info
                    metadata["displayName"] = display_name
                    metadata["providerDisplayName"] = provider_display_names.get(provider, provider.capitalize())
                    metadata["isCloudModel"] = not is_local
                    metadata["canDelete"] = False  # Cloud models can't be deleted
                    
                    models_list.append({
                        "id": model_name,
                        "object": "model",
                        "owned_by": provider,
                        "ready": is_ready,
                        "metadata": metadata
                    })
                    
            # Ensure we always include standard cloud models when API keys exist
            if self._has_valid_keys_for_provider("openai"):
                # Define standard OpenAI models to add if not already present
                standard_openai_models = [
                    {
                        "id": "gpt-4", 
                        "displayName": "GPT-4",
                        "context_length": 8192,
                        "supports_tools": True
                    },
                    {
                        "id": "gpt-4-turbo", 
                        "displayName": "GPT-4 Turbo",
                        "context_length": 128000,
                        "supports_tools": True
                    },
                    {
                        "id": "gpt-3.5-turbo", 
                        "displayName": "GPT-3.5 Turbo",
                        "context_length": 16384,
                        "supports_tools": True
                    }
                ]
                
                existing_ids = {model["id"] for model in models_list}
                for model in standard_openai_models:
                    if model["id"] not in existing_ids:
                        models_list.append({
                            "id": model["id"],
                            "object": "model",
                            "owned_by": "openai",
                            "ready": True,
                            "metadata": {
                                "provider": "openai",
                                "local": False,
                                "isCloudModel": True,
                                "canDelete": False,
                                "displayName": model["displayName"],
                                "providerDisplayName": "OpenAI",
                                "context_length": model["context_length"],
                                "supports_tools": model["supports_tools"]
                            }
                        })
                        
            # Add standard Anthropic models if not already present
            if self._has_valid_keys_for_provider("anthropic"):
                standard_anthropic_models = [
                    {
                        "id": "claude-3-opus", 
                        "displayName": "Claude-3 Opus",
                        "context_length": 200000,
                        "supports_tools": True
                    },
                    {
                        "id": "claude-3-sonnet", 
                        "displayName": "Claude-3 Sonnet",
                        "context_length": 180000,
                        "supports_tools": True
                    },
                    {
                        "id": "claude-3-haiku", 
                        "displayName": "Claude-3 Haiku",
                        "context_length": 150000,
                        "supports_tools": True
                    }
                ]
                
                existing_ids = {model["id"] for model in models_list}
                for model in standard_anthropic_models:
                    if model["id"] not in existing_ids:
                        models_list.append({
                            "id": model["id"],
                            "object": "model",
                            "owned_by": "anthropic",
                            "ready": True,
                            "metadata": {
                                "provider": "anthropic",
                                "local": False,
                                "isCloudModel": True,
                                "canDelete": False,
                                "displayName": model["displayName"],
                                "providerDisplayName": "Anthropic",
                                "context_length": model["context_length"],
                                "supports_tools": model["supports_tools"]
                            }
                        })
                        
            return models_list
            
        except (ImportError, Exception) as e:
            if DEBUG >= 1:
                logging.error(f"Error getting models: {str(e)}")
                
            # Define standard fallback models when API keys exist
            models_list = []
            
            # Add standard OpenAI models if API key exists
            if self._has_valid_keys_for_provider("openai"):
                standard_openai_models = [
                    {
                        "id": "gpt-4", 
                        "displayName": "GPT-4",
                        "context_length": 8192,
                        "supports_tools": True
                    },
                    {
                        "id": "gpt-4-turbo", 
                        "displayName": "GPT-4 Turbo",
                        "context_length": 128000,
                        "supports_tools": True
                    },
                    {
                        "id": "gpt-3.5-turbo", 
                        "displayName": "GPT-3.5 Turbo",
                        "context_length": 16384,
                        "supports_tools": True
                    }
                ]
                
                for model in standard_openai_models:
                    models_list.append({
                        "id": model["id"],
                        "object": "model",
                        "owned_by": "openai",
                        "ready": True,
                        "metadata": {
                            "provider": "openai",
                            "local": False,
                            "isCloudModel": True,
                            "canDelete": False,
                            "displayName": model["displayName"],
                            "providerDisplayName": "OpenAI",
                            "context_length": model["context_length"],
                            "supports_tools": model["supports_tools"]
                        }
                    })
            
            # Add standard Anthropic models if API key exists
            if self._has_valid_keys_for_provider("anthropic"):
                standard_anthropic_models = [
                    {
                        "id": "claude-3-opus", 
                        "displayName": "Claude-3 Opus",
                        "context_length": 200000,
                        "supports_tools": True
                    },
                    {
                        "id": "claude-3-sonnet", 
                        "displayName": "Claude-3 Sonnet",
                        "context_length": 180000,
                        "supports_tools": True
                    },
                    {
                        "id": "claude-3-haiku", 
                        "displayName": "Claude-3 Haiku",
                        "context_length": 150000,
                        "supports_tools": True
                    }
                ]
                
                for model in standard_anthropic_models:
                    models_list.append({
                        "id": model["id"],
                        "object": "model",
                        "owned_by": "anthropic",
                        "ready": True,
                        "metadata": {
                            "provider": "anthropic",
                            "local": False,
                            "isCloudModel": True,
                            "canDelete": False,
                            "displayName": model["displayName"],
                            "providerDisplayName": "Anthropic",
                            "context_length": model["context_length"],
                            "supports_tools": model["supports_tools"]
                        }
                    })
            
            # Also add any models from deployments with enhanced metadata
            for model_name, config in (self.model_deployments or {}).items():
                provider = (
                    "openai" if config.get("model", "").startswith("openai/") else 
                    "anthropic" if config.get("model", "").startswith("anthropic/") else 
                    "ollama" if config.get("model", "").startswith("ollama/") else 
                    "llamacpp" if config.get("model", "").startswith("llamacpp/") else 
                    "litellm"
                )
                
                is_local = provider in ("ollama", "llamacpp", "local")
                has_valid_keys = is_local or self._has_valid_keys_for_provider(provider)
                
                # Format display name
                display_name = model_name
                if provider == "openai" and "-" in model_name:
                    # Extract parts and capitalize
                    parts = model_name.split("-")
                    display_name = parts[0].upper()
                    for part in parts[1:]:
                        if part.isdigit():
                            display_name += "-" + part
                        else:
                            display_name += "-" + part.capitalize()
                elif provider == "anthropic" and "-" in model_name:
                    parts = model_name.split("-")
                    display_name = parts[0].capitalize()
                    for part in parts[1:]:
                        display_name += "-" + part.capitalize()
                
                models_list.append({
                    "id": model_name,
                    "object": "model",
                    "owned_by": provider,
                    "ready": has_valid_keys,
                    "metadata": {
                        "provider": provider,
                        "local": is_local,
                        "isCloudModel": not is_local,
                        "canDelete": False if not is_local else True,
                        "displayName": display_name,
                        "providerDisplayName": provider.capitalize(),
                        "context_length": 8192,  # Default fallback
                        "supports_tools": provider in ["openai", "anthropic"]
                    }
                })
                
            return models_list
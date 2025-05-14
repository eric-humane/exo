"""
LiteLLM integration for exo.

This module provides a service for interacting with LiteLLM, allowing exo to use
various LLM providers through a unified interface, while maintaining proper
resource lifecycle management using the AsyncResource pattern.
"""

import os
import json
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, ClassVar, AsyncIterator
from pathlib import Path

# Import AsyncResource base class
from exo.utils.async_resources import AsyncResource

# Import LiteLLM
import litellm
from litellm import completion, acompletion
from litellm.router import Router
from litellm.exceptions import AuthorizationError, ServiceUnavailableError, BadRequestError

from exo import DEBUG, get_exo_config_dir


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
        display_name: str = "LiteLLM Service"
    ):
        """
        Initialize the LiteLLM service.
        
        Args:
            resource_id: Unique identifier for this service. If not provided, a UUID will be generated.
            config_path: Path to a LiteLLM config file. If not provided, will attempt to load from 
                         default locations.
            load_env_vars: Whether to load provider API keys from environment variables.
            display_name: Display name for this service in logs and UI.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install it with: pip install litellm"
            )
            
        super().__init__(
            resource_id=resource_id or f"litellm-{str(uuid.uuid4())[:8]}",
            resource_type=self.RESOURCE_TYPE,
            display_name=display_name
        )
        
        self.config_path = config_path
        self.load_env_vars = load_env_vars
        self.router: Optional[Router] = None
        self.default_model: Optional[str] = None
    
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
            config = await self._load_config()
            
            # Configure litellm global settings
            litellm.set_verbose = DEBUG >= 2
            if "litellm_settings" in config:
                for key, value in config["litellm_settings"].items():
                    setattr(litellm, key, value)
            
            # Initialize router
            router_config = config.get("router", {})
            self.router = Router(**router_config)
            
            # Add model deployments
            model_deployments = config.get("model_deployments", [])
            for deployment in model_deployments:
                self.router.add_deployment(**deployment)
            
            # Set default model
            self.default_model = config.get("default_model")
            
            # Test connection by sending a simple request
            if model_deployments:
                await self._test_connection()
                
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
        Load LiteLLM configuration from file or environment variables.
        
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
        
        # Load from environment variables if enabled
        if self.load_env_vars:
            # OpenAI
            if openai_api_key := os.environ.get("OPENAI_API_KEY"):
                config["model_deployments"].append({
                    "model_name": "gpt-4",
                    "litellm_params": {
                        "model": "gpt-4",
                        "api_key": openai_api_key
                    }
                })
                
                config["model_deployments"].append({
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": openai_api_key
                    }
                })
                
                # Set default model if not already set
                if not config.get("default_model"):
                    config["default_model"] = "gpt-3.5-turbo"
            
            # Anthropic
            if anthropic_api_key := os.environ.get("ANTHROPIC_API_KEY"):
                config["model_deployments"].append({
                    "model_name": "claude-3-opus",
                    "litellm_params": {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": anthropic_api_key
                    }
                })
                config["model_deployments"].append({
                    "model_name": "claude-3-sonnet",
                    "litellm_params": {
                        "model": "anthropic/claude-3-sonnet-20240229",
                        "api_key": anthropic_api_key
                    }
                })
                
                # Set default model if not already set and OpenAI not configured
                if not config.get("default_model") and not os.environ.get("OPENAI_API_KEY"):
                    config["default_model"] = "claude-3-sonnet"
                    
        return config
    
    async def _test_connection(self, timeout: float = 10.0) -> None:
        """
        Test connection to LLM providers.
        
        Args:
            timeout: Maximum time to wait for a response in seconds.
            
        Raises:
            ConnectionError: If unable to connect to any provider
        """
        if not self.router or not self.router.deployments:
            raise ConnectionError("No LLM providers configured")
            
        model = self.default_model or self.router.deployments[0]["model_name"]
        
        try:
            # Send a minimal request to test connectivity
            response = await asyncio.wait_for(
                self.router.acompletion(
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
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Completion response or async iterator for streaming
            
        Raises:
            ValueError: If arguments are invalid
            ConnectionError: If unable to connect to the provider
            RuntimeError: If the service is not initialized
        """
        await self.ensure_ready()
        
        if not self.router:
            raise RuntimeError("LiteLLM router is not initialized")
            
        actual_model = model or self.default_model
        if not actual_model:
            raise ValueError("No model specified and no default model configured")
        
        try:
            if stream:
                return self.router.acompletion(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
            else:
                return await self.router.acompletion(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs
                )
                
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error in LiteLLM completion: {str(e)}")
            raise
    
    def get_supported_models(self) -> List[Dict[str, Any]]:
        """
        Get list of supported models.
        
        Returns:
            List of model information dictionaries
        """
        if not self.router or not hasattr(self.router, "deployments"):
            return []
            
        return [
            {
                "id": deployment["model_name"],
                "owned_by": "litellm",
                "object": "model"
            }
            for deployment in self.router.deployments
        ]
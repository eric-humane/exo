"""
Model discovery module for dynamically finding available LLM models from various providers.

This module allows exo to dynamically discover and expose models from different
providers based on available API keys, without requiring static configuration.
"""

import os
import logging
import requests
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from exo import DEBUG

class ModelDiscovery:
    """
    Discovers available models from various LLM providers.
    
    This class provides methods to:
    1. Check for API keys and provider availability
    2. Query provider APIs to discover available models
    3. Create LiteLLM model deployments for discovered models
    """
    
    @staticmethod
    def discover_openai_models() -> List[Dict[str, Any]]:
        """
        Query OpenAI API to get available models.
        
        Returns:
            List of model configurations compatible with LiteLLM
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if DEBUG >= 2:
                logging.info("No OpenAI API key found in environment")
            # Return placeholder models when API key isn't available
            return ModelDiscovery._get_openai_placeholder_models()
            
        if DEBUG >= 2:
            logging.info("Found OpenAI API key, attempting to fetch models")
            
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10  # Add timeout to prevent hanging
            )
            
            if response.status_code != 200:
                logging.error(f"Failed to get OpenAI models: {response.text}")
                return ModelDiscovery._get_openai_placeholder_models()
                
            models = response.json().get("data", [])
            
            if DEBUG >= 2:
                logging.info(f"Retrieved {len(models)} OpenAI models")
            
            # Filter for chat models only
            chat_models = [
                model for model in models 
                if (model.get("id", "").startswith(("gpt-3.5", "gpt-4")) 
                    and not model.get("id", "").endswith("-vision"))
            ]
            
            if DEBUG >= 2:
                logging.info(f"Filtered down to {len(chat_models)} chat models")
                if DEBUG >= 3:
                    model_ids = [model.get("id") for model in chat_models]
                    logging.info(f"Chat model IDs: {model_ids}")
            
            return [
                {
                    "model_name": model["id"],
                    "litellm_params": {
                        "model": model["id"],
                        "api_key": "${OPENAI_API_KEY}"
                    },
                    "metadata": {
                        "context_length": ModelDiscovery._get_context_length(model["id"]),
                        "supports_tools": "gpt-4" in model["id"] or "gpt-3.5-turbo" in model["id"],
                        "provider": "openai",
                        "isCloudModel": True
                    },
                    "available": True
                }
                for model in chat_models
            ]
        except Exception as e:
            logging.error(f"Error discovering OpenAI models: {str(e)}")
            return ModelDiscovery._get_openai_placeholder_models()
    
    @staticmethod
    def _get_openai_placeholder_models() -> List[Dict[str, Any]]:
        """
        Get placeholder OpenAI models when API is unavailable.
        
        Returns:
            List of common OpenAI models with availability set based on API key presence
        """
        models = [
            {"id": "gpt-4", "context_window": 8192, "supports_tools": True},
            {"id": "gpt-4-turbo", "context_window": 128000, "supports_tools": True},
            {"id": "gpt-4-vision", "context_window": 128000, "supports_tools": False},
            {"id": "gpt-3.5-turbo", "context_window": 16384, "supports_tools": True}
        ]
        
        # Check if API key is available - if it is, consider the models available
        api_key_exists = os.environ.get("OPENAI_API_KEY") is not None
        
        return [
            {
                "model_name": model["id"],
                "litellm_params": {
                    "model": model["id"],
                    "api_key": "${OPENAI_API_KEY}"
                },
                "metadata": {
                    "context_length": model["context_window"],
                    "supports_tools": model["supports_tools"],
                    "provider": "openai",
                    "isCloudModel": True
                },
                "available": api_key_exists  # Set to True if API key exists
            }
            for model in models
        ]
    
    @staticmethod
    def discover_anthropic_models() -> List[Dict[str, Any]]:
        """
        Query Anthropic API to get available models.
        
        Returns:
            List of model configurations compatible with LiteLLM
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        has_key = api_key is not None
        
        if DEBUG >= 2:
            logging.info(f"Anthropic API key found: {has_key}")
            
        # Anthropic doesn't have a models listing endpoint
        # Use known models that should be available
        models = [
            {
                "id": "claude-3-opus-20240229",
                "name": "claude-3-opus",
                "context_window": 200000,
                "supports_tools": True
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "claude-3-sonnet",
                "context_window": 180000,
                "supports_tools": True
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "claude-3-haiku",
                "context_window": 150000,
                "supports_tools": True
            }
        ]
        
        return [
            {
                "model_name": model["name"],
                "litellm_params": {
                    "model": f"anthropic/{model['id']}",
                    "api_key": "${ANTHROPIC_API_KEY}"
                },
                "metadata": {
                    "context_length": model["context_window"],
                    "supports_tools": model["supports_tools"],
                    "provider": "anthropic",
                    "isCloudModel": True
                },
                "available": has_key
            }
            for model in models
        ]
    
    @staticmethod
    def _get_context_length(model_id: str) -> int:
        """
        Get context length for a model based on its ID.
        
        Args:
            model_id: The OpenAI model ID
            
        Returns:
            Context window size in tokens
        """
        if "gpt-4-turbo" in model_id:
            return 128000
        elif "gpt-4-32k" in model_id:
            return 32768
        elif "gpt-4" in model_id:
            return 8192
        elif "gpt-3.5-turbo-16k" in model_id:
            return 16384
        else:
            return 4096  # Default for gpt-3.5-turbo
    
    @staticmethod
    def discover_ollama_models() -> List[Dict[str, Any]]:
        """
        Query Ollama API to get locally available models.
        
        Returns:
            List of Ollama model configurations compatible with LiteLLM
        """
        # Check if Ollama is installed and running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            available = response.status_code == 200
            models = response.json().get("models", []) if available else []
            
            # Return discovered models
            return [
                {
                    "model_name": f"ollama-{model['name']}",
                    "litellm_params": {
                        "model": f"ollama/{model['name']}",
                        "api_base": "http://localhost:11434"
                    },
                    "metadata": {
                        "context_length": 8192,  # Approximate for most models
                        "supports_tools": False,  # Ollama doesn't support function calling yet
                        "provider": "ollama",
                        "local": True,
                        "isCloudModel": False
                    },
                    "available": True
                }
                for model in models
            ]
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            if DEBUG >= 2:
                logging.info(f"Ollama not detected: {str(e)}")
                
            # Return placeholder models
            return [
                {
                    "model_name": "ollama-llama3",
                    "litellm_params": {
                        "model": "ollama/llama3",
                        "api_base": "http://localhost:11434"
                    },
                    "metadata": {
                        "context_length": 8192,
                        "supports_tools": False,
                        "provider": "ollama",
                        "local": True,
                        "isCloudModel": False
                    },
                    "available": False
                },
                {
                    "model_name": "ollama-mistral",
                    "litellm_params": {
                        "model": "ollama/mistral",
                        "api_base": "http://localhost:11434"
                    },
                    "metadata": {
                        "context_length": 8192,
                        "supports_tools": False,
                        "provider": "ollama",
                        "local": True,
                        "isCloudModel": False
                    },
                    "available": False
                }
            ]
    
    @staticmethod
    def discover_llamacpp_models() -> List[Dict[str, Any]]:
        """
        Detect locally installed llama.cpp models.
        
        Returns:
            List of llama.cpp model configurations compatible with LiteLLM
        """
        # Common paths where llama.cpp models might be found
        model_paths = [
            Path.home() / ".cache" / "llama.cpp",
            Path.home() / "llama.cpp" / "models",
            Path("/usr/local/share/llama.cpp/models"),
            Path("/opt/llama.cpp/models")
        ]
        
        models = []
        for path in model_paths:
            if path.exists() and path.is_dir():
                # Look for .gguf model files
                for model_file in path.glob("**/*.gguf"):
                    model_name = model_file.stem.lower()
                    models.append({
                        "model_name": f"llamacpp-{model_name}",
                        "litellm_params": {
                            "model": "llamacpp/llama",
                            "model_path": str(model_file)
                        },
                        "metadata": {
                            "context_length": 4096,  # Default, actual varies by model
                            "supports_tools": False,
                            "provider": "llamacpp",
                            "local": True,
                            "isCloudModel": False
                        },
                        "available": True
                    })
        
        # If no models found, add placeholder
        if not models:
            models.append({
                "model_name": "llamacpp-local",
                "litellm_params": {
                    "model": "llamacpp/llama"
                },
                "metadata": {
                    "context_length": 4096,
                    "supports_tools": False,
                    "provider": "llamacpp",
                    "local": True,
                    "isCloudModel": False
                },
                "available": False
            })
            
        return models
    
    @staticmethod
    def discover_all_models() -> List[Dict[str, Any]]:
        """
        Discover models from all supported providers.
        
        Returns:
            Combined list of all available models from all providers
        """
        models = []
        
        # Cloud providers
        models.extend(ModelDiscovery.discover_openai_models())
        models.extend(ModelDiscovery.discover_anthropic_models())
        
        # Local providers
        models.extend(ModelDiscovery.discover_ollama_models())
        models.extend(ModelDiscovery.discover_llamacpp_models())
        
        # Sort models: first by availability, then by provider, then by name
        return sorted(
            models,
            key=lambda m: (
                not m.get("available", False),  # Available models first
                m.get("metadata", {}).get("provider", ""),  # Then by provider
                m.get("model_name", "")  # Then by name
            )
        )
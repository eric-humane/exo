# Dynamic Model Discovery Enhancement

This document outlines the approach to modify exo's LiteLLM integration to dynamically discover and list available models from various providers without requiring static configuration.

## Current Limitations

Currently, the system:
- Requires models to be explicitly defined in a config file or through environment variables
- Uses a static list of model mappings that must be updated manually when providers add/remove models
- Can't automatically detect provider capabilities or new model releases

## Proposed Solution

Enhance the LiteLLM integration to:
1. Detect available API keys for providers
2. Query provider APIs to discover available models
3. Dynamically build model deployments based on discovered models
4. Update model information in real-time when requested

## Implementation Details

### 1. Provider API Integration

Create a new module `exo/api/model_discovery.py` that:

```python
import os
import requests
import logging
from typing import List, Dict, Any, Optional

class ModelDiscovery:
    """Discovers available models from various providers."""
    
    @staticmethod
    def discover_openai_models() -> List[Dict[str, Any]]:
        """Query OpenAI API to get available models."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return []
            
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code != 200:
                logging.error(f"Failed to get OpenAI models: {response.text}")
                return []
                
            models = response.json().get("data", [])
            
            # Filter for chat models only
            chat_models = [
                model for model in models 
                if model.get("id", "").startswith(("gpt-3.5", "gpt-4"))
            ]
            
            return [
                {
                    "model_name": model["id"],
                    "litellm_params": {
                        "model": model["id"],
                        "api_key": "${OPENAI_API_KEY}"
                    },
                    "metadata": {
                        "context_length": self._get_context_length(model["id"]),
                        "supports_tools": "gpt-4" in model["id"]
                    }
                }
                for model in chat_models
            ]
        except Exception as e:
            logging.error(f"Error discovering OpenAI models: {str(e)}")
            return []
    
    @staticmethod
    def discover_anthropic_models() -> List[Dict[str, Any]]:
        """Query Anthropic API to get available models."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return []
            
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
                    "supports_tools": model["supports_tools"]
                }
            }
            for model in models
        ]
    
    @staticmethod
    def _get_context_length(model_id: str) -> int:
        """Get context length for a model based on its ID."""
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
    def discover_all_models() -> List[Dict[str, Any]]:
        """Discover models from all supported providers."""
        models = []
        models.extend(ModelDiscovery.discover_openai_models())
        models.extend(ModelDiscovery.discover_anthropic_models())
        # Add other providers like Google, Cohere, etc.
        return models
```

### 2. Update LiteLLMService

Modify `exo/api/litellm_service.py` to incorporate dynamic model discovery:

```python
# In _load_config method
async def _load_config(self) -> Dict[str, Any]:
    """Load configuration with dynamic model discovery."""
    config = {
        "litellm_settings": {},
        "router": {},
        "model_deployments": []
    }
    
    # Try to load from specified config path for custom settings
    if self.config_path:
        config_path = Path(self.config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
    
    # Perform dynamic model discovery
    from exo.api.model_discovery import ModelDiscovery
    discovered_models = ModelDiscovery.discover_all_models()
    
    # Add discovered models to deployments
    if discovered_models:
        existing_names = {d["model_name"] for d in config["model_deployments"]}
        for model in discovered_models:
            if model["model_name"] not in existing_names:
                config["model_deployments"].append(model)
    
    # Set a default model if available
    if config["model_deployments"] and not config.get("default_model"):
        # Prefer small, fast models as default
        preferred_defaults = ["gpt-3.5-turbo", "claude-3-haiku"]
        for model_name in preferred_defaults:
            if any(d["model_name"] == model_name for d in config["model_deployments"]):
                config["default_model"] = model_name
                break
        # If no preferred default, use the first model
        if not config.get("default_model"):
            config["default_model"] = config["model_deployments"][0]["model_name"]
    
    return config
```

### 3. Enhance Model Availability Reporting

Modify the `get_supported_models` method in `LiteLLMService` to include availability status:

```python
def get_supported_models(self) -> List[Dict[str, Any]]:
    """Get list of supported models with availability status."""
    from exo.api.model_discovery import ModelDiscovery
    
    # Always discover all possible models
    all_models = ModelDiscovery.discover_all_models()
    
    # Check which ones are actually available
    available_models = set()
    if self.router and hasattr(self.router, "deployments"):
        available_models = {d["model_name"] for d in self.router.deployments}
    
    # Return all models with availability flag
    return [
        {
            "id": model["model_name"],
            "object": "model",
            "owned_by": "litellm",
            "ready": model["model_name"] in available_models,
            "metadata": model.get("metadata", {})
        }
        for model in all_models
    ]
```

### 4. Enhance ChatGPT API Model Endpoint

Modify the `handle_get_models` method in `ChatGPTAPI` to include all models:

```python
async def handle_get_models(self, request):
    # Start with local models
    models_list = [
        {"id": model_name, "object": "model", "owned_by": "exo", "ready": True} 
        for model_name, _ in model_cards.items()
    ]
    
    # Get both available and unavailable external models
    if self.litellm_adapter:
        try:
            external_models = self.litellm_adapter.get_supported_models()
            # Add models that aren't already in the list
            for model in external_models:
                if not any(m["id"] == model["id"] for m in models_list):
                    models_list.append(model)
        except Exception as e:
            if DEBUG >= 1: print(f"Error getting LiteLLM models: {e}")
    
    return web.json_response({"object": "list", "data": models_list})
```

## Integration with UI

The UI can be enhanced to:
1. Show all possible models, with unavailable ones clearly marked
2. Provide links/instructions to set up API keys for unavailable models
3. Display model capabilities (context size, function calling support, etc.)

## Benefits

This approach:
1. Eliminates the need for static configuration files
2. Automatically discovers new models as they become available
3. Shows users what's possible even when API keys aren't set up
4. Adapts to changes in provider offerings without code changes
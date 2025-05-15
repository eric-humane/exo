# Chat API Modular Architecture

This directory contains a modular implementation of the exo Chat API, which provides OpenAI-compatible API endpoints for chat completions, model management, and image generation.

## Architecture Overview

The API is composed of several modules, each responsible for a specific concern:

1. **Main API Class** (`api.py`): Composes all the modules and provides a unified interface.

2. **Data Models** (`models.py`): Contains data model classes like `Message`, `ChatCompletionRequest`, etc.

3. **Token Handler** (`token_handler.py`): Manages token streaming and processing.

4. **Route Handlers** (`route_handlers.py`): Implements core HTTP route handlers for chat completions.

5. **Model Endpoints** (`model_endpoints.py`): Implements model management endpoints.

6. **Image Endpoints** (`image_endpoints.py`): Implements image generation endpoints.

7. **Utilities** (`utils.py`): Provides utility functions for encoding/decoding, progress bars, etc.

## Backward Compatibility

For backward compatibility, the original `chatgpt_api.py` file now imports and extends the `ChatAPI` class from this modular implementation. This ensures that existing code that imports `ChatGPTAPI` continues to work.

## Extension Points

The modular architecture makes it easier to extend the API:

1. **Adding new endpoints**: Simply add new handler methods to the relevant module.

2. **Supporting new features**: Implement new modules and integrate them into the main API class.

3. **Customizing behavior**: Extend the base classes to override or customize specific behaviors.

## Running Tests

To run tests for the modular API:

```bash
python -m unittest exo/api/chat_api/test_api.py
```

## Example Usage

```python
from exo.api.chat_api import ChatAPI

# Create a ChatAPI instance
api = ChatAPI(
    node=node,
    inference_engine_classname="MLXDynamicShardInferenceEngine",
    response_timeout=30,
    default_model="llama-3.2-1b"
)

# Run the API server
await api.run(host="0.0.0.0", port=8080)
```
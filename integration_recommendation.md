# LiteLLM Integration Recommendation for exo

## Overview

LiteLLM would provide exo with a unified interface for various language models beyond just the locally hosted ones. This integration would allow exo's architecture to interact with hosted models (OpenAI, Anthropic, etc.) and local models (via Ollama, VLLM, etc.) through a single, consistent interface.

## Current Architecture Analysis

After reviewing the codebase, here are the key components relevant to this integration:

1. `exo/api/chatgpt_api.py` - The main API implementation for providing a ChatGPT-compatible API interface
2. `exo/main.py` - Initializes the API and connects it to the node system
3. `exo/utils/async_resources.py` - Provides the AsyncResource pattern which we can leverage

The current implementation directly connects to the exo node system for inference, but there's no abstraction layer for interacting with different model providers.

## Recommended Integration Approach

### 1. Create a LiteLLMResource Class

Create a new class that extends AsyncResource for managing connections to various models through LiteLLM:

```python
# exo/api/litellm_resource.py
import os
import asyncio
import litellm
from litellm import acompletion
from typing import Dict, List, Optional, Union
from exo.utils.async_resources import AsyncResource, ResourceInitializationError

class LiteLLMResource(AsyncResource):
    """
    AsyncResource implementation for LiteLLM, providing a unified interface 
    to various LLM providers and models.
    """
    
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        resource_id: Optional[str] = None,
        initialize_timeout: float = 30.0,
        **model_kwargs
    ):
        """
        Initialize a new LiteLLM resource.
        
        Args:
            model_name: Name of the model to use (e.g., "openai/gpt-4", "ollama/llama2")
            api_base: Optional base URL for the model API
            api_key: Optional API key for the model provider
            resource_id: Optional unique identifier for this resource
            initialize_timeout: Timeout for initialization in seconds
            **model_kwargs: Additional kwargs to pass to the model
        """
        super().__init__(
            resource_id=resource_id or f"litellm-{model_name}",
            initialize_timeout=initialize_timeout
        )
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.model_kwargs = model_kwargs
        self._setup_complete = False
        
    async def _do_initialize(self) -> None:
        """Initialize the LiteLLM connection."""
        try:
            # Test connection with a simple query
            response = await acompletion(
                model=self.model_name,
                messages=[{"role": "system", "content": "Test connection"}],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=5,
                **self.model_kwargs
            )
            self._setup_complete = True
        except Exception as e:
            raise ResourceInitializationError(f"Failed to initialize LiteLLM for {self.model_name}: {str(e)}") from e
    
    async def _do_cleanup(self) -> None:
        """Cleanup the LiteLLM connection."""
        # LiteLLM doesn't require explicit cleanup, but we implement this 
        # to conform to the AsyncResource pattern
        self._setup_complete = False
        
    async def _check_health(self) -> bool:
        """Check if the LiteLLM connection is healthy."""
        try:
            # Send a minimal request to test connectivity
            await acompletion(
                model=self.model_name,
                messages=[{"role": "system", "content": "Health check"}],
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=1,
                **self.model_kwargs
            )
            return True
        except Exception:
            return False
            
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Union[Dict, asyncio.Generator]:
        """
        Generate a completion using the LiteLLM model.
        
        Args:
            messages: The conversation history in the OpenAI format
            temperature: Sampling temperature (0.0 = deterministic)
            stream: Whether to stream the response
            request_id: Optional request ID for tracking
            **kwargs: Additional parameters to pass to LiteLLM
            
        Returns:
            Either a completion result dict or an async generator for streaming
        """
        if not self.is_initialized:
            await self.initialize()
            
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "api_base": self.api_base,
            "api_key": self.api_key,
            **self.model_kwargs,
            **kwargs
        }
        
        # Use litellm's acompletion for async support
        return await acompletion(**completion_kwargs)
```

### 2. Modify ChatGPTAPI to Use LiteLLMResource

Update the ChatGPTAPI class to integrate with LiteLLMResource, allowing it to route requests to either the local exo node system or to external models via LiteLLM:

```python
# In exo/api/chatgpt_api.py

# Add imports
from exo.api.litellm_resource import LiteLLMResource
from typing import Dict, Optional, Union

class ChatGPTAPI:
    def __init__(
        self,
        node: Node,
        inference_engine_classname: str,
        response_timeout: int = 90,
        on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None,
        default_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        litellm_config: Optional[Dict[str, Dict[str, str]]] = None
    ):
        # Existing initialization code...
        
        # Add LiteLLM resources
        self.litellm_resources = {}
        self.litellm_config = litellm_config or {}
        
        # Initialize LiteLLM resources based on config
        for model_name, config in self.litellm_config.items():
            self.litellm_resources[model_name] = LiteLLMResource(
                model_name=config.get("model"),
                api_base=config.get("api_base"),
                api_key=config.get("api_key"),
                **{k: v for k, v in config.items() if k not in ["model", "api_base", "api_key"]}
            )
    
    async def _is_external_model(self, model_name: str) -> bool:
        """Determine if a model should be handled by LiteLLM or local exo."""
        # Check if it's explicitly configured for LiteLLM
        if model_name in self.litellm_resources:
            return True
            
        # Check if it has an explicit provider prefix (e.g., "openai/gpt-4")
        if "/" in model_name:
            provider = model_name.split("/")[0]
            return provider not in ["exo", "local"]
            
        # Default to local exo processing
        return False
    
    # Modify handle_post_chat_completions to use LiteLLM when appropriate
    async def handle_post_chat_completions(self, request):
        data = await request.json()
        if DEBUG >= 2: print(f"[ChatGPTAPI] Handling chat completions request from {request.remote}: {data}")
        stream = data.get("stream", False)
        chat_request = parse_chat_request(data, self.default_model)
        
        # Handle model name remapping for gpt- models
        if chat_request.model and chat_request.model.startswith("gpt-"):
            chat_request.model = self.default_model
            
        # Check if we should use LiteLLM for this model
        if await self._is_external_model(chat_request.model):
            return await self._handle_litellm_completion(chat_request, stream, request)
        
        # Existing local model handling code...
        # [rest of the method continues as before]
    
    async def _handle_litellm_completion(self, chat_request: ChatCompletionRequest, stream: bool, request):
        """Handle completion requests using LiteLLM."""
        model_name = chat_request.model
        
        # Get or create the LiteLLM resource
        if model_name not in self.litellm_resources:
            # If not explicitly configured, create a default resource
            provider, model = model_name.split("/", 1) if "/" in model_name else ("", model_name)
            self.litellm_resources[model_name] = LiteLLMResource(model_name=model_name)
        
        resource = self.litellm_resources[model_name]
        
        try:
            # Ensure resource is initialized
            if not resource.is_initialized:
                await resource.initialize()
                
            request_id = str(uuid.uuid4())
            messages = [message.to_dict() for message in chat_request.messages]
            
            # Add system prompt if set and not already present
            if self.system_prompt and not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            if stream:
                # Handle streaming
                response = web.StreamResponse(
                    status=200,
                    reason="OK",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                await response.prepare(request)
                
                try:
                    completion_generator = await resource.complete(
                        messages=messages,
                        temperature=chat_request.temperature,
                        stream=True,
                        request_id=request_id
                    )
                    
                    async for chunk in completion_generator:
                        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                    return response
                    
                except Exception as e:
                    if DEBUG >= 2:
                        print(f"[ChatGPTAPI] Error in LiteLLM streaming: {e}")
                        traceback.print_exc()
                    return web.json_response(
                        {"detail": f"Error processing prompt with LiteLLM: {str(e)}"},
                        status=500
                    )
            else:
                # Handle non-streaming
                try:
                    completion = await resource.complete(
                        messages=messages,
                        temperature=chat_request.temperature,
                        stream=False,
                        request_id=request_id
                    )
                    return web.json_response(completion)
                    
                except Exception as e:
                    if DEBUG >= 2:
                        print(f"[ChatGPTAPI] Error in LiteLLM completion: {e}")
                        traceback.print_exc()
                    return web.json_response(
                        {"detail": f"Error processing prompt with LiteLLM: {str(e)}"},
                        status=500
                    )
                    
        except Exception as e:
            if DEBUG >= 2:
                print(f"[ChatGPTAPI] Error initializing LiteLLM resource: {e}")
                traceback.print_exc()
            return web.json_response(
                {"detail": f"Error initializing LiteLLM for model {model_name}: {str(e)}"},
                status=500
            )
```

### 3. Update Main.py to Support LiteLLM Configuration

Modify `main.py` to accept LiteLLM configuration options, allowing users to specify external model providers:

```python
# In exo/main.py

# Add to parser arguments
parser.add_argument("--litellm-config", type=str, default=None, help="Path to LiteLLM configuration file")
parser.add_argument("--enable-external-models", action="store_true", help="Enable support for external models via LiteLLM")

# Later in the code during API initialization
litellm_config = None
if args.enable_external_models:
    if args.litellm_config and os.path.exists(args.litellm_config):
        try:
            with open(args.litellm_config, 'r') as f:
                litellm_config = json.load(f)
            print(f"Loaded LiteLLM configuration from {args.litellm_config}")
        except Exception as e:
            print(f"Error loading LiteLLM configuration: {e}")
    else:
        litellm_config = {}  # Empty config allows for dynamic creation of resources

api = ChatGPTAPI(
  node,
  node.inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model,
  system_prompt=args.system_prompt,
  litellm_config=litellm_config
)
```

### 4. Create LiteLLM Configuration Schema

Create a JSON schema for the LiteLLM configuration file:

```json
{
  "openai/gpt-4o": {
    "model": "openai/gpt-4o",
    "api_key": "sk-..."
  },
  "anthropic/claude-3-opus": {
    "model": "anthropic/claude-3-opus",
    "api_key": "sk-ant-..."
  },
  "ollama/llama3": {
    "model": "ollama/llama3",
    "api_base": "http://localhost:11434"
  },
  "local/llama-3-8b-instruct": {
    "model": "ollama/llama-3-8b-instruct",
    "api_base": "http://localhost:11434"
  }
}
```

## Implementation Steps

1. Install the LiteLLM package:
   ```
   pip install litellm
   ```

2. Create the `litellm_resource.py` file with the LiteLLMResource class

3. Modify `chatgpt_api.py` to support LiteLLM models

4. Update `main.py` to accept LiteLLM configuration

5. Create documentation for using external models

## Benefits of This Approach

1. **Unified API**: All models (local and external) accessible through the same API endpoint
2. **Non-Disruptive**: Maintains compatibility with existing code and doesn't require changes to the core architecture
3. **Leverages AsyncResource**: Uses exo's existing async resource pattern for proper lifecycle management
4. **Flexibility**: Allows users to choose between local exo models or external providers
5. **Simple Configuration**: Easy to add new model providers without code changes

## Example Usage

Once implemented, users could:

1. Start exo with external models:
   ```
   exo --enable-external-models --litellm-config ./litellm_config.json
   ```

2. Send requests using the familiar ChatGPT API format, but specifying different providers:
   ```
   curl http://localhost:52415/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "openai/gpt-4o",
       "messages": [{"role": "user", "content": "Hello!"}],
       "temperature": 0.7
     }'
   ```

3. Mix and match local exo models with external ones in the same application
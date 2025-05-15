"""
Route handlers for the chat API.

This module contains HTTP route handlers for the chat API.
"""

import asyncio
import json
import signal
import traceback
import uuid
from typing import Callable, Dict, List, Optional, Any

from aiohttp import web
from exo import DEBUG, VERSION
from exo.helpers import shutdown
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import build_base_shard, get_repo, model_cards, MODEL_CARDS, get_supported_models, get_pretty_name

from .models import (
    ChatCompletionRequest, 
    Message, 
    build_prompt, 
    generate_completion, 
    parse_chat_request,
    parse_message
)


class RouteHandlers:
    """
    Handles HTTP routes for the chat API.
    
    This class provides methods for handling various HTTP endpoints
    like chat completions, token encoding, etc.
    """
    
    def __init__(
        self,
        token_handler,
        node,
        inference_engine_classname: str,
        response_timeout: int = 90,
        default_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_chat_completion_request: Optional[Callable[[str, ChatCompletionRequest, str], None]] = None,
        litellm_adapter = None
    ):
        """
        Initialize the route handlers.
        
        Args:
            token_handler: Handler for token processing
            node: Node for processing
            inference_engine_classname: Inference engine class name
            response_timeout: Response timeout in seconds
            default_model: Default model to use
            system_prompt: System prompt to use
            on_chat_completion_request: Callback for chat completion requests
            litellm_adapter: Adapter for LiteLLM
        """
        self.token_handler = token_handler
        self.node = node
        self.inference_engine_classname = inference_engine_classname
        self.response_timeout = response_timeout
        self.default_model = default_model or "llama-3.2-1b"
        self.system_prompt = system_prompt
        self.on_chat_completion_request = on_chat_completion_request
        self.litellm_adapter = litellm_adapter
        
    def is_valid_model(self, model: str) -> bool:
        """
        Check if a model is valid (exists in model_cards or MODEL_CARDS).
        
        Args:
            model: The model ID to check
            
        Returns:
            True if the model is valid, False otherwise
        """
        return model in model_cards or model in MODEL_CARDS
    
    async def handle_healthcheck(self, request: web.Request) -> web.Response:
        """
        Handle healthcheck endpoint.
        
        Args:
            request: Request object
            
        Returns:
            Response indicating the API is healthy
        """
        return web.json_response({"status": "ok"})
    
    async def handle_quit(self, request: web.Request) -> web.Response:
        """
        Handle quit endpoint.
        
        Args:
            request: Request object
            
        Returns:
            Response confirming quit signal received
        """
        if DEBUG >= 1: 
            print("Received quit signal")
        
        response = web.json_response({"detail": "Quit signal received"}, status=200)
        await response.prepare(request)
        await response.write_eof()
        
        await shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server)
        
        return response
    
    async def handle_root(self, request: web.Request) -> web.FileResponse:
        """
        Handle root endpoint.
        
        Args:
            request: Request object
            
        Returns:
            File response with index.html
        """
        return web.FileResponse(self.static_dir / "index.html")
    
    async def handle_post_chat_token_encode(self, request: web.Request) -> web.Response:
        """
        Handle chat token encoding endpoint.
        
        Args:
            request: Request object
            
        Returns:
            Response with encoded tokens
        """
        data = await request.json()
        model = data.get("model", self.default_model)
        
        if model and model.startswith("gpt-"):  # Handle gpt- model requests
            model = self.default_model
            
        if not model or not self.is_valid_model(model):
            if DEBUG >= 1: 
                print(f"Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
            model = self.default_model
            
        shard = build_base_shard(model, self.inference_engine_classname)
        messages = [parse_message(msg) for msg in data.get("messages", [])]
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
        prompt = build_prompt(tokenizer, messages, data.get("tools", None))
        tokens = tokenizer.encode(prompt)
        
        return web.json_response({
            "length": len(prompt),
            "num_tokens": len(tokens),
            "encoded_tokens": tokens,
            "encoded_prompt": prompt,
        })
    
    async def handle_post_chat_completions(self, request: web.Request) -> web.Response:
        """
        Handle chat completions endpoint.
        
        Args:
            request: Request object
            
        Returns:
            Response with chat completion
        """
        data = await request.json()
        if DEBUG >= 2: 
            print(f"[ChatAPI] Handling chat completions request from {request.remote}: {data}")
            
        stream = data.get("stream", False)
        chat_request = parse_chat_request(data, self.default_model)
        tools = data.get("tools")
        
        # Use LiteLLM for all models when available, providing a unified interface
        use_litellm = False
        
        if self.litellm_adapter and self.litellm_adapter.is_initialized:
            # Check if this model is supported by LiteLLM adapter
            model_mapping = self.litellm_adapter.get_model_mapping_by_id(chat_request.model)
            
            if model_mapping:
                # Model is supported by LiteLLM - use the adapter
                if DEBUG >= 1: 
                    print(f"[ChatAPI] Routing {chat_request.model} request through LiteLLM")
                use_litellm = True
            elif not chat_request.model or not self.is_valid_model(chat_request.model):
                # Unknown model - try to use default
                if DEBUG >= 1: 
                    print(f"[ChatAPI] Unknown model: {chat_request.model}, trying default")
                
                # Try default model with LiteLLM
                default_mapping = self.litellm_adapter.get_model_mapping_by_id(self.default_model)
                if default_mapping:
                    chat_request.model = self.default_model
                    use_litellm = True
                else:
                    # Fall back to local default model
                    if DEBUG >= 1: 
                        print(f"[ChatAPI] Using local default model: {self.default_model}")
                    chat_request.model = self.default_model
        elif not chat_request.model or not self.is_valid_model(chat_request.model):
            # LiteLLM not available and unknown model - fall back to default
            if DEBUG >= 1: 
                print(f"[ChatAPI] Invalid model: {chat_request.model}. Defaulting to {self.default_model}")
            chat_request.model = self.default_model
                
        # For local models
        if not use_litellm:
            shard = build_base_shard(chat_request.model, self.inference_engine_classname)
            if not shard:
                supported_models = [model for model, info in model_cards.items() if self.inference_engine_classname in info.get("repo", {})]
                return web.json_response(
                    {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
                    status=400,
                )

        request_id = str(uuid.uuid4())
        
        if use_litellm:
            # Special handling for LiteLLM routed requests
            # We'll skip tokenization and use a different approach
            try:
                # Add system prompt if set
                if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
                    chat_request.messages.insert(0, Message("system", self.system_prompt))
                    
                # Convert to format needed by LiteLLM
                messages = [msg.to_dict() for msg in chat_request.messages]
                
                # Simplified prompt for visualization only
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                
                # Create token queue for this request
                self.token_handler.create_queue(request_id)
                
                # Notify callback if registered
                if self.on_chat_completion_request:
                    try:
                        self.on_chat_completion_request(request_id, chat_request, prompt)
                    except Exception as e:
                        if DEBUG >= 2: 
                            traceback.print_exc()
                        
                # We'll process this differently and return from here
                return await self._handle_litellm_request(
                    request, request_id, chat_request, messages, 
                    stream, self.response_timeout, tools
                )
                
            except Exception as e:
                if DEBUG >= 2: 
                    print(f"[ChatAPI] Error in LiteLLM processing: {e}")
                    traceback.print_exc()
                # Fall back to standard processing if there's an error
        
        # Standard local processing
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
        if DEBUG >= 4: 
            print(f"[ChatAPI] Resolved tokenizer: {tokenizer}")

        # Add system prompt if set
        if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
            chat_request.messages.insert(0, Message("system", self.system_prompt))

        prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools)
        if self.on_chat_completion_request:
            try:
                self.on_chat_completion_request(request_id, chat_request, prompt)
            except Exception as e:
                if DEBUG >= 2: 
                    traceback.print_exc()

        if DEBUG >= 2: 
            print(f"[ChatAPI] Processing prompt: {request_id=} {shard=} {prompt=}")

        try:
            # Create token queue
            self.token_handler.create_queue(request_id)
            
            # Create a task for processing the prompt
            prompt_task = asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id))
            
            # Start the timeout tracking
            try:
                await asyncio.wait_for(
                    asyncio.shield(prompt_task), 
                    timeout=self.response_timeout
                )
            except asyncio.TimeoutError:
                # Continue processing - we'll handle timeout more gracefully
                if DEBUG >= 1:
                    print(f"[ChatAPI] Initial prompt processing timed out but continuing: {request_id=}")
            except Exception as e:
                # For other exceptions during initial processing, log but continue
                # Some nodes might still be able to produce a response even if initial processing failed
                if DEBUG >= 1:
                    print(f"[ChatAPI] Warning: Initial prompt processing encountered an error: {str(e)}")
                    if DEBUG >= 2:
                        traceback.print_exc()

            if DEBUG >= 2: 
                print(f"[ChatAPI] Waiting for response to finish. timeout={self.response_timeout}s")

            if stream:
                return await self._handle_streaming_response(
                    request, request_id, chat_request, prompt, tokenizer
                )
            else:
                return await self._handle_non_streaming_response(
                    request_id, chat_request, prompt, tokenizer
                )
                
        except asyncio.TimeoutError:
            # Clean up on timeout
            self.token_handler.remove_queue(request_id)
            return web.json_response({"detail": "Response generation timed out"}, status=408)
        except Exception as e:
            # Clean up on other exceptions
            self.token_handler.remove_queue(request_id)
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({"detail": f"Error processing prompt: {str(e)}"}, status=500)
    
    async def _handle_streaming_response(
        self, 
        request: web.Request, 
        request_id: str, 
        chat_request: ChatCompletionRequest, 
        prompt: str, 
        tokenizer
    ) -> web.StreamResponse:
        """
        Handle streaming response for chat completions.
        
        Args:
            request: Request object
            request_id: Unique identifier for the request
            chat_request: Chat completion request
            prompt: Prompt text
            tokenizer: Tokenizer to use
            
        Returns:
            Streaming response
        """
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
            # Initialize first response timeout - slightly longer than subsequent ones
            response_timeout = self.response_timeout
            first_token_received = False
            consecutive_timeouts = 0
            max_consecutive_timeouts = 2  # How many consecutive token timeouts to allow
            
            # Stream tokens while waiting for inference to complete
            while True:
                if DEBUG >= 2: 
                    print(f"[ChatAPI] Waiting for token from queue: {request_id=}")
                try:
                    tokens, is_finished = await self.token_handler.wait_for_tokens(
                        request_id,
                        timeout=response_timeout
                    )
                    
                    # Reset timeout counter when we successfully get tokens
                    consecutive_timeouts = 0
                    
                    # Reduce timeout for subsequent tokens after first success
                    if not first_token_received:
                        first_token_received = True
                        response_timeout = min(response_timeout, 15.0)  # Shorter timeout after first token
                        
                    if DEBUG >= 2: 
                        print(f"[ChatAPI] Got token from queue: {request_id=} {tokens=} {is_finished=}")

                    eos_token_id = None
                    if not eos_token_id and hasattr(tokenizer, "eos_token_id"): 
                        eos_token_id = tokenizer.eos_token_id
                    if not eos_token_id and hasattr(tokenizer, "_tokenizer"): 
                        eos_token_id = tokenizer.special_tokens_map.get("eos_token_id")

                    finish_reason = None
                    if is_finished: 
                        finish_reason = "stop" if tokens[-1] == eos_token_id else "length"
                    if DEBUG >= 2: 
                        print(f"{eos_token_id=} {tokens[-1]=} {finish_reason=}")

                    completion = generate_completion(
                        chat_request,
                        tokenizer,
                        prompt,
                        request_id,
                        tokens,
                        True,  # stream=True
                        finish_reason,
                        "chat.completion",
                    )

                    await response.write(f"data: {json.dumps(completion)}\n\n".encode())

                    if is_finished:
                        break
                        
                except asyncio.TimeoutError:
                    consecutive_timeouts += 1
                    if DEBUG >= 2: 
                        print(f"[ChatAPI] Timeout waiting for token: {request_id=} ({consecutive_timeouts}/{max_consecutive_timeouts})")
                    
                    # Allow a certain number of consecutive timeouts before giving up
                    if consecutive_timeouts > max_consecutive_timeouts:
                        if first_token_received:
                            # If we already sent some tokens, finish the stream with a timeout notification
                            try:
                                timeout_msg = {"error": {"message": "Token generation timed out", "type": "timeout"}}
                                await response.write(f"data: {json.dumps(timeout_msg)}\n\n".encode())
                                await response.write(b'data: {"done": true}\n\n')
                            except Exception:
                                pass  # Ignore errors during error reporting
                            break
                        else:
                            # If no tokens were received yet, return a timeout error
                            await response.write_eof()
                            return web.json_response({"detail": "Response generation timed out"}, status=408)

            await response.write(b'data: {"done": true}\n\n')
            await response.write_eof()
            return response

        except asyncio.CancelledError:
            # Handle client disconnection gracefully
            if DEBUG >= 2: 
                print(f"[ChatAPI] Request was cancelled/client disconnected: {request_id=}")
            # We don't return a response here, just clean up
            raise
            
        except Exception as e:
            if DEBUG >= 2: 
                print(f"[ChatAPI] Error processing prompt: {e}")
                traceback.print_exc()
            
            # If we already started streaming, try to send an error message
            try:
                if first_token_received:
                    error_msg = {"error": {"message": f"Error: {str(e)}", "type": "internal_error"}}
                    await response.write(f"data: {json.dumps(error_msg)}\n\n".encode())
                    await response.write(b'data: {"done": true}\n\n')
                    await response.write_eof()
                    return response
            except Exception:
                pass  # Ignore errors during error reporting
                
            # Otherwise return a JSON error
            return web.json_response(
                {"detail": f"Error processing prompt: {str(e)}"},
                status=500
            )

        finally:
            # Clean up the queue for this request
            self.token_handler.remove_queue(request_id)
    
    async def _handle_non_streaming_response(
        self, 
        request_id: str, 
        chat_request: ChatCompletionRequest, 
        prompt: str, 
        tokenizer
    ) -> web.Response:
        """
        Handle non-streaming response for chat completions.
        
        Args:
            request_id: Unique identifier for the request
            chat_request: Chat completion request
            prompt: Prompt text
            tokenizer: Tokenizer to use
            
        Returns:
            JSON response with completion
        """
        tokens = []
        try:
            # Initialize timeout parameters for non-streaming mode
            response_timeout = self.response_timeout
            first_token_received = False
            consecutive_timeouts = 0
            max_consecutive_timeouts = 2
            
            while True:
                try:
                    _tokens, is_finished = await self.token_handler.wait_for_tokens(
                        request_id, 
                        timeout=response_timeout
                    )
                    
                    # Reset timeout counter and adjust timeout after first success
                    consecutive_timeouts = 0
                    if not first_token_received:
                        first_token_received = True
                        response_timeout = min(response_timeout, 15.0)
                    
                    tokens.extend(_tokens)
                    if is_finished:
                        break
                        
                except asyncio.TimeoutError:
                    consecutive_timeouts += 1
                    if DEBUG >= 2: 
                        print(f"[ChatAPI] Timeout waiting for token in non-streaming mode: {request_id=} ({consecutive_timeouts}/{max_consecutive_timeouts})")
                    
                    # Allow a limited number of consecutive timeouts
                    if consecutive_timeouts > max_consecutive_timeouts:
                        # If we already have tokens, return what we've got with timeout reason
                        if tokens:
                            if DEBUG >= 1: 
                                print(f"[ChatAPI] Returning partial response after timeout: {len(tokens)} tokens")
                            break
                        else:
                            # No tokens received yet, return timeout error
                            if DEBUG >= 1: 
                                print(f"[ChatAPI] No tokens received, returning timeout error")
                            return web.json_response({"detail": "Response generation timed out"}, status=408)
            
            # Determine finish reason
            finish_reason = "length"  # Default reason
            
            # Try to determine if we stopped due to EOS token
            eos_token_id = None
            if not eos_token_id and hasattr(tokenizer, "eos_token_id"): 
                eos_token_id = tokenizer.eos_token_id
            if not eos_token_id and hasattr(tokenizer, "_tokenizer"): 
                eos_token_id = tokenizer.special_tokens_map.get("eos_token_id")

            # Only check for EOS token if we actually have tokens
            if tokens:
                if DEBUG >= 2: 
                    print(f"Checking if end of tokens result {tokens[-1]=} is {eos_token_id=}")

                # Check if the last token is the EOS token
                if tokens[-1] == eos_token_id:
                    finish_reason = "stop"
                # If we had consecutive timeouts, mark as timeout
                elif consecutive_timeouts > 0:
                    finish_reason = "timeout"
            else:
                if DEBUG >= 1: 
                    print(f"Warning: No tokens generated for request {request_id}")
                # Use 'empty' reason when no tokens were generated
                finish_reason = "empty"

            return web.json_response(
                generate_completion(
                    chat_request, 
                    tokenizer, 
                    prompt, 
                    request_id, 
                    tokens, 
                    False,  # stream=False
                    finish_reason, 
                    "chat.completion"
                )
            )
        
        except asyncio.CancelledError:
            # Handle client disconnection gracefully
            if DEBUG >= 2: 
                print(f"[ChatAPI] Non-streaming request was cancelled: {request_id=}")
            raise
            
        except Exception as e:
            if DEBUG >= 2: 
                print(f"[ChatAPI] Error in non-streaming response: {e}")
                traceback.print_exc()
            return web.json_response({"detail": f"Error processing response: {str(e)}"}, status=500)
            
        finally:
            # Clean up the queue for this request
            self.token_handler.remove_queue(request_id)
    
    async def _handle_litellm_request(
        self, request: web.Request, request_id: str, chat_request: ChatCompletionRequest, 
        messages: List[Dict], stream: bool, timeout: float, tools: Optional[List[Dict]] = None
    ) -> web.Response:
        """
        Handle a request using LiteLLM.
        
        Args:
            request: Request object
            request_id: Unique identifier for the request
            chat_request: Chat completion request
            messages: List of message dictionaries
            stream: Whether to stream the response
            timeout: Response timeout in seconds
            tools: Optional tools for function calling
            
        Returns:
            Response with completion from LiteLLM
        """
        try:
            # Initialize LiteLLM adapter if needed
            if not self.litellm_adapter.is_initialized:
                await self.litellm_adapter.initialize()
            
            params = {
                "messages": messages,
                "temperature": chat_request.temperature,
                "stream": stream
            }
            
            if tools:
                params["tools"] = tools
            
            # For streaming responses
            if stream:
                response = web.StreamResponse(
                    status=200,
                    reason="OK",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                await response.prepare(request)
                
                # Create completion iterator using LiteLLM
                try:
                    all_content = ""
                    async for chunk in await self.litellm_adapter.litellm_service.get_completions(
                        model=chat_request.model,
                        **params
                    ):
                        # Format response according to ChatGPT API format
                        if not chunk["choices"]:
                            continue
                            
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            all_content += content
                            
                            # Generate a ChatGPT-like response
                            completion = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": chat_request.model,
                                "system_fingerprint": f"exo_litellm_{VERSION}",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }],
                            }
                            
                            await response.write(f"data: {json.dumps(completion)}\n\n".encode())
                        
                        # For final chunk
                        if chunk["choices"][0].get("finish_reason"):
                            # Send final chunk with finish reason
                            finish_reason = chunk["choices"][0]["finish_reason"]
                            completion = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": chat_request.model,
                                "system_fingerprint": f"exo_litellm_{VERSION}",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }],
                            }
                            await response.write(f"data: {json.dumps(completion)}\n\n".encode())
                            break
                            
                    # End the stream
                    # Use a proper JSON format for the DONE message
                    await response.write(b'data: {"done": true}\n\n')
                    return response
                    
                except Exception as e:
                    if DEBUG >= 1:
                        print(f"[ChatAPI] Error in LiteLLM streaming: {str(e)}")
                        if DEBUG >= 2: traceback.print_exc()
                    
                    # Try to send error message in stream
                    try:
                        error_msg = {"error": {"message": f"Error: {str(e)}", "type": "internal_error"}}
                        await response.write(f"data: {json.dumps(error_msg)}\n\n".encode())
                        await response.write(b'data: {"done": true}\n\n')
                        return response
                    except Exception:
                        # If we can't send error in stream, fall back to JSON response
                        return web.json_response(
                            {"detail": f"Error in LiteLLM streaming: {str(e)}"},
                            status=500
                        )
            
            else:
                # Non-streaming response
                try:
                    response = await self.litellm_adapter.litellm_service.get_completions(
                        model=chat_request.model,
                        **params
                    )
                    
                    # Format the response according to ChatGPT API format
                    completion = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": chat_request.model,
                        "system_fingerprint": f"exo_litellm_{VERSION}",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant", 
                                "content": response["choices"][0]["message"]["content"]
                            },
                            "finish_reason": response["choices"][0]["finish_reason"]
                        }],
                        "usage": response.get("usage", {
                            "prompt_tokens": 0,  # We don't have exact count
                            "completion_tokens": 0,
                            "total_tokens": 0
                        })
                    }
                    
                    return web.json_response(completion)
                    
                except Exception as e:
                    if DEBUG >= 1:
                        print(f"[ChatAPI] Error in LiteLLM completion: {str(e)}")
                        if DEBUG >= 2: traceback.print_exc()
                    return web.json_response(
                        {"detail": f"Error in LiteLLM completion: {str(e)}"},
                        status=500
                    )
        
        except Exception as e:
            if DEBUG >= 1:
                print(f"[ChatAPI] Error handling LiteLLM request: {str(e)}")
                if DEBUG >= 2: traceback.print_exc()
            return web.json_response(
                {"detail": f"Error handling LiteLLM request: {str(e)}"},
                status=500
            )
            
        finally:
            # Clean up the queue for this request
            self.token_handler.remove_queue(request_id)
"""
Model management endpoints for the chat API.

This module contains handlers for model-related endpoints like listing models,
getting model details, downloading models, and managing the model pool.
"""

import json
import traceback
import asyncio
from typing import Dict, List, Optional, Any

from aiohttp import web, ClientConnectionResetError
from exo import DEBUG, VERSION
from exo.models import (
    model_cards, get_supported_models, get_pretty_name, 
    build_full_shard, build_base_shard
)
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import delete_model

from .utils import extract_provider_info


class ModelEndpoints:
    """
    Endpoints for model management in the chat API.
    
    This class handles endpoints for listing models, getting model details,
    downloading models, and managing the model pool.
    """
    
    def __init__(
        self,
        node,
        inference_engine_classname: str,
        default_model: Optional[str] = None,
        litellm_adapter = None
    ):
        """
        Initialize the model endpoints.
        
        Args:
            node: Node for processing
            inference_engine_classname: Inference engine class name
            default_model: Default model to use
            litellm_adapter: Adapter for LiteLLM
        """
        self.node = node
        self.inference_engine_classname = inference_engine_classname
        self.default_model = default_model or "llama-3.2-1b"
        self.litellm_adapter = litellm_adapter
    
    async def handle_get_models(self, request: web.Request) -> web.Response:
        """
        Handle GET /models endpoint.
        
        This endpoint lists all available models.
        
        Args:
            request: Request object
            
        Returns:
            Response with list of models
        """
        # Start with local models
        models_list = []
        for model_name, info in model_cards.items():
            model_data = {
                "id": model_name, 
                "object": "model", 
                "owned_by": "exo", 
                "provider": "exo", 
                "ready": True,
                "capabilities": {
                    "context_window": info.get("context_length", 8192),
                    "max_tokens": info.get("max_tokens", 4096),
                    "supports_tools": info.get("supports_tools", False)
                },
                "metadata": {
                    "local": True,
                    "isCloudModel": False
                }
            }
            models_list.append(model_data)
        
        # Add models from LiteLLM if available
        if self.litellm_adapter and self.litellm_adapter.is_initialized:
            try:
                external_models = self.litellm_adapter.get_supported_models()
                # Only add models that aren't already in the list
                for model in external_models:
                    if not any(m["id"] == model["id"] for m in models_list):
                        # Ensure the model has a provider field
                        if "owned_by" in model and "provider" not in model:
                            if "openai" in model["owned_by"].lower():
                                model["provider"] = "openai"
                            elif "anthropic" in model["owned_by"].lower():
                                model["provider"] = "anthropic"
                            else:
                                model["provider"] = model["owned_by"]
                        
                        # Add capabilities information
                        model["capabilities"] = {
                            "context_window": model.get("context_window", 8192),
                            "max_tokens": model.get("max_tokens", 4096),
                            "supports_tools": model.get("supports_tools", False)
                        }
                            
                        # Ensure cloud models have clear identification
                        if model.get("isCloudModel"):
                            model["isExternal"] = True
                            
                            # Add display name if not present
                            if not model.get("display_name"):
                                model_id = model.get("id", "")
                                if model.get("provider") == "openai" or model_id.startswith("gpt-"):
                                    model["display_name"] = model_id.replace("gpt-", "GPT-")
                                elif model.get("provider") == "anthropic" or model_id.startswith("claude-"):
                                    model["display_name"] = model_id.replace("claude-", "Claude ")
                        
                        models_list.append(model)
            except Exception as e:
                if DEBUG >= 1: 
                    print(f"Error getting LiteLLM models: {e}")
        
        return web.json_response({"object": "list", "data": models_list})
    
    async def handle_get_initial_models(self, request: web.Request) -> web.Response:
        """
        Handle GET /initial_models endpoint.
        
        This endpoint provides initial model data for the UI.
        
        Args:
            request: Request object
            
        Returns:
            Response with initial model data
        """
        model_data = {}
        # Add local models
        for model_id in get_supported_models([[self.inference_engine_classname]]):
            # Get model info from model_cards if available
            model_info = model_cards.get(model_id, {})
            
            model_data[model_id] = {
                "name": get_pretty_name(model_id),
                "downloaded": None,  # Initially unknown
                "download_percentage": None,  # Change from 0 to null
                "total_size": None,
                "total_downloaded": None,
                "loading": True,  # Add loading state
                "provider": "exo",  # Add provider info for local models
                "isCloudModel": False,  # Explicitly mark as local model
                "capabilities": {
                    "context_window": model_info.get("context_length", 8192),
                    "max_tokens": model_info.get("max_tokens", 4096),
                    "supports_tools": model_info.get("supports_tools", False)
                }
            }
        
        # Add models from LiteLLM if available
        if self.litellm_adapter and self.litellm_adapter.is_initialized:
            try:
                external_models = self.litellm_adapter.get_supported_models()
                for model in external_models:
                    model_id = model["id"]
                    # Skip if already added
                    if model_id in model_data:
                        continue
                    
                    # Set provider based on owned_by or id
                    provider = model.get("metadata", {}).get("provider")
                    if not provider:
                        if "owned_by" in model:
                            if "openai" in model["owned_by"].lower():
                                provider = "openai"
                            elif "anthropic" in model["owned_by"].lower():
                                provider = "anthropic"
                            else:
                                provider = model["owned_by"]
                        elif model_id.startswith("gpt-"):
                            provider = "openai"
                        elif model_id.startswith("claude-"):
                            provider = "anthropic"
                        else:
                            provider = "external"
                            
                    model_data[model_id] = {
                        "name": model_id,
                        "downloaded": True,  # Always mark external models as downloaded/ready
                        "download_percentage": 100,  # Always 100% for external models
                        "total_size": None,
                        "total_downloaded": None,
                        "loading": False,
                        "provider": provider,
                        "isCloudModel": True,  # Mark as cloud model
                        "capabilities": {
                            "context_window": model.get("metadata", {}).get("context_length", 8192),
                            "max_tokens": model.get("metadata", {}).get("max_tokens", 4096),
                            "supports_tools": model.get("metadata", {}).get("supports_tools", False)
                        }
                    }
            except Exception as e:
                if DEBUG >= 1: 
                    print(f"Error getting LiteLLM models for initial models: {e}")
                    if DEBUG >= 2:
                        traceback.print_exc()
        
        return web.json_response(model_data)
    
    async def handle_model_support(self, request: web.Request) -> web.Response:
        """
        Handle GET /modelpool endpoint.
        
        This endpoint provides streaming updates about model status.
        
        Args:
            request: Request object
            
        Returns:
            Streaming response with model status updates
        """
        try:
            response = web.StreamResponse(
                status=200, 
                reason='OK', 
                headers={ 
                    'Content-Type': 'text/event-stream', 
                    'Cache-Control': 'no-cache', 
                    'Connection': 'keep-alive' 
                }
            )
            await response.prepare(request)
            
            try:
                # First, send local model information
                async for path, s in self.node.shard_downloader.get_shard_download_status(self.inference_engine_classname):
                    try:
                        # Get model info from model_cards if available
                        model_info = model_cards.get(s.shard.model_id, {})
                        
                        model_data = { 
                            s.shard.model_id: { 
                                "downloaded": s.downloaded_bytes == s.total_bytes, 
                                "download_percentage": 100 if s.downloaded_bytes == s.total_bytes else 100 * float(s.downloaded_bytes) / float(s.total_bytes), 
                                "total_size": s.total_bytes, 
                                "total_downloaded": s.downloaded_bytes,
                                "provider": "exo",  # Add provider info
                                "isCloudModel": False,  # Explicitly mark as local model
                                "capabilities": {
                                    "context_window": model_info.get("context_length", 8192),
                                    "max_tokens": model_info.get("max_tokens", 4096),
                                    "supports_tools": model_info.get("supports_tools", False)
                                }
                            } 
                        }
                        try:
                            await response.write(f"data: {json.dumps(model_data)}\n\n".encode())
                        except ClientConnectionResetError:
                            if DEBUG >= 2:
                                print("Client disconnected during model info streaming")
                            return response
                        except Exception as e:
                            if DEBUG >= 2:
                                print(f"Error writing to stream: {e}")
                            return response
                    except Exception as model_error:
                        if DEBUG >= 1:
                            print(f"Error processing model data: {model_error}")
                        continue
                
                # Then, if LiteLLM is available, send external model information
                if self.litellm_adapter and self.litellm_adapter.is_initialized:
                    try:
                        external_models = self.litellm_adapter.get_supported_models()
                        for model in external_models:
                            try:
                                model_id = model["id"]
                                
                                # Set provider based on owned_by or id
                                provider = model.get("metadata", {}).get("provider")
                                if not provider:
                                    if "owned_by" in model:
                                        if "openai" in model["owned_by"].lower():
                                            provider = "openai"
                                        elif "anthropic" in model["owned_by"].lower():
                                            provider = "anthropic"
                                        else:
                                            provider = model["owned_by"]
                                    elif model_id.startswith("gpt-"):
                                        provider = "openai"
                                    elif model_id.startswith("claude-"):
                                        provider = "anthropic"
                                    else:
                                        provider = "external"
                                
                                model_data = {
                                    model_id: {
                                        "name": model_id,
                                        "downloaded": True,  # Always mark online models as downloaded
                                        "download_percentage": 100,  # Always 100% for online models
                                        "total_size": None,
                                        "total_downloaded": None,
                                        "provider": provider,
                                        "isCloudModel": True,  # Mark as cloud model
                                        "capabilities": {
                                            "context_window": model.get("metadata", {}).get("context_length", 8192),
                                            "max_tokens": model.get("metadata", {}).get("max_tokens", 4096),
                                            "supports_tools": model.get("metadata", {}).get("supports_tools", False)
                                        }
                                    }
                                }
                                try:
                                    await response.write(f"data: {json.dumps(model_data)}\n\n".encode())
                                except ClientConnectionResetError:
                                    if DEBUG >= 2:
                                        print("Client disconnected during external model info streaming")
                                    return response
                                except Exception as e:
                                    if DEBUG >= 2:
                                        print(f"Error writing to stream: {e}")
                                    return response
                            except Exception as model_error:
                                if DEBUG >= 1:
                                    print(f"Error processing external model: {model_error}")
                                continue
                    except Exception as e:
                        if DEBUG >= 1:
                            print(f"Error adding LiteLLM models to stream: {e}")
                
                # End of stream with proper JSON format
                try:
                    await response.write(b'data: {"done": true}\n\n')
                except ClientConnectionResetError:
                    if DEBUG >= 2:
                        print("Client disconnected before end of stream")
                except Exception as e:
                    if DEBUG >= 2:
                        print(f"Error writing end of stream: {e}")
            except ClientConnectionResetError:
                if DEBUG >= 2:
                    print("Client disconnected during model info streaming")
                return response
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Error streaming model info: {e}")
                    if DEBUG >= 2:
                        traceback.print_exc()
            return response

        except Exception as e:
            print(f"Error in handle_model_support: {str(e)}")
            traceback.print_exc()
            return web.json_response({"detail": f"Server error: {str(e)}"}, status=500)
    
    async def handle_get_download_progress(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/download/progress endpoint.
        
        This endpoint provides information about ongoing downloads.
        
        Args:
            request: Request object
            
        Returns:
            Response with download progress
        """
        progress_data = {}
        for node_id, progress_event in self.node.node_download_progress.items():
            if isinstance(progress_event, RepoProgressEvent):
                if progress_event.status != "in_progress": continue
                progress_data[node_id] = progress_event.to_dict()
            else:
                print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
        return web.json_response(progress_data)
    
    async def handle_delete_model(self, request: web.Request) -> web.Response:
        """
        Handle DELETE /models/{model_name} endpoint.
        
        This endpoint deletes a model from the local storage.
        
        Args:
            request: Request object
            
        Returns:
            Response confirming deletion
        """
        model_id = request.match_info.get('model_name')
        try:
            if await delete_model(model_id, self.inference_engine_classname): 
                return web.json_response({
                    "status": "success", 
                    "message": f"Model {model_id} deleted successfully"
                })
            else: 
                return web.json_response({
                    "detail": f"Model {model_id} files not found"
                }, status=404)
        except Exception as e:
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({
                "detail": f"Error deleting model: {str(e)}"
            }, status=500)
    
    async def handle_post_download(self, request: web.Request) -> web.Response:
        """
        Handle POST /download endpoint.
        
        This endpoint initiates a model download.
        
        Args:
            request: Request object
            
        Returns:
            Response confirming download started
        """
        try:
            data = await request.json()
            model_name = data.get("model")
            if not model_name: 
                return web.json_response({
                    "error": "model parameter is required"
                }, status=400)
            if model_name not in model_cards: 
                return web.json_response({
                    "error": f"Invalid model: {model_name}. Supported models: {list(model_cards.keys())}"
                }, status=400)
            shard = build_full_shard(model_name, self.inference_engine_classname)
            if not shard: 
                return web.json_response({
                    "error": f"Could not build shard for model {model_name}"
                }, status=400)
            # Create an asyncio task to handle the download
            asyncio.create_task(
                self.node.inference_engine.shard_downloader.ensure_shard(
                    shard, self.inference_engine_classname
                )
            )

            return web.json_response({
                "status": "success", 
                "message": f"Download started for model: {model_name}"
            })
        except Exception as e:
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_topology(self, request: web.Request) -> web.Response:
        """
        Handle GET /v1/topology endpoint.
        
        This endpoint provides information about the node topology.
        
        Args:
            request: Request object
            
        Returns:
            Response with topology information
        """
        try:
            topology = self.node.current_topology
            if topology:
                return web.json_response(topology.to_json())
            else:
                return web.json_response({})
        except Exception as e:
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({
                "detail": f"Error getting topology: {str(e)}"
            }, status=500)
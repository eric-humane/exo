"""
Main API class for the chat API.

This module contains the main API class that composes all the other modules.
"""

import asyncio
from pathlib import Path
from typing import Callable, Optional

import aiohttp_cors
from aiohttp import web

from exo import DEBUG
from exo.orchestration import Node
from exo.helpers import get_exo_images_dir

from .token_handler import TokenHandler
from .route_handlers import RouteHandlers
from .model_endpoints import ModelEndpoints
from .image_endpoints import ImageEndpoints


class ChatAPI:
    """
    Main API class for the chat API.
    
    This class composes all the other modules and provides a unified interface
    for the chat API.
    """
    
    def __init__(
        self,
        node: Node,
        inference_engine_classname: str,
        response_timeout: int = 90,
        on_chat_completion_request: Optional[Callable] = None,
        default_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        enable_litellm: bool = True,
        litellm_config_path: Optional[str] = None
    ):
        """
        Initialize the chat API.
        
        Args:
            node: Node for processing
            inference_engine_classname: Inference engine class name
            response_timeout: Response timeout in seconds
            on_chat_completion_request: Callback for chat completion requests
            default_model: Default model to use
            system_prompt: System prompt to use
            enable_litellm: Whether to enable LiteLLM integration
            litellm_config_path: Path to LiteLLM config file
        """
        self.node = node
        self.inference_engine_classname = inference_engine_classname
        self.response_timeout = response_timeout
        self.on_chat_completion_request = on_chat_completion_request
        self.default_model = default_model or "llama-3.2-1b"
        self.system_prompt = system_prompt
        
        # Create web application
        self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
        
        # Initialize token handler
        self.token_handler = TokenHandler()
        
        # Initialize LiteLLM integration if enabled
        self.litellm_adapter = None
        if enable_litellm:
            try:
                if DEBUG >= 1: 
                    print("Initializing LiteLLM integration")
                from exo.api.litellm_service import LiteLLMService
                from exo.api.litellm_adapter import LiteLLMAdapter
                
                litellm_service = LiteLLMService(config_path=litellm_config_path)
                self.litellm_adapter = LiteLLMAdapter(node, litellm_service=litellm_service)
                # Don't await initialization here - we'll do it asynchronously in run()
            except Exception as e:
                if DEBUG >= 1: 
                    print(f"Failed to initialize LiteLLM integration: {e}")
                    if DEBUG >= 2: 
                        import traceback
                        traceback.print_exc()
        
        # Initialize endpoint modules
        self.route_handlers = RouteHandlers(
            self.token_handler,
            node,
            inference_engine_classname,
            response_timeout,
            default_model,
            system_prompt,
            on_chat_completion_request,
            self.litellm_adapter
        )
        
        self.model_endpoints = ModelEndpoints(
            node,
            inference_engine_classname,
            default_model,
            self.litellm_adapter
        )
        
        self.image_endpoints = ImageEndpoints(
            node,
            inference_engine_classname,
            response_timeout
        )
        
        # Get the callback system and register our handler
        self.token_callback = node.on_token.register("chatgpt-api-token-handler")
        self.token_callback.on_next_async(self.handle_tokens)
        
        # Add static routes
        if "__compiled__" not in globals():
            self.static_dir = Path(__file__).parent.parent.parent / "tinychat"
            self.route_handlers.static_dir = self.static_dir
        
        # Set up routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up routes for the API."""
        # Set up CORS with more secure configuration
        cors = aiohttp_cors.setup(self.app)
        
        # Get allowed origins from environment or default to localhost
        allowed_origins = self.get_allowed_origins()
        
        # Configure CORS options
        cors_options_map = {}
        for origin in allowed_origins:
            cors_options_map[origin] = aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers=["Content-Type", "Content-Length", "Accept", "X-Requested-With"],
                allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                max_age=3600,
            )
            
        # Define routes with their handlers
        routes = [
            # Model endpoints
            web.get("/models", self.model_endpoints.handle_get_models),
            web.get("/v1/models", self.model_endpoints.handle_get_models),
            web.get("/initial_models", self.model_endpoints.handle_get_initial_models),
            web.get("/modelpool", self.model_endpoints.handle_model_support),
            web.delete("/models/{model_name}", self.model_endpoints.handle_delete_model),
            web.post("/download", self.model_endpoints.handle_post_download),
            web.get("/v1/download/progress", self.model_endpoints.handle_get_download_progress),
            web.get("/v1/topology", self.model_endpoints.handle_get_topology),
            web.get("/topology", self.model_endpoints.handle_get_topology),
            
            # Chat/completions endpoints
            web.post("/chat/token/encode", self.route_handlers.handle_post_chat_token_encode),
            web.post("/v1/chat/token/encode", self.route_handlers.handle_post_chat_token_encode),
            web.post("/chat/completions", self.route_handlers.handle_post_chat_completions),
            web.post("/v1/chat/completions", self.route_handlers.handle_post_chat_completions),
            
            # Image endpoints
            web.post("/v1/image/generations", self.image_endpoints.handle_post_image_generations),
            web.post("/create_animation", self.image_endpoints.handle_create_animation),
            
            # Utility endpoints
            web.get("/healthcheck", self.route_handlers.handle_healthcheck),
            web.post("/quit", self.route_handlers.handle_quit),
        ]
        
        # Add routes with CORS
        for route_def in routes:
            route = self.app.router.add_route(route_def.method, route_def.path, route_def.handler)
            cors.add(route, cors_options_map)
        
        # Add static routes
        if "__compiled__" not in globals():
            self.app.router.add_get("/", self.route_handlers.handle_root)
            self.app.router.add_static("/", self.static_dir, name="static")
        
        # Always add images route, regardless of compilation status
        self.app.router.add_static("/images/", self.image_endpoints.images_dir, name="static_images")
        
        # Add middlewares
        self.app.middlewares.append(self._timeout_middleware)
        self.app.middlewares.append(self._log_request)
    
    async def _timeout_middleware(self, app, handler):
        """
        Middleware for timeout handling.
        
        Args:
            app: Application
            handler: Handler function
            
        Returns:
            Middleware function
        """
        async def middleware(request):
            try:
                return await asyncio.wait_for(
                    handler(request), 
                    timeout=self.response_timeout
                )
            except asyncio.TimeoutError:
                return web.json_response(
                    {"detail": "Request timed out"}, 
                    status=408
                )

        return middleware
    
    async def _log_request(self, app, handler):
        """
        Middleware for request logging.
        
        Args:
            app: Application
            handler: Handler function
            
        Returns:
            Middleware function
        """
        async def middleware(request):
            if DEBUG >= 2: 
                print(f"Received request: {request.method} {request.path}")
            return await handler(request)

        return middleware
    
    async def handle_tokens(self, request_id: str, tokens, is_finished: bool):
        """
        Handle tokens from the node.
        
        Args:
            request_id: Request ID
            tokens: Tokens
            is_finished: Whether this is the final batch
            
        Returns:
            None
        """
        return await self.token_handler.handle_tokens(request_id, tokens, is_finished)
    
    async def run(self, host: str = "0.0.0.0", port: int = 52415):
        """
        Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            
        Returns:
            None
        """
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        # Initialize LiteLLM adapter if available
        if self.litellm_adapter and not self.litellm_adapter.is_initialized:
            try:
                await self.litellm_adapter.initialize()
                if DEBUG >= 1: 
                    print("LiteLLM integration initialized successfully")
            except Exception as e:
                if DEBUG >= 1: 
                    print(f"Failed to initialize LiteLLM integration: {e}")
                    if DEBUG >= 2: 
                        import traceback
                        traceback.print_exc()
        
        print(f"Chat API server running at http://{host}:{port}")
    
    def get_allowed_origins(self):
        """
        Get the list of allowed origins for CORS.
        
        This method provides a secure default CORS configuration that:
        1. Only allows specific origins (not wildcards)
        2. Can be configured via the EXO_CORS_ALLOWED_ORIGINS environment variable
        3. Defaults to localhost origins if not specified
        
        The EXO_CORS_ALLOWED_ORIGINS should be a comma-separated list of allowed origins,
        e.g., "http://localhost:3000,https://example.com"
        
        Returns:
            List of allowed origins
        """
        import os
        
        # Get origins from environment variable, or use defaults
        cors_origins = os.environ.get("EXO_CORS_ALLOWED_ORIGINS", "")
        
        if cors_origins:
            # Split by comma and strip whitespace
            origins = [origin.strip() for origin in cors_origins.split(",")]
        else:
            # Default to common development and localhost origins
            origins = [
                "http://localhost:52415",
                "http://127.0.0.1:52415",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
            
            # Add local network address if running on a local network
            local_ip = os.environ.get("EXO_LOCAL_IP")
            if local_ip:
                origins.append(f"http://{local_ip}:52415")
                origins.append(f"https://{local_ip}:52415")
        
        if DEBUG >= 2:
            print(f"CORS allowed origins: {origins}")
        
        return origins
        
    async def cleanup(self):
        """
        Clean up resources.
        
        Returns:
            None
        """
        # Clean up token handler
        self.token_handler.cleanup()
        
        # Clean up LiteLLM adapter
        if self.litellm_adapter and self.litellm_adapter.is_initialized:
            await self.litellm_adapter.cleanup()
        
        # Deregister token callback
        if self.token_callback:
            self.node.on_token.deregister(self.token_callback)
            self.token_callback = None
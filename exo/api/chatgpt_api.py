"""
OpenAI-compatible ChatGPT API for exo.

This module provides an OpenAI-compatible API for the exo framework,
using the modular chat_api implementation.
"""

import asyncio
from typing import Callable, Optional

from exo.orchestration import Node
from exo.api.chat_api import ChatAPI

# Re-export the ChatAPI class
__all__ = ["ChatGPTAPI"]


class ChatGPTAPI(ChatAPI):
    """
    OpenAI-compatible API for chat completions.
    
    This class extends the modular ChatAPI implementation and provides
    backward compatibility with the original monolithic implementation.
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
        Initialize the ChatGPTAPI.
        
        Args:
            node: The exo node
            inference_engine_classname: Name of the inference engine class
            response_timeout: Timeout for responses in seconds
            on_chat_completion_request: Callback for completion requests
            default_model: Default model to use
            system_prompt: System prompt to use
            enable_litellm: Whether to enable LiteLLM integration
            litellm_config_path: Path to LiteLLM config file
        """
        super().__init__(
            node=node,
            inference_engine_classname=inference_engine_classname,
            response_timeout=response_timeout,
            on_chat_completion_request=on_chat_completion_request,
            default_model=default_model,
            system_prompt=system_prompt,
            enable_litellm=enable_litellm,
            litellm_config_path=litellm_config_path
        )
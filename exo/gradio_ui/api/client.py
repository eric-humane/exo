"""
API client for interacting with exo's OpenAI-compatible API endpoints.
"""

import json
import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union

import aiohttp
import requests


class ExoAPIClient:
    """
    Client for interacting with exo's OpenAI-compatible API endpoints.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the ExoAPIClient.
        
        Args:
            base_url: Base URL for the API. If None, uses the current host.
        """
        self.base_url = base_url or "http://localhost:8000"
        self.api_prefix = "/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest"
        })
    
    def get_endpoint_url(self, endpoint: str) -> str:
        """
        Get the full URL for an API endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL
        """
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        
        if endpoint.startswith("v1/"):
            return f"{self.base_url}/{endpoint}"
        else:
            return f"{self.base_url}/{self.api_prefix}/{endpoint}"
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get available models.
        
        Returns:
            Dictionary of model information
        """
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_initial_models(self) -> Dict[str, Any]:
        """
        Get initial models information.
        
        Returns:
            Dictionary of local model information
        """
        response = self.session.get(f"{self.base_url}/initial_models")
        response.raise_for_status()
        return response.json()
    
    def get_topology(self) -> Dict[str, Any]:
        """
        Get network topology information.
        
        Returns:
            Dictionary of topology information
        """
        response = self.session.get(f"{self.base_url}/topology")
        response.raise_for_status()
        return response.json()
    
    def get_download_progress(self) -> Dict[str, Any]:
        """
        Get download progress information.
        
        Returns:
            Dictionary of download progress
        """
        response = self.session.get(self.get_endpoint_url("download/progress"))
        response.raise_for_status()
        return response.json()
    
    def start_model_download(self, model_name: str) -> Dict[str, Any]:
        """
        Start downloading a model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Response from the server
        """
        response = self.session.post(
            f"{self.base_url}/download",
            json={"model": model_name}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a downloaded model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Response from the server
        """
        response = self.session.delete(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        stream: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion (non-streaming).
        
        Args:
            model: Model name
            messages: List of messages
            stream: Whether to stream the response
            **kwargs: Additional parameters for the API
            
        Returns:
            Chat completion response
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        response = self.session.post(
            self.get_endpoint_url("chat/completions"),
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def stream_chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion.
        
        Args:
            model: Model name
            messages: List of messages
            **kwargs: Additional parameters for the API
            
        Yields:
            Chunks of the response
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.get_endpoint_url("chat/completions"),
                json=data,
                headers={
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest"
                }
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    if line.startswith('data:'):
                        line = line[5:].strip()
                    
                    if line == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(line)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue
    
    def generate_image(
        self, 
        model: str, 
        prompt: str, 
        image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image.
        
        Args:
            model: Model name
            prompt: Text prompt
            image_url: Optional URL of an input image
            
        Returns:
            Image generation response
        """
        data = {
            "model": model,
            "prompt": prompt
        }
        
        if image_url:
            data["image_url"] = image_url
        
        response = self.session.post(
            self.get_endpoint_url("image/generations"),
            json=data
        )
        response.raise_for_status()
        return response.json()


# Singleton instance for the API client
api_client = None

def get_api_client(base_url: str = None) -> ExoAPIClient:
    """
    Get the API client instance.
    
    Args:
        base_url: Optional base URL for the API
        
    Returns:
        API client instance
    """
    global api_client
    if api_client is None:
        api_client = ExoAPIClient(base_url)
    return api_client
"""
Tests for the modular Chat API implementation.
"""

import unittest
from unittest.mock import MagicMock, patch

from exo.api.chat_api.api import ChatAPI
from exo.api.chat_api.models import ChatCompletionRequest, ChatCompletionResponse


class TestChatAPI(unittest.TestCase):
    """Test cases for the ChatAPI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_node.on_token = MagicMock()
        self.mock_node.on_token.register = MagicMock()
        
        # Create a ChatAPI instance with mocked dependencies
        self.api = ChatAPI(
            node=self.mock_node,
            inference_engine_classname="DummyInferenceEngine",
            response_timeout=30,
            enable_litellm=False
        )
    
    @patch('exo.api.chat_api.route_handlers.RouteHandlers.handle_post_chat_completions')
    async def test_chat_completions_handler_registered(self, mock_handler):
        """Test that chat completions handler is correctly registered."""
        # Check that the endpoint route is registered
        routes = [route.path for route in self.api.app.router.routes()]
        self.assertIn("/chat/completions", routes)
        self.assertIn("/v1/chat/completions", routes)
    
    @patch('exo.api.chat_api.model_endpoints.ModelEndpoints.handle_get_models')
    async def test_models_endpoint_registered(self, mock_handler):
        """Test that models endpoint is correctly registered."""
        # Check that the endpoint route is registered
        routes = [route.path for route in self.api.app.router.routes()]
        self.assertIn("/models", routes)
        self.assertIn("/v1/models", routes)
    
    def test_token_callback_registered(self):
        """Test that token callback is registered with the node."""
        self.mock_node.on_token.register.assert_called_once_with("chatgpt-api-token-handler")
    
    @patch('aiohttp.web.AppRunner')
    @patch('aiohttp.web.TCPSite')
    async def test_run_initializes_server(self, mock_site, mock_runner):
        """Test that run method initializes the server correctly."""
        # Mock the setup and start methods
        mock_runner_instance = MagicMock()
        mock_runner.return_value = mock_runner_instance
        mock_site_instance = MagicMock()
        mock_site.return_value = mock_site_instance
        
        # Call the run method
        await self.api.run(host="127.0.0.1", port=8080)
        
        # Check that the server is initialized correctly
        mock_runner.assert_called_once_with(self.api.app)
        mock_runner_instance.setup.assert_called_once()
        mock_site.assert_called_once_with(mock_runner_instance, "127.0.0.1", 8080)
        mock_site_instance.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
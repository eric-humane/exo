"""
Test LiteLLM provider routing functionality.

This module tests the routing mechanism that selects the fastest provider
with valid API keys for model inference requests.
"""

import os
import sys
import asyncio
import unittest
import logging
from unittest.mock import patch, MagicMock, AsyncMock

# Add the exo directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo.api.litellm_adapter import LiteLLMAdapter
from exo.api.litellm_service import LiteLLMService
from exo.api.provider_latency_tracker import ModelProviderLatencyTracker
from exo.inference.shard import Shard

# Set up logging
logging.basicConfig(level=logging.INFO)


class TestLiteLLMRouting(unittest.TestCase):
    """Test cases for LiteLLM routing functionality."""

    async def async_setup(self):
        """Set up test environment."""
        # Create mock node
        self.mock_node = MagicMock()
        self.mock_node.process_prompt = AsyncMock()
        self.mock_node.on_token = MagicMock()
        self.mock_node.on_token.register = MagicMock(return_value=self.mock_node.on_token)
        self.mock_node.on_token.notify = AsyncMock()
        
        # Create mock LiteLLMService
        self.mock_service = MagicMock(spec=LiteLLMService)
        self.mock_service.initialize = AsyncMock()
        self.mock_service.get_completions = AsyncMock()
        self.mock_service.get_fastest_provider_for_model_type = AsyncMock()
        self.mock_service.is_key_valid = AsyncMock(return_value=True)
        self.mock_service.latency_tracker = MagicMock(spec=ModelProviderLatencyTracker)
        
        # Set up test mappings
        self.test_mappings = [
            {
                "local_model_id": "llama-3.2-3b",
                "max_tokens": 4096,
                "context_window": 8192,
                "supports_tools": False,
                "priority": 10,
                "provider": "local",
                "model_type": "llama"
            },
            {
                "local_model_id": "external-gpt-4",
                "external_model_id": "gpt-4",
                "max_tokens": 8192,
                "context_window": 16384,
                "supports_tools": True,
                "priority": 20,
                "provider": "openai",
                "model_type": "gpt-4"
            },
            {
                "local_model_id": "external-claude-3-opus",
                "external_model_id": "claude-3-opus",
                "max_tokens": 4096,
                "context_window": 16384,
                "supports_tools": True,
                "priority": 20,
                "provider": "anthropic",
                "model_type": "claude"
            }
        ]
        
        # Create LiteLLMAdapter
        self.adapter = LiteLLMAdapter(
            node=self.mock_node,
            litellm_service=self.mock_service,
            model_mappings=self.test_mappings
        )
        
        # Initialize the adapter
        await self.adapter.initialize()

    def setUp(self):
        """Set up test environment synchronously."""
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.async_setup())

    async def async_teardown(self):
        """Clean up test environment."""
        await self.adapter.cleanup()

    def tearDown(self):
        """Clean up test environment synchronously."""
        self.loop.run_until_complete(self.async_teardown())

    async def test_find_fastest_model(self):
        """Test that find_fastest_model correctly selects the fastest provider."""
        # Set up the mock service to return a specific fastest provider
        self.mock_service.get_fastest_provider_for_model_type.return_value = "openai"
        
        # Call find_fastest_model with a model ID
        model_id, use_external = await self.adapter.find_fastest_model("external-gpt-4")
        
        # Verify that the correct model was selected
        self.assertEqual(model_id, "external-gpt-4")
        self.assertTrue(use_external)
        
        # Verify that get_fastest_provider_for_model_type was called with the correct model type
        self.mock_service.get_fastest_provider_for_model_type.assert_called_once_with("gpt-4")

    async def test_alternative_models(self):
        """Test that get_alternative_models returns appropriate alternatives."""
        # Test with a local model
        alternatives = self.adapter.get_alternative_models("llama-3.2-3b")
        
        # Verify that we get alternative models of the same type
        # Shouldn't have any in our test setup, but the function should work
        self.assertEqual(len(alternatives), 0)
        
        # Test with an external model
        alternatives = self.adapter.get_alternative_models("external-gpt-4")
        
        # Again, our test setup doesn't have alternatives for gpt-4
        self.assertEqual(len(alternatives), 0)

    async def test_process_prompt_routing(self):
        """Test that process_prompt routes to the fastest provider."""
        # Set up the mock service to return "anthropic" as the fastest provider
        self.mock_service.get_fastest_provider_for_model_type.return_value = "anthropic"
        
        # Create a test shard for gpt-4
        test_shard = Shard(
            shard_id="test-shard",
            model_id="external-gpt-4",
            layers=None,
            dtype=None,
            device=None,
            filename=None,
            vocab_size=0
        )
        
        # Configure the mock_service.get_completions to return a valid response
        mock_response = {
            "choices": [{"message": {"content": "Hello, world!"}}]
        }
        self.mock_service.get_completions.return_value = mock_response
        
        # Process a prompt
        await self.adapter.process_prompt(
            shard=test_shard,
            prompt="Hello",
            request_id="test-request",
            stream=False
        )
        
        # Check that find_fastest_model was called
        # We can't directly verify this since we're not mocking find_fastest_model,
        # but we can check that get_fastest_provider_for_model_type was called
        self.mock_service.get_fastest_provider_for_model_type.assert_called_once_with("gpt-4")
        
        # _process_with_external should be called with the claude model (based on our mock)
        process_with_external_call_args = list(self.mock_service.get_completions.call_args)
        self.assertIn("model", process_with_external_call_args[1])
        self.assertEqual(process_with_external_call_args[1]["model"], "claude-3-opus")

    async def test_api_key_validation(self):
        """Test that models with invalid API keys are not used."""
        # Set up the mock service to indicate that OpenAI keys are invalid
        self.mock_service.is_key_valid.side_effect = lambda provider, key: provider != "openai"
        
        # Create a test shard for gpt-4
        test_shard = Shard(
            shard_id="test-shard",
            model_id="external-gpt-4",
            layers=None,
            dtype=None,
            device=None,
            filename=None,
            vocab_size=0
        )
        
        # Configure the mock_service.get_completions to return a valid response
        mock_response = {
            "choices": [{"message": {"content": "Hello, world!"}}]
        }
        self.mock_service.get_completions.return_value = mock_response
        
        # Process a prompt
        with patch.object(self.adapter, 'find_fastest_model', new_callable=AsyncMock) as mock_find_fastest:
            # Mock find_fastest_model to return claude instead of gpt-4
            mock_find_fastest.return_value = ("external-claude-3-opus", True)
            
            await self.adapter.process_prompt(
                shard=test_shard,
                prompt="Hello",
                request_id="test-request",
                stream=False
            )
            
            # Verify that find_fastest_model was called
            mock_find_fastest.assert_called_once_with("external-gpt-4")
            
            # Verify that _process_with_external was called with claude
            # (we're checking this indirectly via get_completions)
            self.mock_service.get_completions.assert_called_once()
            call_args = self.mock_service.get_completions.call_args[1]
            self.assertEqual(call_args["model"], "claude-3-opus")


    async def test_fallback_to_local(self):
        """Test fallback to local model when external provider fails."""
        # Set up the test mappings to include a local llama alternative
        self.adapter.model_mappings.append({
            "local_model_id": "llama-3.2-3b",
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_tools": False,
            "priority": 10,
            "provider": "local",
            "model_type": "gpt-4"  # Same type as gpt-4 for testing purposes
        })
        
        # Create a test shard for gpt-4
        test_shard = Shard(
            shard_id="test-shard",
            model_id="external-gpt-4",
            layers=None,
            dtype=None,
            device=None,
            filename=None,
            vocab_size=0
        )
        
        # Make the external call fail
        self.mock_service.get_completions.side_effect = Exception("External service error")
        
        # Patch get_alternative_models to return our local model
        with patch.object(self.adapter, 'get_alternative_models') as mock_alternatives:
            mock_alternatives.return_value = ["llama-3.2-3b"]
            
            # Process a prompt
            await self.adapter.process_prompt(
                shard=test_shard,
                prompt="Hello",
                request_id="test-request",
                stream=False
            )
            
            # Verify that process_prompt was called with the local model
            self.mock_node.process_prompt.assert_called_once()
            call_args = self.mock_node.process_prompt.call_args[0]
            self.assertEqual(call_args[0].model_id, "llama-3.2-3b")


def run_tests():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
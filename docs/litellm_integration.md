# LiteLLM Integration for exo

This document describes the integration between exo and LiteLLM, allowing exo to seamlessly work with both local and cloud-hosted language models.

## Overview

The LiteLLM integration allows exo to:

1. Route requests to external providers (OpenAI, Anthropic, etc.) through a unified API
2. Load balance between local exo nodes and external API providers
3. Intelligently route requests to the fastest client with valid API keys
4. Fallback to external providers when local models aren't available
5. Use LiteLLM's standardized error handling and retry logic
6. Support function calling and tools across different model providers

## Architecture

The integration consists of three main components:

1. **LiteLLMService**: A service class implementing the AsyncResource pattern that manages connections to LiteLLM.
2. **LiteLLMAdapter**: An adapter that bridges exo's Node system with the LiteLLM service.
3. **ChatGPTAPI**: Enhanced to use the LiteLLM adapter for external models.

![Architecture Diagram]

## Installation

To use the LiteLLM integration, you need to install the LiteLLM package:

```bash
pip install litellm
```

## Configuration

### Command Line Arguments

The integration adds the following command line arguments to exo:

- `--enable-litellm`: Enable or disable LiteLLM integration (default: enabled)
- `--litellm-config`: Path to a LiteLLM configuration file

Example:

```bash
exo --enable-litellm --litellm-config=~/litellm_config.json
```

### Configuration File

You can configure the LiteLLM integration using a JSON configuration file. Here's an example:

```json
{
  "litellm_settings": {
    "set_verbose": false,
    "cache": false,
    "timeout": 120
  },
  "default_model": "gpt-3.5-turbo",
  "router": {
    "num_retries": 3,
    "retry_delay": 0.5,
    "timeout": 30
  },
  "model_deployments": [
    {
      "model_name": "gpt-4",
      "litellm_params": {
        "model": "gpt-4",
        "api_key": "${OPENAI_API_KEY}"
      }
    },
    {
      "model_name": "claude-3-opus",
      "litellm_params": {
        "model": "anthropic/claude-3-opus-20240229",
        "api_key": "${ANTHROPIC_API_KEY}"
      }
    },
    {
      "model_name": "ollama-llama3",
      "litellm_params": {
        "model": "ollama/llama3",
        "api_base": "http://localhost:11434"
      }
    }
  ]
}
```

The configuration file supports:

- **litellm_settings**: Global settings for LiteLLM
- **default_model**: Default model to use when none is specified
- **router**: Settings for the LiteLLM router (retries, timeouts, etc.)
- **model_deployments**: List of model deployments to register

### Environment Variables

The integration will automatically use these environment variables if available:

- `OPENAI_API_KEY`: API key for OpenAI
- `ANTHROPIC_API_KEY`: API key for Anthropic

## Usage

Once configured, the integration works automatically. When a request comes in through the ChatGPT API:

1. If the model is a local exo model, it's processed by exo nodes
2. If the model is a cloud model (starts with "gpt-", "claude-", etc.), it's routed to LiteLLM
3. If the model is unknown, it tries LiteLLM first, then falls back to local

### Examples

Using a local model:
```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Using an external model:
```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Advanced Features

### Distributed Load Balancing

You can set up multiple LiteLLM deployments across different exo nodes to distribute load. Each node can:

1. Handle local model inference for its assigned layers
2. Route external API requests via LiteLLM
3. Fallback to other nodes/providers when overloaded

### Function Calling / Tools

The integration supports OpenAI's function calling format. Example:

```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

## Smart Routing

The LiteLLM integration includes intelligent routing capabilities:

1. **Latency-Based Routing**: Automatically routes requests to the fastest provider based on recent latency measurements
2. **API Key Validation**: Only routes to providers with valid API keys
3. **Provider Failover**: Automatically falls back to alternative providers when primary providers fail
4. **Model Type Matching**: Finds equivalent models across providers for routing flexibility

When a request is made:
1. The adapter identifies the fastest provider with valid API keys for the requested model type
2. If the requested model is from that provider, it's used directly
3. If not, the adapter finds an equivalent model from the fastest provider
4. If external providers fail, the adapter falls back to local models when possible

This provides optimal performance while maintaining reliability.

## Limitations

Current limitations of the LiteLLM integration:

1. Tokenization for external models isn't precise, so token counts may be estimates
2. Function calling is only supported with models that natively support it
3. Some provider-specific parameters aren't exposed in the API
4. Latency tracking requires multiple requests to build accurate performance profiles

## Troubleshooting

If you encounter issues:

1. Enable debug logging with `DEBUG=2 exo`
2. Check the LiteLLM configuration file for errors
3. Verify API keys are correct and have sufficient permissions
4. Ensure network connectivity to external API providers
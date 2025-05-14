# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation and Setup

```bash
# Fresh installation with venv (recommended)
source install.sh

# Alternative manual installation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Setup for Mac M1/M2/M3 optimization
./configure_mlx.sh
```

### Running exo

```bash
# Run basic exo instance with automatic discovery
exo

# Run a specific model with a prompt
exo run llama-3.2-3b --prompt "Your prompt here"

# Run with different inference engine
exo --inference-engine tinygrad

# Run with LiteLLM integration for external models
exo --litellm-config=~/litellm_config.json

# Run with debug logs
DEBUG=9 exo
TINYGRAD_DEBUG=2 exo  # For tinygrad specific debugging
```

### Testing

```bash
# Run specific test file
python3 -m exo.inference.test_inference_engine

# Run tokenizer and model helper tests
python3 ./test/test_tokenizers.py
python3 ./test/test_model_helpers.py

# Run tinygrad tests
RUN_TINYGRAD=1 python3 -m exo.inference.test_inference_engine

# Run async improvements tests
./run_async_tests.sh
# Or directly:
python3 -m exo.tests.test_async_improvements
```

### Code Formatting

```bash
# Install formatting requirements
pip install -e '.[formatting]'

# Format the codebase
python3 format.py ./exo
```

### Building for Distribution

```bash
# Build the exo package with Nuitka
python3 scripts/build_exo.py
```

## Architecture Overview

exo is a framework for running AI models across distributed devices. It's designed to allow running LLMs across multiple heterogeneous devices by partitioning the model.

### Key Components

1. **Node System**: The core of the distributed architecture, implemented in `exo/orchestration/node.py`. Each device becomes a node in the network, connecting peer-to-peer rather than using a master-worker architecture.

2. **Discovery Modules**: Mechanisms for nodes to discover each other:
   - UDP Discovery (`exo/networking/udp/udp_discovery.py`) - Default discovery mechanism
   - Tailscale Discovery (`exo/networking/tailscale/tailscale_discovery.py`)
   - Manual Discovery (`exo/networking/manual/manual_discovery.py`)

3. **Inference Engines**: Backends that handle model execution:
   - MLX (`exo/inference/mlx/sharded_inference_engine.py`) - For Apple Silicon
   - tinygrad (`exo/inference/tinygrad/inference.py`) - Cross-platform
   - Support for others is in development (PyTorch, llama.cpp)

4. **Model Partitioning**: Implements strategies for splitting models across devices:
   - Ring Memory Weighted (`exo/topology/ring_memory_weighted_partitioning_strategy.py`) - Default strategy that allocates layers based on device memory

5. **API Layer**: Provides a ChatGPT-compatible API (`exo/api/chatgpt_api.py`) for integrating with applications.

6. **LiteLLM Integration**: Connects to cloud-based LLM providers (`exo/api/litellm_service.py` and `exo/api/litellm_adapter.py`).

7. **Download System**: Handles downloading model shards from Hugging Face, found in `exo/download/`.

### Data Flow

1. A client sends a prompt to any node via the API
2. The receiving node coordinates with other nodes to partition the model
3. Each node downloads its assigned model shards (if not already cached)
4. Inference runs in a ring topology with each node processing its layers
5. Results are streamed back to the client

## Development Guidelines

1. **Python Version**: Python 3.12+ is required (asyncio issues in earlier versions)

2. **Environment Variables**:
   - `DEBUG`: Set logging level (0-9)
   - `TINYGRAD_DEBUG`: Set tinygrad logging level (1-6)
   - `EXO_HOME`: Set custom model storage location (default: `~/.cache/exo/downloads`)
   - `HF_ENDPOINT`: Set custom Hugging Face endpoint
   - `OPENAI_API_KEY`: API key for OpenAI (when using LiteLLM integration)
   - `ANTHROPIC_API_KEY`: API key for Anthropic (when using LiteLLM integration)

3. **Hardware Support**:
   - Apple Silicon: Uses MLX by default
   - NVIDIA GPUs: Uses tinygrad by default
   - AMD GPUs: Uses tinygrad by default
   - CPUs: Fallback for all platforms

4. **Resource Management**:
   - Use the AsyncResource pattern for all resource lifecycle management
   - Call `ensure_ready()` before using a resource
   - See `docs/async_resource_pattern.md` for guidance
   - Implement `_do_initialize()`, `_do_cleanup()`, and `_do_health_check()` methods
   - Follow the standardization plan in `docs/async_resource_standardization.md`

5. **Platform-Specific Notes**:
   - For Apple Silicon, run `./configure_mlx.sh` to optimize GPU memory allocation
   - For Linux with NVIDIA, ensure CUDA and cuDNN are properly installed

6. **LiteLLM Integration**:
   - See `docs/litellm_integration.md` for detailed documentation
   - Configure with a JSON file (example in `exo/api/litellm_config_example.json`)
   - Set API keys via environment variables or config file
   - Use `--enable-litellm=false` to disable the integration if needed
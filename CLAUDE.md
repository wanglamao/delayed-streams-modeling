# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the Delayed Streams Modeling (DSM) repository by Kyutai Labs, containing Speech-to-Text (STT) and Text-to-Speech (TTS) models. It provides research implementations and production-ready components for real-time speech processing using the Delayed Streams Modeling technique.

**Models:**
- `kyutai/stt-1b-en_fr` - 1B params, English+French, 0.5s delay, semantic VAD
- `kyutai/stt-2.6b-en` - 2.6B params, English-only, 2.5s delay
- `kyutai/tts-1.6b-en_fr` - 1.6B params, English+French TTS

## Development Commands

### Code Quality

This repository uses Ruff for linting and formatting, managed via pre-commit hooks:

```bash
# Install pre-commit hooks
uvx pre-commit install

# Run linting and formatting checks manually
uvx ruff check
uvx ruff format --check

# Auto-fix linting issues and format
uvx ruff check --fix
uvx ruff format
```

### Rust Development (stt-rs/)

```bash
cd stt-rs

# Check, lint, and test
cargo check
cargo clippy
cargo test

# Build and run standalone STT client
cargo run --features cuda -r -- ../audio/bria.mp3

# With timestamps and VAD
cargo run --features cuda -r -- ../audio/bria.mp3 --timestamps --vad
```

### Running Python Scripts

All Python scripts use PEP 723 inline dependencies and are run with `uv`:

```bash
# STT with word-level timestamps
uv run scripts/stt_from_file_pytorch.py --hf-repo kyutai/stt-2.6b-en audio/bria.mp3

# STT with semantic VAD (1B model only)
uv run scripts/stt_from_file_pytorch.py --hf-repo kyutai/stt-1b-en_fr --vad audio/bria.mp3

# STT with prompting (spelling adaptation)
uv run scripts/stt_from_file_with_prompt_pytorch.py \
  --hf-repo kyutai/stt-2.6b-en \
  --file audio/bria.mp3 \
  --prompt_file audio/loona.mp3 \
  --prompt_text "Loonah"

# Dataset evaluation
uv run scripts/stt_evaluate_on_dataset.py --dataset meanwhile --hf-repo kyutai/stt-2.6b-en

# TTS from text file
uv run scripts/tts_pytorch.py text_input.txt audio_output.wav

# Streaming TTS
echo "text to speak" | uv run scripts/tts_pytorch_streaming.py audio_output.wav
```

### Running the Rust Server

The production Rust server is in the separate `moshi-server` crate (from the kyutai-labs/moshi repo):

```bash
# Install the server
cargo install --features cuda moshi-server

# Run STT server
moshi-server worker --config configs/config-stt-en_fr-hf.toml

# Run TTS server
moshi-server worker --config configs/config-tts.toml

# Connect to server from Python
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3
uv run scripts/tts_rust_server.py text_input.txt audio_output.wav
```

### Quick Module Commands

```bash
# Using moshi module directly (no script needed)
uvx --with moshi python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en audio/bria.mp3

# Using moshi-mlx for Apple Silicon
uvx --with moshi-mlx python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx audio/bria.mp3 --temp 0
```

## Architecture

### Three Implementation Paths

1. **PyTorch** (`scripts/*_pytorch.py`, `*_pytorch.ipynb`)
   - For research and experimentation
   - Direct tensor streaming with `mimi.streaming()` and `lm_gen.streaming()`
   - Dependencies: `moshi` Python package (v0.2.11+)

2. **Rust Server** (`moshi-server` crate, external repo)
   - For production deployment
   - WebSocket-based streaming, handles 64-400 concurrent streams on H100/L40S
   - 3x real-time processing speed
   - Configured via TOML files in `configs/`

3. **MLX** (`scripts/*_mlx.py`)
   - For Apple Silicon (Mac/iPhone)
   - Hardware-accelerated inference with quantization support (4-bit, 8-bit)
   - Dependencies: `moshi-mlx` package (v0.2.6+)

### Core Components

- **Mimi**: Neural audio codec (encoder/decoder), processes audio at 24kHz, outputs audio tokens at 12.5Hz
- **LM (Language Model)**: Transformer-based sequence-to-sequence model (Moshi architecture)
- **LMGen**: Generator for autoregressive inference with streaming support
- **Semantic VAD**: Voice activity detection for end-of-turn detection (1B model only)

### Key Technical Details

**Delayed Streams Modeling (DSM):**
- Streaming inference processes audio in chunks (frame_size = 1920 samples at 24kHz)
- Audio silence prefix (default 1s) and delay suffix (0.5s-2.5s depending on model) are added for proper context
- Text tokens are delayed relative to audio tokens by `asr_delay_in_tokens`

**Streaming Pattern (PyTorch):**
```python
with mimi.streaming(batch_size=1), lm_gen.streaming(batch_size=1):
    for audio_chunk in audio_chunks:
        audio_tokens = mimi.encode(audio_chunk)
        text_tokens = lm_gen.step(audio_tokens)
```

**Word-Level Timestamps:**
- The `end_of_padding_id` token (id=0) marks word boundaries
- Timestamps are derived from token positions and the Mimi frame rate (12.5Hz)
- Offset calculation accounts for silence prefix and audio delay

**Classifier-Free Guidance (TTS):**
- The TTS model uses CFG distillation with a fixed coefficient of 2.0
- Voice embeddings are loaded from `.safetensors` files in the voice folder

### Configuration Files

TOML configs in `configs/` define:
- Model paths (Hugging Face Hub URLs using `hf://` or `hf-snapshot://` prefixes)
- Batch sizes (adjust based on GPU memory)
- Transformer architecture (d_model, num_heads, num_layers, etc.)
- Audio codec settings (n_q quantization levels, 8-32 typical range)
- Temperature and sampling parameters

### Script Dependencies

Python scripts declare dependencies via PEP 723 inline metadata:
```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi==0.2.11",
#     "julius",
#     "librosa",
#     "soundfile",
# ]
# ///
```

## Troubleshooting

- **Torch compilation errors**: Set `NO_TORCH_COMPILE=1`
- **Sentencepiece/cmake issues**: Set `CMAKE_POLICY_VERSION_MINIMUM=3.5` or use gcc-13
- **Rust sentencepiece build errors**: Set `CXXFLAGS="-include cstdint"`

## Contributing Notes

- This is a research codebase; new features are generally not accepted (bug fixes welcome)
- Pre-commit hooks are mandatory (Ruff linting/formatting, nbstripout for notebooks)
- CLA required for contributions
- Rust code: Apache license; Python code: MIT license

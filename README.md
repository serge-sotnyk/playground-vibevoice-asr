# playground-vibevoice-asr

Experimental repo for evaluating [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR-HF) as a local transcription and diarization backend for recruiter-harness workflow.

## Setup

```bash
# Install dependencies
uv sync

# Install PyTorch with CUDA support
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Usage

CLI entry point will be added during implementation. See `docs/vibevoice_asr_experiment_plan.md` for the full experiment plan.

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests
uv run pytest
```

## Hardware Requirements

- NVIDIA GPU with CUDA support (tested on RTX 3090)
- Windows

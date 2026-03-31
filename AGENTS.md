## Overview

**playground-vibevoice-asr** — experimental repo for evaluating VibeVoice-ASR as a local transcription and diarization backend.

- Python 3.13+, managed with uv
- Target hardware: Windows + NVIDIA RTX 3090
- Specification: `docs/vibevoice_asr_experiment_plan.md`

## Commands

```bash
# Install dependencies
uv sync

# Install PyTorch with CUDA support (run once after uv sync)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
├── pyproject.toml
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── docs/
│   ├── vibevoice_asr_experiment_plan.md   # Experiment specification
│   ├── commands/                          # AI workflow command templates
│   └── features/                          # Feature plans
└── src/                                   # Source code (TBD)
```

## Code Style

- All code and comments in English
- Python 3.13+
- Formatter and linter: ruff
- Type annotations where practical
- Tests: pytest
- Keep modules small and explicit, avoid premature abstractions
- Prefer clear errors over silent fallbacks

## Development Notes

- Local execution only, Windows workstation with RTX 3090
- PyTorch must be installed with CUDA support from the PyTorch index
- bitsandbytes bundles its own CUDA binaries (no system CUDA dependency for bnb itself)
- Use MCP context7 for up-to-date library documentation

# playground-vibevoice-asr

Experimental repo for evaluating [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR-HF) as a local transcription and diarization backend for recruiter-harness workflow.

## Setup

```bash
# Install all dependencies (PyTorch CUDA index is selected automatically on Windows/Linux)
uv sync
```

## Usage

Run the CLI through `uv run`.

```bash
uv run vv-trans --audio "C:\Users\Serge\Documents\Sound Recordings\my_voice_example_en.mp3" --runtime 4bit
```

By default the result is written next to the audio file with the suffix `.trans.json`.

Examples:

- `my_voice_example_en.mp3` -> `my_voice_example_en.mp3.trans.json`
- `Recording (2).mp3` -> `Recording (2).mp3.trans.json`

You can override the output path explicitly:

```bash
uv run vv-trans --audio "C:\Users\Serge\Documents\Sound Recordings\my_voice_example_ru.mp3" --runtime 4bit --output "results\ru.json"
```

Optional prompt:

```bash
uv run vv-trans --audio "C:\Users\Serge\Documents\Sound Recordings\Nata-sample.m4a" --runtime 4bit --prompt "Interview between two speakers"
```

The JSON output is written in UTF-8 with indentation for human-readable inspection.

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

## Notes

- The first implementation only supports the `4bit` runtime.
- For local audio files, `--output` is optional and defaults to `<audio>.trans.json`.
- For non-local inputs such as URLs, pass `--output` explicitly.

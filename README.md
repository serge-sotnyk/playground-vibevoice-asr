# playground-vibevoice-asr

Experimental repo for evaluating [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR-HF) as a local transcription and diarization backend for recruiter-harness workflow.

## Setup

```bash
# Install all dependencies (PyTorch CUDA index is selected automatically on Windows/Linux)
uv sync
```

## Usage

Run the CLI through `uv run`.

### Single file

```bash
uv run vv-trans "C:\Users\Serge\Documents\Sound Recordings\my_voice_example_en.mp3"
```

By default the result is written next to the audio file with the suffix `.trans.<runtime>.json`.

Short flags are available: `-o` (output), `-p` (prompt), `-r` (runtime), `-m` (model-id).

Examples (default `-r 4`):

- `my_voice_example_en.mp3` -> `my_voice_example_en.mp3.trans.4.json`
- `Recording (2).mp3` -> `Recording (2).mp3.trans.4.json`

### Runtime modes

Four quantization/precision modes are available:

```bash
# 4-bit (default) — lowest VRAM, bitsandbytes NF4 quantization
uv run vv-trans file.mp3 --runtime 4

# 8-bit — TorchAO INT8 weight-only quantization (faster)
uv run vv-trans file.mp3 --runtime 8

# 8-bit — Quanto INT8 quantization (better macOS support)
uv run vv-trans file.mp3 --runtime 8q

# 16-bit — full bfloat16 precision, no quantization
uv run vv-trans file.mp3 --runtime 16
```

Different runtimes produce separate output files (`.trans.4.json`, `.trans.8.json`, `.trans.8q.json`, `.trans.16.json`), so you can compare them side by side.

### Chunk size (VRAM optimization)

For large audio or heavy runtimes (16-bit), use `--chunk-seconds` to reduce peak VRAM usage:

```bash
uv run vv-trans file.mp3 --runtime 16 --chunk-seconds 30
```

### Batch processing (directory)

Process all audio files (`.mp3`, `.wav`, `.m4a`, `.flac`) in a directory:

```bash
uv run vv-trans "C:\Users\Serge\Documents\Sound Recordings"
```

The model is loaded once and reused for all files. Existing outputs are skipped (use `--force` to overwrite). If a file fails, processing continues with the rest.

```bash
# Force re-processing of all files
uv run vv-trans ./audio_dir/ --force

# Batch with specific runtime
uv run vv-trans ./audio_dir/ --runtime 16 --chunk-seconds 30
```

Note: `--output` cannot be used with directory input.

### Explicit output path

```bash
uv run vv-trans file.mp3 --output "results/custom.json"
```

When `--output` is specified, the path is used as-is (no runtime suffix injection).

### Optional prompt

```bash
uv run vv-trans interview.mp3 -p "Interview between two speakers"
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

- For local audio files, `--output` is optional and defaults to `<audio>.trans.<runtime>.json`.
- For non-local inputs such as URLs, pass `--output` explicitly.

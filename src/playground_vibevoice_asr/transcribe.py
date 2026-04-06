from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    VibeVoiceAsrForConditionalGeneration,
)

from playground_vibevoice_asr.output_schema import TranscriptOutput, TranscriptSegment

DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"

RUNTIME_4 = "4"
RUNTIME_8 = "8"       # TorchAO Int8WeightOnly
RUNTIME_8Q = "8q"     # Quanto INT8
RUNTIME_16 = "16"
SUPPORTED_RUNTIMES = [RUNTIME_4, RUNTIME_8, RUNTIME_8Q, RUNTIME_16]

SAMPLE_RATE = 24000  # VibeVoice native sample rate


def _get_peak_vram_mb() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated() / (1024 * 1024))


def _reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def load_model(
    runtime: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[Any, Any, float]:
    """Load model and processor. Returns (model, processor, load_seconds)."""
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"Unsupported runtime: {runtime}")

    _reset_vram_stats()
    started_at = time.perf_counter()

    processor = AutoProcessor.from_pretrained(model_id)

    if runtime == RUNTIME_4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    elif runtime == RUNTIME_8:
        from torchao.quantization import Int8WeightOnlyConfig as TorchaoInt8Config
        from transformers import TorchAoConfig

        quantization_config = TorchAoConfig(quant_type=TorchaoInt8Config())
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    elif runtime == RUNTIME_8Q:
        from transformers import QuantoConfig

        quantization_config = QuantoConfig(weights="int8")
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:  # RUNTIME_16
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
        )

    load_seconds = time.perf_counter() - started_at
    return model, processor, load_seconds


def _quantization_type(runtime: str) -> str | None:
    if runtime == RUNTIME_4:
        return "nf4"
    if runtime == RUNTIME_8:
        return "int8-torchao"
    if runtime == RUNTIME_8Q:
        return "int8-quanto"
    return None


def transcribe_audio(
    *,
    audio: str,
    model: Any,
    processor: Any,
    runtime: str,
    model_load_seconds: float,
    prompt: str | None = None,
    chunk_seconds: int | None = None,
) -> TranscriptOutput:
    started_at = time.perf_counter()

    inputs = processor.apply_transcription_request(audio=audio, prompt=prompt)
    inputs = inputs.to(model.device, model.dtype)

    generate_kwargs: dict[str, Any] = dict(inputs)
    if chunk_seconds is not None:
        generate_kwargs["acoustic_tokenizer_chunk_size"] = chunk_seconds * SAMPLE_RATE

    output_ids = model.generate(**generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]

    inference_seconds = time.perf_counter() - started_at

    segments = parse_segments(parsed)
    full_text = " ".join(
        segment.text for segment in segments if not is_non_speech_marker(segment.text)
    )
    metadata = {
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_vram_mb": _get_peak_vram_mb(),
        "quantization_type": _quantization_type(runtime),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "segment_count": len(segments),
    }
    return TranscriptOutput(
        source=audio,
        model_id=str(getattr(model.config, "_name_or_path", "")),
        runtime=runtime,
        prompt=prompt,
        full_text=full_text,
        segments=segments,
        metadata=metadata,
    )


def parse_segments(parsed: Any) -> list[TranscriptSegment]:
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"Expected non-empty parsed transcription list, got: {type(parsed).__name__}"
        )

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Segment {index} is not a dict: {type(item).__name__}")

        missing = {key for key in ("Start", "End", "Content") if key not in item}
        if missing:
            raise ValueError(
                f"Segment {index} is missing required keys: {sorted(missing)}"
            )

        text = str(item["Content"]).strip()
        if not text:
            raise ValueError(f"Segment {index} has empty content")

        speaker = item.get("Speaker")
        if speaker is None and not is_non_speech_marker(text):
            speaker = 0

        segments.append(
            TranscriptSegment(
                speaker=None if speaker is None else f"speaker_{speaker}",
                start=float(item["Start"]),
                end=float(item["End"]),
                text=text,
            )
        )

    return segments


def is_non_speech_marker(text: str) -> bool:
    return text.startswith("[") and text.endswith("]")


def write_transcript(output: TranscriptOutput, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

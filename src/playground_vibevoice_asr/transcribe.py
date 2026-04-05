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
RUNTIME_4BIT = "4bit"


def transcribe_audio(
    *,
    audio: str,
    runtime: str,
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str | None = None,
) -> TranscriptOutput:
    if runtime != RUNTIME_4BIT:
        raise ValueError(f"Unsupported runtime: {runtime}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    started_at = time.perf_counter()
    inputs = processor.apply_transcription_request(audio=audio, prompt=prompt)
    inputs = inputs.to(model.device, model.dtype)
    output_ids = model.generate(**inputs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    duration_seconds = time.perf_counter() - started_at

    segments = parse_segments(parsed)
    full_text = " ".join(
        segment.text for segment in segments if not is_non_speech_marker(segment.text)
    )
    metadata = {
        "processing_duration_seconds": round(duration_seconds, 3),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "segment_count": len(segments),
    }
    return TranscriptOutput(
        source=audio,
        model_id=model_id,
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

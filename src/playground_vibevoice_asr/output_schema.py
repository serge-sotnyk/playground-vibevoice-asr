from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    speaker: str | None
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptOutput:
    source: str
    model_id: str
    runtime: str
    prompt: str | None
    full_text: str
    segments: list[TranscriptSegment]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload


def default_output_path(audio: str) -> Path:
    audio_path = Path(audio)
    if not audio_path.exists():
        raise ValueError("--output is required when --audio is not a local file path")
    return audio_path.with_name(f"{audio_path.name}.trans.json")

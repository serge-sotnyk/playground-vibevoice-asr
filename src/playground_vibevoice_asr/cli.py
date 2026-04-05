from __future__ import annotations

import argparse
from pathlib import Path

from playground_vibevoice_asr.output_schema import default_output_path
from playground_vibevoice_asr.transcribe import (
    DEFAULT_MODEL_ID,
    RUNTIME_4BIT,
    transcribe_audio,
    write_transcript,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with VibeVoice-ASR")
    parser.add_argument("--audio", required=True, help="Local audio path or remote URL")
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <audio>.trans.json for local files",
    )
    parser.add_argument("--prompt", help="Optional prompt to guide transcription")
    parser.add_argument(
        "--runtime",
        choices=[RUNTIME_4BIT],
        default=RUNTIME_4BIT,
        help="Runtime mode",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output) if args.output else default_output_path(args.audio)
    transcript = transcribe_audio(
        audio=args.audio,
        runtime=args.runtime,
        model_id=args.model_id,
        prompt=args.prompt,
    )
    write_transcript(transcript, output_path)
    print(output_path)


if __name__ == "__main__":
    main()

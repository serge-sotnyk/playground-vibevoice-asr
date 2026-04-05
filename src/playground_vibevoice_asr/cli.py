from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playground_vibevoice_asr.output_schema import default_output_path
from playground_vibevoice_asr.transcribe import (
    DEFAULT_MODEL_ID,
    SUPPORTED_RUNTIMES,
    RUNTIME_4,
    load_model,
    transcribe_audio,
    write_transcript,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


def find_audio_files(directory: Path) -> list[Path]:
    files = [
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    files.sort()
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with VibeVoice-ASR")
    parser.add_argument(
        "--audio",
        required=True,
        help="Local audio path, remote URL, or directory of audio files",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <audio>.trans.<runtime>.json for local files",
    )
    parser.add_argument("--prompt", help="Optional prompt to guide transcription")
    parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIMES,
        default=RUNTIME_4,
        help="Runtime mode (default: 4)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        help="Audio chunk size in seconds for model.generate(). "
        "Smaller chunks use less VRAM but run slower. "
        "Default: model default (60s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files (batch mode)",
    )
    return parser


def _process_single(args: argparse.Namespace) -> None:
    model, processor, load_time = load_model(args.runtime, args.model_id)
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(args.audio, args.runtime)
    )
    result = transcribe_audio(
        audio=args.audio,
        model=model,
        processor=processor,
        runtime=args.runtime,
        model_load_seconds=load_time,
        prompt=args.prompt,
        chunk_seconds=args.chunk_seconds,
    )
    write_transcript(result, output_path)
    print(output_path)


def _process_directory(args: argparse.Namespace) -> None:
    if args.output:
        print("Error: --output cannot be used with directory input", file=sys.stderr)
        sys.exit(1)

    audio_dir = Path(args.audio)
    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) in {audio_dir}")
    model, processor, load_time = load_model(args.runtime, args.model_id)

    ok, failed, skipped = 0, 0, 0
    for audio_file in audio_files:
        output_path = default_output_path(str(audio_file), args.runtime)
        if output_path.exists() and not args.force:
            print(f"Skipping (exists): {output_path}")
            skipped += 1
            continue
        try:
            result = transcribe_audio(
                audio=str(audio_file),
                model=model,
                processor=processor,
                runtime=args.runtime,
                model_load_seconds=load_time,
                prompt=args.prompt,
                chunk_seconds=args.chunk_seconds,
            )
            write_transcript(result, output_path)
            print(output_path)
            ok += 1
        except Exception as e:
            print(f"Error processing {audio_file}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {ok} ok, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    audio_path = Path(args.audio)
    if audio_path.is_dir():
        _process_directory(args)
    else:
        _process_single(args)


if __name__ == "__main__":
    main()

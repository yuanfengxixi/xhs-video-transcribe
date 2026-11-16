"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .transcriber import transcribe_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xhs-video-transcribe",
        description="Create a text transcript and SRT subtitles from a local video or audio file.",
    )
    parser.add_argument("input", type=Path, help="local video or audio file")
    parser.add_argument(
        "--out-dir", type=Path, help="output directory (default: <input directory>/transcripts)"
    )
    parser.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    parser.add_argument("--language", default="zh", help="language code, or 'auto' to detect (default: zh)")
    parser.add_argument("--device", default="cpu", help="inference device, such as cpu or cuda (default: cpu)")
    parser.add_argument(
        "--compute-type", default="int8", help="faster-whisper compute type (default: int8)"
    )
    parser.add_argument("--keep-audio", action="store_true", help="keep the extracted WAV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.expanduser()
    output_dir = args.out_dir or input_path.parent / "transcripts"
    if not input_path.is_file():
        print(f"error: input media file does not exist: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() != ".wav" and shutil.which("ffmpeg") is None:
        print("error: ffmpeg was not found; install ffmpeg and retry", file=sys.stderr)
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        parser.exit(2, "faster-whisper is not installed; run `python -m pip install .`\n")

    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        result = transcribe_media(
            input_path,
            output_dir,
            model,
            language=args.language,
            keep_audio=args.keep_audio,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote {result.segment_count} segments")
    print(f"text: {result.text_path}")
    print(f"subtitles: {result.subtitle_path}")
    if result.audio_path:
        print(f"audio: {result.audio_path}")
    return 0

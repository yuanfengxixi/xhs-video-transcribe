"""Audio extraction and transcript file generation."""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Segment(Protocol):
    start: float
    end: float
    text: str


class WhisperModel(Protocol):
    def transcribe(self, audio: str, **options: object) -> tuple[Iterable[Segment], object]: ...


@dataclass(frozen=True)
class TranscriptResult:
    text_path: Path
    subtitle_path: Path
    segment_count: int
    audio_path: Path | None


def format_srt_timestamp(seconds: float) -> str:
    """Format a non-negative time as an SRT timestamp, rounding to milliseconds."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timestamp must be a finite, non-negative number")

    total_ms = math.floor(seconds * 1000 + 0.5)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_transcript(
    segments: Iterable[Segment], text_path: Path, subtitle_path: Path
) -> int:
    """Write non-empty segments to UTF-8 text and SRT files; return the count."""
    text_lines: list[str] = []
    subtitle_blocks: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.end < segment.start:
            raise ValueError("segment end must not precede its start")

        index = len(text_lines) + 1
        text_lines.append(text)
        subtitle_blocks.append(
            f"{index}\n{format_srt_timestamp(segment.start)} --> "
            f"{format_srt_timestamp(segment.end)}\n{text}"
        )

    text_path.write_text("\n".join(text_lines) + ("\n" if text_lines else ""), encoding="utf-8")
    subtitle_path.write_text("\n\n".join(subtitle_blocks) + ("\n" if subtitle_blocks else ""), encoding="utf-8")
    return len(text_lines)


def extract_audio(media_path: Path, wav_path: Path) -> None:
    """Extract mono 16 kHz PCM audio with ffmpeg, without invoking a shell."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg was not found; install ffmpeg and retry") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        raise RuntimeError(f"ffmpeg could not extract audio: {detail or error}") from error


def transcribe_media(
    media_path: Path,
    output_dir: Path,
    model: WhisperModel,
    *,
    language: str = "zh",
    keep_audio: bool = False,
) -> TranscriptResult:
    """Transcribe a local media file and create adjacent `.txt` and `.srt` files."""
    media_path = media_path.expanduser()
    if not media_path.is_file():
        raise FileNotFoundError(f"input media file does not exist: {media_path}")

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{media_path.stem}.txt"
    subtitle_path = output_dir / f"{media_path.stem}.srt"
    input_resolved = media_path.resolve()
    if text_path.resolve() == input_resolved or subtitle_path.resolve() == input_resolved:
        raise ValueError("output files would overwrite the input media")
    language_option = None if language.lower() == "auto" else language

    if media_path.suffix.lower() == ".wav":
        segments, _ = model.transcribe(
            str(media_path), language=language_option, vad_filter=True, beam_size=5
        )
        count = write_transcript(segments, text_path, subtitle_path)
        return TranscriptResult(
            text_path, subtitle_path, count, media_path if keep_audio else None
        )

    if keep_audio:
        audio_path = output_dir / f"{media_path.stem}.wav"
        extract_audio(media_path, audio_path)
        segments, _ = model.transcribe(
            str(audio_path), language=language_option, vad_filter=True, beam_size=5
        )
        count = write_transcript(segments, text_path, subtitle_path)
        return TranscriptResult(text_path, subtitle_path, count, audio_path)

    with tempfile.TemporaryDirectory(prefix="xhs-transcribe-", dir=output_dir) as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        extract_audio(media_path, audio_path)
        segments, _ = model.transcribe(
            str(audio_path), language=language_option, vad_filter=True, beam_size=5
        )
        count = write_transcript(segments, text_path, subtitle_path)
    return TranscriptResult(text_path, subtitle_path, count, None)

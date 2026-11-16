from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xhs_video_transcribe.transcriber import (
    extract_audio,
    format_srt_timestamp,
    transcribe_media,
    write_transcript,
)


class FakeModel:
    def __init__(self, segments: list[SimpleNamespace]) -> None:
        self.segments = segments
        self.audio_path: str | None = None
        self.options: dict[str, object] = {}

    def transcribe(self, audio: str, **options: object):
        self.audio_path = audio
        self.options = options
        return iter(self.segments), SimpleNamespace(language="zh", duration=1.0)


class TranscriberTests(unittest.TestCase):
    def test_srt_timestamp_rounds_across_second_and_minute_boundaries(self) -> None:
        self.assertEqual(format_srt_timestamp(1.2345), "00:00:01,235")
        self.assertEqual(format_srt_timestamp(59.9996), "00:01:00,000")
        self.assertEqual(format_srt_timestamp(3600), "01:00:00,000")

    def test_srt_timestamp_rejects_negative_and_non_finite_values(self) -> None:
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                format_srt_timestamp(value)

    def test_write_transcript_skips_blank_segments_and_numbers_outputs_contiguously(self) -> None:
        segments = [
            SimpleNamespace(start=0.1, end=0.9, text="  你好  "),
            SimpleNamespace(start=0.9, end=1.1, text="  "),
            SimpleNamespace(start=1.2, end=2.0, text="世界"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "sample.txt"
            subtitle_path = Path(temp_dir) / "sample.srt"

            count = write_transcript(segments, text_path, subtitle_path)

            self.assertEqual(count, 2)
            self.assertEqual(text_path.read_text(encoding="utf-8"), "你好\n世界\n")
            self.assertEqual(
                subtitle_path.read_text(encoding="utf-8"),
                "1\n00:00:00,100 --> 00:00:00,900\n你好\n\n"
                "2\n00:00:01,200 --> 00:00:02,000\n世界\n",
            )

    def test_write_transcript_rejects_reversed_segment_times(self) -> None:
        segment = SimpleNamespace(start=2.0, end=1.0, text="bad timing")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "end must not precede"):
                write_transcript(
                    [segment], Path(temp_dir) / "sample.txt", Path(temp_dir) / "sample.srt"
                )

    def test_extract_audio_calls_ffmpeg_without_a_shell(self) -> None:
        with patch("xhs_video_transcribe.transcriber.subprocess.run") as run:
            extract_audio(Path("input video.mp4"), Path("output.wav"))

        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-nostdin", command)
        self.assertEqual(command[command.index("-i") + 1], "input video.mp4")
        self.assertEqual(run.call_args.kwargs["check"], True)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_transcribe_media_uses_auto_language_and_cleans_temporary_audio(self) -> None:
        model = FakeModel([SimpleNamespace(start=0.0, end=1.0, text="hello")])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "clip.mp4"
            media.write_bytes(b"test media")
            out_dir = root / "out"

            def fake_extract(_media_path: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            with patch("xhs_video_transcribe.transcriber.extract_audio", fake_extract):
                result = transcribe_media(media, out_dir, model, language="auto")

            self.assertEqual(model.options["language"], None)
            self.assertTrue(result.text_path.is_file())
            self.assertTrue(result.subtitle_path.is_file())
            self.assertIsNone(result.audio_path)
            self.assertFalse(Path(model.audio_path or "").exists())

    def test_transcribe_media_can_keep_extracted_audio(self) -> None:
        model = FakeModel([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "clip.mp4"
            media.write_bytes(b"test media")

            def fake_extract(_media_path: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            with patch("xhs_video_transcribe.transcriber.extract_audio", fake_extract):
                result = transcribe_media(media, root / "out", model, keep_audio=True)

            self.assertIsNotNone(result.audio_path)
            self.assertTrue(result.audio_path and result.audio_path.is_file())
            self.assertEqual(model.options["language"], "zh")

    def test_wav_input_is_transcribed_without_ffmpeg(self) -> None:
        model = FakeModel([SimpleNamespace(start=0.0, end=1.0, text="hello")])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav = root / "clip.wav"
            wav.write_bytes(b"wav")

            with patch("xhs_video_transcribe.transcriber.extract_audio") as extract:
                result = transcribe_media(wav, root / "out", model)

            extract.assert_not_called()
            self.assertEqual(model.audio_path, str(wav))
            self.assertIsNone(result.audio_path)

    def test_output_name_that_would_overwrite_the_input_is_rejected(self) -> None:
        model = FakeModel([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_input = root / "notes.txt"
            text_input.write_text("keep this input", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overwrite the input"):
                transcribe_media(text_input, root, model)

            self.assertEqual(text_input.read_text(encoding="utf-8"), "keep this input")


if __name__ == "__main__":
    unittest.main()

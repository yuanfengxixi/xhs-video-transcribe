from __future__ import annotations

import unittest

from xhs_video_transcribe.cli import build_parser


class CliTests(unittest.TestCase):
    def test_defaults_and_user_options(self) -> None:
        args = build_parser().parse_args(["clip.mp4", "--language", "auto", "--keep-audio"])
        self.assertEqual(args.model, "small")
        self.assertEqual(args.language, "auto")
        self.assertTrue(args.keep_audio)
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()

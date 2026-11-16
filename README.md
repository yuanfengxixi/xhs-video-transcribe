# XHS Video Transcribe

A small cross-platform CLI that turns a **local video or audio file** into a UTF-8 text transcript and an SRT subtitle file using faster-whisper.

This is an unofficial community tool and is not affiliated with Xiaohongshu. It does not log in to Xiaohongshu, download media, or contact a platform. Provide a media file that you are allowed to process.

## Requirements

- Python 3.11 or newer
- `ffmpeg` available on `PATH` for video and non-WAV audio inputs; WAV inputs are read directly
- A faster-whisper model. The first run downloads the selected model from Hugging Face; later runs can use its local cache.

Install the project in a virtual environment:

```sh
python -m venv .venv
```

Activate the environment (PowerShell: `\.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`), then install:

```sh
python -m pip install .
```

## Usage

```sh
xhs-video-transcribe path/to/video.mp4
xhs-video-transcribe path/to/audio.m4a --language auto --out-dir transcripts
xhs-video-transcribe path/to/video.mp4 --model medium --device cuda --compute-type float16 --keep-audio
```

By default, the CLI uses the `small` model on CPU with `int8` compute and assumes Chinese audio. It writes `<input-name>.txt` and `<input-name>.srt` into a `transcripts/` folder beside the input. A temporary WAV file is removed when transcription finishes; `--keep-audio` keeps it in the output directory.

Run the tests without installing faster-whisper or downloading a model:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

In PowerShell, set `$env:PYTHONPATH = "src"` before running the same unittest command.

## Notes

- Use `--language auto` for language detection, or supply a faster-whisper language code such as `en` or `ja`.
- Model files are downloaded by faster-whisper and can be large. Select a smaller model for lower memory use.
- SRT timestamps are rounded to milliseconds. Empty speech segments are omitted and remaining cues are numbered consecutively.
- The project contains no video, audio, or transcript samples.

## License

MIT. See [LICENSE](LICENSE). faster-whisper, CTranslate2, FFmpeg, and model weights keep their own licenses and terms.

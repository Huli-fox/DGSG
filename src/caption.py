#!/usr/bin/env python3
"""Generate SRT or VTT subtitle files from audio/video using Deepgram STT.

Usage:
    python src/caption.py recording.mp3
    python src/caption.py video.mp4 --language multi --format srt
    python src/caption.py recording.mp3 --format vtt --output captions.vtt
    python src/caption.py --url https://example.com/audio.wav
    python src/caption.py recording.mp3 --diarize --format srt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from deepgram import DeepgramClient
from deepgram_captions import DeepgramConverter, srt, webvtt

try:
    from .env_utils import load_env
except ImportError:
    from env_utils import load_env


def _build_transcribe_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for Deepgram SDK v6 listen.v1.media calls."""
    params: dict[str, Any] = {
        "model": kwargs.get("model", "nova-3"),
        "smart_format": kwargs.get("smart_format", True),
        # utterances=True splits the transcript at natural pauses — required
        # for subtitle timing.
        "utterances": kwargs.get("utterances", True),
        "diarize": kwargs.get("diarize", False),
        "tag": kwargs.get("tag", "deepgram-examples"),
    }

    language = (kwargs.get("language") or "").strip()
    detect_language = bool(kwargs.get("detect_language", False))
    if language.lower() == "auto" or detect_language:
        params["detect_language"] = True
    elif language:
        params["language"] = language

    timeout = kwargs.get("timeout_seconds")
    if timeout is not None:
        params["request_options"] = {"timeout_in_seconds": int(timeout)}

    return params


def _response_to_dict(response: Any) -> dict:
    """Dump an SDK response model to a plain dict, dropping unset (None) fields.

    deepgram-captions decides whether to emit speaker labels by checking for the
    presence of a "speaker" key, so a plain model_dump() (which keeps
    "speaker": None when diarize is off) would produce "<v Speaker None>" cues.
    """
    return response.model_dump(exclude_none=True)


def transcribe_file(client: DeepgramClient, file_path: str | Path, **kwargs: Any) -> dict:
    """Transcribe a local audio/video file and return the raw response dict.

    Uses transcribe_file() which uploads the bytes to Deepgram. The SDK
    auto-detects the audio format from the file header.
    """
    audio_bytes = Path(file_path).read_bytes()
    params = _build_transcribe_params(kwargs)
    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        **params,
    )
    return _response_to_dict(response)


def transcribe_url(client: DeepgramClient, url: str, **kwargs: Any) -> dict:
    """Transcribe audio from a public URL."""
    params = _build_transcribe_params(kwargs)
    response = client.listen.v1.media.transcribe_url(
        url=url,
        **params,
    )
    return _response_to_dict(response)


def generate_captions(
    dg_response: dict,
    fmt: str = "srt",
    line_length: int | None = None,
) -> str:
    """Convert a Deepgram response dict into SRT or VTT subtitle text."""
    converter = DeepgramConverter(dg_response)
    if fmt == "vtt":
        return webvtt(converter, line_length=line_length)
    return srt(converter, line_length=line_length)


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(
        description="Generate SRT/VTT subtitles from audio/video using Deepgram",
        epilog="Examples:\n"
               "  python src/caption.py recording.mp3\n"
               "  python src/caption.py video.mp4 --language multi\n"
               "  python src/caption.py --url https://example.com/audio.wav --format vtt\n"
               "  python src/caption.py interview.wav --diarize --output interview.srt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("file", nargs="?", help="Local audio/video file to transcribe")
    source.add_argument("--url", help="Public URL of audio to transcribe (Deepgram fetches it server-side)")

    parser.add_argument(
        "--format",
        choices=["srt", "vtt"],
        default="srt",
        help="Output format: srt (SubRip, default) or vtt (WebVTT)",
    )
    parser.add_argument("--output", "-o", help="Output file path (default: stdout or <input>.srt/.vtt)")
    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model (default: nova-3). Use nova-3-medical for medical audio",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. multi, auto, zh-CN, en, ja). Omit for model default",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (adds speaker labels to subtitles)",
    )
    parser.add_argument(
        "--line-length",
        type=int,
        default=None,
        help="Max characters per subtitle line (default: library decides)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print(
            "Error: DEEPGRAM_API_KEY environment variable is not set.\n"
            "Get a free API key at https://console.deepgram.com/",
            file=sys.stderr,
        )
        sys.exit(1)

    client = DeepgramClient(api_key=api_key)

    if args.url:
        print(f"Transcribing URL: {args.url}", file=sys.stderr)
        dg_response = transcribe_url(
            client,
            args.url,
            model=args.model,
            language=args.language,
            diarize=args.diarize,
        )
    else:
        if not Path(args.file).exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        print(f"Transcribing file: {args.file}", file=sys.stderr)
        dg_response = transcribe_file(
            client,
            args.file,
            model=args.model,
            language=args.language,
            diarize=args.diarize,
        )

    captions = generate_captions(dg_response, fmt=args.format, line_length=args.line_length)

    if args.output:
        output_path = args.output
    elif args.file:
        output_path = str(Path(args.file).with_suffix(f".{args.format}"))
    else:
        print(captions)
        return

    Path(output_path).write_text(captions, encoding="utf-8")
    print(f"Subtitles written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

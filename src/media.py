"""Media file discovery and FFmpeg audio extraction utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
})

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
})

SUPPORTED_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class MediaExtractionError(RuntimeError):
    """Raised when audio extraction via FFmpeg fails."""


class ExtractionCancelled(RuntimeError):
    """Raised when FFmpeg extraction is cancelled by the user."""


def find_ffmpeg() -> str | None:
    """Return path to ffmpeg executable if available on PATH."""
    return shutil.which("ffmpeg")


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_supported_media(path: Path, include_audio: bool = True) -> bool:
    ext = path.suffix.lower()
    if include_audio:
        return ext in SUPPORTED_EXTENSIONS
    return ext in VIDEO_EXTENSIONS


def collect_media_files(
    paths: Iterable[str | Path],
    recursive: bool = True,
    include_audio: bool = True,
) -> list[Path]:
    """Expand files and directories into a deduplicated, sorted list of media files."""
    collected: list[Path] = []
    seen: set[Path] = set()

    for item in paths:
        p = Path(item)
        if not p.exists():
            continue
        if p.is_file():
            resolved = p.resolve()
            if is_supported_media(resolved, include_audio=include_audio) and resolved not in seen:
                seen.add(resolved)
                collected.append(resolved)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            dir_files = sorted(
                (f.resolve() for f in iterator if f.is_file() and is_supported_media(f, include_audio=include_audio)),
                key=lambda x: str(x).lower(),
            )
            for resolved in dir_files:
                if resolved not in seen:
                    seen.add(resolved)
                    collected.append(resolved)

    return collected


def extract_audio(
    video_path: Path,
    output_path: Path,
    ffmpeg_bin: str | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Extract lightweight mono 16kHz MP3 audio from a video file using FFmpeg.

    Keeping the audio compact dramatically reduces upload time and avoids hitting
    memory or payload limits on large video files.
    """
    binary = ffmpeg_bin or find_ffmpeg()
    if not binary:
        raise MediaExtractionError("未找到 ffmpeg，请确认已安装并加入系统 PATH。")

    if cancel_event and cancel_event.is_set():
        raise ExtractionCancelled("任务已取消")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(output_path),
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )

    try:
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise ExtractionCancelled("任务已取消")
            time.sleep(0.1)

        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        if proc.returncode != 0:
            err_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            detail = err_text.splitlines()[-1] if err_text else f"退出码 {proc.returncode}"
            raise MediaExtractionError(f"FFmpeg 提取音频失败: {detail}")

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise MediaExtractionError("FFmpeg 未生成有效音频（视频可能不包含音轨）")

        return output_path
    finally:
        if proc.stderr:
            proc.stderr.close()

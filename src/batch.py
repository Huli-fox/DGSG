"""Batch subtitle generation engine for local video and audio files."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from deepgram import DeepgramClient

try:
    from .caption import generate_captions, transcribe_file
    from .env_utils import PROJECT_ROOT
    from .media import (
        ExtractionCancelled,
        extract_audio,
        find_ffmpeg,
        is_video_file,
    )
except ImportError:
    from caption import generate_captions, transcribe_file
    from env_utils import PROJECT_ROOT
    from media import (
        ExtractionCancelled,
        extract_audio,
        find_ffmpeg,
        is_video_file,
    )


class FileStatus(str, Enum):
    PENDING = "等待中"
    EXTRACTING = "提取音频"
    TRANSCRIBING = "转录中"
    DONE = "已完成"
    SKIPPED = "已跳过"
    FAILED = "失败"
    CANCELLED = "已取消"


@dataclass
class BatchOptions:
    api_key: str
    output_format: str = "srt"
    model: str = "nova-3"
    language: str = "multi"
    diarize: bool = False
    line_length: int | None = None
    output_dir: Path | None = None
    skip_existing: bool = True
    extract_audio_for_video: bool = True
    max_workers: int = 2
    timeout_seconds: int = 300


@dataclass
class FileResult:
    index: int
    source: Path
    output: Path | None
    status: FileStatus
    message: str = ""
    elapsed: float = 0.0


ProgressCallback = Callable[[int, FileStatus, str, Path | None], None]


def build_output_paths(
    files: Sequence[Path],
    fmt: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Compute target subtitle paths, avoiding filename collisions when using a shared output_dir."""
    if output_dir is None:
        return [f.with_suffix(f".{fmt}") for f in files]

    results: list[Path] = []
    used: dict[str, int] = {}
    for f in files:
        base_stem = f.stem
        key = base_stem.lower()
        count = used.get(key, 0)
        used[key] = count + 1
        final_stem = base_stem if count == 0 else f"{base_stem}_{count + 1}"
        results.append(output_dir / f"{final_stem}.{fmt}")
    return results


class BatchProcessor:
    """Process a batch of local media files into SRT/VTT subtitles concurrently."""

    def __init__(
        self,
        options: BatchOptions,
        client_factory: Callable[[str], DeepgramClient] | None = None,
        transcribe_fn: Callable[..., dict] | None = None,
        temp_base_dir: Path | None = None,
    ) -> None:
        self.options = options
        self._client_factory = client_factory or (lambda key: DeepgramClient(api_key=key))
        self._transcribe_fn = transcribe_fn or transcribe_file
        self._temp_base_dir = temp_base_dir or PROJECT_ROOT

    def run(
        self,
        files: Sequence[Path],
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[FileResult]:
        if not files:
            return []

        if self.options.output_dir is not None:
            self.options.output_dir.mkdir(parents=True, exist_ok=True)

        target_paths = build_output_paths(
            files,
            self.options.output_format,
            self.options.output_dir,
        )
        ffmpeg_bin = find_ffmpeg() if self.options.extract_audio_for_video else None
        workers = max(1, min(int(self.options.max_workers), len(files)))
        results: list[FileResult | None] = [None] * len(files)

        def emit(idx: int, status: FileStatus, message: str = "", out: Path | None = None) -> None:
            if on_progress:
                on_progress(idx, status, message, out)

        # Keep temp files inside workspace (or custom base) and always clean up.
        self._temp_base_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dgsg_tmp_", dir=str(self._temp_base_dir)) as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            def process_one(idx: int, src: Path, dst: Path) -> FileResult:
                start = time.monotonic()
                if cancel_event and cancel_event.is_set():
                    emit(idx, FileStatus.CANCELLED, "已取消", None)
                    return FileResult(idx, src, None, FileStatus.CANCELLED, "已取消", 0.0)

                if not src.is_file():
                    msg = "文件不存在"
                    emit(idx, FileStatus.FAILED, msg, None)
                    return FileResult(idx, src, None, FileStatus.FAILED, msg, time.monotonic() - start)

                if self.options.skip_existing and dst.is_file():
                    msg = f"已存在: {dst.name}"
                    emit(idx, FileStatus.SKIPPED, msg, dst)
                    return FileResult(idx, src, dst, FileStatus.SKIPPED, msg, time.monotonic() - start)

                temp_audio: Path | None = None
                try:
                    upload_path = src
                    if self.options.extract_audio_for_video and is_video_file(src) and ffmpeg_bin:
                        emit(idx, FileStatus.EXTRACTING, "FFmpeg 提取音频中...", None)
                        temp_audio = tmp_dir / f"track_{idx}_{threading.get_ident()}.mp3"
                        upload_path = extract_audio(
                            src,
                            temp_audio,
                            ffmpeg_bin=ffmpeg_bin,
                            cancel_event=cancel_event,
                        )

                    if cancel_event and cancel_event.is_set():
                        emit(idx, FileStatus.CANCELLED, "已取消", None)
                        return FileResult(idx, src, None, FileStatus.CANCELLED, "已取消", time.monotonic() - start)

                    emit(idx, FileStatus.TRANSCRIBING, "上传并转录中...", None)
                    client = self._client_factory(self.options.api_key)
                    dg_response = self._transcribe_fn(
                        client,
                        upload_path,
                        model=self.options.model,
                        language=self.options.language,
                        diarize=self.options.diarize,
                        timeout_seconds=self.options.timeout_seconds,
                    )

                    captions = generate_captions(
                        dg_response,
                        fmt=self.options.output_format,
                        line_length=self.options.line_length,
                    )

                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(captions, encoding="utf-8")

                    elapsed = time.monotonic() - start
                    msg = f"{dst.name} ({elapsed:.1f}s)"
                    emit(idx, FileStatus.DONE, msg, dst)
                    return FileResult(idx, src, dst, FileStatus.DONE, msg, elapsed)

                except ExtractionCancelled:
                    elapsed = time.monotonic() - start
                    emit(idx, FileStatus.CANCELLED, "已取消", None)
                    return FileResult(idx, src, None, FileStatus.CANCELLED, "已取消", elapsed)
                except Exception as exc:
                    elapsed = time.monotonic() - start
                    err_msg = str(exc).strip() or exc.__class__.__name__
                    # Keep table message concise on a single line
                    short_msg = err_msg.splitlines()[0][:160]
                    emit(idx, FileStatus.FAILED, short_msg, None)
                    return FileResult(idx, src, None, FileStatus.FAILED, err_msg, elapsed)
                finally:
                    if temp_audio is not None and temp_audio.exists():
                        temp_audio.unlink(missing_ok=True)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(process_one, idx, src, dst): idx
                    for idx, (src, dst) in enumerate(zip(files, target_paths))
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    results[idx] = future.result()

        return [r for r in results if r is not None]

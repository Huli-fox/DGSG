"""Offline unit tests for batch video subtitle generation (no live API key required)."""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepgram import ListenV1Response

from src.batch import BatchOptions, BatchProcessor, FileStatus, build_output_paths
from src.caption import _build_transcribe_params, generate_captions, transcribe_file
from src.env_utils import read_env_file, save_api_key
from src.media import (
    ExtractionCancelled,
    MediaExtractionError,
    collect_media_files,
    extract_audio,
    find_ffmpeg,
)

SAMPLE_DG_RESPONSE = {
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "Hello world. This is a batch test.",
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5, "punctuated_word": "Hello"},
                            {"word": "world", "start": 0.5, "end": 1.0, "punctuated_word": "world."},
                            {"word": "this", "start": 1.2, "end": 1.5, "punctuated_word": "This"},
                            {"word": "is", "start": 1.5, "end": 1.7, "punctuated_word": "is"},
                            {"word": "a", "start": 1.7, "end": 1.8, "punctuated_word": "a"},
                            {"word": "batch", "start": 1.8, "end": 2.2, "punctuated_word": "batch"},
                            {"word": "test", "start": 2.2, "end": 2.6, "punctuated_word": "test."},
                        ],
                    }
                ]
            }
        ],
        "utterances": [
            {
                "start": 0.0,
                "end": 1.0,
                "transcript": "Hello world.",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "punctuated_word": "Hello"},
                    {"word": "world", "start": 0.5, "end": 1.0, "punctuated_word": "world."},
                ],
                "speaker": 0,
                "channel": 0,
            },
            {
                "start": 1.2,
                "end": 2.6,
                "transcript": "This is a batch test.",
                "words": [
                    {"word": "this", "start": 1.2, "end": 1.5, "punctuated_word": "This"},
                    {"word": "is", "start": 1.5, "end": 1.7, "punctuated_word": "is"},
                    {"word": "a", "start": 1.7, "end": 1.8, "punctuated_word": "a"},
                    {"word": "batch", "start": 1.8, "end": 2.2, "punctuated_word": "batch"},
                    {"word": "test", "start": 2.2, "end": 2.6, "punctuated_word": "test."},
                ],
                "speaker": 1,
                "channel": 0,
            },
        ],
    }
}


def _make_sdk_response(diarize: bool) -> ListenV1Response:
    """Build a real SDK response model (not a hand-written dict) like the API returns."""
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9, "punctuated_word": "Hello"},
        {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.9, "punctuated_word": "world."},
    ]
    utterance = {
        "start": 0.0,
        "end": 1.0,
        "confidence": 0.9,
        "channel": 0,
        "transcript": "Hello world.",
        "words": words,
        "id": "u1",
    }
    if diarize:
        for w in words:
            w["speaker"] = 0
        utterance["speaker"] = 0
    return ListenV1Response.model_validate(
        {
            "metadata": {
                "request_id": "req-1",
                "sha256": "abc",
                "created": "2024-01-01T00:00:00Z",
                "duration": 1.0,
                "channels": 1,
                "models": ["m"],
                "model_info": {"m": {"name": "nova-3", "version": "1", "arch": "nova-3"}},
            },
            "results": {
                "channels": [
                    {"alternatives": [{"transcript": "Hello world.", "confidence": 0.9, "words": words}]}
                ],
                "utterances": [utterance],
            },
        }
    )


def _fake_client(response: ListenV1Response) -> SimpleNamespace:
    media = SimpleNamespace(transcribe_file=lambda **_kwargs: response)
    return SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(media=media)))


class TestBatchSubtitleTool(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix=".test_tmp_", dir=str(PROJECT_ROOT))
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_env_read_and_save_preserves_comments(self) -> None:
        env_file = self.tmp_dir / ".env"
        env_file.write_text("# header comment\nOTHER_KEY=foo\nDEEPGRAM_API_KEY=old\n", encoding="utf-8")

        save_api_key("new_secret_123", env_path=env_file)
        parsed = read_env_file(env_file)
        self.assertEqual(parsed["DEEPGRAM_API_KEY"], "new_secret_123")
        self.assertEqual(parsed["OTHER_KEY"], "foo")
        self.assertIn("# header comment", env_file.read_text(encoding="utf-8"))

    def test_collect_media_files_recursive_and_dedup(self) -> None:
        sub = self.tmp_dir / "nested"
        sub.mkdir()
        v1 = self.tmp_dir / "a.mp4"
        a1 = self.tmp_dir / "b.mp3"
        v2 = sub / "c.mkv"
        txt = sub / "notes.txt"
        for f in (v1, a1, v2, txt):
            f.write_bytes(b"dummy")

        non_rec = collect_media_files([self.tmp_dir], recursive=False)
        self.assertEqual(non_rec, [v1.resolve(), a1.resolve()])

        rec = collect_media_files([self.tmp_dir, v1], recursive=True)
        self.assertEqual(rec, [v1.resolve(), a1.resolve(), v2.resolve()])

        video_only = collect_media_files([self.tmp_dir], recursive=True, include_audio=False)
        self.assertEqual(video_only, [v1.resolve(), v2.resolve()])

    def test_build_output_paths_handles_collisions(self) -> None:
        f1 = self.tmp_dir / "d1" / "clip.mp4"
        f2 = self.tmp_dir / "d2" / "clip.mkv"
        out_dir = self.tmp_dir / "out"

        same_dir_paths = build_output_paths([f1, f2], "srt", output_dir=None)
        self.assertEqual(same_dir_paths, [f1.with_suffix(".srt"), f2.with_suffix(".srt")])

        shared_paths = build_output_paths([f1, f2], "vtt", output_dir=out_dir)
        self.assertEqual(shared_paths, [out_dir / "clip.vtt", out_dir / "clip_2.vtt"])

        # Same stem in the same folder (different container, different case) must not share a target.
        d1 = self.tmp_dir / "d1"
        same_folder = build_output_paths([d1 / "a.mp4", d1 / "A.mkv"], "srt")
        self.assertEqual(same_folder, [d1 / "a.srt", d1 / "A_2.srt"])

        # A source literally named "clip_2" must not collide with a generated "_2" suffix.
        f3 = self.tmp_dir / "d3" / "clip_2.mp4"
        tricky = build_output_paths([f1, f2, f3], "srt", output_dir=out_dir)
        self.assertEqual(tricky, [out_dir / "clip.srt", out_dir / "clip_2.srt", out_dir / "clip_2_2.srt"])

    def test_transcribe_file_output_has_no_null_speaker_labels(self) -> None:
        audio = self.tmp_dir / "a.mp3"
        audio.write_bytes(b"fake-audio")

        plain = transcribe_file(_fake_client(_make_sdk_response(diarize=False)), audio)
        vtt_text = generate_captions(plain, fmt="vtt")
        srt_text = generate_captions(plain, fmt="srt")
        self.assertIn("Hello world.", vtt_text)
        self.assertNotIn("Speaker None", vtt_text)
        self.assertNotIn("<v ", vtt_text)
        self.assertNotIn("[speaker", srt_text)

        # Speaker labels must still appear when diarization is actually on.
        diarized = transcribe_file(_fake_client(_make_sdk_response(diarize=True)), audio)
        self.assertIn("<v Speaker 0>", generate_captions(diarized, fmt="vtt"))
        self.assertIn("[speaker 0]", generate_captions(diarized, fmt="srt"))

    def test_transcribe_params_language_modes(self) -> None:
        p_multi = _build_transcribe_params({"language": "multi", "model": "nova-3"})
        self.assertEqual(p_multi["language"], "multi")
        self.assertNotIn("detect_language", p_multi)

        p_auto = _build_transcribe_params({"language": "auto", "timeout_seconds": 180})
        self.assertTrue(p_auto["detect_language"])
        self.assertNotIn("language", p_auto)
        self.assertEqual(p_auto["request_options"], {"timeout_in_seconds": 180})

    def test_caption_generation_srt_and_vtt(self) -> None:
        srt_text = generate_captions(SAMPLE_DG_RESPONSE, fmt="srt")
        self.assertIn("-->", srt_text)
        self.assertIn("Hello world.", srt_text)

        vtt_text = generate_captions(SAMPLE_DG_RESPONSE, fmt="vtt")
        self.assertTrue(vtt_text.startswith("WEBVTT"))
        self.assertIn("This is a batch test.", vtt_text)

    @unittest.skipUnless(find_ffmpeg(), "ffmpeg not installed")
    def test_ffmpeg_extract_audio_from_synthetic_video(self) -> None:
        ffmpeg_bin = find_ffmpeg()
        assert ffmpeg_bin is not None
        video = self.tmp_dir / "sample.mp4"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x120:d=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-shortest",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                str(video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio_out = self.tmp_dir / "extracted.mp3"
        result_path = extract_audio(video, audio_out, ffmpeg_bin=ffmpeg_bin)
        self.assertEqual(result_path, audio_out)
        self.assertTrue(audio_out.is_file())
        self.assertGreater(audio_out.stat().st_size, 500)

    @unittest.skipUnless(find_ffmpeg(), "ffmpeg not installed")
    def test_ffmpeg_extract_audio_does_not_hang_on_noisy_stderr(self) -> None:
        ffmpeg_bin = find_ffmpeg()
        assert ffmpeg_bin is not None
        clean = self.tmp_dir / "clean.ts"
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=s=320x240:d=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                "-shortest", "-c:v", "mpeg4", "-c:a", "mp2", "-f", "mpegts", str(clean),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        data = bytearray(clean.read_bytes())
        rng = random.Random(1)
        for _ in range(len(data) // 200):
            data[rng.randrange(4096, len(data))] = rng.randrange(256)
        corrupted = self.tmp_dir / "corrupted.ts"
        corrupted.write_bytes(bytes(data))

        # Precondition: the error output must overflow any OS pipe buffer (64 KiB on Linux, less on Windows),
        # otherwise this test would not exercise the deadlock.
        probe = subprocess.run(
            [
                ffmpeg_bin, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(corrupted), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "64k", str(self.tmp_dir / "probe.mp3"),
            ],
            capture_output=True,
        )
        self.assertGreater(len(probe.stderr), 64 * 1024)

        out = self.tmp_dir / "out.mp3"
        cancel = threading.Event()
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["result"] = extract_audio(corrupted, out, ffmpeg_bin=ffmpeg_bin, cancel_event=cancel)
            except Exception as exc:  # noqa: BLE001 - recorded and asserted below
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=60)
        if worker.is_alive():
            cancel.set()
            worker.join(timeout=10)
            self.fail("extract_audio hung on a corrupted video (stderr pipe deadlock)")
        # Success or a clean MediaExtractionError are both fine; hanging/cancel is not.
        error = outcome.get("error")
        self.assertTrue(error is None or isinstance(error, MediaExtractionError), repr(error))
        self.assertNotIsInstance(error, ExtractionCancelled)

    @unittest.skipUnless(find_ffmpeg(), "ffmpeg not installed")
    def test_ffmpeg_extract_audio_reports_stderr_detail(self) -> None:
        bogus = self.tmp_dir / "not_a_video.mp4"
        bogus.write_bytes(b"this is not a video" * 100)
        with self.assertRaises(MediaExtractionError) as ctx:
            extract_audio(bogus, self.tmp_dir / "x.mp3", ffmpeg_bin=find_ffmpeg())
        msg = str(ctx.exception)
        self.assertIn("FFmpeg 提取音频失败", msg)
        # The detail must come from captured stderr, not the exit-code fallback.
        self.assertNotIn("退出码", msg)

    def test_batch_processor_skip_failure_isolation_and_cancel(self) -> None:
        ok_file = self.tmp_dir / "good.mp3"
        skip_file = self.tmp_dir / "already.mp3"
        bad_file = self.tmp_dir / "fail.mp3"
        for f in (ok_file, skip_file, bad_file):
            f.write_bytes(b"fake-audio")
        (self.tmp_dir / "already.srt").write_text("existing", encoding="utf-8")

        def fake_transcribe(_client, path, **_kwargs):
            if Path(path).name == "fail.mp3":
                raise RuntimeError("mock API failure")
            return SAMPLE_DG_RESPONSE

        opts = BatchOptions(
            api_key="dummy-key",
            output_format="srt",
            skip_existing=True,
            extract_audio_for_video=False,
            max_workers=2,
        )
        processor = BatchProcessor(
            opts,
            client_factory=lambda _k: object(),  # type: ignore[arg-type]
            transcribe_fn=fake_transcribe,
            temp_base_dir=self.tmp_dir,
        )
        progress_events: list[tuple[int, FileStatus]] = []
        results = processor.run(
            [ok_file, skip_file, bad_file],
            on_progress=lambda idx, status, _msg, _out: progress_events.append((idx, status)),
        )

        by_name = {r.source.name: r for r in results}
        self.assertEqual(by_name["good.mp3"].status, FileStatus.DONE)
        self.assertTrue((self.tmp_dir / "good.srt").is_file())
        self.assertEqual(by_name["already.mp3"].status, FileStatus.SKIPPED)
        self.assertEqual((self.tmp_dir / "already.srt").read_text(encoding="utf-8"), "existing")
        self.assertEqual(by_name["fail.mp3"].status, FileStatus.FAILED)
        self.assertTrue(any(st == FileStatus.DONE for _, st in progress_events))

        # Test pre-cancelled run
        cancel_ev = threading.Event()
        cancel_ev.set()
        cancelled = processor.run([ok_file], cancel_event=cancel_ev)
        self.assertEqual(cancelled[0].status, FileStatus.CANCELLED)

    def test_gui_module_imports_and_initializes(self) -> None:
        import tkinter as tk
        from src.gui import SubtitleBatchApp

        root = tk.Tk()
        root.withdraw()
        try:
            app = SubtitleBatchApp(root)
            self.assertEqual(len(app.items), 0)
        finally:
            root.destroy()

    def test_gui_recovers_when_batch_run_raises(self) -> None:
        import tkinter as tk
        from src.gui import SubtitleBatchApp

        media = self.tmp_dir / "a.mp3"
        media.write_bytes(b"fake-audio")
        blocker = self.tmp_dir / "blocker.txt"
        blocker.write_text("x", encoding="utf-8")

        root = tk.Tk()
        root.withdraw()
        try:
            app = SubtitleBatchApp(root)
            app.api_key_var.set("dummy-key")
            # Parent is a regular file, so BatchProcessor.run() fails creating the output dir.
            app.output_dir_var.set(str(blocker / "out"))
            app._add_paths([media])

            with mock.patch("src.gui.messagebox.showerror") as showerror:
                app._on_start()
                assert app._worker_thread is not None
                app._worker_thread.join(timeout=10)
                self.assertFalse(app._worker_thread.is_alive())
                app._poll_events()

            self.assertFalse(app._running)
            self.assertEqual(str(app.btn_start.cget("state")), tk.NORMAL)
            self.assertEqual(str(app.btn_cancel.cget("state")), tk.DISABLED)
            self.assertEqual(app.items[0].status, FileStatus.FAILED)
            showerror.assert_called_once()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

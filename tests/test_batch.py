"""Offline unit tests for batch video subtitle generation (no live API key required)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch import BatchOptions, BatchProcessor, FileStatus, build_output_paths
from src.caption import _build_transcribe_params, generate_captions
from src.env_utils import read_env_file, save_api_key
from src.media import collect_media_files, extract_audio, find_ffmpeg

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


if __name__ == "__main__":
    unittest.main()

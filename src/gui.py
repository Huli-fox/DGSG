"""Minimal Tkinter GUI for batch video subtitle generation with Deepgram."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from .batch import BatchOptions, BatchProcessor, FileResult, FileStatus
    from .env_utils import get_api_key, save_api_key
    from .media import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, collect_media_files, find_ffmpeg
except ImportError:
    from batch import BatchOptions, BatchProcessor, FileResult, FileStatus
    from env_utils import get_api_key, save_api_key
    from media import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, collect_media_files, find_ffmpeg


LANGUAGE_CHOICES = [
    ("multi - 多语混合 (nova-3 推荐)", "multi"),
    ("auto - 自动检测单语种", "auto"),
    ("zh-CN - 中文(简体)", "zh-CN"),
    ("en - 英语", "en"),
    ("ja - 日语", "ja"),
    ("ko - 韩语", "ko"),
    ("es - 西班牙语", "es"),
    ("fr - 法语", "fr"),
    ("de - 德语", "de"),
]

MODEL_CHOICES = [
    "nova-3",
    "nova-2",
    "nova-3-medical",
]


@dataclass
class QueueItem:
    source: Path
    output: Path | None = None
    status: FileStatus = FileStatus.PENDING
    detail: str = ""


def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _parse_language_value(display_text: str) -> str:
    text = display_text.strip()
    if " - " in text:
        return text.split(" - ", 1)[0].strip()
    return text


def _open_in_explorer(path: Path) -> None:
    if not path.exists():
        return
    if os.name == "nt":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R" if path.is_file() else "", str(path)])
    else:
        target = path.parent if path.is_file() else path
        subprocess.Popen(["xdg-open", str(target)])


class SubtitleBatchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Deepgram 批量视频字幕生成工具")
        self.root.geometry("920x660")
        self.root.minsize(780, 540)

        self.ffmpeg_path = find_ffmpeg()
        self.items: list[QueueItem] = []
        self.item_iids: list[str] = []
        self._event_queue: queue.Queue[tuple] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._running = False

        self._init_vars()
        self._build_ui()
        self._poll_events()

    def _init_vars(self) -> None:
        self.api_key_var = tk.StringVar(value=get_api_key())
        self.show_key_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=True)

        self.format_var = tk.StringVar(value="srt")
        self.language_var = tk.StringVar(value=LANGUAGE_CHOICES[0][0])
        self.model_var = tk.StringVar(value="nova-3")
        self.workers_var = tk.IntVar(value=2)
        self.diarize_var = tk.BooleanVar(value=False)
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.extract_audio_var = tk.BooleanVar(value=bool(self.ffmpeg_path))
        self.output_dir_var = tk.StringVar(value="")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.summary_var = tk.StringVar(value="就绪：请添加视频或音频文件")

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 1. API Key bar
        api_frame = ttk.LabelFrame(main, text="Deepgram API Key", padding=(8, 6))
        api_frame.pack(fill=tk.X, pady=(0, 8))

        self.api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show="*")
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Checkbutton(
            api_frame,
            text="显示",
            variable=self.show_key_var,
            command=self._toggle_key_visibility,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            api_frame,
            text="保存到 .env",
            command=self._on_save_key,
        ).pack(side=tk.LEFT)

        # 2. File list toolbar & Treeview
        list_frame = ttk.LabelFrame(main, text="待处理文件", padding=(8, 6))
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        toolbar = ttk.Frame(list_frame)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        self.btn_add_files = ttk.Button(toolbar, text="添加视频/音频", command=self._on_add_files)
        self.btn_add_files.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_add_folder = ttk.Button(toolbar, text="添加文件夹", command=self._on_add_folder)
        self.btn_add_folder.pack(side=tk.LEFT, padx=(0, 6))

        self.chk_recursive = ttk.Checkbutton(toolbar, text="含子目录", variable=self.recursive_var)
        self.chk_recursive.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_remove = ttk.Button(toolbar, text="移除选中", command=self._on_remove_selected)
        self.btn_remove.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear = ttk.Button(toolbar, text="清空列表", command=self._on_clear_files)
        self.btn_clear.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="双击条目可定位文件", foreground="#666666").pack(side=tk.RIGHT)

        tree_container = ttk.Frame(list_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "dir", "status", "detail")
        self.tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("name", text="文件名")
        self.tree.heading("dir", text="所在目录")
        self.tree.heading("status", text="状态")
        self.tree.heading("detail", text="详情 / 输出")

        self.tree.column("name", width=230, anchor=tk.W)
        self.tree.column("dir", width=240, anchor=tk.W)
        self.tree.column("status", width=90, anchor=tk.CENTER)
        self.tree.column("detail", width=260, anchor=tk.W)

        y_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # 3. Options panel
        opt_frame = ttk.LabelFrame(main, text="字幕参数", padding=(8, 6))
        opt_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(opt_frame)
        row1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row1, text="格式:").pack(side=tk.LEFT)
        self.cmb_format = ttk.Combobox(
            row1, textvariable=self.format_var, values=("srt", "vtt"), state="readonly", width=6
        )
        self.cmb_format.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(row1, text="语言:").pack(side=tk.LEFT)
        self.cmb_lang = ttk.Combobox(
            row1,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_CHOICES],
            width=26,
        )
        self.cmb_lang.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(row1, text="模型:").pack(side=tk.LEFT)
        self.cmb_model = ttk.Combobox(row1, textvariable=self.model_var, values=MODEL_CHOICES, width=14)
        self.cmb_model.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(row1, text="并发:").pack(side=tk.LEFT)
        self.spn_workers = ttk.Spinbox(row1, from_=1, to=5, textvariable=self.workers_var, width=4)
        self.spn_workers.pack(side=tk.LEFT, padx=(4, 0))

        row2 = ttk.Frame(opt_frame)
        row2.pack(fill=tk.X, pady=(0, 6))

        self.chk_diarize = ttk.Checkbutton(row2, text="说话人分离 (diarize)", variable=self.diarize_var)
        self.chk_diarize.pack(side=tk.LEFT, padx=(0, 14))

        self.chk_skip = ttk.Checkbutton(row2, text="跳过已有字幕", variable=self.skip_existing_var)
        self.chk_skip.pack(side=tk.LEFT, padx=(0, 14))

        ffmpeg_label = "视频先提取音频 (FFmpeg)" if self.ffmpeg_path else "视频先提取音频 (未检测到 FFmpeg)"
        self.chk_ffmpeg = ttk.Checkbutton(row2, text=ffmpeg_label, variable=self.extract_audio_var)
        if not self.ffmpeg_path:
            self.chk_ffmpeg.configure(state=tk.DISABLED)
        self.chk_ffmpeg.pack(side=tk.LEFT)

        row3 = ttk.Frame(opt_frame)
        row3.pack(fill=tk.X)

        ttk.Label(row3, text="输出目录:").pack(side=tk.LEFT)
        self.ent_out = ttk.Entry(row3, textvariable=self.output_dir_var)
        self.ent_out.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))

        self.btn_pick_out = ttk.Button(row3, text="选择目录", command=self._on_pick_output_dir)
        self.btn_pick_out.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_reset_out = ttk.Button(
            row3, text="同原视频目录", command=lambda: self.output_dir_var.set("")
        )
        self.btn_reset_out.pack(side=tk.LEFT)

        # 4. Control & progress
        ctrl_frame = ttk.Frame(main)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_start = ttk.Button(ctrl_frame, text="开始批量生成", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_cancel = ttk.Button(ctrl_frame, text="取消", command=self._on_cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=(0, 12))

        self.progress_bar = ttk.Progressbar(ctrl_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Label(ctrl_frame, textvariable=self.summary_var).pack(side=tk.RIGHT)

        # 5. Compact log box
        log_frame = ttk.LabelFrame(main, text="日志", padding=(6, 4))
        log_frame.pack(fill=tk.X)

        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        if self.ffmpeg_path:
            self._append_log(f"已检测到 FFmpeg: {self.ffmpeg_path}")
        else:
            self._append_log("未检测到 FFmpeg：大视频将直接上传原文件，建议安装 FFmpeg 以加快上传。")

    def _toggle_key_visibility(self) -> None:
        self.api_entry.configure(show="" if self.show_key_var.get() else "*")

    def _on_save_key(self) -> None:
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showwarning("提示", "请先输入 DEEPGRAM_API_KEY。")
            return
        save_api_key(key)
        self._append_log("已将 DEEPGRAM_API_KEY 保存到项目根目录 .env")

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {item.source for item in self.items}
        added = 0
        for p in paths:
            if p in existing:
                continue
            existing.add(p)
            item = QueueItem(source=p)
            self.items.append(item)
            iid = self.tree.insert(
                "",
                tk.END,
                values=(p.name, str(p.parent), item.status.value, ""),
            )
            self.item_iids.append(iid)
            added += 1

        self.summary_var.set(f"共 {len(self.items)} 个文件待处理")
        if added:
            self._append_log(f"已添加 {added} 个媒体文件（当前共 {len(self.items)} 个）")

    def _on_add_files(self) -> None:
        video_pats = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
        all_pats = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        selected = filedialog.askopenfilenames(
            title="选择视频或音频文件",
            filetypes=[
                ("视频与音频", all_pats),
                ("仅视频", video_pats),
                ("所有文件", "*.*"),
            ],
        )
        if not selected:
            return
        files = collect_media_files(selected, recursive=False, include_audio=True)
        self._add_paths(files)

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择包含视频的文件夹")
        if not folder:
            return
        files = collect_media_files([folder], recursive=self.recursive_var.get(), include_audio=True)
        if not files:
            messagebox.showinfo("提示", "该文件夹下未找到支持的视频或音频文件。")
            return
        self._add_paths(files)

    def _on_remove_selected(self) -> None:
        selected_iids = set(self.tree.selection())
        if not selected_iids:
            return
        new_items: list[QueueItem] = []
        new_iids: list[str] = []
        for item, iid in zip(self.items, self.item_iids):
            if iid in selected_iids:
                self.tree.delete(iid)
            else:
                new_items.append(item)
                new_iids.append(iid)
        self.items = new_items
        self.item_iids = new_iids
        self.summary_var.set(f"共 {len(self.items)} 个文件待处理")

    def _on_clear_files(self) -> None:
        for iid in self.item_iids:
            self.tree.delete(iid)
        self.items.clear()
        self.item_iids.clear()
        self.progress_var.set(0.0)
        self.summary_var.set("就绪：请添加视频或音频文件")

    def _on_pick_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="选择字幕输出目录")
        if folder:
            self.output_dir_var.set(folder)

    def _on_tree_double_click(self, _event: tk.Event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        try:
            idx = self.item_iids.index(iid)
        except ValueError:
            return
        item = self.items[idx]
        target = item.output if (item.output and item.output.exists()) else item.source
        _open_in_explorer(target)

    def _set_controls_state(self, running: bool) -> None:
        self._running = running
        state = tk.DISABLED if running else tk.NORMAL
        for widget in (
            self.btn_add_files,
            self.btn_add_folder,
            self.chk_recursive,
            self.btn_remove,
            self.btn_clear,
            self.cmb_format,
            self.cmb_lang,
            self.cmb_model,
            self.spn_workers,
            self.chk_diarize,
            self.chk_skip,
            self.ent_out,
            self.btn_pick_out,
            self.btn_reset_out,
            self.btn_start,
        ):
            widget.configure(state=state)

        if self.ffmpeg_path:
            self.chk_ffmpeg.configure(state=state)
        self.btn_cancel.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _on_start(self) -> None:
        if self._running:
            return
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("缺少 API Key", "请先填写 DEEPGRAM_API_KEY。")
            return
        if not self.items:
            messagebox.showwarning("列表为空", "请先添加至少一个视频或音频文件。")
            return

        out_dir_str = self.output_dir_var.get().strip()
        output_dir = Path(out_dir_str) if out_dir_str else None
        try:
            workers = max(1, min(5, int(self.workers_var.get())))
        except Exception:
            workers = 2

        options = BatchOptions(
            api_key=api_key,
            output_format=self.format_var.get().strip() or "srt",
            model=self.model_var.get().strip() or "nova-3",
            language=_parse_language_value(self.language_var.get()),
            diarize=self.diarize_var.get(),
            output_dir=output_dir,
            skip_existing=self.skip_existing_var.get(),
            extract_audio_for_video=self.extract_audio_var.get() and bool(self.ffmpeg_path),
            max_workers=workers,
        )

        # Reset statuses in UI
        for item, iid in zip(self.items, self.item_iids):
            item.status = FileStatus.PENDING
            item.output = None
            item.detail = ""
            self.tree.set(iid, "status", FileStatus.PENDING.value)
            self.tree.set(iid, "detail", "")

        self.progress_var.set(0.0)
        self._cancel_event.clear()
        self._set_controls_state(True)

        files = [item.source for item in self.items]
        self._append_log(
            f"开始批量任务：{len(files)} 个文件 | 格式={options.output_format} | "
            f"语言={options.language or 'default'} | 模型={options.model} | 并发={options.max_workers}"
        )

        def worker() -> None:
            results: list[FileResult] = []
            error: str | None = None

            def on_progress(idx: int, status: FileStatus, message: str, out: Path | None) -> None:
                self._event_queue.put(("progress", idx, status, message, out))

            try:
                processor = BatchProcessor(options)
                results = processor.run(files, cancel_event=self._cancel_event, on_progress=on_progress)
            except Exception as exc:
                error = str(exc).strip() or exc.__class__.__name__
            finally:
                # Always notify the UI, otherwise the controls stay disabled forever.
                self._event_queue.put(("finished", results, error))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        if not self._running:
            return
        self._cancel_event.set()
        self.btn_cancel.configure(state=tk.DISABLED)
        self._append_log("正在取消剩余任务，请稍候...")

    def _update_summary(self) -> None:
        total = len(self.items)
        if total == 0:
            self.progress_var.set(0.0)
            self.summary_var.set("就绪：请添加视频或音频文件")
            return
        terminal = {FileStatus.DONE, FileStatus.SKIPPED, FileStatus.FAILED, FileStatus.CANCELLED}
        done = sum(1 for i in self.items if i.status == FileStatus.DONE)
        skipped = sum(1 for i in self.items if i.status == FileStatus.SKIPPED)
        failed = sum(1 for i in self.items if i.status == FileStatus.FAILED)
        finished = sum(1 for i in self.items if i.status in terminal)
        self.progress_var.set((finished / total) * 100.0)
        self.summary_var.set(f"进度 {finished}/{total} | 成功 {done} | 跳过 {skipped} | 失败 {failed}")

    def _mark_unfinished_failed(self, message: str) -> None:
        terminal = {FileStatus.DONE, FileStatus.SKIPPED, FileStatus.FAILED, FileStatus.CANCELLED}
        for item, iid in zip(self.items, self.item_iids):
            if item.status in terminal:
                continue
            item.status = FileStatus.FAILED
            item.detail = message
            self.tree.set(iid, "status", FileStatus.FAILED.value)
            self.tree.set(iid, "detail", message)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_events(self) -> None:
        try:
            while True:
                event = self._event_queue.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, idx, status, message, out = event
                    if 0 <= idx < len(self.items):
                        item = self.items[idx]
                        item.status = status
                        item.detail = message
                        if out is not None:
                            item.output = out
                        iid = self.item_iids[idx]
                        self.tree.set(iid, "status", status.value)
                        self.tree.set(iid, "detail", message)
                        if status == FileStatus.FAILED:
                            self._append_log(f"[失败] {item.source.name}: {message}")
                        elif status == FileStatus.DONE:
                            self._append_log(f"[完成] {item.source.name} -> {message}")
                        self._update_summary()
                elif kind == "finished":
                    _, results, error = event
                    if error is not None:
                        self._mark_unfinished_failed(f"任务中止: {error}")
                    self._set_controls_state(False)
                    self._update_summary()
                    if error is not None:
                        self._append_log(f"[错误] 批量任务异常终止：{error}")
                        messagebox.showerror("批量任务异常终止", error)
                        continue
                    done = sum(1 for r in results if r.status == FileStatus.DONE)
                    skipped = sum(1 for r in results if r.status == FileStatus.SKIPPED)
                    failed = sum(1 for r in results if r.status == FileStatus.FAILED)
                    cancelled = sum(1 for r in results if r.status == FileStatus.CANCELLED)
                    self._append_log(
                        f"批量任务结束：成功 {done}，跳过 {skipped}，失败 {failed}，取消 {cancelled}"
                    )
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_events)


def launch_gui() -> None:
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    SubtitleBatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

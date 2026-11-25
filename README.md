# Deepgram 批量视频字幕生成工具

基于 Deepgram 官方字幕生成示例改写的轻量级批量视频/音频字幕生成工具，支持通过极简桌面界面（Python 内置 `tkinter`）批量将视频或音频转为 `.srt` / `.vtt` 字幕文件。

## 主要功能

- **批量处理视频与音频**：支持多选文件或选择整个文件夹（可选递归子目录），支持常见视频（`.mp4` / `.mkv` / `.mov` / `.avi` / `.webm` 等）与音频格式。
- **FFmpeg 自动提取轻量音频**：当系统已安装 `ffmpeg` 时，自动先从视频中提取轻量化单声道 MP3 临时音轨再上传至 Deepgram，大幅降低上传体积与耗时，处理完成后自动清理临时文件。
- **并发队列与容错**：支持自定义并发数、单个文件失败不影响其余任务、支持跳过已有字幕、支持随时取消。
- **灵活参数**：支持 `srt` / `vtt` 格式切换、多语种选择（`multi` / `auto` / `zh-CN` / `en` / `ja` 等）、模型选择（默认 `nova-3`）、说话人分离（`diarize`）、自定义输出目录或默认保存到原视频同目录。
- **保留原有 CLI**：仍可直接通过命令行使用 [src/caption.py](src/caption.py)。

## 环境准备

- Python 3.10+
- Deepgram API Key（可在 [Deepgram Console](https://console.deepgram.com/) 免费获取）
- 推荐安装 `ffmpeg` 并加入系统 `PATH`（用于视频提取轻量音频）

### 在项目虚拟环境中安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动极简桌面 UI

```powershell
.\.venv\Scripts\python.exe app.py
```

首次使用时在顶部填入 `DEEPGRAM_API_KEY`（可点击「保存到 .env」持久化，下次自动加载），添加视频或文件夹后点击「开始批量生成」即可。

## 命令行用法（兼容原示例）

```powershell
# 转录单个本地文件（默认生成同名字幕 .srt）
.\.venv\Scripts\python.exe src/caption.py recording.mp3

# 指定语言和格式
.\.venv\Scripts\python.exe src/caption.py video.mp4 --language multi --format srt

# 转录公开 URL 并输出 WebVTT
.\.venv\Scripts\python.exe src/caption.py --url https://dpgr.am/spacewalk.wav --format vtt

# 开启说话人分离
.\.venv\Scripts\python.exe src/caption.py interview.wav --diarize --output interview.srt
```

## 运行测试

```powershell
# 运行离线单元测试（包含 FFmpeg 音频提取、批处理引擎、GUI 初始化测试）
.\.venv\Scripts\python.exe -m unittest tests/test_batch.py -v

# 运行原官方在线集成测试（需要设置 DEEPGRAM_API_KEY）
.\.venv\Scripts\python.exe tests/test_example.py
```

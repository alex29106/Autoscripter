"""媒体解码：用 ffmpeg 把任意音/视频转成 16kHz 单声道 wav。

所有路径用 pathlib 处理，兼容中文/日文/空格路径。
中间文件（解码出的 wav、Demucs 分离出的人声）放在项目的 .cache/wav、.cache/vocals 里，
先写到临时名字、成功后再改名（中途中断不会留下半个文件被下次当成好的复用），
转写完由调用方用 remove_intermediate 删掉。
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("subtitle.media")

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"
WAV_DIR = CACHE_ROOT / "wav"
VOCALS_DIR = CACHE_ROOT / "vocals"

# 支持的输入扩展名（ffmpeg 能解码的常见格式）
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".ts", ".m4v", ".wmv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "未找到 ffmpeg，请先安装（conda install -c conda-forge ffmpeg）。"
        )
    return exe


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTS


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    """文件内容 SHA1（用于缓存键）。大文件只读前后若干块以提速。"""
    h = hashlib.sha1()
    size = path.stat().st_size
    with path.open("rb") as f:
        if size <= 8 * chunk:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        else:
            # 头 + 尾 + 大小，足以区分文件且快
            h.update(f.read(4 * chunk))
            f.seek(-4 * chunk, 2)
            h.update(f.read(4 * chunk))
            h.update(str(size).encode())
    return h.hexdigest()[:16]


def decode_to_wav(src: Path, out_dir: Path | None = None, sample_rate: int = 16000) -> Path:
    """把 src 解码为 16k 单声道 PCM wav，返回输出路径。

    out_dir 为 None 时写入项目的 .cache/wav。已有完整的同名结果时直接复用。
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = Path(out_dir) if out_dir is not None else WAV_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_{file_sha1(src)}_{sample_rate}.wav"

    if out.exists() and out.stat().st_size > 0:
        log.debug("复用已解码 wav: %s", out)
        return out

    part = out.with_name(out.stem + ".part.wav")       # 扩展名仍是 .wav，ffmpeg 据此选输出格式
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",                      # 丢弃视频
        "-ac", "1",                 # 单声道
        "-ar", str(sample_rate),    # 采样率
        "-acodec", "pcm_s16le",
        str(part),
    ]
    log.info("ffmpeg 解码: %s", src.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg 解码失败 ({src.name}):\n{proc.stderr}")
    os.replace(part, out)
    return out


def remove_intermediate(*paths) -> None:
    """删掉转写用过的中间文件（只删 .cache/wav、.cache/vocals 里的，别处的文件一律不动）。"""
    for p in paths:
        if not p:
            continue
        p = Path(p)
        try:
            rp = p.resolve()
            if WAV_DIR.resolve() in rp.parents or VOCALS_DIR.resolve() in rp.parents:
                p.unlink(missing_ok=True)
                d = p.parent                         # Demucs 的输出是 vocals/htdemucs/<名字>/vocals.wav，顺手删空目录
                while d != VOCALS_DIR and VOCALS_DIR.resolve() in d.resolve().parents and not any(d.iterdir()):
                    d.rmdir()
                    d = d.parent
        except Exception as e:
            log.debug("删除中间文件失败 %s: %s", p, e)


def separate_vocals(src_wav: Path, model_dir: Path, out_dir: Path | None = None) -> Path:
    """用 Demucs 提取人声（去 BGM/音效），返回人声 wav 路径。

    只适合背景音乐/音效很响、人声正常音量的音频。人声音量很小的音频不要用：实测会把小音量人声分坏
    （部分语句丢失或变成读音相近的错句），所以默认不开。

    注意：Demucs(torch) 与 faster-whisper(ctranslate2) 的 cuDNN 在同进程会符号冲突
    （崩溃，exit 9），因此在**独立子进程**里运行 demucs CLI 做隔离。
    模型与缓存都指向 model_dir（不写 C 盘）；输出放在项目 .cache/vocals，先写到临时目录，成功后再移过去。
    """
    import sys
    from .vram import has_cuda

    model_dir = Path(model_dir)
    hub = model_dir / ".torch_hub"
    hub.mkdir(parents=True, exist_ok=True)

    out_root = Path(out_dir) if out_dir is not None else VOCALS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    stem = Path(src_wav).stem
    expected = out_root / "htdemucs" / stem / "vocals.wav"
    if expected.exists() and expected.stat().st_size > 0:
        log.debug("复用已分离人声: %s", expected)
        return expected

    part_root = out_root / f"_part_{stem}"
    shutil.rmtree(part_root, ignore_errors=True)
    env = os.environ.copy()
    env["TORCH_HOME"] = str(hub)
    env.setdefault("HF_HOME", str(model_dir / ".hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(model_dir / ".hf_cache"))
    env["PYTHONUTF8"] = "1"
    device = "cuda" if has_cuda() else "cpu"
    cmd = [
        sys.executable, "-m", "demucs", "--two-stems", "vocals",
        "-n", "htdemucs", "--segment", "7", "-d", device,
        "-o", str(part_root), str(src_wav),
    ]
    log.info("Demucs 人声分离(子进程, device=%s): %s", device, Path(src_wav).name)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    produced = part_root / "htdemucs" / stem / "vocals.wav"
    try:
        if proc.returncode != 0 or not produced.exists():
            raise RuntimeError(
                f"Demucs 人声分离失败 (rc={proc.returncode})。"
                f"\nstderr 末尾:\n{(proc.stderr or '')[-800:]}"
            )
        expected.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced, expected)
    finally:
        shutil.rmtree(part_root, ignore_errors=True)
    return expected


def iter_media_files(root: Path, recursive: bool) -> list[Path]:
    """列出文件夹下的媒体文件。"""
    root = Path(root)
    if root.is_file():
        return [root] if is_media_file(root) else []
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and is_media_file(p))

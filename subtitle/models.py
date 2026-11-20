"""模型注册表 + HuggingFace 本地下载。

设计要点:
- 所有模型统一下载到 model_dir 下的子目录，每个模型一个子目录。
- HF_HOME / HF_HUB_CACHE 也指向 model_dir，避免缓存写到 C 盘。
- 支持 HF_ENDPOINT 镜像（如 https://hf-mirror.com），支持断点续传。
- 运行时优先用本地已下载目录；不存在则触发 snapshot_download。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("subtitle.models")


# ---------------------------------------------------------------------------
# 模型注册表：name -> (repo_id, 本地子目录名)
# 仓库名已在 HuggingFace 上确认（见 README 的"模型仓库"一节）。
# ---------------------------------------------------------------------------
ASR_MODELS: dict[str, tuple[str, str]] = {
    # faster-whisper (CTranslate2)
    "whisper":            ("Systran/faster-whisper-large-v3",            "faster-whisper-large-v3"),
    "whisper-large-v3":   ("Systran/faster-whisper-large-v3",            "faster-whisper-large-v3"),
    "whisper-turbo":      ("deepdml/faster-whisper-large-v3-turbo-ct2",  "faster-whisper-large-v3-turbo"),
    "whisper-medium":     ("Systran/faster-whisper-medium",              "faster-whisper-medium"),
    "whisper-small":      ("Systran/faster-whisper-small",               "faster-whisper-small"),
    # 日语专用 kotoba-whisper（CTranslate2 格式，可被 faster-whisper 加载）
    "kotoba-whisper":     ("kotoba-tech/kotoba-whisper-v2.0-faster",     "kotoba-whisper-v2.0-faster"),
    # SenseVoice（通过 funasr 加载）
    "sensevoice":         ("FunAudioLLM/SenseVoiceSmall",                "SenseVoiceSmall"),
    # funasr VAD（SenseVoice 分段/时间戳需要）
    "fsmn-vad":           ("funasr/fsmn-vad",                            "fsmn-vad"),
}

MT_MODELS: dict[str, tuple[str, str]] = {
    # 默认：腾讯 Hy-MT2 翻译专用模型（Apache-2.0，2026-05 发布；8GB 显存下 4-bit 现场量化）
    "hymt2":      ("tencent/Hy-MT2-7B",                   "Hy-MT2-7B"),
    "hymt2-1.8b": ("tencent/Hy-MT2-1.8B",                 "Hy-MT2-1.8B"),   # 快速档
    # 通用 LLM 备用
    "qwen3-8b":   ("unsloth/Qwen3-8B-bnb-4bit",           "Qwen3-8B-bnb-4bit"),  # Qwen3-8B 的 NF4 预量化版
    "qwen3b":     ("Qwen/Qwen2.5-3B-Instruct",            "Qwen2.5-3B-Instruct"),
    "qwen7b":     ("Qwen/Qwen2.5-7B-Instruct",            "Qwen2.5-7B-Instruct"),
    # 回退
    "nllb":       ("facebook/nllb-200-distilled-600M",    "nllb-200-distilled-600M"),
    # 旧版混元翻译模型（腾讯混元社区许可，实测不如 Hy-MT2，仅保留兼容）
    "hunyuan":    ("tencent/Hunyuan-MT-7B",               "Hunyuan-MT-7B"),
}

# download --all-recommended 的推荐组合
RECOMMENDED_ASR = ["whisper", "kotoba-whisper", "sensevoice", "fsmn-vad"]
RECOMMENDED_MT = ["hymt2", "nllb"]

# 固定下载版本：repo_id -> commit。这些是开发和测试时实际用的版本，重新下载时拿到的是同一份权重
# （仓库作者以后更新了也不受影响）。没列出的仓库下载最新版。想要最新版可以删掉对应的行。
MODEL_REVISIONS: dict[str, str] = {
    "Systran/faster-whisper-large-v3":        "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "kotoba-tech/kotoba-whisper-v2.0-faster": "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc",
    "tencent/Hy-MT2-7B":                      "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223",
    "tencent/Hy-MT2-1.8B":                    "9a341cd1b679d3efd23b46e847b01745a71ed792",
    "unsloth/Qwen3-8B-bnb-4bit":              "1deaf68f694c40dbce295da300851729d759b21a",
    "Qwen/Qwen2.5-3B-Instruct":               "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "facebook/nllb-200-distilled-600M":       "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
}


def configure_hf_env(model_dir: Path) -> None:
    """让 HF 缓存指向 model_dir，避免写 C 盘。必须在 import 任何 HF 库前调用。"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = model_dir / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    # 启用更快/更稳的传输（若安装了 hf_transfer 则生效）
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        log.info("使用 HF 镜像站: %s", endpoint)


def _resolve(name: str, registry: dict) -> tuple[str, str]:
    if name not in registry:
        raise KeyError(
            f"未知模型别名 '{name}'。可用: {sorted(registry.keys())}"
        )
    return registry[name]


def local_path(name: str, model_dir: Path, registry: Optional[dict] = None) -> Path:
    """返回某模型的本地目录（不保证已下载）。"""
    registry = registry or {**ASR_MODELS, **MT_MODELS}
    _, subdir = _resolve(name, registry)
    return Path(model_dir) / subdir


def is_downloaded(name: str, model_dir: Path, registry: Optional[dict] = None) -> bool:
    """判断模型是否已下载完成（目录存在且含权重文件）。

    只看权重文件：只有 config/json 的目录多半是下载中断，应继续下载（snapshot_download 支持断点续传）。"""
    p = local_path(name, model_dir, registry)
    if not p.is_dir():
        return False
    patterns = ["*.bin", "*.safetensors", "*.pt", "*.onnx"]
    for pat in patterns:
        if any(p.rglob(pat)):
            return True
    return False


def ensure_model(
    name: str,
    model_dir: Path,
    registry: Optional[dict] = None,
    force: bool = False,
) -> Path:
    """确保模型已下载到本地，返回本地目录。已存在则跳过。

    使用 snapshot_download(local_dir=...)，支持断点续传与 HF_ENDPOINT 镜像。
    """
    registry = registry or {**ASR_MODELS, **MT_MODELS}
    repo_id, subdir = _resolve(name, registry)
    target = Path(model_dir) / subdir

    if not force and is_downloaded(name, model_dir, registry):
        log.info("模型已存在，跳过下载: %s -> %s", name, target)
        return target

    # 延迟导入，避免未安装时在 import 阶段报错
    from huggingface_hub import snapshot_download

    revision = MODEL_REVISIONS.get(repo_id)
    log.info("下载模型 %s (%s @ %s) -> %s", name, repo_id, revision[:10] if revision else "最新版", target)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        resume_download=True,          # 断点续传
        local_dir_use_symlinks=False,  # Windows 下用真实文件，避免符号链接问题
    )
    log.info("下载完成: %s", name)
    return target


def ensure_models(names: list[str], model_dir: Path, force: bool = False) -> dict[str, Path]:
    """批量确保多个模型（可混合 ASR/MT 别名）。"""
    registry = {**ASR_MODELS, **MT_MODELS}
    out: dict[str, Path] = {}
    for n in names:
        out[n] = ensure_model(n, model_dir, registry, force=force)
    return out

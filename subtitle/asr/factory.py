"""ASR 后端工厂：按名称构造后端实例。"""
from __future__ import annotations

from pathlib import Path

from .base import ASRBackend
from .whisper_backend import FasterWhisperBackend, KotobaWhisperBackend
from .sensevoice_backend import SenseVoiceBackend

# 用户可见的 --asr 取值
ASR_CHOICES = ["whisper", "kotoba-whisper", "sensevoice"]

# whisper 的尺寸别名 -> models 注册表别名
_WHISPER_ALIASES = {
    "whisper": "whisper",
    "whisper-large-v3": "whisper",
    "whisper-turbo": "whisper-turbo",
    "whisper-medium": "whisper-medium",
    "whisper-small": "whisper-small",
}


def build_asr(name: str, model_dir: Path, cfg: dict) -> ASRBackend:
    name = name.lower()
    if name in _WHISPER_ALIASES:
        return FasterWhisperBackend(model_dir, cfg, model_alias=_WHISPER_ALIASES[name],
                                    default_language=None, name="whisper")
    if name == "kotoba-whisper":
        return KotobaWhisperBackend(model_dir, cfg)
    if name == "sensevoice":
        return SenseVoiceBackend(model_dir, cfg)
    raise ValueError(f"未知 ASR 后端: {name}. 可用: {ASR_CHOICES}")

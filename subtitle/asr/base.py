"""ASR 后端统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..segments import Transcript


class ASRBackend(ABC):
    """所有 ASR 后端的基类。

    生命周期: __init__(配置) -> load() -> transcribe()* -> unload()。
    transcribe 接收已解码的 16k 单声道 wav 路径，返回统一的 Transcript。
    """

    name: str = "base"

    def __init__(self, model_dir: Path, cfg: dict):
        self.model_dir = Path(model_dir)
        self.cfg = cfg
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """加载模型到内存/显存。"""

    @abstractmethod
    def transcribe(self, wav_path: Path, src_lang: Optional[str] = None) -> Transcript:
        """转写单个 wav。src_lang 为 None 表示自动检测。"""

    def unload(self) -> None:
        """卸载模型并释放显存。"""
        self._model = None
        from ..vram import release_cuda
        release_cuda()

"""MT 后端工厂 + 显存/加载失败自动回退。

回退到 NLLB-600M 会让译文质量明显变差，所以每次回退都用红字提示，并把原因记在后端的 fallback 属性上
（运行结束的汇总里会再列一次）。
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import MTBackend
from .qwen_backend import QwenMTBackend
from .hymt_backend import HyMTBackend
from .nllb_backend import NLLBMTBackend
from ..vram import total_vram_gb, has_cuda
from .. import console

log = logging.getLogger("subtitle.mt.factory")

MT_CHOICES = ["hymt2", "hymt2-1.8b", "qwen3-8b", "qwen3b", "qwen7b", "nllb", "hunyuan"]
_LARGE = {"hymt2", "qwen7b", "qwen3-8b", "hunyuan"}   # 7B/8B 级，即使 4-bit 也需要约 5~6GB


def build_mt(name: str, model_dir: Path, cfg: dict) -> MTBackend:
    name = name.lower()
    if name in ("hymt2", "hymt2-1.8b", "hunyuan"):
        return HyMTBackend(model_dir, cfg, alias=name)
    if name in ("qwen3b", "qwen7b", "qwen3-8b"):
        return QwenMTBackend(model_dir, cfg, alias=name)
    if name == "nllb":
        return NLLBMTBackend(model_dir, cfg)
    raise ValueError(f"未知 MT 后端: {name}. 可用: {MT_CHOICES}")


class _FallbackOnLoad(MTBackend):
    """包装器：首选后端加载失败（显存不足、bitsandbytes 不可用等）时自动换成 NLLB，并红字提示。"""

    def __init__(self, primary: MTBackend, model_dir: Path, cfg: dict):
        super().__init__(model_dir, cfg)
        self._inner = primary
        self.name = primary.name
        self.requested = primary.name
        self.fallback = ""

    def load(self) -> None:
        try:
            self._inner.load()
        except Exception as e:
            log.error("MT 后端 %s 加载失败（%s），回退到 NLLB-200-distilled-600M", self._inner.name, e)
            self.fallback = f"{self._inner.name} 加载失败（{type(e).__name__}: {str(e)[:200]}）"
            console.warn(f"翻译模型 {self._inner.name} 加载失败（{type(e).__name__}: {str(e)[:200]}），"
                         f"改用低质量的备用模型 NLLB-200-distilled-600M。译文质量会明显变差；"
                         f"如果不想要这个结果，请先解决上面的错误再重新运行")
            try:
                self._inner.unload()
            except Exception:
                pass
            self._inner = NLLBMTBackend(self.model_dir, self.cfg)
            self.name = self._inner.name
            self._inner.load()

    @property
    def load_plan(self):
        return getattr(self._inner, "load_plan", None)

    def __getattr__(self, item):        # 其余方法/属性全部转发给实际后端
        if item == "_inner":
            raise AttributeError(item)
        return getattr(self._inner, item)

    def translate_batch(self, *a, **k):
        return self._inner.translate_batch(*a, **k)

    def translate_sentence(self, *a, **k):
        return self._inner.translate_sentence(*a, **k)

    def translate_units(self, *a, **k):
        return self._inner.translate_units(*a, **k)

    def translate_all(self, *a, **k):
        return self._inner.translate_all(*a, **k)

    def unload(self) -> None:
        self._inner.unload()


def build_mt_with_fallback(name: str, model_dir: Path, cfg: dict) -> MTBackend:
    """按显存可行性选择后端，必要时回退到 nllb；加载时真的失败也回退。两种回退都红字提示。"""
    name = name.lower()
    vram = total_vram_gb()
    if name in _LARGE and has_cuda() and vram < 6.0:
        log.warning("显存 %.1fGB 不足以运行 %s（即使 4-bit），回退到 NLLB-600M", vram, name)
        console.warn(f"显存只有 {vram:.1f}GB，放不下 {name}（4-bit 也要约 6GB），"
                     f"改用低质量的备用模型 NLLB-200-distilled-600M。可以试试 --mt hymt2-1.8b（小模型，需要约 3GB）")
        mt = NLLBMTBackend(model_dir, cfg)
        mt.requested = name
        mt.fallback = f"显存 {vram:.1f}GB 不足以运行 {name}"
        return mt
    primary = build_mt(name, model_dir, cfg)
    if isinstance(primary, NLLBMTBackend):
        primary.requested, primary.fallback = name, ""
        return primary
    return _FallbackOnLoad(primary, model_dir, cfg)

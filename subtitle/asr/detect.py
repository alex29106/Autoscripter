"""语种检测：用 Whisper 对音频前 N 秒做语言识别。"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import models
from ..vram import asr_compute_settings

log = logging.getLogger("subtitle.asr.detect")


class LanguageDetector:
    """复用一个小/快的 faster-whisper 模型做语种检测。

    默认用与主 ASR 相同的 large-v3；若只为检测可在 config 里换 small 更省时。
    """

    def __init__(self, model_dir: Path, cfg: dict, model_alias: str = "whisper"):
        self.model_dir = Path(model_dir)
        self.cfg = cfg
        self.model_alias = model_alias
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        path = models.ensure_model(self.model_alias, self.model_dir, models.ASR_MODELS)
        device, compute_type = asr_compute_settings()
        try:
            self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        except Exception as e:
            from .. import console
            console.warn(f"语种检测模型在 GPU 上加载失败（{type(e).__name__}），改用 CPU，会慢很多")
            self._model = WhisperModel(str(path), device="cpu", compute_type="int8")

    def detect(self, wav_path: Path) -> tuple[str, float]:
        """返回 (lang_iso, probability)。只看前 detect_seconds 秒。

        VAD 跟随 asr.vad_filter（默认关）：Silero VAD 可能听不见音量很小的人声，开着它检测语种时
        这类音频会被整段当成静音。"""
        if self._model is None:
            self.load()
        acfg = self.cfg.get("asr", {})
        detect_seconds = acfg.get("detect_seconds", 30)
        # faster-whisper 没有直接截断参数，这里用 clip_timestamps 限制解码窗口
        _segments, info = self._model.transcribe(
            str(wav_path),
            language=None,
            clip_timestamps=[0.0, float(detect_seconds)],
            without_timestamps=True,
            vad_filter=bool(acfg.get("vad_filter", False)),
        )
        lang = info.language or "unknown"
        prob = getattr(info, "language_probability", 0.0) or 0.0
        return lang, prob

    def unload(self) -> None:
        self._model = None
        from ..vram import release_cuda
        release_cuda()

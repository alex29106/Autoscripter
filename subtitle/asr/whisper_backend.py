"""faster-whisper 后端（whisper 默认 + kotoba-whisper 复用）。

kotoba-whisper-v2.0-faster 是 CTranslate2 格式，可直接用 faster-whisper 的
WhisperModel 加载，因此两者共用本类，仅模型路径/默认语言不同。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import ASRBackend
from ..segments import Segment, Transcript, Word
from .. import models
from ..vram import asr_compute_settings

log = logging.getLogger("subtitle.asr.whisper")


def _is_repetitive(text: str) -> bool:
    """高重复的音效/幻听（如"ゴトン ゴトン…"或"あ あ あ…"）。"""
    s = text.replace(" ", "")
    if len(s) >= 12 and len(set(s)) <= 3:
        return True
    toks = text.split()
    if len(toks) >= 6 and len(set(toks)) <= 2:
        return True
    return False


class FasterWhisperBackend(ASRBackend):
    name = "whisper"

    # 子类/实例可覆盖
    model_alias = "whisper"
    default_language: Optional[str] = None  # None=自动检测
    # kotoba 等 distil 模型的 alignment heads 不支持词级时间戳（开启会原生崩溃），
    # 这类后端置 False，退回片段级时间戳。
    supports_word_timestamps = True

    def __init__(self, model_dir: Path, cfg: dict, model_alias: Optional[str] = None,
                 default_language: Optional[str] = None, name: Optional[str] = None):
        super().__init__(model_dir, cfg)
        if model_alias:
            self.model_alias = model_alias
        if default_language is not None:
            self.default_language = default_language
        if name:
            self.name = name

    def load(self) -> None:
        from faster_whisper import WhisperModel

        path = models.ensure_model(self.model_alias, self.model_dir, models.ASR_MODELS)
        device, compute_type = asr_compute_settings()
        log.info("加载 faster-whisper: %s (device=%s, compute=%s)",
                 self.model_alias, device, compute_type)
        try:
            self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        except Exception as e:
            # GPU 加载失败时退回 CPU int8（结果一样，但慢很多），红字提示
            from .. import console
            log.warning("GPU 加载失败 (%s)，退回 CPU int8", e)
            console.warn(f"语音识别模型 {self.model_alias} 在 GPU 上加载失败（{type(e).__name__}: {str(e)[:200]}），"
                         f"改用 CPU 运行，会慢很多")
            self._model = WhisperModel(str(path), device="cpu", compute_type="int8")

    def transcribe(self, wav_path: Path, src_lang: Optional[str] = None) -> Transcript:
        if self._model is None:
            self.load()

        acfg = self.cfg.get("asr", {})
        language = src_lang or self.default_language  # None -> 自动检测
        want_wt = acfg.get("word_timestamps", True) and self.supports_word_timestamps
        if acfg.get("word_timestamps", True) and not self.supports_word_timestamps:
            log.info("后端 %s 不支持词级时间戳，使用片段级时间戳", self.name)

        want_vad = acfg.get("vad_filter", False)
        vad_params = None
        if want_vad:
            # 调低阈值以免漏掉带 BGM/轻声的语音
            vad_params = {
                "threshold": float(acfg.get("vad_threshold", 0.2)),
                "min_silence_duration_ms": int(acfg.get("vad_min_silence_ms", 700)),
            }
        kwargs = dict(
            language=language,
            task="transcribe",
            beam_size=acfg.get("beam_size", 5),
            vad_filter=want_vad,
            vad_parameters=vad_params,
            word_timestamps=want_wt,
            condition_on_previous_text=acfg.get("condition_on_previous_text", False),
            no_speech_threshold=acfg.get("no_speech_threshold", 0.6),
            compression_ratio_threshold=acfg.get("compression_ratio_threshold", 2.4),
            log_prob_threshold=acfg.get("log_prob_threshold", -1.0),
        )
        # 关 VAD 时启用幻听静音跳过
        if not want_vad and acfg.get("hallucination_silence_threshold") is not None:
            kwargs["hallucination_silence_threshold"] = float(acfg["hallucination_silence_threshold"])
        segments_iter, info = self._model.transcribe(str(wav_path), **kwargs)

        # 段质量过滤阈值（滤掉 BGM/音效幻听，如"咔嚓/啊啊/零"这类噪声段）
        drop_nsp = float(acfg.get("drop_no_speech_prob", 0.85))
        drop_lp = float(acfg.get("drop_avg_logprob", -1.2))
        drop_cr = float(acfg.get("drop_compression_ratio", 2.5))

        out: list[Segment] = []
        dropped = 0
        for seg in segments_iter:
            nsp = getattr(seg, "no_speech_prob", 0.0) or 0.0
            lp = getattr(seg, "avg_logprob", 0.0) or 0.0
            cr = getattr(seg, "compression_ratio", 0.0) or 0.0
            text = seg.text.strip()
            # 非语音/低置信/高重复（如"ゴトン ゴトン…"音效）-> 丢弃
            if (not text or nsp > drop_nsp or lp < drop_lp
                    or cr > drop_cr or _is_repetitive(text)):
                dropped += 1
                continue
            words = None
            if getattr(seg, "words", None):
                words = [Word(start=w.start, end=w.end, text=w.word) for w in seg.words
                         if w.start is not None and w.end is not None]
                if not words:
                    words = None
            out.append(Segment(start=seg.start, end=seg.end, text=text, words=words))

        lang = info.language or language or "unknown"
        prob = getattr(info, "language_probability", None)
        log.info("转写完成: %d 段（过滤 %d 段低置信/非语音）, 语种=%s (p=%.2f)",
                 len(out), dropped, lang, prob or 0.0)
        return Transcript(language=lang, backend=self.name, segments=out,
                          duration=getattr(info, "duration", 0.0) or 0.0,
                          language_probability=None if language else (float(prob) if prob is not None else None),
                          language_forced=bool(language), dropped=dropped)


class KotobaWhisperBackend(FasterWhisperBackend):
    """日语专用。底层仍是 faster-whisper，强制日语。"""
    name = "kotoba-whisper"
    model_alias = "kotoba-whisper"
    default_language = "ja"
    supports_word_timestamps = False  # distil 模型，词级对齐会崩溃

    def __init__(self, model_dir: Path, cfg: dict):
        super().__init__(model_dir, cfg, model_alias="kotoba-whisper",
                         default_language="ja", name="kotoba-whisper")

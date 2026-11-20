"""SenseVoiceSmall 后端（通过 funasr 加载）。

SenseVoice 不提供词级时间戳，因此采用 fsmn-vad 先做语音活动检测拿到分段
[start_ms, end_ms]，再对每个分段用 SenseVoice 转写，分段时间戳作为片段时间戳
（符合规格：SenseVoice 不支持词级时间戳时退而使用 VAD 分段时间戳）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import ASRBackend
from ..segments import Segment, Transcript
from .. import models
from ..vram import has_cuda

log = logging.getLogger("subtitle.asr.sensevoice")

# src_lang(ISO) -> SenseVoice 语言码
_LANG_MAP = {"zh": "zh", "yue": "yue", "en": "en", "ja": "ja", "ko": "ko"}


class SenseVoiceBackend(ASRBackend):
    name = "sensevoice"

    def __init__(self, model_dir: Path, cfg: dict):
        super().__init__(model_dir, cfg)
        self._vad = None

    def load(self) -> None:
        from funasr import AutoModel

        sv_path = models.ensure_model("sensevoice", self.model_dir, models.ASR_MODELS)
        vad_path = models.ensure_model("fsmn-vad", self.model_dir, models.ASR_MODELS)
        device = "cuda:0" if has_cuda() else "cpu"
        log.info("加载 SenseVoice + fsmn-vad (device=%s)", device)

        # SenseVoice 主模型（不带内部 vad，自己控制分段以获得时间戳）
        self._model = AutoModel(
            model=str(sv_path),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )
        # 独立 VAD 模型
        self._vad = AutoModel(
            model=str(vad_path),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    def _vad_segments(self, wav_path: Path) -> list[tuple[float, float]]:
        """返回 [(start_sec, end_sec), ...]。"""
        res = self._vad.generate(input=str(wav_path), disable_pbar=True)
        # funasr vad 返回 [{'value': [[s_ms,e_ms], ...]}]
        segs: list[tuple[float, float]] = []
        if res and isinstance(res, list):
            value = res[0].get("value", [])
            for pair in value:
                if len(pair) == 2:
                    segs.append((pair[0] / 1000.0, pair[1] / 1000.0))
        return segs

    def transcribe(self, wav_path: Path, src_lang: Optional[str] = None) -> Transcript:
        if self._model is None:
            self.load()

        import soundfile as sf
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        sv_lang = _LANG_MAP.get((src_lang or "").lower(), "auto")

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        vad_segs = self._vad_segments(wav_path)
        if not vad_segs:
            # 整段兜底
            vad_segs = [(0.0, len(audio) / sr)]

        out: list[Segment] = []
        for (s, e) in vad_segs:
            i0 = max(0, int(s * sr))
            i1 = min(len(audio), int(e * sr))
            if i1 <= i0:
                continue
            chunk = audio[i0:i1]
            res = self._model.generate(
                input=chunk,
                cache={},
                language=sv_lang,
                use_itn=True,
                batch_size_s=60,
                disable_pbar=True,
            )
            raw = res[0]["text"] if res else ""
            text = rich_transcription_postprocess(raw).strip()
            if text:
                out.append(Segment(start=round(s, 3), end=round(e, 3), text=text, words=None))

        lang = src_lang or "unknown"
        log.info("SenseVoice 转写完成: %d 段", len(out))
        return Transcript(language=lang, backend=self.name, segments=out,
                          duration=len(audio) / sr, language_forced=bool(src_lang), dropped=0)

    def unload(self) -> None:
        self._model = None
        self._vad = None
        from ..vram import release_cuda
        release_cuda()

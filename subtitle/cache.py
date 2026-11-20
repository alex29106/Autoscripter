"""转写中间结果缓存。

缓存键 = 文件名 + 文件哈希 + ASR 后端名 + 源语言 + 标签。换翻译模型时直接复用转写缓存，不重复转写。
标签（cli 生成）包含 VAD / 人声分离设置，以及会影响转写结果的 ASR 参数：
这些参数和默认值不同时，标签里多一段参数哈希，改了参数就会重新转写，而不是悄悄复用旧结果；
参数等于默认值时不加这一段。
缓存为 JSON，存于 model_dir 外的项目 .cache 目录；先写临时文件再改名，中途中断不会留下半个文件。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .media import file_sha1
from .segments import Transcript

log = logging.getLogger("subtitle.cache")

# 会影响转写结果的 ASR 参数及其默认值（与 config.yaml 的默认值一致）。
# 只用于 faster-whisper 系后端；vad_threshold 已经在 VAD 标签里，detect_seconds 只影响 --lang-adapt 的语种检测。
ASR_PARAM_DEFAULTS = {
    "beam_size": 5,
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "hallucination_silence_threshold": 2.0,
    "drop_no_speech_prob": 0.85,
    "drop_avg_logprob": -1.2,
    "drop_compression_ratio": 2.5,
    "word_timestamps": True,
    "vad_min_silence_ms": 700,
}
_WHISPER_FAMILY = {"whisper", "kotoba-whisper"}


def asr_param_tag(acfg: dict, backend: str) -> str:
    """ASR 参数与默认值相同时返回 ""，否则返回 "p" + 参数哈希（8 位）。"""
    if backend not in _WHISPER_FAMILY:
        return ""
    vals = {}
    for k, default in ASR_PARAM_DEFAULTS.items():
        v = acfg.get(k, default)
        if k == "vad_min_silence_ms" and not acfg.get("vad_filter", False):
            v = default                                   # 不开 VAD 时这个参数不起作用
        if k == "hallucination_silence_threshold" and acfg.get("vad_filter", False):
            v = default                                   # 开 VAD 时不传这个参数（见 whisper_backend）
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, (int, float)) and v is not None:
            v = float(v)
            default = float(default)
        if v != default:
            vals[k] = v
    if not vals:
        return ""
    h = hashlib.sha1(json.dumps(vals, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"p{h}"


class TranscriptCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, src_file: Path, backend: str, src_lang: str | None, tag: str = "") -> str:
        h = file_sha1(Path(src_file))
        lang = src_lang or "auto"
        suffix = f"_{tag}" if tag else ""
        return f"{Path(src_file).stem}_{h}_{backend}_{lang}{suffix}"

    def path_for(self, src_file: Path, backend: str, src_lang: str | None, tag: str = "") -> Path:
        return self.cache_dir / f"{self._key(src_file, backend, src_lang, tag)}.json"

    def load(self, src_file: Path, backend: str, src_lang: str | None, tag: str = "") -> Transcript | None:
        p = self.path_for(src_file, backend, src_lang, tag)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not data.get("segments"):
                log.warning("缓存 %s 是空转写（0 个片段），忽略并重新转写", p.name)
                return None
            log.info("命中转写缓存: %s", p.name)
            return Transcript.from_dict(data)
        except Exception as e:
            log.warning("缓存读取失败，忽略: %s (%s)", p.name, e)
            return None

    def save(self, src_file: Path, transcript: Transcript, src_lang: str | None, tag: str = "") -> Path:
        p = self.path_for(src_file, transcript.backend, src_lang, tag)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        log.debug("写入转写缓存: %s", p.name)
        return p

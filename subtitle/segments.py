"""后端无关的内部数据结构。

所有 ASR 后端统一输出 ``list[Segment]``，后续断句/翻译/写字幕都只依赖这里的
结构，不依赖任何具体后端。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Word:
    """词级时间戳（可选，部分后端不提供）。"""
    start: float
    end: float
    text: str


@dataclass
class Segment:
    """一条转写片段。

    start/end 单位为秒。words 为可选的词级时间戳列表，用于时间轴校准；
    SenseVoice 等不支持词级时间戳的后端 words 为 None。
    """
    start: float
    end: float
    text: str
    words: Optional[list[Word]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        words = d.get("words")
        word_objs = [Word(**w) for w in words] if words else None
        return cls(start=d["start"], end=d["end"], text=d["text"], words=word_objs)


@dataclass
class Transcript:
    """整段转写结果 + 元数据，可序列化为缓存 JSON。

    language_probability / language_forced / dropped（旧版本的缓存里没有，读出来是 None）：
    识别语种的把握、语种是不是用户指定的、被质量过滤丢掉的片段数——运行时醒目地显示出来。"""
    language: str                 # 检测/指定的源语言（ISO 639-1，中文为 zh）
    backend: str                  # 实际使用的 ASR 后端名
    segments: list[Segment] = field(default_factory=list)
    duration: float = 0.0
    language_probability: Optional[float] = None
    language_forced: Optional[bool] = None
    dropped: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "duration": self.duration,
            "language_probability": self.language_probability,
            "language_forced": self.language_forced,
            "dropped": self.dropped,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            language=d["language"],
            backend=d["backend"],
            duration=d.get("duration", 0.0),
            segments=[Segment.from_dict(s) for s in d["segments"]],
            language_probability=d.get("language_probability"),
            language_forced=d.get("language_forced"),
            dropped=d.get("dropped"),
        )

"""翻译单元：把相邻的“句子组”（CueGroup）合并成完整的一句话再交给翻译模型。

为什么需要：字幕条是给人看的显示单位（≤20 字、≤7 秒），一句日语常被切成 2~3 条，
ASR 也常把一句话切成两个零间隔的片段（如「私がただれた責任」|「取ってください」）。
把碎片单独翻译会拆散搭配、产生错位。

合并规则（全部可在 config.yaml 的 translation 段调整）：
  - 相邻两组的时间间隔 <= merge_gap_sec，且前一组的句子没有句末标点（。！？…）；
  - 日语（装了 fugashi 时）：前一组在语法上已经说完（以终助词、感叹词、用言/助动词的终止形等结尾）就不合并。
    不开 VAD 时 Whisper 的片段首尾相接、常常不带标点，只看“间隔 + 标点”会把一串独立的短句粘成一句。
    只看前一句自己，不把后一句接上一起分析：试过，分析器会把连起来的文字当成一句话，偏向“没说完”
    （见 is_complete_ja 的说明）；
  - 间隔按“估计的开口时间”算（Whisper 把停顿算进后一个词的开头时，片段之间名义上的间隔是 0）；
  - 合并后的文字 <= max_unit_chars，且来自的 ASR 片段数 <= max_unit_segments；
  - 合并处插入一个「、」作为软分隔（不然模型容易把两句读成一句）。
翻译单元只决定“送给模型的文字”，不改变任何字幕条的切分和时间轴。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .segments import Segment
from .segmenter import SENTENCE_END, _get_tagger

_SOFT_SEP = "、"


@dataclass
class Unit:
    text: str                                   # 送给翻译模型的整句（保留 ASR 原始标点）
    cues: list[Segment] = field(default_factory=list)
    seg_indices: list[int] = field(default_factory=list)
    group_sizes: list[int] = field(default_factory=list)    # 合并进来的每个句子组各有几条字幕（排版用）


_FINAL_FORMS = ("終止形", "命令形", "意志推量形")


def is_complete_ja(sentence: str):
    """这句日语在语法上是否已经说完。True / False；判断不了（没装 fugashi、空句）返回 None。

    只看这一句自己的最后一个词：感叹词、终助词（よ/ね/な/か…）、用言或助动词的终止形/命令形/意志形 → 说完了；
    名词、格助词/接续助词、连用形、副词等 → 可能还没说完（「私がただれた責任」|「取ってください」）。

    已知的漏判（宁可少合并：少合并只是少一点上下文，多合并会把不相干的两句话揉在一起翻译）：
      - 终止形和连体形同形，被拆到两个片段的定语从句（「思いを寄せていた」|「彼女に」）会被当成已说完；
      - 句末的 か/の 单独看一律是终助词，间接疑问（「何を言ってるか」|「わからなくても…」）会被当成已说完。
    “把后一句接上再分析一次”来分辨这两种情况的做法试过，没有采用：分析器把连起来的文字当成一句话，
    会把真正说完的句子也判成没说完（「楽しかった」+「ちょっと…」→ 连体形；句末的「の」+「ねえ…」→ 不是终助词）。"""
    tagger = _get_tagger()
    if tagger is None:
        return None
    last = None
    for w in tagger(sentence or ""):
        if w.feature.pos1 not in ("補助記号", "空白"):
            last = w
    if last is None:
        return None
    f = last.feature
    if f.pos1 == "感動詞":
        return True
    if f.pos1 == "助詞":
        return f.pos2 == "終助詞"
    if f.pos1 in ("動詞", "形容詞", "助動詞"):
        return str(getattr(f, "cForm", "") or "").startswith(_FINAL_FORMS)
    return False


def build_units(groups, cfg: dict, lang: str = "") -> list[Unit]:
    tcfg = cfg.get("translation", {})
    check_complete = (lang or "").lower().split("-")[0] == "ja" and tcfg.get("merge_require_incomplete", True)
    merge_gap = float(tcfg.get("merge_gap_sec", 0.3))
    max_chars = int(tcfg.get("max_unit_chars", 60))
    max_segs = int(tcfg.get("max_unit_segments", 3))

    units: list[Unit] = []
    prev_sent = ""
    for g in groups:
        sent = (g.sentence or "").strip()
        if not sent or not g.cues:
            continue
        if units:
            u = units[-1]
            start = max(g.cues[0].start, getattr(g, "voiced_start", None) or g.cues[0].start)
            gap = start - u.cues[-1].end
            prev = u.text.rstrip()
            segs = set(u.seg_indices) | {getattr(g, "seg_index", -1)}
            if (gap <= merge_gap and prev and prev[-1] not in SENTENCE_END
                    and len(u.text) + len(sent) + 1 <= max_chars and len(segs) <= max_segs
                    and not (check_complete and is_complete_ja(prev_sent))):
                sep = "" if prev[-1] in "、，, " else _SOFT_SEP
                u.text = prev + sep + sent
                u.cues.extend(g.cues)
                u.seg_indices.append(getattr(g, "seg_index", -1))
                u.group_sizes.append(len(g.cues))
                prev_sent = sent
                continue
        units.append(Unit(text=sent, cues=list(g.cues), seg_indices=[getattr(g, "seg_index", -1)],
                          group_sizes=[len(g.cues)]))
        prev_sent = sent
    return units

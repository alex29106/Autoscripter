"""断句 + 翻译阶段编排（CLI、图形界面与评测脚本共用同一条代码路径）。

输入一份 Transcript 和已加载的 MT 后端，输出：
  - src_cues      原文字幕条（时间轴由断句器决定）
  - tgt_mono      单语译文文件用的译文条
  - tgt_bilingual 双语文件用的译文条（条数和时间轴与 src_cues 完全一致）
  - 句子模式（默认，translation.granularity: sentence）：整句翻译，再排成字幕条。
      单语（translation.mono_display）：
        sentence（默认）：译文自己排版，不要求与原文条一一对应——整句放得下就是一条
            （时间从这句的第一条原文条到最后一条），放不下就在译文的标点/词边界切成尽量少的几条，
            每条的时间取它覆盖的那几条原文条。所以单语文件的条数可以比原文文件少。
            按规则排不出来时用 distribute.layout_mono_relaxed：宁可某条超字数，
            也不把同一段译文放到两条上；只有译文短到切不出足够的段数时，才让后一条“延续显示”同一段文字，并计数。
        repeat（旧）：条数与原文相同，整句 ≤ 每条字数上限时这句的每一条都显示整句；
        split（旧）：条数与原文相同，按各条原文字数比例切分。
      双语：按各条原文字数比例切分（同一句话按时间铺开，不追求逐行直译）。
  - 逐条模式（旧，granularity: cue）：每条单独翻译，两份译文相同。

“同一句话有没有被重复显示”按来源判断，而不是看文字（说话人自己把一句话说两遍是正常的，
那是两个翻译单元、两次翻译，各显示一次）：每个翻译单元的译文，在单语文件里排出的各条（不算延续显示的条）
依次拼起来必须正好等于这句译文——不多（重复）不少（丢字）。做不到的单元记在 layout_stats["mismatch"]。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .segments import Segment, Transcript
from .segmenter import build_cue_groups, clean_text, max_chars_for

log = logging.getLogger("subtitle.pipeline")

# 比较“拼起来是否等于整句”时忽略的字符：空白，以及会被清洗成空格的逗号/句号/停顿符
_IGNORE = re.compile(r"[\s，。、；：,.;:…～~]")


@dataclass
class StageResult:
    src_cues: list[Segment]
    tgt_mono: list[Segment]
    tgt_bilingual: list[Segment]
    unit_stats: dict | None = None      # 翻译方式计数：sentence / no-context / cue-fallback / source，error=出错次数
    units: list | None = None           # 调试/评测用：每个翻译单元一个 dict（原文、译文、处理方式、排版方式、各条文字）
    layout_stats: dict | None = None    # 排版计数，见 _new_layout_stats
    skipped: str = ""                   # 没有翻译时的原因


def _base(lang: str) -> str:
    return (lang or "").lower().split("-")[0]


def _cues(src_cues: list[Segment], texts: list[str], tgt: str) -> list[Segment]:
    return [Segment(start=c.start, end=c.end, text=clean_text(t, tgt) or t, words=None)
            for c, t in zip(src_cues, texts)]


def same_text(pieces, text) -> bool:
    """各条文字依次拼起来是不是正好等于整句（忽略空白和会被清洗掉的标点）。"""
    return _IGNORE.sub("", "".join(pieces)) == _IGNORE.sub("", text or "")


def _new_layout_stats() -> dict:
    return {
        "one": 0,          # 整句放进一条的单元
        "split": 0,        # 按规则切成几条的单元
        "relaxed": 0,      # 按规则排不出来、用了退路的单元
        "hold": 0,         # 延续显示的条（译文太短切不开，后一条继续显示同一段文字）
        "over": 0,         # 超出每条字数上限的单语条
        "mismatch": 0,     # 单语条拼起来 ≠ 整句译文的单元（重复显示或丢字）；默认方式下应为 0
        "bi_repeat": 0,    # 双语文件里因为译文切不开而重复显示整句的条
        # 另外按排版方式计数：cue（逐条翻译/保留原文）、repeat / proportional（旧的单语方式）
    }


def segment_and_translate(transcript: Transcript, mt, cfg: dict,
                          src_lang: str, tgt_lang: str, progress=None) -> StageResult:
    """断句并翻译。mt 为 None 或源/目标同语种时不翻译（译文条 = 原文条，skipped 写明原因）。
    progress(已完成句数, 总句数) 可选（图形界面显示进度用）。"""
    groups = build_cue_groups(transcript, cfg)
    src_cues = [c for g in groups for c in g.cues]
    src, tgt = _base(src_lang), _base(tgt_lang)
    if mt is None:
        return StageResult(src_cues, list(src_cues), list(src_cues), skipped="没有加载翻译模型")
    if src == tgt:
        return StageResult(src_cues, list(src_cues), list(src_cues),
                           skipped=f"源语种（{src}）与目标语种相同，没有翻译")

    tcfg = cfg.get("translation", {})
    if tcfg.get("granularity", "sentence") == "cue":
        translated = mt.translate_all([c.text for c in src_cues], src, tgt)
        tgt_cues = _cues(src_cues, translated, tgt)
        return StageResult(src_cues, tgt_cues, list(tgt_cues))

    from .units import build_units
    from .distribute import (split_proportional, display_mono, layout_mono, layout_mono_relaxed,
                             display_width)
    units = build_units(groups, cfg, src)
    log.info("%d 条字幕 -> %d 个翻译单元（%d 个跨多条）", len(src_cues), len(units),
             sum(len(u.cues) > 1 for u in units))
    results = (mt.translate_units(units, src, tgt, progress=progress) if progress
               else mt.translate_units(units, src, tgt))
    scfg = cfg.get("segmentation", {})
    max_chars = max_chars_for(tgt, cfg)
    max_dur = float(scfg.get("max_duration_sec", 7.0))
    mode = tcfg.get("mono_display", "sentence")
    mono_cues: list[Segment] = []
    bi_texts: list[str] = []
    detail = []
    ls = _new_layout_stats()
    for k, (u, r) in enumerate(zip(units, results)):
        cue_texts = [c.text for c in u.cues]
        if r["per_cue"] is not None:            # 退回了逐条翻译（或保留原文）：与原文条一一对应
            pieces = [(i, i + 1, t, False) for i, t in enumerate(r["per_cue"])]
            bi = list(r["per_cue"])
            how = "cue"
        else:
            bi = split_proportional(r["text"], cue_texts)
            if not same_text(bi, r["text"]):
                ls["bi_repeat"] += len(bi) - 1
            lay = (layout_mono(r["text"], u.cues, max_chars, max_dur, group_sizes=u.group_sizes)
                   if mode == "sentence" else None)
            if lay is not None:
                pieces = [(a, b, t, False) for a, b, t in lay]
                how = "one" if len(lay) == 1 else "split"
            elif mode == "sentence":            # 按规则排不出来：退路（不重复、不丢字；躲不开时延续显示并计数）
                pieces, info = layout_mono_relaxed(r["text"], u.cues, max_chars, max_dur,
                                                   group_sizes=u.group_sizes)
                how = "relaxed"
                ls["hold"] += info["hold"]
                log.info("单元 %d 按规则排不出来，用退路排成 %d 条（延续显示 %d 条，超字数 %d 条）：%r",
                         k, len(pieces), info["hold"], info["over"], r["text"][:40])
            elif mode == "repeat":              # 旧方式（需要用 --mono-display repeat 明确选择）
                pieces = [(i, i + 1, t, False) for i, t in enumerate(display_mono(r["text"], cue_texts, max_chars))]
                how = "repeat"
            else:                               # split：与原文条一一对应，按比例切分
                pieces = [(i, i + 1, t, False) for i, t in enumerate(bi)]
                how = "proportional"
            if not same_text([t for _a, _b, t, held in pieces if not held], r["text"]):
                ls["mismatch"] += 1
                if mode == "sentence":
                    log.warning("单元 %d 的单语条拼起来不等于整句译文（重复或丢字）：%r -> %r",
                                k, r["text"], [t for _a, _b, t, _h in pieces])
        ls[how] = ls.get(how, 0) + 1
        mono_cues += [Segment(start=u.cues[a].start, end=u.cues[b - 1].end,
                              text=clean_text(t, tgt) or t, words=None) for a, b, t, _h in pieces]
        bi_texts += bi
        detail.append(dict(src=u.text, zh=r["text"] if r["text"] is not None else " / ".join(r["per_cue"]),
                           how=r["how"], src_cues=len(u.cues), mono_cues=len(pieces), layout=how,
                           start=u.cues[0].start, end=u.cues[-1].end,
                           pieces=[[t, held] for _a, _b, t, held in pieces]))
    assert len(bi_texts) == len(src_cues), "双语译文条数必须与原文条数一致"
    ls["over"] = sum(1 for c in mono_cues if display_width(c.text) > max_chars)
    stats: dict = {}
    for r in results:
        stats[r["how"]] = stats.get(r["how"], 0) + 1
    errors = sum(r.get("errors", 0) for r in results)
    if errors:
        stats["error"] = errors
    if len(mono_cues) != len(src_cues):
        log.info("单语译文排成 %d 条（原文 %d 条）", len(mono_cues), len(src_cues))
    return StageResult(src_cues, mono_cues, _cues(src_cues, bi_texts, tgt), stats, detail, ls)

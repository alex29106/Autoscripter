"""评测打分（规则的修改记录见 data/scorer_changelog.txt）。

对一段“只看中文的人读到的”译文（若干条连起来）判断：
  meaning_ok : 每组关键意思正则都命中、没有命中已知错误译法、不含日文假名
  frag       : 多条的句子里，只有 0~1 个汉字的条数（如「是」「了」「。」）
  added      : 译文长度远超参考译文（>1.8 倍 + 4 字），疑似把上下文或编造内容塞了进来
"""
from __future__ import annotations

import re

KANA = re.compile(r"[぀-ヿ]")
CJK = re.compile(r"[一-鿿]")
_STRIP_REF = re.compile(r"[\s，。！？、…／/（）()“”]")
_STRIP_OUT = re.compile(r"[\s，。！？、…／/（）()“”,.!?]")


def has_kana(s: str) -> bool:
    return bool(KANA.search(s or ""))


def score(item: dict, per_cue: list[str]) -> dict:
    pc = [p or "" for p in per_cue]
    joined = "".join(pc)
    hits = [bool(re.search(g, joined)) for g in item["must"]]
    bad = [p for p in item.get("must_not", []) if re.search(p, joined)]
    kana = any(has_kana(p) for p in pc)
    n = len(pc)
    frag = sum(1 for p in pc if len(CJK.findall(p)) <= 1) if n > 1 else 0
    ref_len = len(_STRIP_REF.sub("", item.get("ref", "")))
    out_len = len(_STRIP_OUT.sub("", joined))
    added = bool(ref_len) and out_len > 1.8 * ref_len + 4
    return dict(meaning_ok=all(hits) and not bad and not kana,
                must_cov=sum(hits) / len(hits), must_not_hit=bad, kana=kana, frag=frag, added=added)


def overlapping_text(item: dict, cues_all: dict, out_cues: list) -> list[str]:
    """按时间重叠取出与评测条目对应的输出字幕文本（适用于任何断句方式）。

    out_cues: [(start, end, text)]。取与条目时间段重叠 >= 该输出条自身时长 50% 的条，
    或盖住条目时间段 50% 以上的条（单语文件一句话排成一条时，一条可能比评测条目长）；
    相邻完全相同的文字只保留一份。
    注意：因为去了重，这个分数看不见“同一句译文被重复显示”（例如 repeat 方式）。
    重复显示由 run.py 的结构检查（pipeline 的 mismatch，按翻译单元核对）负责，不靠这个分数。
    """
    f, rng = item["src"].split("#")
    a, b = map(int, rng.split("-"))
    t0, t1 = cues_all[f][a - 1]["start"], cues_all[f][b - 1]["end"]
    sel = [z for (s, e, z) in out_cues
           if min(e, t1) - max(s, t0) >= 0.5 * min(max(e - s, 1e-3), max(t1 - t0, 1e-3))]
    out = []
    for z in sel:
        if not out or out[-1] != z:
            out.append(z)
    return out

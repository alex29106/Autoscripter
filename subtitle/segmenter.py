"""字幕断句。

把 ASR 的 Transcript 切成最终字幕条（cue），满足：
- 每条不超过语种对应的最大字符数（zh/ja 20、en 42）。
- 每条显示时长不超过 max_duration（默认 7s）。
- 时间戳优先用词级时间戳校准；无词级时间戳时在片段内按字符线性插值。
- 中/日/粤字幕去掉逗号、句号（停顿用空格分隔），保留问号和感叹号。
- 日语：装了 fugashi + unidic-lite 时按形态素/文节边界切分，并让同一句切出的各条长度均衡；
  没装时退回旧的启发式规则（标点 + 常见助词 + 字符数）。
- 显示时长不足 min_duration_sec 的条向后面的空隙延长（不与下一条重叠）。
- 句内停顿（仅日语形态素模式）：Whisper 常把停顿算进相邻的词里，使某个词长达数秒。
  判断“这句要不要切、在哪切、每条有多长”时用收紧后的时间线（一个词最多算 字数 × max_char_sec 秒的发音，
  多出来的部分视为它前面的停顿；同一个形态素内部不留间隔），切点优先落在停顿处。
  ASR 文本里的空格也是切点。写出的时间轴仍以原始时间戳为准；只有一条超过 max_duration 时，
  才取其中“发音最多”的 max_duration 窗口（通常是收掉开头的停顿，而不是截掉结尾）。
"""
from __future__ import annotations

import logging
import math

from .segments import Segment, Transcript

log = logging.getLogger("subtitle.segmenter")

# 句末标点（会强制在此处收束一条）
SENTENCE_END = set("。．.!！?？…～~")
# 停顿标点（可切分点）
PAUSE = set("、，,；;：:・")
# 日语常见助词/接续，靠近上限时可在其后切分（仅旧规则使用）
JA_PARTICLES = set("はがをにへとももでやのかねよわねえ")
# 需要去标点、空格分隔停顿的语种
CJK_LANGS = {"zh", "ja", "yue"}
# 清洗时要从 CJK 文本中移除（替换为空格）的标点：逗号、句号、停顿符等；问号/感叹号保留
_KEEP_IN_DISPLAY = set("!！?？")
_CJK_STRIP = (SENTENCE_END | PAUSE) - _KEEP_IN_DISPLAY


def _char_timeline(seg: Segment) -> list[tuple]:
    """把一个片段展开成逐字符 (char, start, end, word_index) 时间线（没有词级时间戳时 word_index 为 -1）。"""
    chars: list[tuple] = []
    if seg.words:
        for wi, w in enumerate(seg.words):
            wt = w.text
            n = len(wt)
            if n == 0:
                continue
            dur = max(w.end - w.start, 1e-3)
            nv = sum(1 for ch in wt if ch != " ") or n      # 词里的空格（Whisper 的词常以空格开头）不占时长
            k = 0
            for ch in wt:
                if ch == " " and nv != n:
                    t = w.start + dur * k / nv
                    chars.append((ch, t, t, wi))
                else:
                    chars.append((ch, w.start + dur * k / nv, w.start + dur * (k + 1) / nv, wi))
                    k += 1
    if not chars:
        text = seg.text
        n = len(text) or 1
        dur = max(seg.end - seg.start, 1e-3)
        for k, ch in enumerate(text):
            chars.append((ch, seg.start + dur * k / n, seg.start + dur * (k + 1) / n, -1))
    return chars


def _tokens(chars, lang) -> list[tuple[int, int, bool]]:
    """切成原子片段 (start_idx, end_idx_excl, ends_sentence)，保留全部字符。"""
    toks: list[tuple[int, int, bool]] = []
    start = 0
    N = len(chars)
    for i in range(N):
        ch = chars[i][0]
        boundary = False
        ends_sentence = False
        if ch in SENTENCE_END:
            boundary = True
            ends_sentence = True
        elif ch in PAUSE:
            boundary = True
        elif lang == "ja" and ch in JA_PARTICLES:
            boundary = True
        elif lang not in CJK_LANGS and ch == " ":
            boundary = True
        if boundary:
            toks.append((start, i + 1, ends_sentence))
            start = i + 1
    if start < N:
        toks.append((start, N, False))
    return toks


def _hard_split(s: int, e: int, max_chars: int) -> list[tuple[int, int]]:
    """原子片段本身超长时，按 max_chars 硬切。"""
    if e - s <= max_chars:
        return [(s, e)]
    out = []
    i = s
    while i < e:
        j = min(i + max_chars, e)
        out.append((i, j))
        i = j
    return out


def _pack(chars, toks, max_chars: int, max_dur: float) -> list[tuple[int, int]]:
    """把原子片段打包成 cue 的字符区间，受 max_chars / max_dur 约束。"""
    cues: list[tuple[int, int]] = []
    cur_s = cur_e = None

    def flush():
        nonlocal cur_s, cur_e
        if cur_s is not None:
            cues.append((cur_s, cur_e))
            cur_s = cur_e = None

    for (s, e, ends_sent) in toks:
        for (ps, pe) in _hard_split(s, e, max_chars):
            if cur_s is None:
                cur_s, cur_e = ps, pe
            else:
                new_chars = pe - cur_s
                new_dur = chars[pe - 1][2] - chars[cur_s][1]
                if new_chars > max_chars or new_dur > max_dur:
                    flush()
                    cur_s, cur_e = ps, pe
                else:
                    cur_e = pe
        if ends_sent:
            flush()
    flush()
    return cues


def _collapse_runs(raw: str) -> str:
    """折叠异常重复：同一标点/字符连续出现 >3 次压成 1 个（防 MT 退化产生的垃圾串）。"""
    import re
    # 连续 4+ 个相同非字母数字字符 -> 1 个
    raw = re.sub(r"([^\w一-鿿぀-ヿ])\1{3,}", r"\1", raw)
    # 连续 6+ 个相同字符（含文字）-> 压到 2 个
    raw = re.sub(r"(.)\1{5,}", r"\1\1", raw)
    return raw


def _clean(raw: str, lang: str) -> str:
    raw = _collapse_runs(raw.strip())
    if lang in CJK_LANGS:
        out = []
        for ch in raw:
            if ch in _CJK_STRIP:
                out.append(" ")
            else:
                out.append(ch)
        raw = "".join(out)
    # 折叠多余空格
    raw = " ".join(raw.split())
    return raw.strip()


def clean_text(raw: str, lang: str) -> str:
    """对外清洗接口：CJK 去句末标点、停顿转空格、折叠空白。"""
    return _clean(raw, (lang or "").lower().split("-")[0])


def _sentence_spans(chars) -> list[tuple[int, int]]:
    """把字符时间线按句末标点切成句子区间 [(s,e), ...]，覆盖全部字符。"""
    spans = []
    start = 0
    N = len(chars)
    for i in range(N):
        if chars[i][0] in SENTENCE_END:
            spans.append((start, i + 1))
            start = i + 1
    if start < N:
        spans.append((start, N))
    return spans


def max_chars_for(lang: str, cfg: dict) -> int:
    """某语种每条字幕最多几个字：config 的 segmentation.max_chars[语种]，没写的语种用 default。
    断句（原文）和译文排版（pipeline）都用这一个函数，规则只有一份。"""
    maxc = cfg.get("segmentation", {}).get("max_chars", {})
    base_lang = (lang or "").lower().split("-")[0]
    return int(maxc.get(base_lang, maxc.get("default", 42)))


def _resolve_params(lang: str, cfg: dict):
    scfg = cfg.get("segmentation", {})
    base_lang = (lang or "").lower().split("-")[0]
    max_dur = float(scfg.get("max_duration_sec", 7.0))
    return base_lang, max_chars_for(lang, cfg), max_dur


def _max_char_sec(cfg: dict) -> float:
    """一个字符最多算多少秒的发音（超出部分视为停顿）；0 表示不做这项修正。"""
    return float(cfg.get("segmentation", {}).get("max_char_sec", 1.0))


class CueGroup:
    """一句话对应的一组源 cue + 原始整句文本（供翻译）。

    sentence 保留 ASR 原始标点，让翻译模型拿到完整、良构的句子；
    cues 是切好的源字幕条（已清洗、带时间轴）；seg_index 是它来自第几个 ASR 片段；
    voiced_start 是估计的“实际开口时间”（开头的停顿被 Whisper 算进第一个词时，它晚于 cues[0].start）。
    """
    __slots__ = ("sentence", "cues", "seg_index", "voiced_start")

    def __init__(self, sentence: str, cues: list[Segment], seg_index: int = -1,
                 voiced_start: float | None = None):
        self.sentence = sentence
        self.cues = cues
        self.seg_index = seg_index
        self.voiced_start = cues[0].start if voiced_start is None and cues else voiced_start


# ---------------------------------------------------------------------------
# 日语：形态素边界 + 均衡切分（需要 fugashi + unidic-lite）
# ---------------------------------------------------------------------------
_TAGGER = None
_TAGGER_TRIED = False
# 这些词性前面不能断开（会把词尾、助词、助动词、标点甩到下一条开头）
_DEP_POS = {"助詞", "助動詞", "接尾辞", "補助記号"}
# 在这些接续助词之后断开最自然（从句边界）
_CLAUSE_END = {"て", "で", "が", "けど", "けれど", "けれども", "から", "ので", "のに", "し", "ば", "と", "たら", "ながら"}


def _get_tagger():
    global _TAGGER, _TAGGER_TRIED
    if not _TAGGER_TRIED:
        _TAGGER_TRIED = True
        try:
            import fugashi
            _TAGGER = fugashi.Tagger()
        except Exception as e:  # 未安装或词典缺失：退回旧规则
            log.warning("未能加载 fugashi/unidic-lite（%s），日语断句退回旧的启发式规则", e)
            _TAGGER = None
    return _TAGGER


def _ja_tokens(text: str) -> list[tuple[int, int, str, str, str]]:
    """形态素列表 [(起点, 终点, pos1, pos2, 表层形)]。"""
    toks = []
    pos = 0
    for w in _get_tagger()(text):
        s = text.find(w.surface, pos)
        if s < 0:
            continue
        toks.append((s, s + len(w.surface), w.feature.pos1, w.feature.pos2, w.surface))
        pos = s + len(w.surface)
    return toks


def _ja_break_points(text: str, toks=None) -> dict[int, float]:
    """返回允许断开的位置 {i: 加分}，表示可以在 text[i] 之前断开。"""
    if toks is None:
        toks = _ja_tokens(text)
    out: dict[int, float] = {}
    for k in range(1, len(toks)):
        s, _e, pos1, pos2, _sur = toks[k]
        _ps, _pe, ppos1, _ppos2, psur = toks[k - 1]
        if pos1 in _DEP_POS:
            continue
        # 〜てくれる / 〜ている / 〜てしまう：て形后面的补助动词与前面是一个文节
        if pos2 == "非自立可能" and psur in ("て", "で"):
            continue
        bonus = 0.0
        if psur in PAUSE or ppos1 == "補助記号":
            bonus = 0.5
        elif ppos1 == "助詞" and psur in _CLAUSE_END:
            bonus = 0.3
        elif ppos1 == "助詞" and psur == "の":     # 连体「の」之后尽量不断（名词短语内部）
            bonus = -0.3
        out[s] = bonus
    return out


def _balanced_cuts(n: int, cands: dict[int, float], max_chars: int, k: int, too_long=None):
    """在候选断点里选 k-1 个，使各段长度尽量相等（动态规划）；做不到返回 None。

    too_long(a, b) 为真表示 [a, b) 这一段的显示时长超限，不允许。"""
    ideal = n / k
    pts = [0] + sorted(cands) + [n]
    inf = float("inf")
    best = {(0, 0): (0.0, None)}
    for j in range(1, k + 1):
        for b in pts[1:]:
            if (j < k and b == n) or (j == k and b != n):
                continue
            cur = (inf, None)
            for a in pts:
                if a >= b or (j - 1, a) not in best or b - a > max_chars:
                    continue
                if too_long is not None and too_long(a, b):
                    continue
                cost = best[(j - 1, a)][0] + (b - a - ideal) ** 2 - cands.get(b, 0.0) * ideal ** 2 * 0.25
                if cost < cur[0]:
                    cur = (cost, a)
            if cur[1] is not None:
                best[(j, b)] = cur
    if (k, n) not in best:
        return None
    cuts, b, j = [], n, k
    while j > 0:
        a = best[(j, b)][1]
        cuts.append(b)
        b, j = a, j - 1
    return sorted(cuts)[:-1]


# 句内停顿：不短于这么多秒的无声间隔可以作为切点（即使后面是助词），停顿越长越优先
_PAUSE_CUT_SEC = 0.5
_NO_PAUSE_CUT = {"助動詞", "接尾辞", "補助記号"}


def _tight_times(sub, idx, toks, char_sec: float):
    """收紧后的逐字符时间 (st, en, w_st, w_en)（只含非空格字符，与 idx 对齐）：

    1) 一个词的时长超过 字数 × char_sec 时，把它的开始时间收到 结束 - 字数 × char_sec
       （多出来的算作词前的停顿）；
    2) 同一个形态素内部不留间隔：前面的字贴到后面的字上（Whisper 的子词切分与形态素不一致时，
       停顿会被算到一个词的中间，如「一|応」）。
    w_st / w_en 是只做了第 1 步的时间（对“实际在发音的时间”的估计，一条字幕超时、要取舍时用）。
    “结束时间可靠”来自 Whisper 的对齐方式（停顿多被算进后一个词的开头），并非每次都成立，
    所以这套时间只用来决定怎么切，不直接写进字幕。"""
    st = [sub[i][1] for i in idx]
    en = [sub[i][2] for i in idx]
    wid = [sub[i][3] if len(sub[i]) > 3 else -1 for i in idx]
    if char_sec > 0:
        k = 0
        while k < len(idx):
            j = k
            while wid[k] >= 0 and j + 1 < len(idx) and wid[j + 1] == wid[k]:
                j += 1
            cnt = j - k + 1
            w_end = en[j]
            if wid[k] >= 0 and w_end - st[k] > cnt * char_sec:
                w_start = w_end - cnt * char_sec
                for m in range(cnt):
                    st[k + m] = w_start + char_sec * m
                    en[k + m] = w_start + char_sec * (m + 1)
            k = j + 1
    w_st, w_en = list(st), list(en)
    for (a, b, _p1, _p2, _sur) in toks:
        for i in range(min(b, len(idx)) - 2, a - 1, -1):
            if st[i + 1] - en[i] > 1e-3:
                d = en[i] - st[i]
                en[i] = st[i + 1]
                st[i] = en[i] - d
    return st, en, w_st, w_en


def _best_window(start: float, end: float, max_dur: float, voiced) -> tuple[float, float]:
    """[start, end] 超过 max_dur 时，取覆盖发音时间（voiced 里各区间）最多的 max_dur 窗口，并列时取最早的。"""
    lo, hi = start, end - max_dur
    cand = {lo, hi}
    for s, e in voiced:
        for x in (s, e, s - max_dur, e - max_dur):
            if lo < x < hi:
                cand.add(x)

    def cover(w: float) -> float:
        return round(sum(max(0.0, min(e, w + max_dur) - max(s, w)) for s, e in voiced), 4)

    best = max(sorted(cand), key=cover)
    return best, best + max_dur


def _split_sentence_ja(sub, max_chars: int, max_dur: float, char_sec: float = 0.0):
    """把一句（字符时间线 sub）切成若干 [a, b) 区间（sub 的下标，含空格字符）。

    条数取“每条都不超过 max_chars 个字、不超过 max_dur 秒”的最小值；切点在文节边界或句内停顿处，
    停顿处优先，其次让各条字数均衡。ASR 文本里的空格（Whisper 自己标出的停顿）也是切点。
    单独一个形态素就超时的（如拖得很长的拟声词）不切进词里，整个词作为一条，时长由调用方取舍。
    返回 (区间列表, 发音开始时间, 发音结束时间)，后两者是对“实际在发音的时间”的估计，
    按 sub 的下标对齐（空格字符沿用原值），供调用方在一条字幕超时时取舍。"""
    idx = [i for i, c in enumerate(sub) if c[0] != " "]
    text = "".join(sub[i][0] for i in idx)
    n = len(text)
    t_st = [c[1] for c in sub]
    t_en = [c[2] for c in sub]
    if n == 0:
        return [], t_st, t_en
    spaces = [k for k in range(1, n) if idx[k] - idx[k - 1] > 1]       # text[k] 前面有空格
    toks = []
    for a, b in zip([0] + spaces, spaces + [n]):                       # 空格两侧不会是同一个词
        toks.extend((a + s, a + e, p1, p2, sur) for s, e, p1, p2, sur in _ja_tokens(text[a:b]))
    st, en, w_st, w_en = _tight_times(sub, idx, toks, char_sec)
    for k, i in enumerate(idx):
        t_st[i], t_en[i] = w_st[k], w_en[k]
    tok_of = [-1] * n
    for ti, (a, b, _p1, _p2, _sur) in enumerate(toks):
        for i in range(a, min(b, n)):
            tok_of[i] = ti

    def too_long(a: int, b: int) -> bool:
        if tok_of[a] >= 0 and tok_of[a] == tok_of[b - 1]:      # 单个形态素：切不动，不算超时
            return False
        return en[b - 1] - st[a] > max_dur + 1e-6

    if n <= max_chars and not too_long(0, n):
        return [(0, len(sub))], t_st, t_en
    cands = _ja_break_points(text, toks)
    for b in spaces:
        cands[b] = max(cands.get(b, 0.0), 0.5)
    for (b, _e, pos1, _pos2, _sur) in toks:     # 句内停顿：可以在助词前切，但不切进词里、不切在助动词/词尾前
        if b == 0 or pos1 in _NO_PAUSE_CUT:
            continue
        gap = st[b] - en[b - 1]
        if gap >= _PAUSE_CUT_SEC:
            cands[b] = max(cands.get(b, 0.0), min(gap, 2.0))
    cuts = None
    kk = max(2, math.ceil(n / max_chars))
    while cuts is None and kk <= n:
        cuts = _balanced_cuts(n, cands, max_chars, kk, too_long)
        kk += 1
    if cuts is None:
        # 没有合法切法（如超过 max_chars 的片假名串、没有词级时间戳的超长片段）：按字数均分，时长由调用方兜底截断
        k = min(n, max(2, math.ceil(n / max_chars), math.ceil((en[-1] - st[0]) / max_dur)))
        cuts = sorted({round(n * j / k) for j in range(1, k)} - {0, n})
    bounds = [0] + cuts + [n]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        lo = idx[a]
        hi = idx[b - 1] + 1
        spans.append((lo, hi))
    # 让区间首尾相接（把中间的空格并入前一段），保证覆盖整句
    fixed = []
    for j, (lo, hi) in enumerate(spans):
        lo = 0 if j == 0 else fixed[-1][1]
        hi = len(sub) if j == len(spans) - 1 else hi
        fixed.append((lo, hi))
    return fixed, t_st, t_en


def _use_ja_morphology(base_lang: str, cfg: dict) -> bool:
    if base_lang != "ja":
        return False
    if not cfg.get("segmentation", {}).get("ja_morphology", True):
        return False
    return _get_tagger() is not None


def build_cue_groups(transcript: Transcript, cfg: dict) -> list[CueGroup]:
    """按 ASR 片段（内部再按句末标点拆句）切 cue。每句一个 CueGroup。全局保证时间轴单调、不重叠。"""
    base_lang, max_chars, max_dur = _resolve_params(transcript.language, cfg)
    morph = _use_ja_morphology(base_lang, cfg)
    char_sec = _max_char_sec(cfg)

    groups: list[CueGroup] = []
    prev_end = 0.0
    for si, seg in enumerate(transcript.segments):
        if not seg.text.strip():
            continue
        chars = _char_timeline(seg)
        if not chars:
            continue
        for (ss, se) in _sentence_spans(chars):
            sub = chars[ss:se]
            sentence_raw = "".join(c[0] for c in sub).strip()
            if not sentence_raw:
                continue
            if morph:
                spans, t_st, t_en = _split_sentence_ja(sub, max_chars, max_dur, char_sec)
            else:
                spans = _pack(sub, _tokens(sub, base_lang), max_chars, max_dur)
                t_st, t_en = [c[1] for c in sub], [c[2] for c in sub]
            cues: list[Segment] = []
            voiced = None
            for (a, b) in spans:
                text = _clean("".join(sub[k][0] for k in range(a, b)), base_lang)
                if not text:
                    continue
                vis = [k for k in range(a, b) if sub[k][0] != " "]      # 首尾的空格不参与计时
                if voiced is None:
                    voiced = t_st[vis[0]]
                start = max(sub[vis[0]][1], prev_end)
                end = max(sub[vis[-1]][2], start + 0.05)
                if end - start > max_dur:
                    # 超过 max_dur：只显示其中发音最多的那一段（多数情况是收掉开头的停顿）
                    start, end = _best_window(start, end, max_dur, [(t_st[k], t_en[k]) for k in vis])
                cues.append(Segment(start=round(start, 3), end=round(end, 3), text=text, words=None))
                prev_end = end
            if cues:
                groups.append(CueGroup(sentence=sentence_raw, cues=cues, seg_index=si,
                                       voiced_start=max(voiced, cues[0].start)))
    _extend_short_cues([c for g in groups for c in g.cues], cfg)
    return groups


def _extend_short_cues(cues: list[Segment], cfg: dict) -> None:
    """显示不足 min_duration_sec 的条向后面的空隙延长（不与下一条重叠，不超过 max_duration）。

    Whisper 的词级时间戳在片段末尾偶尔会把最后几个词压到几十毫秒内，导致字幕一闪而过。"""
    scfg = cfg.get("segmentation", {})
    min_dur = float(scfg.get("min_duration_sec", 0.6))
    max_dur = float(scfg.get("max_duration_sec", 7.0))
    if min_dur <= 0:
        return
    for i, c in enumerate(cues):
        if c.end - c.start >= min_dur:
            continue
        limit = cues[i + 1].start if i + 1 < len(cues) else c.start + min_dur
        new_end = min(c.start + min_dur, limit, c.start + max_dur)
        if new_end > c.end:
            c.end = round(new_end, 3)


def segment_transcript(transcript: Transcript, cfg: dict) -> list[Segment]:
    """把整段转写切成最终字幕 cue（扁平 Segment 列表）。"""
    cues: list[Segment] = []
    for g in build_cue_groups(transcript, cfg):
        cues.extend(g.cues)
    return cues

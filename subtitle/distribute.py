"""把一个翻译单元的整句译文排成字幕条。

几种方式：
  layout_mono(...)         单语文件（默认）：译文自己排版，不要求与原文条一一对应。用最少的条数，
                           每条 <= max_chars 字、<= max_dur 秒，每条的时间取它覆盖的那几条原文条的起止；
                           整句放得下就只出一条。翻译单元由几个句子组合并而成时，一条字幕要么落在
                           一个句子组内部，要么正好由几个完整的句子组组成（不把上一句的尾巴和下一句的前半拼成一条）。
  layout_mono_relaxed(...) layout_mono 排不出来时的退路：同样的搜索，约束换成按固定优先级比较的代价，一定有解。
                           宁可某条超字数，也不把同一段译文放到两条上；唯一躲不开的情况（译文能切出的段数比
                           时长逼出来的条数还少）明确标成“延续显示”，由调用方计数。
  split_proportional(...)  按各条原文字数的比例找切点，条数与原文条相同（双语文件使用；切不开时每条放整句）
  display_mono(...)        旧的单语方式：整句不超过 max_chars 时每条都显示整句，否则按比例切分

切点规则（通用的中文断行规则，不针对任何样本）：
  - 优先落在中文标点之后，其次 jieba 词边界，实在不行才在词中间；
  - 某条不能以标点、闭括号/闭引号或附着虚词（了/的/吗/呢/吧/啊…）开头，不能以开括号结尾；
  - 尽量不以“不/没/很/在/把/被/对…”这类要和后文连读的字结尾；
  - 句子够长时每条至少 2 个汉字（避免出现只有「。」「了」的条）。
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

log = logging.getLogger("subtitle.distribute")

_PUNCT = set("，。！？、…；：,.!?;:~～—―")
_BOUND = set("了的吗呢吧啊呀哦啦嘛着过们么得地")
_OPEN = set("「『（(《“‘")
_CLOSE = set("」』）)》”’")
_FWD = set("不没很太最更都也还就才再又只把被给让向对和跟与在为从")
_CJK = re.compile(r"[一-鿿]")
_JIEBA = None


def _jieba():
    """延迟加载 jieba，并把它的词典缓存放进项目的 .cache（默认在系统临时目录，曾出现长时间卡住）。"""
    global _JIEBA
    if _JIEBA is None:
        import jieba
        root = Path(__file__).resolve().parent.parent / ".cache"
        root.mkdir(parents=True, exist_ok=True)
        jieba.dt.tmp_dir = str(root)
        jieba.setLogLevel(logging.WARNING)
        _JIEBA = jieba
    return _JIEBA


def _cjk(s: str) -> int:
    return len(_CJK.findall(s))


def _weights(src_cues: list[str]) -> list[float]:
    w = [max(1, len(c.replace(" ", ""))) for c in src_cues]
    tot = sum(w)
    out, acc = [], 0
    for x in w[:-1]:
        acc += x
        out.append(acc / tot)
    return out


def _is_cjk_text(s: str) -> bool:
    core = re.sub(r"\s", "", s)
    return bool(core) and _cjk(core) >= 0.3 * len(core)


def _split_spaced(text: str, src_cues: list[str]) -> list[str]:
    """非中日韩译文（如英文）：只在词与词之间（空格处）切，优先标点后，按比例。"""
    n = len(src_cues)
    words = text.split()
    if len(words) < n:
        return [text] * n
    targets = _weights(src_cues)
    L = len(text)
    starts, pos = [], 0
    for w in words:                     # 每个词在原文中的起点
        pos = text.find(w, pos)
        starts.append(pos)
        pos += len(w)
    cands = {}
    for k in range(1, len(words)):
        cands[k] = 0.0 if words[k - 1][-1] in _PUNCT else 0.12
    best = {(0, 0): (0.0, None)}
    inf = float("inf")
    for j in range(1, n):
        for b in sorted(cands):
            cur = (inf, None)
            for (jj, a), (cost, _) in list(best.items()):
                if jj != j - 1 or a >= b:
                    continue
                c = cost + abs(starts[b] / L - targets[j - 1]) + cands[b]
                if c < cur[0]:
                    cur = (c, a)
            if cur[1] is not None:
                best[(j, b)] = cur
    final = min(((cost, a) for (jj, a), (cost, _) in best.items() if jj == n - 1 and a < len(words)),
                default=(inf, None))
    if final[1] is None:
        return [text] * n
    cuts, b, j = [], final[1], n - 1
    while j > 0:
        cuts.append(b)
        b = best[(j, b)][1]
        j -= 1
    cuts = sorted(cuts)
    out, prev = [], 0
    for c in cuts + [len(words)]:
        out.append(" ".join(words[prev:c]))
        prev = c
    return out


def _zh_cut_points(zh: str) -> dict[int, float]:
    """中文译文里可以断开的位置 {p: 代价}，表示在 zh[p] 之前断开。代价越小越好。"""
    L = len(zh)
    cands: dict[int, float] = {p: 1.0 for p in range(1, L)}       # 词中间：最后手段
    pos = 0
    for tok in _jieba().lcut(zh):
        pos += len(tok)
        if 0 < pos < L:
            cands[pos] = min(cands[pos], 0.12)                    # 词边界
    for i, ch in enumerate(zh):
        if ch in _PUNCT and 0 < i + 1 < L:
            cands[i + 1] = 0.0                                    # 标点之后
    cands = {p: c for p, c in cands.items()
             if zh[p] not in _PUNCT and zh[p] not in _BOUND and zh[p] not in _CLOSE and zh[p - 1] not in _OPEN}
    for p in list(cands):
        if zh[p - 1] in _FWD:
            cands[p] += 0.5
        elif zh[p - 1] == "的":            # 句中“的”多为连体，与后面的名词连读（“护卫的 / 委托”）
            cands[p] += 0.4
    return cands


def _spaced_cut_points(text: str) -> dict[int, float]:
    """用空格分词的译文（英文、韩文等）：只能在词与词之间断开，优先标点之后。"""
    cands: dict[int, float] = {}
    for m in re.finditer(r"\s+(?=\S)", text):
        if m.start() > 0:
            cands[m.end()] = 0.0 if text[m.start() - 1] in _PUNCT else 0.12
    return cands


def _ja_cut_points(text: str) -> dict[int, float]:
    """日文译文：在文节边界断开（和断句器切日语原文用的是同一套形态素规则）。没装 fugashi 时没有可用的切点。"""
    from . import segmenter
    if segmenter._get_tagger() is None:
        return {}
    return {p: max(0.0, 0.12 - 0.24 * bonus) for p, bonus in segmenter._ja_break_points(text).items()
            if 0 < p < len(text)}


_KANA = re.compile(r"[぀-ヿ]")
_JK = re.compile(r"[぀-ヿ一-鿿]")


def _text_kind(text: str) -> str:
    """译文按哪种文字的规则断行：ja（含假名）/ zh（汉字为主）/ spaced（用空格分词：英文、韩文等）。"""
    if _KANA.search(text):
        return "ja"
    return "zh" if _is_cjk_text(text) else "spaced"


# 数“实字”的函数（一条字幕至少要有几个实字）；用空格分词的文字不需要
_CORE = {"zh": _cjk, "ja": lambda s: len(_JK.findall(s))}
_CUTS = {"zh": _zh_cut_points, "ja": _ja_cut_points, "spaced": _spaced_cut_points}


def layout_mono(text: str, cues, max_chars: int = 20, max_dur: float = 7.0, min_cjk: int = 2,
                group_sizes=None):
    """单语字幕排版：把整句译文排成尽量少的几条，返回 [(a, b, 文字)]，表示这一条覆盖 cues[a:b]。

    cues 是这个翻译单元的原文条（带时间轴）。约束：每条显示字数 <= max_chars，
    时长（第一条原文条的开始到最后一条的结束）<= max_dur。条数取能满足约束的最小值；
    条数相同时，让“译文切到的比例”尽量接近“原文条累计字数的比例”（同一句话按说话进度铺开），
    切点按中文断行规则选。整句放得下就返回一条。
    group_sizes（可选）：这个单元由几个句子组（ASR 片段里的一句）合并而成、每组各几条。给了以后，
    一条字幕要么落在一个句子组内部，要么正好由几个完整的句子组组成。原因：句子组的边界是说话人
    自己的停顿/换气处，是最自然的切点；“上一组 + 下一组的前半”拼成一条，等于放着自然的切点不用、
    却把下一句从中间切开，下一句的前半还会提前显示出来。
    译文是什么文字就按什么文字的规则找切点（中文 / 日文文节 / 空格分词）。
    排不出来时返回 None（译文太长，条数用到和原文条一样多仍超字数；或者译文能切出的段数不够），
    由调用方改用 layout_mono_relaxed。"""
    text = (text or "").strip()
    m = len(cues)
    if m == 0:
        return []
    if m == 1 or not text:
        return [(0, m, text)]
    kind = _text_kind(text)
    core = _CORE.get(kind)
    length = display_len if core else (lambda t: len(t.strip()))

    def dur(a: int, b: int) -> float:
        return cues[b - 1].end - cues[a].start

    owner = None
    if group_sizes and len(group_sizes) > 1 and sum(group_sizes) == m:
        owner = [j for j, n in enumerate(group_sizes) for _ in range(n)]

    def allowed(a: int, b: int) -> bool:
        """cues[a:b] 能不能排成一条：在同一个句子组内，或者正好是几个完整的句子组。"""
        if owner is None or owner[a] == owner[b - 1]:
            return True
        return (a == 0 or owner[a - 1] != owner[a]) and (b == m or owner[b] != owner[b - 1])

    total = length(text)
    if total <= max_chars and dur(0, m) <= max_dur + 1e-6:
        return [(0, m, text)]
    L = len(text)
    cands = _CUTS[kind](text)
    pts = sorted(cands)
    w = [max(1, len(c.text.replace(" ", ""))) for c in cues]
    tot = sum(w)
    frac = [0.0]
    for x in w:
        frac.append(frac[-1] + x / tot)
    inf = float("inf")
    # 时长决定的最少条数：每条的时长 = 它覆盖的第一条原文条的开始到最后一条的结束（含中间的停顿），
    # 所以按“每段不超过 max_dur”贪心切出的段数才是下限（用 整句跨度 ÷ max_dur 的话，
    # 句中有长停顿时会高估，把排得下的句子判成排不出来）
    g_min = max(2, math.ceil(total / max_chars), len(_hold_ranges(cues, 0, m, max_dur)))
    for g in range(g_min, m + 1):
        need = (min_cjk if core(text) >= min_cjk * g else 1) if core else 0
        # best[(j, i, p)]：前 j 条覆盖 cues[:i] 和 text[:p] 的最小代价
        layers: list[dict] = [{(0, 0): (0.0, None)}]
        for j in range(1, g + 1):
            layer: dict = {}
            last = j == g
            for i in ([m] if last else range(j, m - (g - j) + 1)):
                for p in ([L] if last else pts):
                    cur = (inf, None)
                    for (i0, p0), (cost, _) in layers[j - 1].items():
                        if i0 >= i or p0 >= p or dur(i0, i) > max_dur + 1e-6 or not allowed(i0, i):
                            continue
                        piece = text[p0:p]
                        if length(piece) > max_chars or (need and core(piece) < need):
                            continue
                        c = cost if last else cost + abs(p / L - frac[i]) + cands[p]
                        if c < cur[0]:
                            cur = (c, (i0, p0))
                    if cur[1] is not None:
                        layer[(i, p)] = cur
            layers.append(layer)
            if not layer:
                break
        if len(layers) == g + 1 and (m, L) in layers[g]:
            out, key = [], (m, L)
            for j in range(g, 0, -1):
                prev = layers[j][key][1]
                out.append((prev[0], key[0], text[prev[1]:key[1]].strip()))
                key = prev
            return out[::-1]
    return None


def _hold_ranges(cues, a: int, b: int, max_dur: float) -> list[tuple[int, int]]:
    """把 cues[a:b] 切成尽量少的几段，每段的时长 <= max_dur（单独一条原文条就超时的不再切）。"""
    out, s = [], a
    for i in range(a + 1, b + 1):
        if i == b or cues[i].end - cues[s].start > max_dur + 1e-6:
            out.append((s, i))
            s = i
    return out


def layout_mono_relaxed(text: str, cues, max_chars: int = 20, max_dur: float = 7.0, min_cjk: int = 2,
                        group_sizes=None):
    """layout_mono 排不出来时的退路。返回 (排版, 说明)。

    排版是 [(a, b, 文字, held)]：这一条覆盖 cues[a:b]；held=True 表示它没有自己的译文，
    是上一条的文字继续显示。说明是 {"hold": 延续显示的条数, "over": 超字数的条数, "group": 跨句子组的条数}。

    搜索空间和 layout_mono 一样（译文切成几段，每段覆盖连续的几条原文条），但把硬约束换成按下面的
    优先级逐项比较的代价，所以一定有解（最差是整句一段）：
      1. 延续显示的条数最少。一段译文覆盖的原文条超过 max_dur 时，这段时间要拆成几条，后面几条只能
         继续显示同一段文字。只有“译文能切出的段数比时长逼出来的条数还少”（很长的一段话只译出几个字）
         时才躲不开；
      2. 超出 max_chars 的字数（平方和）最小：译文比这几条原文条容得下的长时，宁可某条超字数，
         也不把同一段译文放到两条上、不丢字；
      3. 跨句子组的条数最少（见 layout_mono 的 group_sizes）；
      4. 实字不足 min_cjk 个的碎片条最少；5. 条数最少；6. 切点代价（与 layout_mono 相同）。
    任何情况下，held=False 的各条文字首尾相接、合起来就是整句译文（不丢字、不重复）。"""
    text = (text or "").strip()
    m = len(cues)
    info = {"hold": 0, "over": 0, "group": 0}
    if m == 0:
        return [], info
    L = len(text)
    kind = _text_kind(text) if L else "spaced"
    core = _CORE.get(kind)
    cands = _CUTS[kind](text) if L else {}
    pts = sorted(cands)
    # 前缀和：text[:p] 里的显示字数 / 实字数（中日文）
    vis, cor = [0], [0]
    for ch in text:
        vis.append(vis[-1] + (display_len(ch) if core else 0))
        cor.append(cor[-1] + (core(ch) if core else 0))

    def length(p0: int, p: int) -> int:
        return vis[p] - vis[p0] if core else len(text[p0:p].strip())

    owner = None
    if group_sizes and len(group_sizes) > 1 and sum(group_sizes) == m:
        owner = [j for j, n in enumerate(group_sizes) for _ in range(n)]

    def crosses(a: int, b: int) -> int:
        if owner is None or owner[a] == owner[b - 1]:
            return 0
        return 0 if (a == 0 or owner[a - 1] != owner[a]) and (b == m or owner[b] != owner[b - 1]) else 1

    w = [max(1, len(c.text.replace(" ", ""))) for c in cues]
    tot = sum(w)
    frac = [0.0]
    for x in w:
        frac.append(frac[-1] + x / tot)

    # best[i][p] = (代价, 前驱)：若干段覆盖 cues[:i] 和 text[:p] 的最小代价。
    # 代价按优先级逐项比较：(延续, 超字数², 跨组, 碎片, 条数, 切点)。i、p 都只增不减，按 i 从小到大推即可。
    best: list[dict] = [dict() for _ in range(m + 1)]
    best[0][0] = ((0, 0, 0, 0, 0, 0.0), None)
    for i0 in range(m):
        for p0, (cost, _) in list(best[i0].items()):
            nxt = [q for q in pts if q > p0]
            for i in range(i0 + 1, m + 1):
                h = len(_hold_ranges(cues, i0, i, max_dur)) - 1
                g = crosses(i0, i)
                for p in ((L,) if i == m else nxt):
                    nc = cor[p] - cor[p0]
                    if L and (p <= p0 or (core and nc < 1)):
                        continue
                    over = max(0, length(p0, p) - max_chars)
                    tiny = 1 if core and nc < min_cjk else 0
                    step = 0.0 if i == m else abs(p / L - frac[i]) + cands[p]
                    new = (cost[0] + h, cost[1] + over * over, cost[2] + g, cost[3] + tiny, cost[4] + 1,
                           cost[5] + step)
                    old = best[i].get(p)
                    if old is None or new < old[0]:
                        best[i][p] = (new, (i0, p0))
    if L not in best[m]:                 # 整句没有一个实字（如纯标点）：整句作为一段
        best[m][L] = ((0, 0, 0, 0, 1, 0.0), (0, 0))
    pieces, key = [], (m, L)
    while key != (0, 0):
        prev = best[key[0]][key[1]][1]
        pieces.append((prev[0], key[0], prev[1], key[1]))
        key = prev
    out = []
    for a, b, p0, p in pieces[::-1]:
        t = text[p0:p].strip()
        info["group"] += crosses(a, b)
        for k, (s, e) in enumerate(_hold_ranges(cues, a, b, max_dur)):
            out.append((s, e, t, k > 0))
            info["hold"] += k > 0
            info["over"] += length(p0, p) > max_chars
    return out, info


def display_width(text: str) -> int:
    """一条字幕的显示字数：中日文不计空白和会被清洗掉的标点，其他文字（英文、韩文等）按字符数。"""
    return display_len(text) if _text_kind(text or "") != "spaced" else len((text or "").strip())


def split_proportional(zh: str, src_cues: list[str], min_cjk: int = 2) -> list[str]:
    """双语文件用：按各条原文字数的比例把整句译文切成 len(src_cues) 段（条数必须与原文相同）。
    译文短到切不出这么多段（如一个词的译文对两条原文）时，每条都放整句——只在双语文件里发生，
    调用方会计数（pipeline 的 bi_repeat）。"""
    n = len(src_cues)
    zh = (zh or "").strip()
    if n <= 1:
        return [zh]
    kind = _text_kind(zh)
    if kind == "spaced":                        # 英文、韩文等用空格分词的译文
        return _split_spaced(zh, src_cues)
    L = len(zh)
    if L < n:                                   # 太短，切不开：每条都放整句
        return [zh] * n
    core = _CORE[kind]
    targets = _weights(src_cues)
    cands = _CUTS[kind](zh)
    need = min_cjk if core(zh) >= min_cjk * n else 1
    pts = sorted(cands)
    inf = float("inf")
    best = {(0, 0): (0.0, None)}
    for j in range(1, n):
        for b in pts:
            cur = (inf, None)
            for (jj, a), (cost, _) in list(best.items()):
                if jj != j - 1 or a >= b or core(zh[a:b]) < need:
                    continue
                c = cost + abs(b / L - targets[j - 1]) + cands[b]
                if c < cur[0]:
                    cur = (c, a)
            if cur[1] is not None:
                best[(j, b)] = cur
    final = (inf, None)
    for (jj, a), (cost, _) in best.items():
        if jj == n - 1 and core(zh[a:]) >= need and cost < final[0]:
            final = (cost, a)
    if final[1] is None:
        return _split_plain(zh, targets)
    cuts, b, j = [], final[1], n - 1
    while j > 0:
        cuts.append(b)
        b = best[(j, b)][1]
        j -= 1
    cuts = sorted(cuts)
    out, prev = [], 0
    for p in cuts + [L]:
        out.append(zh[prev:p].strip())
        prev = p
    return out


def _split_plain(zh: str, targets: list[float]) -> list[str]:
    """约束无法满足时的退路：按比例在字符位置切（保证每条非空）。"""
    L = len(zh)
    cuts, last = [], 0
    for k, t in enumerate(targets):
        p = min(max(last + 1, round(t * L)), L - (len(targets) - k))
        cuts.append(p)
        last = p
    out, prev = [], 0
    for p in cuts + [L]:
        out.append(zh[prev:p].strip() or zh)
        prev = p
    return out


def display_len(zh: str) -> int:
    """字幕里实际显示的字数：不计空白和会被清洗成空格的逗号/句号/停顿符（保留的 ？！ 计入）。"""
    return len(re.sub(r"[\s，。、；：,.;:…～~]", "", zh or ""))


def display_mono(zh: str, src_cues: list[str], max_chars: int = 20) -> list[str]:
    """单语字幕：整句显示字数不超过 max_chars 时每条都显示整句，否则按比例切分。"""
    if len(src_cues) <= 1:
        return [(zh or "").strip()]
    if display_len(zh) <= max_chars:
        return [(zh or "").strip()] * len(src_cues)
    return split_proportional(zh, src_cues)

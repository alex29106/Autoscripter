"""VTT 字幕格式校验脚本。

检查:
  - WEBVTT 头
  - 序号从 1 连续递增
  - 时间戳格式 HH:MM:SS.mmm --> HH:MM:SS.mmm
  - 时间轴递增且不重叠（start < end，且 start >= 上一条 end）
  - 文件名符合规则 <stem>_<code><ext>.vtt
  - （目录模式）同一源文件的原文文件与双语文件条数、时间轴一致；
    单语译文文件可以比原文少（一句话排成一条），但每一条必须正好由连续的若干条原文条合并而成，
    且合起来覆盖全部原文条（不丢条、不错位）
另外给出“警告”（不影响返回码，提示可能的质量问题）:
  - 显示不足 0.3 秒的条
  - 中文轨（_cn 文件、双语文件的中文行）里出现日文假名（多半是没翻译）
  - 中文轨里只有 0~1 个汉字的条（碎片，如「是」「了」）
  - 单语译文里相邻两条文字相同。只看文件分不清“说话人自己说了两遍”和“同一句译文被放到了两条上”，
    所以同目录里有对应的原文文件时，按原文分两类报：原文也相同（多半是说话人重复，正常）/
    原文不同（可疑：同一段译文重复显示，或译文太短、后一条延续显示）。
    能确定是不是重复显示的检查在 pipeline 里（按翻译单元核对，见 subtitle/pipeline.py 的 mismatch）。

用法:
  python verify_vtt.py <某个.vtt 或 目录>
返回码 0 表示格式全部通过，非 0 表示有格式错误。
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 管道/文件里不乱码
except Exception:
    pass

KANA = re.compile(r"[぀-ヿ]")
CJK = re.compile(r"[一-鿿]")
SHORT_WARN_SEC = 0.3

TS = r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
TS_LINE = re.compile(rf"^{TS} --> {TS}$")
# 文件名: <stem>_<code><原扩展>.vtt，code 为 cn/iso 或 双语 a-b
NAME_RE = re.compile(r"^.+_([a-z]{2,3}(?:-[a-z]{2,3})?)\.[^.]+\.vtt$")


def _max_cue_sec() -> float:
    """每条字幕最长显示时长：取项目 config.yaml 的 segmentation.max_duration_sec（读不到时用规格里的 7 秒）。"""
    try:
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8"))
        return float(cfg["segmentation"]["max_duration_sec"])
    except Exception:
        return 7.0


MAX_CUE_SEC = _max_cue_sec()
_DUR_TOL = 0.02


def _to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _zh_lines(path: Path, text_lines: list[str]) -> list[str]:
    """该条里属于中文轨的文字行：_cn 文件取全部；双语文件 <a>-cn 默认第 1 行是译文（--swap-lines 时第 2 行）。"""
    m = re.match(r"^.+_([a-z]{2,3})(?:-([a-z]{2,3}))?\.[^.]+\.vtt$", path.name)
    if not m:
        return []
    a, b = m.group(1), m.group(2)
    if b is None:
        return text_lines if a == "cn" else []
    if b == "cn":   # 无法从文件本身判断是否 swap：两行里不含假名的那行当作中文
        return [t for t in text_lines if not KANA.search(t)] or text_lines[:1]
    return []


def quality_warnings(path: Path) -> list[str]:
    """质量警告（不影响返回码）。"""
    warns = []
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except Exception:
        return warns
    blocks = [b for b in re.split(r"\n\s*\n", "\n".join(text.split("\n")[1:]).strip("\n")) if b.strip()]
    short = kana = tiny = 0
    for b in blocks:
        bl = b.split("\n")
        if len(bl) < 3:
            continue
        m = TS_LINE.match(bl[1].strip())
        if m and _to_sec(*m.group(5, 6, 7, 8)) - _to_sec(*m.group(1, 2, 3, 4)) < SHORT_WARN_SEC:
            short += 1
        for t in _zh_lines(path, bl[2:]):
            if KANA.search(t):
                kana += 1
            if len(CJK.findall(t)) <= 1:
                tiny += 1
    if short:
        warns.append(f"{short} 条显示不足 {SHORT_WARN_SEC} 秒")
    if kana:
        warns.append(f"中文轨有 {kana} 条含日文假名（可能未翻译）")
    if tiny:
        warns.append(f"中文轨有 {tiny} 条只有 0~1 个汉字")
    return warns


def validate_file(path: Path):
    """返回 (errors: list[str], cue_count: int)。"""
    errors: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("文件含 BOM（应为 UTF-8 无 BOM）")
    if b"\r\n" in raw:
        errors.append("含 CRLF 换行（应统一为 \\n）")
    text = raw.decode("utf-8-sig")

    lines = text.split("\n")
    if not lines or lines[0].strip() != "WEBVTT":
        errors.append("首行不是 WEBVTT")
        return errors, 0

    # 按空行切块（跳过头部）
    body = "\n".join(lines[1:]).strip("\n")
    blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]

    prev_end = -1.0
    expected_idx = 1
    for b in blocks:
        bl = b.split("\n")
        if len(bl) < 3:
            errors.append(f"第 {expected_idx} 块行数不足: {bl!r}")
            expected_idx += 1
            continue
        # 序号
        if bl[0].strip() != str(expected_idx):
            errors.append(f"序号不连续：期望 {expected_idx}，得到 {bl[0].strip()!r}")
        # 时间戳
        m = TS_LINE.match(bl[1].strip())
        if not m:
            errors.append(f"第 {expected_idx} 条时间戳格式错误: {bl[1]!r}")
        else:
            start = _to_sec(*m.group(1, 2, 3, 4))
            end = _to_sec(*m.group(5, 6, 7, 8))
            if end <= start:
                errors.append(f"第 {expected_idx} 条 end<=start ({start}->{end})")
            if start < prev_end - 1e-6:
                errors.append(f"第 {expected_idx} 条与上一条重叠 (start {start} < 上一条 end {prev_end})")
            if end - start > MAX_CUE_SEC + _DUR_TOL:
                errors.append(f"第 {expected_idx} 条时长 {end - start:.2f}s 超过 {MAX_CUE_SEC}s")
            prev_end = end
        # 文本
        if not any(x.strip() for x in bl[2:]):
            errors.append(f"第 {expected_idx} 条文本为空")
        expected_idx += 1

    # 文件名规则
    if not NAME_RE.match(path.name):
        errors.append(f"文件名不符合规则: {path.name}")

    return errors, len(blocks)


def _cue_times(path: Path) -> list[tuple[int, int]]:
    """读出每条的 (开始, 结束)，单位毫秒；格式不对的条跳过。"""
    return [(s, e) for s, e, _t in _read_cues(path)]


def _read_cues(path: Path) -> list[tuple[int, int, str]]:
    """读出每条的 (开始毫秒, 结束毫秒, 文字)；文字多行时用换行连起来。格式不对的块跳过。"""
    out = []
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except Exception:
        return out
    body = "\n".join(text.split("\n")[1:]).strip("\n")
    for b in re.split(r"\n\s*\n", body):
        bl = b.split("\n")
        if len(bl) < 3:
            continue
        m = TS_LINE.match(bl[1].strip())
        if m:
            out.append((round(_to_sec(*m.group(1, 2, 3, 4)) * 1000), round(_to_sec(*m.group(5, 6, 7, 8)) * 1000),
                        "\n".join(x.strip() for x in bl[2:] if x.strip())))
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s", "", t)


def adjacent_repeats(tgt, src=None) -> tuple[int, int, int]:
    """单语译文 tgt 里相邻两条文字相同的次数。给了原文条 src（tgt 的每条由连续的原文条合并而成）时，
    按这两条各自覆盖的原文分类，返回 (原文也相同, 原文不同, 没有原文可比)。"""
    same = diff = unknown = 0
    for (s1, e1, t1), (s2, e2, t2) in zip(tgt, tgt[1:]):
        if not _norm(t1) or _norm(t1) != _norm(t2):
            continue
        if not src:
            unknown += 1
            continue
        a = "".join(t for s, e, t in src if s >= s1 and e <= e1)
        b = "".join(t for s, e, t in src if s >= s2 and e <= e2)
        if _norm(a) and _norm(a) == _norm(b):
            same += 1
        else:
            diff += 1
    return same, diff, unknown


def _repeat_warning(same: int, diff: int, unknown: int) -> str:
    if unknown:
        return f"有 {unknown} 处相邻两条文字相同（没有原文文件可比，分不清是不是说话人重复）"
    if not (same or diff):
        return ""
    parts = []
    if diff:
        parts.append(f"原文不同 {diff} 处（可疑：同一段译文重复显示，或译文太短、后一条延续显示）")
    if same:
        parts.append(f"原文也相同 {same} 处（多半是说话人自己重复，正常）")
    return "相邻两条文字相同：" + "；".join(parts)


def check_alignment(src: list[tuple[int, int]], tgt: list[tuple[int, int]]) -> str:
    """译文条是否正好由连续的原文条合并而成并覆盖全部原文条。通过返回 ""，否则返回原因。"""
    i = 0
    for k, (s, e) in enumerate(tgt, 1):
        if i >= len(src):
            return f"第 {k} 条之后没有可对应的原文条"
        if s != src[i][0]:
            return f"第 {k} 条的开始时间不在原文条边界上（或跳过了原文条）"
        while i < len(src) and src[i][1] < e:
            i += 1
        if i >= len(src) or src[i][1] != e:
            return f"第 {k} 条的结束时间不在原文条边界上"
        i += 1
    if i != len(src):
        return f"最后 {len(src) - i} 条原文没有对应的译文"
    return ""


def validate_dir(d: Path):
    all_ok = True
    vtts = sorted(d.rglob("*.vtt"))
    if not vtts:
        print(f"目录下没有 .vtt 文件: {d}")
        return False

    counts_by_source = defaultdict(dict)
    paths_by_source = defaultdict(dict)
    for v in vtts:
        errors, n = validate_file(v)
        status = "OK" if not errors else "FAIL"
        print(f"[{status}] {v.name}  ({n} 条)")
        for e in errors:
            print(f"    - {e}")
        for w in quality_warnings(v):
            print(f"    ! 警告: {w}")
        if errors:
            all_ok = False
        # 归组：去掉 _<code> 还原源名
        m = re.match(r"^(.+)_([a-z]{2,3}(?:-[a-z]{2,3})?)(\.[^.]+)\.vtt$", v.name)
        if m:
            src_key = (v.parent, m.group(1), m.group(3))
            counts_by_source[src_key][m.group(2)] = n
            paths_by_source[src_key][m.group(2)] = v

    # 同源一致性：原文 == 双语（条数和时间轴）；单语译文由连续的原文条合并而成
    for key, codes in counts_by_source.items():
        (parent, stem, ext) = key
        paths = paths_by_source[key]
        problems = []
        pairs = [c for c in codes if "-" in c]
        for c in pairs:
            a, b = c.split("-", 1)
            ref = _cue_times(paths[c])
            if a in codes and _cue_times(paths[a]) != ref:
                problems.append(f"原文({a}) 与双语({c}) 的条数或时间轴不一致")
            if b in codes:
                why = check_alignment(ref, _cue_times(paths[b]))
                if why:
                    problems.append(f"单语译文({b}) 与原文条对不上：{why}")
        if not pairs and len(set(codes.values())) > 1:
            # 没有双语文件时分不清哪个是原文：只提示，不判错
            print(f"[提示] {stem}{ext} 各文件条数不同（单语译文可以比原文少）: {codes}")
        if problems:
            all_ok = False
            print(f"[FAIL] 同源一致性 {stem}{ext}: {codes}")
            for pb in problems:
                print(f"    - {pb}")
        elif pairs:
            print(f"[OK] 同源一致性 {stem}{ext}: {codes}")
        # 中文单语文件里的相邻重复：找一个与它对得上的原文文件（它的每条由连续的原文条合并而成），按原文分类
        if "cn" in codes:
            tgt = _read_cues(paths["cn"])
            src = None
            for a in codes:
                if a != "cn" and "-" not in a:
                    cand = _read_cues(paths[a])
                    if not check_alignment([(s, e) for s, e, _ in cand], [(s, e) for s, e, _ in tgt]):
                        src = cand
                        break
            w = _repeat_warning(*adjacent_repeats(tgt, src))
            if w:
                print(f"    ! 警告: {paths['cn'].name}: {w}")

    return all_ok


def main(argv) -> int:
    if not argv:
        print("用法: python verify_vtt.py <.vtt 或 目录>")
        return 2
    target = Path(argv[0])
    if target.is_dir():
        ok = validate_dir(target)
    elif target.is_file():
        errors, n = validate_file(target)
        ok = not errors
        print(f"[{'OK' if ok else 'FAIL'}] {target.name} ({n} 条)")
        for e in errors:
            print(f"    - {e}")
        for w in quality_warnings(target):
            print(f"    ! 警告: {w}")
        if re.match(r"^.+_cn\.[^.]+\.vtt$", target.name):
            w = _repeat_warning(*adjacent_repeats(_read_cues(target)))
            if w:
                print(f"    ! 警告: {w}")
    else:
        print(f"路径不存在: {target}")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

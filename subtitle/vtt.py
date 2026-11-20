"""WEBVTT 字幕写出 + 文件名规则。

文件名: <原名(无扩展)>_<语言代码><原扩展名>.vtt，如 xxx_cn.mp3.vtt
语言代码: 中文用 cn，其余用 ISO 639-1。
双语组合码: <src>-<tgt>，如 ja-cn。
编码 UTF-8 无 BOM，换行统一 \n。
"""
from __future__ import annotations

from pathlib import Path

from .segments import Segment


def display_code(lang: str, cfg: dict) -> str:
    """ISO 639-1 -> 文件名用语言代码（中文 zh->cn）。"""
    mapping = cfg.get("lang_code_display", {"zh": "cn"})
    base = (lang or "").lower().split("-")[0]
    return mapping.get(base, base)


def output_name(src_file: Path, lang_code: str) -> str:
    """生成 <stem>_<lang_code><suffix>.vtt。"""
    src_file = Path(src_file)
    return f"{src_file.stem}_{lang_code}{src_file.suffix}.vtt"


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600 * 1000)
    m, ms = divmod(ms, 60 * 1000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _safe_line(text: str) -> str:
    """压成单行且非空，防止空行/内嵌换行破坏 VTT 块结构。"""
    s = " ".join(str(text).split())
    return s if s else "·"


def _write(path: Path, blocks: list[str]) -> None:
    """以 UTF-8 无 BOM、\\n 换行写出。"""
    content = "WEBVTT\n\n" + "\n\n".join(blocks)
    if blocks:
        content += "\n"
    # newline='' 禁止 Windows 下 \n->\r\n 转换；encoding utf-8 无 BOM
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def write_mono(cues: list[Segment], out_path: Path) -> Path:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{_fmt_ts(c.start)} --> {_fmt_ts(c.end)}\n{_safe_line(c.text)}")
    _write(out_path, blocks)
    return out_path


def write_bilingual(src_cues: list[Segment], tgt_cues: list[Segment],
                    out_path: Path, swap: bool = False) -> Path:
    """双语：每条两行。默认译文在上、原文在下；swap=True 调换。

    src_cues/tgt_cues 必须一一对应（条数相同，时间轴取原文）。
    """
    assert len(src_cues) == len(tgt_cues), "双语写出要求原文/译文条数一致"
    blocks = []
    for i, (sc, tc) in enumerate(zip(src_cues, tgt_cues), 1):
        top, bottom = (sc.text, tc.text) if swap else (tc.text, sc.text)
        blocks.append(
            f"{i}\n{_fmt_ts(sc.start)} --> {_fmt_ts(sc.end)}\n{_safe_line(top)}\n{_safe_line(bottom)}")
    _write(out_path, blocks)
    return out_path

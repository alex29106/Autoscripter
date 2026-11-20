"""MT 后端统一接口 + 两种翻译编排。

1) 句子单元（默认，translate_units）：每个翻译单元（完整的一句话）单独翻译一次，
   做确定性校验；不合格时去掉上下文重试一次；仍不合格才对这一句退回逐条翻译。
   译文如何分配回各条字幕由 subtitle/distribute.py 负责（条数、时间轴不变）。
2) 逐条（旧模式，translate_all）：按批翻译字幕条，要求按条返回 JSON；条数不符时二分重试，
   最终单条兜底。保留用于对照和兜底。
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

log = logging.getLogger("subtitle.mt")

_KANA = re.compile(r"[぀-ヿ]")
_CJK = re.compile(r"[一-鿿㐀-䶿]")
_LATIN = re.compile(r"[A-Za-z]")
_CORE = re.compile(r"[\s、。，,.!！?？…～~「」『』（）()“”\"'：:；;・\-]")
_CJK_TARGETS = {"zh", "yue", "ja", "ko"}


def check_unit_output(src: str, out: str, tgt_lang: str) -> str:
    """对一个翻译单元的译文做确定性校验；合格返回空串，否则返回原因。"""
    out = (out or "").strip()
    if not out:
        return "empty"
    if tgt_lang in ("zh", "yue"):
        if _KANA.search(out):
            return "kana"                       # 未翻译的日文（常见于片假名拟声词）
        cjk, latin = len(_CJK.findall(out)), len(_LATIN.findall(out))
        if latin > 8 and latin > cjk:
            return "not-target-language"        # 整句输出成了英文等
    s_len, o_len = len(_CORE.sub("", src)), len(_CORE.sub("", out))
    if tgt_lang in _CJK_TARGETS:
        if o_len > 2.2 * s_len + 6:
            return "too-long"                   # 多半是把上下文也一起翻了
    elif o_len > 6 * s_len + 20:
        return "too-long"
    if s_len >= 8 and o_len < 0.15 * s_len:
        return "too-short"
    return ""

# ISO 639-1 -> 语言英文名（提示词用）
LANG_NAMES = {
    "zh": "Simplified Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "yue": "Cantonese", "fr": "French", "de": "German", "es": "Spanish",
    "ru": "Russian", "it": "Italian", "pt": "Portuguese", "auto": "the source language",
}


class CountMismatch(Exception):
    """翻译返回条数与输入不一致。"""


class MTFailure(RuntimeError):
    """翻译模型连续出错（显存不足、模型损坏等）。继续下去只会得到一份几乎全是原文的“译文”，
    所以停止翻译这个文件，由调用方把它记为失败。"""


class MTBackend(ABC):
    name = "base"

    def __init__(self, model_dir: Path, cfg: dict):
        self.model_dir = Path(model_dir)
        self.cfg = cfg
        self._model = None
        self._errors = 0             # 逐条翻译里出错（抛异常）的次数，累计
        self._last_error = None

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def translate_batch(self, lines: list[str], ctx_before: list[str],
                        ctx_after: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """翻译一批。必须返回与 lines 等长的列表，否则抛 CountMismatch。"""

    def unload(self) -> None:
        self._model = None
        from ..vram import release_cuda
        release_cuda()

    # ---- 句子单元模式 ----
    # 每个后端默认给多少个前/后单元作上下文（子类可覆盖；config 的 translation.context_before/after 优先）
    ctx_before_units = 3
    ctx_after_units = 2

    def translate_sentence(self, text: str, ctx_before: list[str], ctx_after: list[str],
                           src_lang: str, tgt_lang: str) -> str:
        """翻译一个完整的句子单元，返回译文（纯文本）。默认借用批量接口翻一条。"""
        out = self.translate_batch([text], ctx_before, ctx_after, src_lang, tgt_lang)
        return out[0] if out else ""

    def translate_units(self, units, src_lang: str, tgt_lang: str, progress=None) -> list[dict]:
        """逐个翻译单元。返回与 units 等长的列表，每项为
        {"text": 整句译文 或 None, "per_cue": 逐条译文 或 None, "how": "sentence|no-context|cue-fallback|source",
         "errors": 这一句翻译时出错（抛异常）的次数}。
        "per_cue" 只在退回逐条翻译时给出（此时不再需要分配）。
        连续 translation.max_consecutive_errors 句都因为出错（不是译文不合格）而只能保留原文时，抛 MTFailure。
        progress(已完成, 总数) 可选，图形界面用来显示进度。"""
        from .. import console
        tcfg = self.cfg.get("translation", {})
        nb = int(tcfg.get("context_before", self.ctx_before_units))
        na = int(tcfg.get("context_after", self.ctx_after_units))
        max_consec = int(tcfg.get("max_consecutive_errors", 3))
        results: list[dict] = []
        stats = {"sentence": 0, "no-context": 0, "cue-fallback": 0, "source": 0, "error": 0}
        consec = 0
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(units), desc="翻译", unit="句")
        except Exception:
            pbar = None
        try:
            for k, u in enumerate(units):
                before = [x.text for x in units[max(0, k - nb):k]] if nb > 0 else []
                after = [x.text for x in units[k + 1:k + 1 + na]] if na > 0 else []
                res = None
                errors, last_err = 0, None
                for how, cb, ca in (("sentence", before, after), ("no-context", [], [])):
                    if how == "no-context" and not (cb or before or after):
                        continue                        # 本来就没有上下文，不必重复
                    try:
                        out = self.translate_sentence(u.text, cb, ca, src_lang, tgt_lang)
                    except Exception as e:
                        out, why = "", f"error: {type(e).__name__}: {e}"
                        errors, last_err = errors + 1, e
                    else:
                        why = check_unit_output(u.text, out, tgt_lang)
                    if not why:
                        res = {"text": out.strip(), "per_cue": None, "how": how}
                        break
                    (log.warning if last_err is not None else log.info)(
                        "单元 %d 译文不合格（%s），%s：%r -> %r", k, why,
                        "去掉上下文重试" if how == "sentence" else "退回逐条翻译", u.text[:40], (out or "")[:60])
                if res is None:
                    cue_texts = [c.text for c in u.cues]
                    e0 = self._errors
                    per = self._translate_resilient(cue_texts, before[-1:], after[:1], src_lang, tgt_lang)
                    if self._errors > e0:
                        errors, last_err = errors + self._errors - e0, self._last_error
                    if all(not check_unit_output(s, t, tgt_lang) for s, t in zip(cue_texts, per)):
                        res = {"text": None, "per_cue": per, "how": "cue-fallback"}
                    else:
                        console.warn(f"第 {k + 1} 句没翻译出来，保留原文：{u.text[:40]}")
                        res = {"text": None, "per_cue": cue_texts, "how": "source"}
                res["errors"] = errors
                stats[res["how"]] += 1
                stats["error"] += errors
                results.append(res)
                consec = consec + 1 if (res["how"] == "source" and errors) else 0
                if max_consec > 0 and consec >= max_consec:
                    raise MTFailure(f"连续 {consec} 句翻译出错（最后一次：{type(last_err).__name__}: {last_err}），"
                                    f"停止翻译这个文件")
                if pbar:
                    pbar.update(1)
                if progress:
                    progress(k + 1, len(units))
        finally:
            if pbar:
                pbar.close()
        log.info("句子单元翻译完成：%s", stats)
        return results

    # ---- 逐条模式（旧）：对外主入口 ----
    def translate_all(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not texts:
            return []
        tcfg = self.cfg.get("translation", {})
        batch_size = int(tcfg.get("batch_size", 16))
        ctx_n = int(tcfg.get("context_lines", 2))

        results: list[Optional[str]] = [None] * len(texts)
        e0 = self._errors
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(texts), desc="翻译", unit="line")
        except Exception:
            pbar = None

        i = 0
        while i < len(texts):
            j = min(i + batch_size, len(texts))
            batch = texts[i:j]
            ctx_before = texts[max(0, i - ctx_n):i]
            ctx_after = texts[j:j + ctx_n]
            out = self._translate_resilient(batch, ctx_before, ctx_after, src_lang, tgt_lang)
            for k, v in enumerate(out):
                results[i + k] = v
            if pbar:
                pbar.update(j - i)
            i = j

        if pbar:
            pbar.close()
        # 空译文兜底为原文，保证每条非空（空文本会破坏 VTT 结构）
        out = []
        for r, src in zip(results, texts):
            if r is None or not str(r).strip():
                log.warning("空译文，使用原文兜底: %r", src)
                out.append(src)
            else:
                out.append(r)
        failed = sum(1 for r, src in zip(out, texts) if r == src)
        if failed:
            from .. import console
            console.warn(f"逐条翻译：{failed}/{len(texts)} 条保留了原文（没翻译出来）")
        if self._errors > e0 and failed >= max(3, len(texts) // 2):
            raise MTFailure(f"逐条翻译有 {failed}/{len(texts)} 条出错（最后一次：{self._last_error}），停止翻译这个文件")
        return out

    def _translate_resilient(self, lines: list[str], ctx_before: list[str],
                             ctx_after: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """条数不符时二分重试，最终单条兜底。保证返回等长列表。"""
        if not lines:
            return []
        try:
            out = self.translate_batch(lines, ctx_before, ctx_after, src_lang, tgt_lang)
            if len(out) == len(lines):
                return out
            raise CountMismatch(f"期望 {len(lines)} 条，得到 {len(out)} 条")
        except Exception as e:      # 条数不符或其它错误（显存/模型）：二分重试，最终单条兜底
            if len(lines) == 1:
                log.warning("单条翻译失败，使用原文兜底: %s", e)
                self._errors += 1
                self._last_error = e
                return [lines[0]]
            log.warning("翻译失败(%s)，二分重试 %d 条", e, len(lines))
            mid = len(lines) // 2
            left = self._translate_resilient(lines[:mid], ctx_before, lines[mid:], src_lang, tgt_lang)
            right = self._translate_resilient(lines[mid:], lines[:mid], ctx_after, src_lang, tgt_lang)
            return left + right

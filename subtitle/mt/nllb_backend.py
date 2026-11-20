"""NLLB-200 翻译后端（回退方案，逐句翻译天然 1:1 对齐）。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import MTBackend
from .. import models
from ..vram import has_cuda

log = logging.getLogger("subtitle.mt.nllb")

# ISO 639-1 -> NLLB FLORES-200 代码
_NLLB_LANG = {
    "zh": "zho_Hans", "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "yue": "yue_Hant", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "ru": "rus_Cyrl", "it": "ita_Latn", "pt": "por_Latn",
}


class NLLBMTBackend(MTBackend):
    name = "nllb"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = models.ensure_model("nllb", self.model_dir, models.MT_MODELS)
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        if has_cuda():
            self._model = self._model.to("cuda").half()
        self._model.eval()
        from .. import console
        console.key(f"翻译模型 nllb（NLLB-200-distilled-600M）加载完成（{'GPU' if has_cuda() else 'CPU'}）")

    def _code(self, lang: str) -> str:
        if lang not in _NLLB_LANG:
            raise ValueError(f"NLLB 不支持的语种代码: {lang}")
        return _NLLB_LANG[lang]

    def translate_batch(self, lines, ctx_before, ctx_after, src_lang, tgt_lang):
        import torch
        if self._model is None:
            self.load()

        src_code = self._code(src_lang) if src_lang in _NLLB_LANG else None
        tgt_code = self._code(tgt_lang)

        tok = self._tokenizer
        if src_code:
            tok.src_lang = src_code
        inputs = tok(lines, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        bos = tok.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, forced_bos_token_id=bos, max_length=512, num_beams=4,
                no_repeat_ngram_size=3, repetition_penalty=1.3)
        out = tok.batch_decode(gen, skip_special_tokens=True)
        # seq2seq 逐句 1:1，长度必然匹配
        return [s.strip() for s in out]

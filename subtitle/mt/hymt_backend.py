"""腾讯混元翻译模型后端：Hy-MT2（tencent/Hy-MT2-7B、Hy-MT2-1.8B）与旧版 Hunyuan-MT-7B。

与 Qwen 后端的区别：
  - 这是翻译专用模型，官方没有 system prompt，按官方指令模板一次翻一段文字（不用 JSON）；
  - 句子单元模式只给“前文”作背景（实测给后文会把后一句的内容一起翻进来）；
  - 旧版 Hunyuan-MT-7B 给上下文时会把上下文也翻出来，所以它不给上下文；
  - 逐条模式（--mt-granularity cue）下每条单独翻译，返回条数天然一致。
加载（显存方案、4-bit 量化）复用 Qwen 后端的实现。
模板来源：https://huggingface.co/tencent/Hy-MT2-7B 的模型卡（Default / Terminology / Structured Data 2 背景信息）。
"""
from __future__ import annotations

import logging
import re

from .base import LANG_NAMES
from .qwen_backend import QwenMTBackend

log = logging.getLogger("subtitle.mt.hymt")

ZH_NAMES = {"zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语", "yue": "粤语", "fr": "法语", "de": "德语",
            "es": "西班牙语", "ru": "俄语", "it": "意大利语", "pt": "葡萄牙语"}


class HyMTBackend(QwenMTBackend):
    def __init__(self, model_dir, cfg, alias: str = "hymt2"):
        super().__init__(model_dir, cfg, alias=alias)
        self.family = "hunyuan" if alias == "hunyuan" else "hymt2"
        self.ctx_before_units = 0 if self.family == "hunyuan" else 3
        self.ctx_after_units = 0

    # ---- 提示词 ----
    def _terms(self, text: str) -> list[tuple[str, str]]:
        pairs = self.cfg.get("translation", {}).get("glossary_pairs") or []
        return [(a, b) for a, b in pairs if a and a in text]

    def _prompt(self, text: str, ctx: list[str], src_lang: str, tgt_lang: str) -> str:
        terms = self._terms(text)
        if "zh" in (tgt_lang, src_lang):          # 官方：涉及中文的方向用中文指令
            lang = ZH_NAMES.get(tgt_lang, tgt_lang)
            if self.family == "hunyuan":
                return f"把下面的文本翻译成{lang}，不要额外解释。\n\n{text}"
            head = ""
            if terms:
                head = "参考下面的翻译：\n" + "\n".join(f"{a} 翻译成 {b}" for a, b in terms) + "\n\n"
            if ctx:
                return (head + "【背景信息】\n" + "\n".join(ctx) +
                        f"\n\n请结合背景信息将以下文本翻译为{lang}。\n\n【待翻译文本】\n{text}")
            return head + f"将以下文本翻译为{lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        lang = LANG_NAMES.get(tgt_lang, tgt_lang)
        if self.family == "hunyuan":
            return f"Translate the following segment into {lang}, without additional explanation.\n\n{text}"
        head = ""
        if terms:
            head = ("Reference the following translations:\n" +
                    "\n".join(f"{a} translates to {b}" for a, b in terms) + "\n\n")
        if ctx:
            return (head + "[Background Information]\n" + "\n".join(ctx) +
                    f"\n\nPlease translate the following text into {lang}, taking the provided background "
                    f"information into consideration.\n\n[Source Text]\n{text}")
        return head + (f"Translate the following text into {lang}. Note that you should only output the "
                       f"translated result without any additional explanation:\n\n{text}")

    # ---- 推理 ----
    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch
        if self._model is None:
            self.load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self._model.device)
        gcfg = self.cfg.get("translation", {}).get("generation", {}) or {}
        sample = bool(gcfg.get("do_sample", False))
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=sample,
                temperature=gcfg.get("temperature") if sample else None,
                top_p=gcfg.get("top_p") if sample else None,
                top_k=gcfg.get("top_k") if sample else None,
                repetition_penalty=float(gcfg.get("repetition_penalty", 1.05)),
                pad_token_id=self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id)
        out = self._tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return out.strip()

    def translate_sentence(self, text, ctx_before, ctx_after, src_lang, tgt_lang) -> str:
        prompt = self._prompt(text, list(ctx_before), src_lang, tgt_lang)
        out = self._generate(prompt, max_new_tokens=min(512, 4 * len(text) + 64))
        return re.sub(r"^\s*(【译文】|译文[:：])\s*", "", out).strip()

    def translate_batch(self, lines, ctx_before, ctx_after, src_lang, tgt_lang):
        # 逐条模式：每条单独翻译（翻译模型不适合一次返回 JSON 数组），条数天然一致
        return [self.translate_sentence(t, [], [], src_lang, tgt_lang) for t in lines]

"""transformers 因果语言模型翻译后端（Qwen2.5 / Qwen3；Hy-MT2 / Hunyuan-MT 继承它，见 hymt_backend.py）。

- 按显存选择全精度 / bitsandbytes 8-bit / 4-bit（见 vram.mt_load_plan；translation.precision 或 --mt-precision 可强制）。
- 批量翻译，附前后上下文，要求模型返回 JSON 字符串数组，并校验条数。
- 提示词来自 prompts/translate.txt，便于修改。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .base import MTBackend, CountMismatch, LANG_NAMES
from .. import models
from ..vram import mt_load_plan

log = logging.getLogger("subtitle.mt.qwen")


def _has_kana(s: str) -> bool:
    """是否含日文假名（平假名 U+3040-309F / 片假名 U+30A0-30FF）。"""
    return any("぀" <= ch <= "ヿ" for ch in s)


class QwenMTBackend(MTBackend):
    def __init__(self, model_dir: Path, cfg: dict, alias: str = "qwen3b"):
        super().__init__(model_dir, cfg)
        self.alias = alias
        self.name = alias
        self._tokenizer = None
        self._prompt_tmpl = None

    def _load_prompt(self) -> str:
        if self._prompt_tmpl is None:
            tcfg = self.cfg.get("translation", {})
            pf = Path(tcfg.get("prompt_file", "prompts/translate.txt"))
            if not pf.is_absolute():
                pf = Path(self.cfg["_project_root"]) / pf
            self._prompt_tmpl = pf.read_text(encoding="utf-8")
        return self._prompt_tmpl

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .. import console
        path = models.ensure_model(self.alias, self.model_dir, models.MT_MODELS)
        plan = mt_load_plan(self.alias, self.cfg.get("translation", {}).get("precision", "auto"))
        self.load_plan = plan
        console.key(f"翻译模型 {self.alias}：{plan['reason']}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))

        kwargs = {}
        dtype = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}.get(plan["dtype"], torch.float16)
        if plan.get("prequantized"):
            # 仓库本身就是 4-bit 量化权重（quantization_config 在其 config.json 里），直接加载
            kwargs["device_map"] = "auto"
        elif plan.get("load_in_8bit"):
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["dtype"] = torch.float16      # int8 矩阵乘按 fp16 计算；不指定时按 bf16 加载，每次都要转换并刷警告
            kwargs["device_map"] = "auto"
        elif plan["load_in_4bit"]:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        elif plan["device_map"] == "cpu":
            kwargs["dtype"] = torch.float32
        else:
            kwargs["dtype"] = dtype
            kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        self._model.eval()

    def _chat_kwargs(self) -> dict:
        # Qwen3 默认会先输出 <think> 思考过程；翻译不需要，关掉
        return {"enable_thinking": False} if "qwen3" in self.alias else {}

    def _build_prompt(self, lines, ctx_before, ctx_after, src_lang, tgt_lang) -> str:
        tmpl = self._load_prompt()
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines))
        pairs = self.cfg.get("translation", {}).get("glossary_pairs") or []
        joined = "".join(lines)
        used = [(a, b) for a, b in pairs if a and a in joined]   # 只注入本次待译文字里出现的词条
        glossary = "; ".join(f"{a} -> {b}" for a, b in used) if used else "(none)"
        return tmpl.format(
            src_lang_name=LANG_NAMES.get(src_lang, src_lang),
            tgt_lang_name=LANG_NAMES.get(tgt_lang, tgt_lang),
            glossary=glossary,
            context_before="\n".join(ctx_before) or "(none)",
            context_after="\n".join(ctx_after) or "(none)",
            n=len(lines),
            numbered_lines=numbered,
        )

    @staticmethod
    def _parse_json_array(text: str, n: int) -> list[str]:
        """从模型输出中解析出字符串数组。"""
        # 去掉可能的 ```json 围栏
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        # 抓取第一个 [...] 块
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m.group(0)
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError("not a list")
        out = []
        for x in arr:
            if isinstance(x, dict):
                # 容错：模型偶尔返回 {"translation": "..."} / {"subtitle": "..."} 之类
                vals = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
                out.append(" ".join(vals).strip())
            elif isinstance(x, list):
                out.append(" ".join(str(v) for v in x).strip())
            else:
                out.append(str(x).strip())
        return out

    def translate_batch(self, lines, ctx_before, ctx_after, src_lang, tgt_lang):
        import torch
        if self._model is None:
            self.load()

        prompt = self._build_prompt(lines, ctx_before, ctx_after, src_lang, tgt_lang)
        messages = [
            {"role": "system", "content": "You are a precise subtitle translation engine that returns only JSON."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self._chat_kwargs())
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # 解码参数与 Hy-MT2 后端共用 config 的 translation.generation（默认值相同：贪心、1.05）
        gcfg = self.cfg.get("translation", {}).get("generation", {}) or {}
        sample = bool(gcfg.get("do_sample", False))
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=min(2048, 64 * len(lines) + 256),
                do_sample=sample,
                temperature=gcfg.get("temperature") if sample else None,
                top_p=gcfg.get("top_p") if sample else None,
                top_k=gcfg.get("top_k") if sample else None,
                repetition_penalty=float(gcfg.get("repetition_penalty", 1.05)),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = gen[0][inputs["input_ids"].shape[1]:]
        out_text = self._tokenizer.decode(gen, skip_special_tokens=True)
        out_text = re.sub(r"^\s*<think>.*?</think>\s*", "", out_text, flags=re.S)

        try:
            arr = self._parse_json_array(out_text, len(lines))
        except Exception as e:
            raise CountMismatch(f"JSON 解析失败: {e}; 原始输出前200字: {out_text[:200]!r}")
        if len(arr) != len(lines):
            raise CountMismatch(f"期望 {len(lines)} 条，解析出 {len(arr)} 条")
        # 多条批次里有空译文时，触发二分重试（单独翻译往往能拿到结果）；
        # 单条仍空则交由上层回退原文。
        if len(lines) > 1 and any(not s.strip() for s in arr):
            raise CountMismatch("批次中存在空译文，拆小重试")
        # 目标为中文却混入日文假名 -> 疑似未翻译/漂移，拆小重试
        if tgt_lang.lower().startswith("zh") and len(lines) > 1 and any(_has_kana(s) for s in arr):
            raise CountMismatch("输出疑似未翻译（含日文假名），拆小重试")
        return arr

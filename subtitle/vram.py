"""CUDA / 显存检测，决定 ASR 计算精度与 MT 量化方案。"""
from __future__ import annotations

import gc
import logging

log = logging.getLogger("subtitle.vram")


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def total_vram_gb() -> float:
    """返回 0 号 GPU 总显存（GB），无 CUDA 返回 0。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def free_vram_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def asr_compute_settings() -> tuple[str, str]:
    """返回 (device, compute_type) 供 faster-whisper 使用。

    有 GPU -> cuda/float16；否则 cpu/int8。
    """
    if has_cuda():
        return "cuda", "float16"
    return "cpu", "int8"


# MT 别名 -> (全精度门槛 GB, 8-bit 门槛 GB 或 None, 全精度 dtype, 仓库是否已是 4-bit 预量化)
# 按总显存选精度（ASR 在加载 MT 前已卸载）：>= 全精度门槛用 bf16/fp16；>= 8-bit 门槛用 bitsandbytes 8-bit；
# 否则 bitsandbytes 4-bit（NF4）。门槛 = 权重大小 + 约 2~3GB（桌面常驻显存、激活值、CUDA 上下文）。
# 7B 级的全精度门槛是 18GB（bf16 权重约 15GB，16GB 卡放不下还要留给桌面）；
# 8-bit 一档约 7.5GB 权重，给 10GB 以上的卡。只有 4-bit 一档在 8GB 卡上实测过。
_MT_SIZES = {
    "qwen3b":     (7.0, None, "float16", False),      # ~6.2GB fp16（8GB 卡实测可用）
    "qwen7b":     (18.0, 10.0, "float16", False),     # ~15GB fp16
    "hunyuan":    (18.0, 10.0, "bfloat16", False),    # Hunyuan-MT-7B
    "hymt2":      (18.0, 10.0, "bfloat16", False),    # Hy-MT2-7B：bf16 权重 15GB；4-bit 实测模型占 4.65GB
    "hymt2-1.8b": (5.0, None, "bfloat16", False),     # ~3.6GB
    "qwen3-8b":   (0.0, None, "float16", True),       # unsloth/Qwen3-8B-bnb-4bit，权重已是 NF4
}
PRECISIONS = ["auto", "full", "int8", "nf4"]
_PREC_NAMES = {"full": "全精度", "int8": "bitsandbytes 8-bit", "nf4": "bitsandbytes 4-bit（NF4）"}


def mt_load_plan(backend: str, precision: str = "auto") -> dict:
    """根据显存为 MT 模型给出加载方案。precision：auto（按显存自动选）/ full / int8 / nf4（强制）。

    返回 dict: {device_map, dtype, load_in_4bit, load_in_8bit, prequantized, precision, reason}
    显存不足以放下 4-bit 时，调用方（mt/factory.py）会回退到 NLLB；加载时真的失败也会回退。
    """
    vram = total_vram_gb()
    free = free_vram_gb()
    name = backend.lower()
    precision = (precision or "auto").lower()
    base = {"device_map": "cuda", "load_in_4bit": False, "load_in_8bit": False, "prequantized": False}
    if not has_cuda():
        plan = dict(base, device_map="cpu", dtype="float32", precision="cpu",
                    reason="没有可用的 CUDA 显卡，用 CPU 运行（很慢）")
        log.info("MT 加载方案(%s): %s", backend, plan["reason"])
        return plan

    need_full, need_int8, dtype, preq = _MT_SIZES.get(name, (0.0, None, "float16", False))   # 未列出的（如 NLLB）按小模型处理
    if dtype == "bfloat16":
        try:
            import torch
            if not torch.cuda.is_bf16_supported():
                dtype = "float16"
        except Exception:
            dtype = "float16"
    mem = f"显存共 {vram:.1f}GB（当前空闲 {free:.1f}GB）"
    if preq:
        return dict(base, dtype=dtype, prequantized=True, precision="nf4",
                    reason=f"{mem}，仓库本身是 4-bit 预量化权重，直接加载")
    if precision == "auto":
        if vram >= need_full:
            precision = "full"
        elif need_int8 is not None and vram >= need_int8:
            precision = "int8"
        else:
            precision = "nf4"
        why = "按显存自动选择"
    else:
        why = "按设置强制使用"
    if precision == "full":
        plan = dict(base, dtype=dtype, precision="full", reason=f"{mem}，{why}全精度（{dtype}）")
    elif precision == "int8":
        plan = dict(base, dtype="float16", load_in_8bit=True, precision="int8",
                    reason=f"{mem}，{why} {_PREC_NAMES['int8']}")
    else:
        plan = dict(base, dtype="float16", load_in_4bit=True, precision="nf4",
                    reason=f"{mem}，{why} {_PREC_NAMES['nf4']}")
    if why == "按显存自动选择" and precision != "full":
        plan["reason"] += f"；显存 >= {need_full:g}GB 时会自动用全精度"
        if precision == "nf4" and need_int8 is not None:
            plan["reason"] += f"，>= {need_int8:g}GB 时用 8-bit"
    log.info("MT 加载方案(%s): %s", backend, plan["reason"])
    return plan


def release_cuda() -> None:
    """释放显存缓存。用于 ASR 卸载后、加载 MT 前。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

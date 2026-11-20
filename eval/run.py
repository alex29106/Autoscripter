"""翻译回归评测：在缓存的真实转写上跑项目自己的“断句 + 翻译”代码路径，然后
  1) 用 verify_vtt 做格式校验（原文/双语条数一致、单语译文与原文条对得上、时间轴不重叠等）；
  2) 统计整文件的结构性指标（碎片条、假名、相邻重复、超长、过短的条）；
  3) 按时间重叠把评测条目对应的译文抠出来打分（开发集 / 测试集）。主指标看单语中文文件，
     双语文件的中文行同样打分并列出。

用法（在项目根目录、激活 subtitle-whisper 环境后）：
  python eval/run.py --mt hymt2                      # 默认：3 个文件，开发集
  python eval/run.py --mt qwen3b --set translation.granularity=cue --tag old
  python eval/run.py --mt hymt2 --split test         # 测试集：只在最终验证时看
不需要原始音频（直接读 .cache 里的转写缓存）。输出写到 eval/out/<tag>/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

from subtitle.cli import load_config, setup_logging  # noqa: E402
from subtitle.segments import Transcript  # noqa: E402
from subtitle import vtt, models  # noqa: E402
import scoring  # noqa: E402


def _apply_overrides(cfg: dict, sets: list[str]) -> None:
    for s in sets:
        key, val = s.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            val = json.loads(val)
        except Exception:
            pass
        node[parts[-1]] = val


def _stats(res, max_chars: int = 20) -> dict:
    cjk = lambda t: len(scoring.CJK.findall(t))
    bi = [c.text for c in res.tgt_bilingual]
    mono = [c.text for c in res.tgt_mono]
    units = res.units or []
    ls = res.layout_stats or {}
    return dict(
        cues=len(res.src_cues),
        mono_cues=len(mono),
        units=len(units),
        multi_cue_units=sum(1 for u in units if u["src_cues"] > 1),
        tiny_zh_bilingual=sum(1 for t in bi if cjk(t) <= 1),
        tiny_zh_mono=sum(1 for t in mono if cjk(t) <= 1),
        kana=sum(1 for t in bi + mono if scoring.has_kana(t)),
        adjacent_identical_bilingual=sum(1 for a, b in zip(bi, bi[1:]) if a == b),
        adjacent_identical_mono=sum(1 for a, b in zip(mono, mono[1:]) if a == b),
        over_long_mono=sum(1 for t in mono if len(t.replace(" ", "")) > max_chars),
        under_0_5s=sum(1 for c in res.src_cues if c.end - c.start < 0.5),
        # 按来源的检查（见 pipeline）：mismatch = 单语条拼起来 ≠ 整句译文的单元（重复显示或丢字），默认方式下必须为 0
        mismatch=ls.get("mismatch", 0),
        relaxed=ls.get("relaxed", 0),
        hold=ls.get("hold", 0),
        bi_repeat=ls.get("bi_repeat", 0),
        mt=dict(res.unit_stats or {}),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", default=None, help="MT 后端别名（默认取 config.yaml）")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--files", default="2-1,1-1,gen")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", action="append", default=[], help="覆盖配置，如 translation.granularity=cue")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    cfg = load_config()
    _apply_overrides(cfg, args.set)
    mt_name = args.mt or cfg.get("defaults", {}).get("mt")
    tag = args.tag or mt_name
    out_dir = HERE / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    data = HERE / "data"
    items = json.loads((data / "evalset.json").read_text(encoding="utf-8"))
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    cues_all = json.loads((data / "cues_all.json").read_text(encoding="utf-8"))
    transcripts = json.loads((data / "transcripts.json").read_text(encoding="utf-8"))

    model_dir = Path(cfg.get("model_dir") or ROOT / "model")
    models.configure_hf_env(model_dir)
    from subtitle.mt.factory import build_mt_with_fallback
    from subtitle.pipeline import segment_and_translate
    t_load = time.time()
    mt = build_mt_with_fallback(mt_name, model_dir, cfg)
    mt.load()
    load_s = time.time() - t_load

    summary = dict(mt=mt_name, backend=getattr(mt, "name", "?"), tag=tag, split=args.split,
                   overrides=args.set, load_s=round(load_s, 1), files={})
    rows = []
    try:
        for key in args.files.split(","):
            meta = transcripts[key]
            tr = Transcript.from_dict(json.loads((ROOT / ".cache" / meta["cache"]).read_text(encoding="utf-8")))
            t0 = time.time()
            res = segment_and_translate(tr, mt, cfg, "ja", "zh")
            secs = time.time() - t0
            media = Path(meta["media"])
            vtt.write_mono(res.src_cues, out_dir / vtt.output_name(media, "ja"))
            vtt.write_mono(res.tgt_mono, out_dir / vtt.output_name(media, "cn"))
            vtt.write_bilingual(res.src_cues, res.tgt_bilingual, out_dir / vtt.output_name(media, "ja-cn"))
            st = _stats(res, int(cfg.get("segmentation", {}).get("max_chars", {}).get("zh", 20)))
            st["translate_s"] = round(secs, 1)
            summary["files"][key] = st
            (out_dir / f"units_{key}.json").write_text(
                json.dumps(res.units or [], ensure_ascii=False, indent=1), encoding="utf-8")
            out_cues = [(c.start, c.end, c.text) for c in res.tgt_mono]
            bi_cues = [(c.start, c.end, c.text) for c in res.tgt_bilingual]
            for it in items:
                if it["src"] == "synthetic" or not it["src"].startswith(key + "#"):
                    continue
                if args.split != "all" and split[it["id"]] != args.split:
                    continue
                pcs = scoring.overlapping_text(it, cues_all, out_cues)
                sc = scoring.score(it, pcs)
                pcs_bi = scoring.overlapping_text(it, cues_all, bi_cues)
                rows.append(dict(id=it["id"], file=key, per_cue=pcs, **sc, per_cue_bilingual=pcs_bi,
                                 meaning_ok_bilingual=scoring.score(it, pcs_bi)["meaning_ok"]))
    finally:
        mt.unload()

    import verify_vtt  # project root
    print("\n=== 格式校验 ===")
    summary["vtt_ok"] = bool(verify_vtt.validate_dir(out_dir))
    # 结构检查：默认的单语排版下，每句译文在单语文件里只出现一次（旧的 repeat 方式本来就会重复，不算）
    expect_no_repeat = (cfg.get("translation", {}).get("mono_display", "sentence") != "repeat"
                        and cfg.get("translation", {}).get("granularity", "sentence") == "sentence")
    summary["structure_ok"] = (not expect_no_repeat) or all(st["mismatch"] == 0 for st in summary["files"].values())
    n = len(rows)
    ok = sum(r["meaning_ok"] for r in rows)
    summary["meaning"] = f"{ok}/{n}"
    summary["meaning_bilingual"] = f"{sum(r['meaning_ok_bilingual'] for r in rows)}/{n}"
    summary["frag"] = sum(r["frag"] for r in rows)
    summary["added"] = sum(bool(r["added"]) for r in rows)
    summary["failures"] = [dict(id=r["id"], text=" / ".join(r["per_cue"])) for r in rows if not r["meaning_ok"]]
    summary["failures_bilingual"] = [dict(id=r["id"], text=" / ".join(r["per_cue_bilingual"]))
                                     for r in rows if not r["meaning_ok_bilingual"]]
    (out_dir / "summary.json").write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\n=== {tag}  (mt={mt_name}, backend={summary['backend']}, split={args.split}, load {summary['load_s']}s) ===")
    for k, st in summary["files"].items():
        print(f"  {k:4s} {st}")
    print(f"  意思正确（单语中文文件，按时间对齐）: {summary['meaning']}   双语文件中文行: {summary['meaning_bilingual']}"
          f"   碎片条 {summary['frag']}  疑似塞入多余内容 {summary['added']}"
          f"   格式校验: {'OK' if summary['vtt_ok'] else 'FAIL'}"
          f"   重复显示检查: {'OK' if summary['structure_ok'] else 'FAIL（有单元的译文被重复显示或丢字，见 mismatch）'}")
    for fl in summary["failures"]:
        print(f"    ✗ {fl['id']}: {fl['text']}")
    for fl in summary["failures_bilingual"]:
        print(f"    ✗(双语) {fl['id']}: {fl['text']}")
    return 0 if summary["vtt_ok"] and summary["structure_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

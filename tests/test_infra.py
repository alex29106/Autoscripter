"""缓存键、校验脚本、显存方案、回退提示、中间文件清理的单元测试（不需要 GPU，不加载任何模型）。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subtitle.cli import load_config, build_parser, cmd_run  # noqa: E402
from subtitle.cache import asr_param_tag, TranscriptCache  # noqa: E402
from subtitle.segments import Segment, Transcript  # noqa: E402
from subtitle import media, vram, console  # noqa: E402
import verify_vtt  # noqa: E402

CFG = load_config()
TMP = ROOT / ".cache" / "test_tmp"


class TestCacheKey(unittest.TestCase):
    def test_default_params_keep_old_names(self):
        # 参数等于默认值：不加参数哈希，旧缓存文件名不变
        self.assertEqual(asr_param_tag(CFG["asr"], "whisper"), "")

    def test_changed_params_change_key(self):
        a = dict(CFG["asr"], drop_no_speech_prob=0.95)
        b = dict(CFG["asr"], drop_no_speech_prob=0.9)
        ta, tb = asr_param_tag(a, "whisper"), asr_param_tag(b, "whisper")
        self.assertTrue(ta.startswith("p") and tb.startswith("p"))
        self.assertNotEqual(ta, tb)
        self.assertEqual(ta, asr_param_tag(dict(a), "whisper"))           # 同样的参数同样的键
        # 不起作用的参数不影响键：不开 VAD 时的 vad_min_silence_ms；非 whisper 系后端
        self.assertEqual(asr_param_tag(dict(CFG["asr"], vad_min_silence_ms=300), "whisper"), "")
        self.assertEqual(asr_param_tag(a, "sensevoice"), "")
        # 整数和浮点写法不算改动
        self.assertEqual(asr_param_tag(dict(CFG["asr"], beam_size=5.0), "whisper"), "")

    def test_save_is_atomic_and_backward_compatible(self):
        TMP.mkdir(parents=True, exist_ok=True)
        try:
            src = TMP / "x.wav"
            src.write_bytes(b"RIFF" + b"\0" * 100)
            cache = TranscriptCache(TMP)
            tr = Transcript(language="ja", backend="whisper", duration=1.0,
                            segments=[Segment(0.0, 1.0, "はい")], language_probability=0.93,
                            language_forced=False, dropped=2)
            p = cache.save(src, tr, None, "novad")
            self.assertTrue(p.exists())
            self.assertFalse(list(TMP.glob("*.part")))
            back = cache.load(src, "whisper", None, "novad")
            self.assertEqual((back.language_probability, back.language_forced, back.dropped), (0.93, False, 2))
            # 旧格式（没有新字段）照样能读
            d = json.loads(p.read_text(encoding="utf-8"))
            for k in ("language_probability", "language_forced", "dropped"):
                d.pop(k)
            old = Transcript.from_dict(d)
            self.assertIsNone(old.language_probability)
            self.assertEqual(old.segments[0].text, "はい")
        finally:
            shutil.rmtree(TMP, ignore_errors=True)


class TestIntermediateFiles(unittest.TestCase):
    def test_only_deletes_inside_cache_dirs(self):
        TMP.mkdir(parents=True, exist_ok=True)
        try:
            outside = TMP / "keep.wav"
            outside.write_bytes(b"x")
            media.WAV_DIR.mkdir(parents=True, exist_ok=True)
            inside = media.WAV_DIR / "unit_test_tmp.wav"
            inside.write_bytes(b"x")
            media.remove_intermediate(outside, inside, None)
            self.assertTrue(outside.exists())
            self.assertFalse(inside.exists())
        finally:
            shutil.rmtree(TMP, ignore_errors=True)

    def test_default_dirs_inside_project(self):
        self.assertIn(ROOT.resolve(), media.WAV_DIR.resolve().parents)
        self.assertIn(ROOT.resolve(), media.VOCALS_DIR.resolve().parents)


class TestVerifyRepeats(unittest.TestCase):
    def test_classify_by_source(self):
        src = [(0, 1000, "はい"), (1000, 2000, "はい"), (2000, 3000, "行く"), (3000, 4000, "よ")]
        tgt = [(0, 1000, "好的"), (1000, 2000, "好的"), (2000, 3000, "走"), (3000, 4000, "走")]
        self.assertEqual(verify_vtt.adjacent_repeats(tgt, src), (1, 1, 0))
        self.assertEqual(verify_vtt.adjacent_repeats(tgt), (0, 0, 2))
        self.assertIn("原文不同 1 处", verify_vtt._repeat_warning(1, 1, 0))
        self.assertEqual(verify_vtt._repeat_warning(0, 0, 0), "")

    def test_max_cue_sec_from_config(self):
        self.assertEqual(verify_vtt.MAX_CUE_SEC, float(CFG["segmentation"]["max_duration_sec"]))


class TestVramPlan(unittest.TestCase):
    def _plan(self, total, precision="auto", name="hymt2"):
        with mock.patch.object(vram, "has_cuda", return_value=True), \
             mock.patch.object(vram, "total_vram_gb", return_value=total), \
             mock.patch.object(vram, "free_vram_gb", return_value=total - 1.0):
            return vram.mt_load_plan(name, precision)

    def test_tiers(self):
        self.assertEqual(self._plan(24.0)["precision"], "full")
        self.assertEqual(self._plan(16.0)["precision"], "int8")       # 16GB 放不下 15GB 的 bf16 权重 + 桌面
        self.assertTrue(self._plan(12.0)["load_in_8bit"])
        p8 = self._plan(8.0)
        self.assertEqual((p8["precision"], p8["load_in_4bit"]), ("nf4", True))   # 8GB 卡
        self.assertEqual(self._plan(8.0, name="qwen3b")["precision"], "full")    # 8GB 卡（fp16）
        self.assertEqual(self._plan(8.0, name="hymt2-1.8b")["precision"], "full")

    def test_forced(self):
        self.assertEqual(self._plan(8.0, "int8")["precision"], "int8")
        self.assertEqual(self._plan(24.0, "nf4")["precision"], "nf4")

    def test_small_gpu_falls_back_loudly(self):
        from subtitle.mt import factory
        seen = []
        console.add_sink(lambda lv, t: seen.append((lv, t)))
        try:
            with mock.patch.object(factory, "has_cuda", return_value=True), \
                 mock.patch.object(factory, "total_vram_gb", return_value=4.0):
                mt = factory.build_mt_with_fallback("hymt2", ROOT, CFG)
        finally:
            console._sinks.clear()
        self.assertEqual(mt.name, "nllb")
        self.assertTrue(mt.fallback)
        self.assertTrue(any(lv == "warn" and "NLLB" in t for lv, t in seen), seen)


class TestRun(unittest.TestCase):
    def test_no_input_files(self):
        TMP.mkdir(parents=True, exist_ok=True)
        try:
            args = build_parser(CFG).parse_args(["run", str(TMP)])
            self.assertEqual(cmd_run(args, json.loads(json.dumps(CFG))), 2)
        finally:
            shutil.rmtree(TMP, ignore_errors=True)


# 样本音频（不随仓库发布）：放一个已有转写缓存的音频在这里才会跑，否则跳过
SAMPLE = ROOT / "sample" / "example.mp3"


@unittest.skipUnless(SAMPLE.exists(), "缺少样本文件")
class TestRunPipeline(unittest.TestCase):
    """命令行编排：用缓存的转写 + 假翻译模型跑完整流程（不需要 GPU）。"""

    def _run(self, mt, extra=()):
        from subtitle import cli
        from subtitle.mt import factory
        out = TMP / "out"
        args = build_parser(CFG).parse_args(["run", str(SAMPLE), "--src-lang", "ja", "--mode", "bilingual",
                                             "--output-dir", str(out), *extra])
        cfg = json.loads(json.dumps(CFG))
        with mock.patch.object(factory, "build_mt_with_fallback", return_value=mt), \
             mock.patch.object(vram, "release_cuda", lambda: None):
            code = cli.cmd_run(args, cfg)
        return code, out

    def setUp(self):
        cache = TranscriptCache(ROOT / ".cache")
        if not cache.path_for(SAMPLE, "whisper", "ja", "novad").exists():
            self.skipTest("缺少这个样本的转写缓存")
        TMP.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_ok_writes_valid_files(self):
        from test_translation_pipeline import FakeMT
        code, out = self._run(FakeMT(), extra=["--recursive"])
        self.assertEqual(code, 0)
        files = sorted(p.name for p in out.glob("*.vtt"))
        self.assertEqual(len(files), 3, files)
        ok = verify_vtt.validate_dir(out)
        self.assertTrue(ok)

    def test_mt_failure_marks_file_failed(self):
        from test_translation_pipeline import FakeMT
        seen = []
        console.add_sink(lambda lv, t: seen.append((lv, t)))
        try:
            code, out = self._run(FakeMT(fail=True))
        finally:
            console._sinks.clear()
        self.assertEqual(code, 1)                                        # 有文件失败：返回 1
        self.assertFalse(list(out.glob("*.vtt")) if out.exists() else [])  # 不写出全是原文的“译文”
        self.assertTrue(any(lv == "error" and "连续" in t for lv, t in seen), seen)

    def test_missing_path_is_reported_and_skipped(self):
        from test_translation_pipeline import FakeMT
        from subtitle import cli
        seen = []
        console.add_sink(lambda lv, t: seen.append((lv, t)))
        try:
            out = TMP / "out2"
            args = build_parser(CFG).parse_args(["run", str(TMP / "不存在.mp3"), str(SAMPLE), "--src-lang", "ja",
                                                 "--mode", "mono-tgt", "--output-dir", str(out)])
            from subtitle.mt import factory
            with mock.patch.object(factory, "build_mt_with_fallback", return_value=FakeMT()), \
                 mock.patch.object(vram, "release_cuda", lambda: None):
                res = cli.run_pipeline(args, json.loads(json.dumps(CFG)))
        finally:
            console._sinks.clear()
        self.assertTrue(any(lv == "warn" and "路径不存在" in t for lv, t in seen))
        self.assertEqual([r.status for r in res.reports], ["ok"])
        self.assertEqual(len(res.reports[0].outputs), 1)
        self.assertEqual(res.reports[0].language, "ja")


if __name__ == "__main__":
    unittest.main()

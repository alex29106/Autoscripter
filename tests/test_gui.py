"""图形界面的辅助函数（不需要 GPU、不初始化 CUDA、不启动网页服务）。

运行：python -m unittest discover -s tests -p "test_gui.py" -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subtitle import gui  # noqa: E402
import verify_vtt  # noqa: E402

TMP = ROOT / ".cache" / "test_tmp_gui"
VTT = ("WEBVTT\n\n"
       "1\n00:00:01.000 --> 00:00:02.500\n第一句\n\n"
       "2\n00:00:04.000 --> 00:00:05.000\n第二句\nそのまま\n\n"     # 双语：两行
       "3\n00:00:05.000 --> 00:00:06.000\n第三句\n\n"               # 和上一条首尾相接
       "4\n00:01:10.000 --> 00:01:12.000\n最后一句\n")              # 和上一条隔了一分多钟


class TestPlayerSubtitles(unittest.TestCase):
    """网页播放器里每条字幕显示到下一条开始；下载的字幕文件不变。"""

    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)
        self.src = TMP / "x_cn.wav.vtt"
        self.src.write_bytes(VTT.encode("utf-8"))
        self.made: list[str] = []

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)
        for p in self.made:
            Path(p).unlink(missing_ok=True)

    def test_held_until_next_cue(self):
        out = gui._display_vtt(self.src)
        self.made.append(out)
        self.assertIn(gui.GRADIO_TMP.resolve(), Path(out).resolve().parents)    # 副本在项目 .cache/gradio 里
        self.assertEqual(self.src.read_bytes(), VTT.encode("utf-8"))             # 原文件一个字节都不变
        # 逐字节比对：WEBVTT 头、序号连续、无 BOM、\n 换行、不重叠。
        # （不用 verify_vtt.validate_file：它查的是输出文件的规则——单条 ≤7 秒、文件名——副本按设计不满足）
        expected = ("WEBVTT\n\n"
                    "1\n00:00:01.000 --> 00:00:04.000\n第一句\n\n"               # 2.5 秒 → 延到下一条开始的 4 秒
                    "2\n00:00:04.000 --> 00:00:05.000\n第二句\nそのまま\n\n"     # 本来就首尾相接，不变；两行都保留
                    "3\n00:00:05.000 --> 00:01:10.000\n第三句\n\n"               # 隔得再久也一直显示到下一条
                    "4\n00:01:10.000 --> 00:01:12.000\n最后一句\n")              # 最后一条不变
        self.assertEqual(Path(out).read_bytes(), expected.encode("utf-8"))

    def test_player_gets_display_copy(self):
        media_file = TMP / "x.wav"
        media_file.write_bytes(b"RIFF" + b"\0" * 100)
        a, v, info = gui._player_updates(str(self.src), {str(self.src): str(media_file)})
        self.made.append(a["subtitles"])
        self.assertEqual((a["visible"], v["visible"], info), (True, False, ""))
        self.assertNotEqual(Path(a["subtitles"]).resolve(), self.src.resolve())
        self.assertEqual(verify_vtt._read_cues(Path(a["subtitles"]))[0][:2], (1000, 4000))
        import gradio as gr
        self.assertIsNotNone(gr.Audio(subtitles=a["subtitles"], type="filepath").subtitles)   # Gradio 接受这份副本
        self.assertIsNotNone(gr.Video(subtitles=a["subtitles"]).subtitles)

    def test_unreadable_falls_back_to_original(self):
        empty = TMP / "empty_cn.wav.vtt"
        empty.write_bytes(b"WEBVTT\n")
        self.assertEqual(gui._display_vtt(empty), str(empty))


class TestFilesForPage(unittest.TestCase):
    """项目外的文件先复制进缓存再交给网页（Gradio 不肯给别处的文件）；汇总里写明保存位置。"""

    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_outside_files_are_copied_into_cache(self):
        outside = TMP / "elsewhere" / "a_cn.mp3.vtt"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(VTT.encode("utf-8"))
        fake_cwd, fake_tmp = TMP / "cwd", TMP / "systmp"
        fake_cwd.mkdir()
        fake_tmp.mkdir()
        inside = fake_cwd / "b_cn.mp3.vtt"
        inside.write_bytes(b"x")
        with mock.patch.object(gui.os, "getcwd", return_value=str(fake_cwd)), \
             mock.patch.object(gui.tempfile, "gettempdir", return_value=str(fake_tmp)):
            got = Path(gui._servable(outside))
            try:
                self.assertIn(gui.GRADIO_TMP.resolve(), got.resolve().parents)    # 复制进本次的缓存目录
                self.assertEqual((got.name, got.read_bytes()), (outside.name, outside.read_bytes()))
                self.assertEqual(Path(gui._servable(inside)), inside.resolve())   # 允许的位置：原样返回
                outside.write_bytes(b"WEBVTT\n\nchanged\n")                       # 原文件改了 → 重新复制
                later = time.time() + 10
                os.utime(outside, (later, later))
                self.assertEqual(Path(gui._servable(outside)).read_bytes(), b"WEBVTT\n\nchanged\n")
                self.assertEqual(gui._downloadable([str(outside), str(inside)]), [str(got), str(inside.resolve())])
            finally:
                got.unlink(missing_ok=True)

    def test_summary_says_where_files_are(self):
        rep = NS(status="ok", warnings=[], notes=[], language="ja", lang_forced=False, lang_prob=0.98, units=[1],
                 src_cues=5, mono_cues=3, file=Path("D:/音声/a.mp3"), cached=False,
                 outputs=[Path("D:/音声/a_cn.mp3.vtt"), Path("D:/音声/a_ja.mp3.vtt")])
        job = NS(error="", result=NS(mt_used="", mt_plan="", mt_fallback="", mt_requested="", reports=[rep],
                                     seconds=3.0, ok=True))
        page = gui._final_html(job)
        self.assertIn("字幕已自动保存到：" + str(Path("D:/音声")), page)


class TestOldSessions(unittest.TestCase):
    """直接关窗口或崩溃时 Gradio 来不及删的缓存，下次启动时按“进程已不在”删掉。"""

    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_pid_alive(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertTrue(gui._pid_alive(p.pid))
        finally:
            p.kill()
            p.wait()
        self.assertFalse(gui._pid_alive(p.pid))
        self.assertTrue(gui._pid_alive(os.getpid()))

    def test_only_dead_sessions_and_old_leftovers_are_removed(self):
        root = TMP / "groot"
        dead = root / "s4294967"                                   # 不存在的进程
        (dead / "h").mkdir(parents=True)
        (dead / "h" / "up.mp3").write_bytes(b"0" * 1000)
        alive = root / f"s{os.getpid()}"                           # 还在运行的界面
        alive.mkdir()
        (alive / "up.mp3").write_bytes(b"1")
        mine = root / "s1"                                         # 本次的目录
        mine.mkdir()
        old = root / "0123abcd"                                    # 旧版直接放在 .cache/gradio 里、超过一天的
        old.mkdir()
        (old / "old.mp3").write_bytes(b"2" * 10)
        two_days_ago = time.time() - 2 * 86400
        os.utime(old / "old.mp3", (two_days_ago, two_days_ago))
        os.utime(old, (two_days_ago, two_days_ago))
        (root / "tmpfresh").write_bytes(b"")                       # 旧版的、不到一天：可能旧版界面还开着，不删
        self.assertEqual(gui._clean_old_sessions(root=root, keep=mine), (2, 1010))
        self.assertEqual(sorted(p.name for p in root.iterdir()), sorted([alive.name, "s1", "tmpfresh"]))
        self.assertTrue((alive / "up.mp3").exists())


def tearDownModule():
    # 这个测试进程自己的缓存目录（.cache/gradio/s<进程号>）
    if gui.GRADIO_TMP.name == f"s{os.getpid()}":
        shutil.rmtree(gui.GRADIO_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

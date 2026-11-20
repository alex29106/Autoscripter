"""翻译阶段的单元测试（不需要 GPU，不加载任何模型）。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subtitle.cli import load_config  # noqa: E402
from subtitle.segments import Segment, Transcript, Word  # noqa: E402
from subtitle.segmenter import build_cue_groups, CueGroup, clean_text, _CJK_STRIP, max_chars_for  # noqa: E402
from subtitle.units import build_units, is_complete_ja  # noqa: E402
from subtitle.distribute import (split_proportional, display_mono, layout_mono, layout_mono_relaxed,  # noqa: E402
                                 display_len)
from subtitle.mt.base import MTBackend, MTFailure, check_unit_output  # noqa: E402
from subtitle.pipeline import segment_and_translate, same_text  # noqa: E402

CFG = load_config()
# 开发用的真实转写清单（不随仓库发布）；没有时依赖它的测试跳过
_TRANSCRIPTS_FILE = ROOT / "eval" / "data" / "transcripts.json"
TRANSCRIPTS = (json.loads(_TRANSCRIPTS_FILE.read_text(encoding="utf-8"))
               if _TRANSCRIPTS_FILE.exists() else {})


def _cue(s, e, t):
    return Segment(start=s, end=e, text=t)


def _group(sentence, cues, seg):
    return CueGroup(sentence, cues, seg)


def _covers(src, tgt):
    """tgt 的每一条是否正好由连续的 src 条合并而成，并且合起来覆盖全部 src 条。"""
    i = 0
    for c in tgt:
        if i >= len(src) or abs(c.start - src[i].start) > 1e-6:
            return False
        while i < len(src) and src[i].end < c.end - 1e-6:
            i += 1
        if i >= len(src) or abs(src[i].end - c.end) > 1e-6:
            return False
        i += 1
    return i == len(src)


class FakeMT(MTBackend):
    """确定性的假后端：每个输入字符译成一个“译”，可注入坏输出测试回退，或让每次调用都抛异常。"""
    name = "fake"

    def __init__(self, bad=None, fail=False):
        super().__init__(ROOT, CFG)
        self.bad = bad or {}         # {调用序号: 输出}
        self.fail = fail
        self.calls = 0

    def load(self):
        pass

    def translate_batch(self, lines, ctx_before, ctx_after, src_lang, tgt_lang):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        out = []
        for t in lines:
            self.calls += 1
            out.append(self.bad.get(self.calls, "译" * max(1, len(t.replace(" ", "")) // 2)))
        return out


def _has_fugashi() -> bool:
    return is_complete_ja("そうです") is not None


class TestUnits(unittest.TestCase):
    def test_merge_contiguous_unterminated(self):
        g = [_group("私がただれた責任", [_cue(508.9, 511.5, "私がただれた責任")], 67),
             _group("取ってください", [_cue(511.5, 513.4, "取ってください")], 68)]
        u = build_units(g, CFG)
        self.assertEqual(len(u), 1)
        self.assertEqual(u[0].text, "私がただれた責任、取ってください")   # 软分隔
        self.assertEqual(len(u[0].cues), 2)

    def test_no_merge_after_terminal_punct_or_gap(self):
        g = [_group("見ましたよね?", [_cue(0, 1, "見ましたよね?")], 1),
             _group("だから", [_cue(1.0, 2, "だから")], 2),
             _group("吐いてなかった時のことです", [_cue(5.0, 7, "吐いてなかった時のことです")], 3)]
        u = build_units(g, CFG)
        self.assertEqual([x.text for x in u], ["見ましたよね?", "だから", "吐いてなかった時のことです"])

    def test_complete_sentence_not_merged_ja(self):
        if is_complete_ja("そうです") is None:
            self.skipTest("fugashi 未安装")
        # 前一句已经说完（终助词 / 终止形 / 感叹词）：即使首尾相接、没有标点也不合并
        for a, b in [("今日は寒いですね", "早く帰りましょう"), ("もう食べた", "お腹いっぱい"), ("はい", "分かりました")]:
            g = [_group(a, [_cue(0, 2, a)], 1), _group(b, [_cue(2, 4, b)], 2)]
            self.assertEqual([x.text for x in build_units(g, CFG, "ja")], [a, b], (a, b))
        # 前一句没说完（名词 / 格助词 / 连用形・て形）：合并
        for a, b in [("私がただれた責任", "取ってください"), ("駅の前で", "待っています"), ("手を洗って", "ご飯を食べます")]:
            g = [_group(a, [_cue(0, 2, a)], 1), _group(b, [_cue(2, 4, b)], 2)]
            self.assertEqual(len(build_units(g, CFG, "ja")), 1, (a, b))
        # 不指定语种 / 关掉开关：旧规则
        g = [_group("もう食べた", [_cue(0, 2, "もう食べた")], 1), _group("お腹いっぱい", [_cue(2, 4, "お腹いっぱい")], 2)]
        self.assertEqual(len(build_units(g, CFG)), 1)
        cfg = json.loads(json.dumps(CFG))
        cfg["translation"]["merge_require_incomplete"] = False
        self.assertEqual(len(build_units(g, cfg, "ja")), 1)

    def test_group_sizes_recorded(self):
        if is_complete_ja("そうです") is None:
            self.skipTest("fugashi 未安装")
        # 合并时记下每个句子组各有几条字幕（单语排版用：一条字幕不能跨过半个句子组）
        g = [_group("駅の前で", [_cue(0, 1, "駅の"), _cue(1, 2, "前で")], 1),
             _group("待っています", [_cue(2, 4, "待っています")], 2),
             _group("はい", [_cue(4, 5, "はい")], 3)]
        u = build_units(g, CFG, "ja")
        self.assertEqual([x.group_sizes for x in u], [[2, 1], [1]])
        self.assertEqual([len(x.cues) for x in u], [3, 1])

    def test_final_particle_counts_as_complete(self):
        if is_complete_ja("そうです") is None:
            self.skipTest("fugashi 未安装")
        # 句末的终助词（か/ね/の…）一律算说完，只看这一句自己。间接疑问的 か（「誰が来るか」|「知りません」）
        # 因此不会合并——已知的漏判，有意保留（接上后一句再分析会把真正说完的句子也判成没说完）。
        for a, b in [("行きますか", "お兄様"), ("もう帰るね", "また明日"), ("誰が来るか", "知りません")]:
            self.assertTrue(is_complete_ja(a), a)
            g = [_group(a, [_cue(0, 2, a)], 1), _group(b, [_cue(2, 4, b)], 2)]
            self.assertEqual(len(build_units(g, CFG, "ja")), 2, (a, b))

    def test_voiced_start_gap(self):
        # 片段名义上首尾相接，但后一句开头是一段停顿（Whisper 把停顿算进了第一个词）：按开口时间算间隔，不合并
        a = CueGroup("駅の前で", [_cue(0, 2, "駅の前で")], 1)
        b = CueGroup("待っています", [_cue(2, 6, "待っています")], 2, voiced_start=4.5)
        self.assertEqual(len(build_units([a, b], CFG, "ja")), 2)
        b = CueGroup("待っています", [_cue(2, 6, "待っています")], 2, voiced_start=2.1)
        self.assertEqual(len(build_units([a, b], CFG, "ja")), 1)

    def test_max_segments(self):
        g = [_group("コト", [_cue(i, i + 1, "コト")], i) for i in range(6)]
        u = build_units(g, CFG)
        self.assertTrue(all(len(set(x.seg_indices)) <= CFG["translation"]["max_unit_segments"] for x in u))
        self.assertEqual(sum(len(x.cues) for x in u), 6)


class TestDistribute(unittest.TestCase):
    CASES = [
        ("送到这里就可以了，之后就没问题了。", ["ここまで送っていただければ", "あとは大丈夫です"]),
        ("先去酒馆看看有没有护卫的委托，之后再找住处。", ["護衛の依頼がないか酒場に", "寄ってそれから宿を探す"]),
        ("请为我承担起皮肤溃烂的责任吧。", ["私がただれた責任", "取ってください"]),
        ("这是我出门时母亲交给我的东西。", ["家を出るときに母が持たせてくれたものなの", "です"]),
        ("就算听不懂我在说什么也没关系，就算不明白也没关系。", ["私が何を言ってるか分からなくても", "良いです 私が", "何を言ってるか分からなくても良いです"]),
        ("嗯", ["あ", "の"]),
    ]

    def test_split_invariants(self):
        for zh, src in self.CASES:
            parts = split_proportional(zh, src)
            self.assertEqual(len(parts), len(src), zh)
            self.assertTrue(all(p.strip() for p in parts), (zh, parts))
            if len(zh) >= len(src):
                self.assertEqual("".join(parts).replace(" ", ""), zh.replace(" ", ""), zh)
            for p in parts[1:]:
                if len(zh) >= 2 * len(src):
                    self.assertNotIn(p[0], "，。！？、了的吗呢吧啊", (zh, parts))

    def test_min_two_cjk_when_possible(self):
        parts = split_proportional("这是我出门时母亲交给我的东西。", ["家を出るときに母が持たせてくれたものなの", "です"])
        self.assertTrue(all(len(re.findall(r"[一-鿿]", p)) >= 2 for p in parts), parts)

    def test_english_target_never_splits_words(self):
        en = "If you just take me this far, I will be fine from here on."
        src = ["ここまで送っていただければ", "あとは大丈夫です"]
        parts = split_proportional(en, src)
        self.assertEqual(len(parts), 2)
        self.assertEqual(" ".join(parts).split(), en.split())      # 没有任何词被切开
        self.assertTrue(all(p.strip() for p in parts))
        self.assertEqual(split_proportional("Okay.", ["a", "b"]), ["Okay.", "Okay."])

    def test_display_mono(self):
        src = ["ここまで送っていただければ", "あとは大丈夫です"]
        self.assertEqual(display_mono("送到这里就可以了，之后就没问题了。", src),
                         ["送到这里就可以了，之后就没问题了。"] * 2)
        long = "朝着那仿佛永无止境般漫长的时间终点不断前进，除此之外别无选择。"
        self.assertEqual(len(display_mono(long, src)), 2)
        self.assertNotEqual(display_mono(long, src)[0], long)


class TestLayoutMono(unittest.TestCase):
    def _cues(self, spec):
        return [_cue(s, e, t) for s, e, t in spec]

    def _check(self, text, cues, lay, max_chars=20, max_dur=7.0):
        self.assertIsNotNone(lay)
        self.assertEqual([a for a, _, _ in lay], [0] + [b for _, b, _ in lay[:-1]])     # 首尾相接
        self.assertEqual(lay[-1][1], len(cues))                                          # 覆盖全部原文条
        self.assertEqual("".join(t for _, _, t in lay).replace(" ", ""), text.replace(" ", ""))   # 不丢字
        for a, b, t in lay:
            self.assertGreater(b, a)
            self.assertTrue(t.strip())
            self.assertLessEqual(display_len(t), max_chars, lay)
            self.assertLessEqual(cues[b - 1].end - cues[a].start, max_dur + 1e-6, lay)
            self.assertNotIn(t[0], "，。！？、了的吗呢吧啊」』）”", lay)

    def test_fits_in_one_cue(self):
        cues = self._cues([(0, 2, "ここまで送っていただければ"), (2, 3.5, "あとは大丈夫です")])
        zh = "送到这里就可以了，之后就没问题了。"
        self.assertEqual(layout_mono(zh, cues), [(0, 2, zh)])       # 一句话一条，不再每条重复

    def test_split_by_duration(self):
        cues = self._cues([(0, 3, "周り"), (3, 6, "私とお兄さん以外"), (6, 9, "誰の姿も見えないでしょ")])
        zh = "周围除了我和你以外，看不到任何人吧？"
        lay = layout_mono(zh, cues)
        self._check(zh, cues, lay)
        self.assertEqual(len(lay), 2)                               # 9 秒 > 7 秒：最少 2 条
        self.assertTrue(lay[0][2].endswith("，"), lay)              # 切在标点后

    def test_split_by_length(self):
        cues = self._cues([(0, 1.5, "護衛の依頼がないか"), (1.5, 3, "酒場に寄って"), (3, 4.5, "それから宿を探す"),
                           (4.5, 6, "つもりです")])
        zh = "我打算先去酒馆看看有没有护卫的委托，然后再去找今晚住的地方。"
        lay = layout_mono(zh, cues)
        self._check(zh, cues, lay)
        self.assertEqual(len(lay), 2)                               # 28 字：最少 2 条

    def test_no_cue_straddles_group_boundary(self):
        # 单元 = 句子组 A（1 条）+ 句子组 B（2 条）；B 整组 8 秒放不进一条。
        cues = self._cues([(0, 3, "それでも"), (3, 6.5, "明日の朝になったら"), (6.5, 11, "もう一度だけ話しましょう")])
        zh = "即便如此，等到明天早上，我们再谈最后一次吧"
        lay = layout_mono(zh, cues)                                  # 不知道分组：条数最少（A + B 的前半 拼成一条）
        self._check(zh, cues, lay)
        self.assertEqual([(a, b) for a, b, _ in lay], [(0, 2), (2, 3)])
        lay = layout_mono(zh, cues, group_sizes=[1, 2])              # 知道分组：不把 A 和 B 的前半拼在一起
        self._check(zh, cues, lay)
        self.assertEqual([(a, b) for a, b, _ in lay], [(0, 1), (1, 2), (2, 3)])
        lay = layout_mono(zh, cues, group_sizes=[2, 1])              # 整组 + 整组：可以
        self._check(zh, cues, lay)
        self.assertEqual([(a, b) for a, b, _ in lay], [(0, 2), (2, 3)])
        short = self._cues([(0, 1, "それでも"), (1, 2, "明日の朝になったら"), (2, 3, "もう一度だけ話しましょう")])
        self.assertEqual(layout_mono("即便如此，明早再谈一次吧", short, group_sizes=[1, 2]),
                         [(0, 3, "即便如此，明早再谈一次吧")])      # 整句放得下：仍然只出一条

    def test_long_pause_inside_unit(self):
        # 两条原文各 5 秒、中间停了 6 秒：整句跨度 16 秒，但两条就够（每条不超过 7 秒），不该判成排不出来
        cues = self._cues([(0, 5, "私はここを離れることは無いです"), (11, 16, "あなたも絶対に離れない")])
        zh = "我绝不会离开这里，你也不会离开的。"
        lay = layout_mono(zh, cues)
        self._check(zh, cues, lay)
        self.assertEqual([(a, b) for a, b, _ in lay], [(0, 1), (1, 2)])

    def test_infeasible_returns_none(self):
        cues = self._cues([(0, 1, "あ"), (1, 2, "の")])
        self.assertIsNone(layout_mono("这是一句无论如何也没有办法只用两条二十个字以内的字幕放下来的译文" * 2, cues))   # 两条放不下
        long = self._cues([(0, 5, "駅の"), (5, 10, "前で")])
        self.assertIsNone(layout_mono("嗯", long))                  # 10 秒要两条，一个字切不成两段

    def _check_relaxed(self, text, cues, lay):
        """退路的不变量：首尾相接、覆盖全部原文条、非延续的各条拼起来正好是整句（不重复不丢字）。"""
        self.assertEqual([a for a, _, _, _ in lay], [0] + [b for _, b, _, _ in lay[:-1]])
        self.assertEqual(lay[-1][1], len(cues))
        self.assertFalse(lay[0][3])                                  # 第一条不可能是延续
        self.assertTrue(same_text([t for _, _, t, held in lay if not held], text), lay)
        for a, b, t, held in lay:
            self.assertTrue(t.strip())
            if held:                                                 # 延续显示：文字和上一条相同
                prev = [x for x in lay if x[1] == a][0]
                self.assertEqual(prev[2], t)

    def test_relaxed_too_long_never_repeats(self):
        # 译文比两条原文条容得下的长：宁可超字数，也不把同一段译文放两遍
        cues = self._cues([(0, 3, "あの"), (3, 6, "それで")])
        zh = "这是一句无论如何也没有办法只用两条二十个字以内的字幕放下来的译文，所以只能超出字数上限了"
        self.assertIsNone(layout_mono(zh, cues))
        lay, info = layout_mono_relaxed(zh, cues)
        self._check_relaxed(zh, cues, lay)
        self.assertEqual([(a, b, held) for a, b, _, held in lay], [(0, 1, False), (1, 2, False)])
        self.assertEqual(info["hold"], 0)
        self.assertGreaterEqual(info["over"], 1)
        self.assertLess(abs(display_len(lay[0][2]) - display_len(lay[1][2])), 10, lay)   # 超出的字数摊开

    def test_relaxed_hold_only_when_unavoidable(self):
        # 一个字的译文对 10 秒的两条：切不开，第二条只能延续显示，并计数
        cues = self._cues([(0, 5, "駅の"), (5, 10, "前で")])
        lay, info = layout_mono_relaxed("嗯", cues)
        self.assertEqual(lay, [(0, 1, "嗯", False), (1, 2, "嗯", True)])
        self.assertEqual(info["hold"], 1)
        # 能切开就不延续：两个句子组各 5 秒，译文两句
        lay, info = layout_mono_relaxed("在车站，前面。", cues)
        self._check_relaxed("在车站，前面。", cues, lay)
        self.assertEqual(info["hold"], 0)

    def test_japanese_target_is_laid_out(self):
        # 译文是日文（如英译日）：按日文文节切，而不是拒绝排版、退回每条显示整句
        if not _has_fugashi():
            self.skipTest("fugashi 未安装")
        cues = self._cues([(0, 4, "It is a nice day today"), (4, 8, "shall we go for a walk")])
        ja = "今日はいい天気ですね、散歩に行きましょうか"
        lay = layout_mono(ja, cues)
        self.assertIsNotNone(lay)
        self.assertEqual([(a, b) for a, b, _ in lay], [(0, 1), (1, 2)])
        self.assertTrue(same_text([t for _, _, t in lay], ja))
        self.assertTrue(lay[0][2].endswith("、"), lay)
        self.assertEqual(layout_mono("カタン", self._cues([(0, 1, "a"), (1, 2, "b")])), [(0, 2, "カタン")])
        parts = split_proportional(ja, ["It is a nice day today", "shall we go for a walk"])
        self.assertEqual(len(parts), 2)
        self.assertTrue(same_text(parts, ja), parts)                  # 双语也切开，不再每条放整句

    def test_single_cue_and_english(self):
        self.assertEqual(layout_mono("好的", self._cues([(0, 1, "はい")])), [(0, 1, "好的")])
        cues = self._cues([(0, 4, "ここまで送っていただければ"), (4, 8, "あとは大丈夫です")])
        en = "If you just take me this far, I will be fine from here on."
        lay = layout_mono(en, cues, max_chars=42)
        self.assertEqual(len(lay), 2)
        self.assertEqual(" ".join(t for _, _, t in lay).split(), en.split())     # 不切开单词


class TestCheckUnitOutput(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(check_unit_output("落ち着きます?", "", "zh"), "empty")
        self.assertEqual(check_unit_output("カタン", "カタン", "zh"), "kana")
        self.assertEqual(check_unit_output("あいにく持ち合わせもなくお渡しできるものがない状況です",
                                           "unfortunately, I don't have anything to offer", "zh"), "not-target-language")
        self.assertEqual(check_unit_output("だから", "所以，您能收下这枚戒指吗？它被称为指引之戒，其实还有一对呢", "zh"), "too-long")
        self.assertEqual(check_unit_output("ですからこの指輪を受け取っていただけますか", "所以，能请您收下这枚戒指吗？", "zh"), "")


class TestSegmenter(unittest.TestCase):
    def test_max_chars_for(self):
        """原文断句和译文排版用同一条字数规则：中日粤按字算 20，带地区后缀的同语种一样，没写的语种用 default。"""
        for lang in ("zh", "ja", "yue", "ZH-tw"):
            self.assertEqual(max_chars_for(lang, CFG), 20, lang)
        self.assertEqual(max_chars_for("fr", CFG), CFG["segmentation"]["max_chars"]["default"])

    def test_display_keeps_question_and_exclamation(self):
        self.assertEqual(clean_text("落ち着きます？", "ja"), "落ち着きます？")
        self.assertEqual(clean_text("是吗，好的。", "zh"), "是吗 好的")
        self.assertEqual(clean_text("太好了！", "zh"), "太好了！")

    def test_split_at_pause_inside_long_word(self):
        try:
            import fugashi  # noqa: F401
        except ImportError:
            self.skipTest("fugashi 未安装")
        # 「いい」名义上长 8.4 秒：其实是 6 秒多的停顿 + 2 个字。应在停顿处切开，后一条不截掉结尾
        words = [Word(0.0, 0.6, "今日は"), Word(0.6, 9.0, "いい"), Word(9.0, 9.4, "天気"), Word(9.4, 10.0, "ですね")]
        tr = Transcript(language="ja", backend="t",
                        segments=[Segment(start=0.0, end=10.0, text="今日はいい天気ですね", words=words)])
        groups = build_cue_groups(tr, CFG)
        cues = [c for g in groups for c in g.cues]
        self.assertEqual([c.text for c in cues], ["今日は", "いい天気ですね"])
        self.assertAlmostEqual(cues[1].end, 10.0, places=3)
        self.assertLessEqual(cues[1].end - cues[1].start, CFG["segmentation"]["max_duration_sec"] + 1e-6)
        self.assertLessEqual(cues[0].end, cues[1].start + 1e-6)
        self.assertGreater(groups[0].voiced_start, -1)

    def test_split_at_asr_space(self):
        try:
            import fugashi  # noqa: F401
        except ImportError:
            self.skipTest("fugashi 未安装")
        # ASR 用空格隔开的两段各自 4 秒，合起来超过 7 秒：在空格处切，而不是切进词里
        words = [Word(0.0, 4.0, "ゆっくり"), Word(4.0, 8.0, " ゆっくり")]
        tr = Transcript(language="ja", backend="t",
                        segments=[Segment(start=0.0, end=8.0, text="ゆっくり ゆっくり", words=words)])
        cues = [c for g in build_cue_groups(tr, CFG) for c in g.cues]
        self.assertEqual([c.text for c in cues], ["ゆっくり", "ゆっくり"])

    def test_real_transcripts(self):
        try:
            import fugashi
        except ImportError:
            self.skipTest("fugashi 未安装")
        tagger = fugashi.Tagger()
        for key, meta in TRANSCRIPTS.items():
            p = ROOT / ".cache" / meta["cache"]
            if not p.exists():
                continue
            tr = Transcript.from_dict(json.loads(p.read_text(encoding="utf-8")))
            groups = build_cue_groups(tr, CFG)
            cues = [c for g in groups for c in g.cues]
            for a, b in zip(cues, cues[1:]):
                self.assertLessEqual(a.end, b.start + 1e-6, f"{key}: 时间轴重叠")
            for c in cues:
                self.assertGreater(c.end, c.start)
                self.assertLessEqual(c.end - c.start, CFG["segmentation"]["max_duration_sec"] + 1e-6)
                self.assertLessEqual(len(c.text.replace(" ", "")), CFG["segmentation"]["max_chars"]["ja"])
            for g in groups:                # 一句切成多条时，切点不在词中间、不在助动词/接尾辞前
                if len(g.cues) < 2:
                    continue
                full = "".join(c.text.replace(" ", "") for c in g.cues)
                # 在带标点和空格的原句上分词（去掉后再分词会把「ガタン、ゴトン」并成一个词），再把切点映射回去
                raw = g.sentence
                kept = [i for i, ch in enumerate(raw) if ch not in _CJK_STRIP and ch != " "]
                if "".join(raw[i] for i in kept) != full:
                    continue
                spans, pos = [], 0
                for w in tagger(raw):
                    s = raw.find(w.surface, pos)
                    spans.append((s, s + len(w.surface), w.feature.pos1))
                    pos = s + len(w.surface)
                off = 0
                for c in g.cues[:-1]:
                    off += len(c.text.replace(" ", ""))
                    left, right = kept[off - 1], kept[off]      # 切点两侧的字符在原句中的位置
                    self.assertFalse(any(a <= left and right < b for a, b, _ in spans), f"{key}: 切在词中间 {full}")
                    self.assertFalse(any(a == right and left == right - 1 and p in ("助動詞", "接尾辞")
                                         for a, b, p in spans), f"{key}: 切在助动词前 {full}")


class TestPipeline(unittest.TestCase):
    def _transcript(self):
        meta = TRANSCRIPTS.get("2-1")
        p = ROOT / ".cache" / meta["cache"] if meta else None
        if p is None or not p.exists():
            self.skipTest("缺少转写缓存")
        return Transcript.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def test_counts_and_timeline_identical(self):
        """双语：条数和时间轴与原文完全一致。单语：旧方式（repeat/split/逐条）同样一致。"""
        tr = self._transcript()
        for gran, mode in (("sentence", "repeat"), ("sentence", "split"), ("cue", "sentence")):
            cfg = json.loads(json.dumps(CFG))
            cfg["translation"]["granularity"] = gran
            cfg["translation"]["mono_display"] = mode
            res = segment_and_translate(tr, FakeMT(), cfg, "ja", "zh")
            for tgt in (res.tgt_mono, res.tgt_bilingual):
                self.assertEqual(len(tgt), len(res.src_cues), (gran, mode))
                self.assertEqual([(c.start, c.end) for c in tgt], [(c.start, c.end) for c in res.src_cues])
                self.assertTrue(all(c.text.strip() for c in tgt))

    def test_mono_sentence_layout(self):
        """默认的单语排版：条数可以比原文少，但每条正好由连续的原文条合并而成、覆盖全部原文条。"""
        for key in TRANSCRIPTS:
            p = ROOT / ".cache" / TRANSCRIPTS[key]["cache"]
            if not p.exists():
                continue
            tr = Transcript.from_dict(json.loads(p.read_text(encoding="utf-8")))
            res = segment_and_translate(tr, FakeMT(), CFG, "ja", "zh")
            self.assertEqual(CFG["translation"]["mono_display"], "sentence")
            self.assertEqual([(c.start, c.end) for c in res.tgt_bilingual], [(c.start, c.end) for c in res.src_cues])
            self.assertLessEqual(len(res.tgt_mono), len(res.src_cues))
            self.assertTrue(_covers(res.src_cues, res.tgt_mono), key)
            self.assertTrue(all(c.text.strip() for c in res.tgt_mono))
            for c in res.tgt_mono:
                self.assertLessEqual(c.end - c.start, CFG["segmentation"]["max_duration_sec"] + 1e-6)
            # 一个翻译单元的译文只出现一次：单语条数 = 各单元排出的条数之和；
            # 每个整句翻译的单元，非延续的各条拼起来正好等于它的译文（重复或丢字都会失败）；
            # 整句放得下（≤20 字、≤7 秒）的单元只有一条
            self.assertEqual(sum(u["mono_cues"] for u in res.units), len(res.tgt_mono))
            self.assertEqual(res.layout_stats["mismatch"], 0, key)
            for u in res.units:
                if u["layout"] == "cue":
                    continue
                self.assertTrue(same_text([t for t, held in u["pieces"] if not held], u["zh"]), u)
                if display_len(u["zh"]) <= 20 and u["end"] - u["start"] <= 7.0 + 1e-6:
                    self.assertEqual(u["mono_cues"], 1, u)
            n_held = sum(held for u in res.units for _t, held in u["pieces"])
            self.assertEqual(n_held, res.layout_stats["hold"])

    def test_repeat_is_detected(self):
        """旧的 repeat 方式会把整句放到每一条上：检查必须能看见（mismatch > 0），默认方式为 0。"""
        tr = self._transcript()
        cfg = json.loads(json.dumps(CFG))
        cfg["translation"]["mono_display"] = "repeat"
        res = segment_and_translate(tr, FakeMT(), cfg, "ja", "zh")
        self.assertGreater(res.layout_stats["mismatch"], 0)
        res = segment_and_translate(tr, FakeMT(), CFG, "ja", "zh")
        self.assertEqual(res.layout_stats["mismatch"], 0)

    def test_speaker_repeating_is_not_a_repeat(self):
        """说话人自己把同一句话说两遍：两个翻译单元、各显示一次，相邻两条文字相同是正常的。"""
        tr = Transcript(language="ja", backend="t", segments=[
            Segment(start=0.0, end=1.5, text="ありがとう。"), Segment(start=1.6, end=3.0, text="ありがとう。")])
        res = segment_and_translate(tr, FakeMT(), CFG, "ja", "zh")
        self.assertEqual([c.text for c in res.tgt_mono], ["译译译", "译译译"])
        self.assertEqual(len(res.units), 2)
        self.assertEqual(res.layout_stats["mismatch"], 0)
        self.assertEqual(res.layout_stats["hold"], 0)

    def test_short_translation_holds(self):
        """切不开的短译文（「是的」不能切成「是」「的」）对 10 秒、两条：只能延续显示，记进 hold，不算重复。"""
        if not _has_fugashi():
            self.skipTest("fugashi 未安装")
        tr = Transcript(language="ja", backend="t", segments=[
            Segment(start=0.0, end=5.0, text="駅の前で"), Segment(start=5.0, end=10.0, text="待っています")])
        res = segment_and_translate(tr, FakeMT(bad={1: "是的"}), CFG, "ja", "zh")
        self.assertEqual(len(res.units), 1)
        self.assertEqual([(c.start, c.end, c.text) for c in res.tgt_mono],
                         [(0.0, 5.0, "是的"), (5.0, 10.0, "是的")])
        self.assertEqual(res.layout_stats["hold"], 1)
        self.assertEqual(res.layout_stats["relaxed"], 1)
        self.assertEqual(res.layout_stats["mismatch"], 0)

    def test_consecutive_errors_abort(self):
        """翻译模型一直出错（如显存不足）：不再悄悄写出一份全是原文的“译文”，而是报错。"""
        tr = Transcript(language="ja", backend="t", segments=[
            Segment(start=float(i), end=i + 0.9, text=t) for i, t in enumerate(["はい。", "そうです。", "行きます。", "了解。"])])
        with self.assertRaises(MTFailure):
            segment_and_translate(tr, FakeMT(fail=True), CFG, "ja", "zh")

    def test_same_language_is_reported(self):
        tr = Transcript(language="zh", backend="t", segments=[Segment(start=0.0, end=1.0, text="你好。")])
        res = segment_and_translate(tr, FakeMT(), CFG, "zh", "zh")
        self.assertTrue(res.skipped)

    def test_fallbacks(self):
        tr = Transcript(language="ja", backend="t", segments=[
            Segment(start=0.0, end=2.0, text="私がただれた責任"),
            Segment(start=2.0, end=3.5, text="取ってください")])
        # 第 1 次（带上下文）返回假名 -> 去掉上下文重试；本例没有上下文，所以直接退回逐条翻译
        mt = FakeMT(bad={1: "カタン"})
        res = segment_and_translate(tr, mt, CFG, "ja", "zh")
        self.assertEqual(res.unit_stats.get("cue-fallback"), 1)
        self.assertEqual(len(res.tgt_mono), 2)               # 逐条翻译的单元保持一一对应
        # 全部都坏 -> 保留原文（不丢条）
        mt = FakeMT(bad={i: "カタン" for i in range(1, 20)})
        res = segment_and_translate(tr, mt, CFG, "ja", "zh")
        self.assertEqual(res.unit_stats.get("source"), 1)
        self.assertEqual([c.text for c in res.tgt_mono], [c.text for c in res.src_cues])


if __name__ == "__main__":
    unittest.main()

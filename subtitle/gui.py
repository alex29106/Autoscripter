"""图形界面（Gradio）：python main.py gui

和命令行走同一条代码路径（cli.run_pipeline），命令行能做的这里都能做；界面只是把参数换成控件：
  - 拖入音频/视频文件（可多个），或者填本机路径（文件或文件夹，可多行）；
  - 基本设置 = 命令行参数；高级参数 = config.yaml 里的超参数（只对这一次运行生效，不改 config.yaml）；
  - 运行日志、红字警告（降级/回退/没翻译出来/失败）、运行汇总；
  - 生成的字幕：下载、表格查看（可搜索）、边听边看（播放器里每条字幕显示到下一条开始，下载的文件不变）；
  - “查看已有字幕”：任意 .vtt 的表格 + 格式校验 + 质量警告（同 verify_vtt.py）。
只在本机 127.0.0.1 上监听，不开公网分享，不发统计数据；界面用系统字体（不连 Google Fonts）。
字幕在每个文件完成时自动保存：拖入的文件写到项目的 output 目录，填本机路径的写在原文件旁边，填了输出目录就写那里。
拖入文件的副本、下载用和播放用的副本由 Gradio 存在 .cache/gradio/s<进程号> 里：按 Ctrl+C 退出时删掉；
直接关窗口或崩溃时它来不及删，下次启动界面时由 _clean_old_sessions 删。
"""
from __future__ import annotations

import copy
import hashlib
import html
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from . import console
from .cli import PROJECT_ROOT, load_config, build_parser, run_pipeline, _fmt_dur

GRADIO_ROOT = PROJECT_ROOT / ".cache" / "gradio"
UPLOAD_OUT = PROJECT_ROOT / "output"
# 必须在 import gradio 之前设置。每个界面进程一个子目录，进程退出后能认出哪些是没清掉的
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
GRADIO_TMP = Path(os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_ROOT / f"s{os.getpid()}")))

log = logging.getLogger("subtitle.gui")

MAX_PLAYER_MB = 300        # 超过这个大小的原文件不在网页里播放（浏览器要把整个文件读进来）
LOG_LINES = 400

SRC_LANGS = [("自动识别（auto）", "auto"), ("日语 ja", "ja"), ("英语 en", "en"), ("韩语 ko", "ko"),
             ("中文 zh", "zh"), ("粤语 yue", "yue"), ("法语 fr", "fr"), ("德语 de", "de"),
             ("西班牙语 es", "es"), ("俄语 ru", "ru")]
TGT_LANGS = [("中文 zh", "zh"), ("英语 en", "en"), ("日语 ja", "ja"), ("韩语 ko", "ko")]
MODES = [("只要译文（单语）", "mono-tgt"), ("原文 + 译文 + 双语", "bilingual"), ("只要原文", "mono-src")]
MONO = [("译文自己排版（推荐）", "sentence"), ("旧：条数同原文，按比例切", "split"),
        ("旧：条数同原文，短句每条重复整句", "repeat")]
PREC = [("按显存自动选", "auto"), ("全精度", "full"), ("8-bit", "int8"), ("4-bit", "nf4")]

# 高级参数：(分组, config 路径, 标签, 说明)。默认值取 config.yaml 当前的值。
ADVANCED = [
    ("断句与排版", "segmentation.max_chars.zh", "中日粤文每条最多字数", "规格要求 20；日文、粤语同样适用"),
    ("断句与排版", "segmentation.max_chars.default", "其他语种每条最多字符数", "英文等"),
    ("断句与排版", "segmentation.max_duration_sec", "每条最长显示秒数", "规格要求 7"),
    ("断句与排版", "segmentation.min_duration_sec", "每条最短显示秒数", "不足的向后面的空隙延长；0 关闭"),
    ("断句与排版", "segmentation.max_char_sec", "日语：每字最多算几秒发音", "超出部分视为停顿；0 关闭"),
    ("翻译单元", "translation.merge_gap_sec", "合并相邻句子的最大间隔（秒）", ""),
    ("翻译单元", "translation.merge_require_incomplete", "只合并“话没说完”的句子（日语）", ""),
    ("翻译单元", "translation.max_unit_chars", "一个翻译单元最多字数", ""),
    ("翻译单元", "translation.max_unit_segments", "一个翻译单元最多几个识别片段", ""),
    ("翻译单元", "translation.max_consecutive_errors", "连续出错几句就停止翻译这个文件", "0 = 不停止"),
    ("翻译单元", "translation.generation.repetition_penalty", "重复惩罚（repetition_penalty）",
     "Hy-MT2 / Qwen 有效；回退模型 NLLB 用自己的固定值"),
    ("语音识别（改了会重新转写）", "asr.beam_size", "beam_size", ""),
    ("语音识别（改了会重新转写）", "asr.condition_on_previous_text", "参考上文（容易重复幻听）", ""),
    ("语音识别（改了会重新转写）", "asr.word_timestamps", "词级时间戳", "关掉后断句只能按字数平均估计时间"),
    ("语音识别（改了会重新转写）", "asr.no_speech_threshold", "no_speech_threshold", "Whisper 判定静音的阈值"),
    ("语音识别（改了会重新转写）", "asr.compression_ratio_threshold", "compression_ratio_threshold", ""),
    ("语音识别（改了会重新转写）", "asr.log_prob_threshold", "log_prob_threshold", ""),
    ("语音识别（改了会重新转写）", "asr.hallucination_silence_threshold", "hallucination_silence_threshold",
     "关 VAD 时跳过疑似静音处的幻听（秒）"),
    ("语音识别（改了会重新转写）", "asr.drop_no_speech_prob", "丢弃：no_speech 概率高于", "段级质量过滤"),
    ("语音识别（改了会重新转写）", "asr.drop_avg_logprob", "丢弃：平均对数概率低于", "段级质量过滤"),
    ("语音识别（改了会重新转写）", "asr.drop_compression_ratio", "丢弃：压缩率高于", "段级质量过滤（高重复）"),
    ("语音识别（改了会重新转写）", "asr.vad_min_silence_ms", "VAD 最短静音（毫秒）", "只在打开 VAD 时起作用"),
]
_MIRROR = {"segmentation.max_chars.zh": ["segmentation.max_chars.ja", "segmentation.max_chars.yue"],
           "segmentation.max_chars.default": ["segmentation.max_chars.en"]}

CSS = """
.as-warn { color: #c62828; font-weight: 600; }
.as-key  { color: #9a5b00; }
.as-ok   { color: #2e7d32; font-weight: 600; }
.as-dim  { color: #757575; }
.dark .as-warn { color: #ff7961; }
.dark .as-key  { color: #ffca28; }
.dark .as-ok   { color: #81c784; }
.as-panel { line-height: 1.6; }
.as-panel ul { margin: 0.2em 0 0.6em 1.2em; padding: 0; }
.as-bar { height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin: 4px 0 8px; }
.as-bar > div { height: 100%; background: #1e88e5; }
"""


# ---------------------------------------------------------------------------
# 配置读写小工具
# ---------------------------------------------------------------------------
def _get(cfg: dict, path: str, default=None):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _set(cfg: dict, path: str, value) -> None:
    parts = path.split(".")
    node = cfg
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value


# ---------------------------------------------------------------------------
# 一次运行（后台线程）
# ---------------------------------------------------------------------------
class _Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.stage = "准备中…"
        self.frac = 0.0
        self.result = None
        self.error = None
        self.stop = False
        self.done = threading.Event()

    def add_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > LOG_LINES:
                del self.lines[: len(self.lines) - LOG_LINES]

    def sink(self, level: str, text: str) -> None:
        with self.lock:
            if level in ("warn", "error", "key"):
                self.events.append((level, text))
        prefix = {"warn": "[警告] ", "error": "[错误] "}.get(level, "")
        self.add_line(prefix + text)

    def progress(self, stage: str, done: int, total: int, desc: str = "") -> None:
        total = max(1, total)
        if stage == "翻译句子":
            self.stage = f"正在翻译 {desc}：{done}/{total} 句"
        elif stage == "识别":
            self.stage = f"正在识别 {desc}（第 {done + 1}/{total} 个文件；长文件要几分钟）"
        else:
            self.stage = f"正在处理 {desc}（第 {done + 1}/{total} 个文件）"
        self.frac = min(1.0, done / total)


class _LogHandler(logging.Handler):
    def __init__(self, job: _Job):
        super().__init__(logging.INFO)
        self.job = job
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.job.add_line(self.format(record))
        except Exception:
            pass


_CURRENT: dict = {"job": None}


def _worker(job: _Job, args, cfg) -> None:
    handler = _LogHandler(job)
    logger = logging.getLogger("subtitle")
    logger.addHandler(handler)
    console.add_sink(job.sink)
    try:
        job.result = run_pipeline(args, cfg, progress=job.progress, should_stop=lambda: job.stop)
    except Exception as e:                      # run_pipeline 自己会接住单个文件的错误，这里是意料之外的
        log.exception("运行出错")
        job.error = e
        job.sink("error", f"运行出错：{type(e).__name__}: {e}")
    finally:
        console.remove_sink(job.sink)
        logger.removeHandler(handler)
        job.done.set()


# ---------------------------------------------------------------------------
# 显示
# ---------------------------------------------------------------------------
def _esc(s) -> str:
    return html.escape(str(s))


def _events_html(events, limit: int = 30) -> str:
    items = [(lv, t) for lv, t in events if lv in ("warn", "error")]
    if not items:
        return ""
    more = f"<li class='as-dim'>……另有 {len(items) - limit} 条，见日志</li>" if len(items) > limit else ""
    return ("<div><b>需要注意</b><ul>" + "".join(f"<li class='as-warn'>{_esc(t)}</li>" for _, t in items[:limit])
            + more + "</ul></div>")


def _running_html(job: _Job) -> str:
    keys = [t for lv, t in job.events if lv == "key"][-6:]
    return ("<div class='as-panel'>"
            f"<div><b>{_esc(job.stage)}</b></div>"
            f"<div class='as-bar'><div style='width:{job.frac * 100:.0f}%'></div></div>"
            + "".join(f"<div class='as-key'>{_esc(t)}</div>" for t in keys)
            + _events_html(job.events) + "</div>")


def _final_html(job: _Job) -> str:
    res = job.result
    if res is None:
        return f"<div class='as-panel as-warn'>运行出错：{_esc(job.error)}</div>"
    out = ["<div class='as-panel'>"]
    if res.mt_used:
        used = res.mt_used + (f"（{res.mt_plan}）" if res.mt_plan else "")
        if res.mt_fallback:
            out.append(f"<div class='as-warn'>翻译模型：要求 {_esc(res.mt_requested)}，实际用的是 {_esc(used)}"
                       f"——{_esc(res.mt_fallback)}</div>")
        else:
            out.append(f"<div class='as-key'>翻译模型：{_esc(used)}</div>")
    if not res.reports:
        out.append("<div class='as-warn'>没有找到可处理的音频/视频文件</div>")
    for r in res.reports:
        if r.status == "ok":
            cls = "as-warn" if r.warnings else "as-ok"
            lang = r.language + ("（指定）" if r.lang_forced else
                                 (f"（把握 {r.lang_prob:.0%}）" if r.lang_prob is not None else ""))
            body = (f"原文 {r.src_cues} 条 → 译文 {r.mono_cues} 条" if r.units or r.mono_cues != r.src_cues
                    else f"{r.src_cues} 条")
            out.append(f"<div class='{cls}'>✔ {_esc(r.file.name)} · 语种 {_esc(lang)} · {body}"
                       f"{' · 用了转写缓存' if r.cached else ''}</div>")
            if r.warnings or r.notes:
                out.append("<ul>" + "".join(f"<li class='as-warn'>{_esc(w)}</li>" for w in r.warnings)
                           + "".join(f"<li class='as-key'>{_esc(w)}</li>" for w in r.notes) + "</ul>")
        elif r.status == "failed":
            out.append(f"<div class='as-warn'>✘ {_esc(r.file.name)}：{_esc(r.error)}</div>")
        else:
            out.append(f"<div class='as-dim'>· {_esc(r.file.name)}：没有处理（已停止）</div>")
    dirs = sorted({str(Path(p).parent) for r in res.reports for p in r.outputs})
    if dirs:
        out.append("<div class='as-key' style='margin-top:6px'>字幕已自动保存到："
                   + "；".join(_esc(d) for d in dirs) + "</div>")
    n_ok = sum(r.status == "ok" for r in res.reports)
    tail = f"{n_ok}/{len(res.reports)} 个文件完成，用时 {_fmt_dur(res.seconds)}"
    out.append(f"<div class='{'as-ok' if res.ok else 'as-warn'}' style='margin-top:6px'>{tail}</div></div>")
    return "".join(out)


def _vtt_rows(path) -> list[list]:
    import verify_vtt
    rows = []
    for i, (s, e, t) in enumerate(verify_vtt._read_cues(Path(path)), 1):
        rows.append([i, _ts(s), _ts(e), t.replace("\n", " / ")])
    return rows


def _ts(ms: int) -> str:
    s, ms = divmod(int(ms), 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{ms:03d}"


def _vtt_check_html(path: Path) -> str:
    import verify_vtt
    errors, n = verify_vtt.validate_file(path)
    warns = verify_vtt.quality_warnings(path)
    rep = ""
    if path.name.endswith(".vtt") and "_cn." in path.name:
        src = None
        stem, rest = path.name.rsplit("_cn.", 1)
        for cand in path.parent.iterdir():              # 同目录里找原文文件：<名字>_<语种>.<扩展>.vtt
            if cand.name.startswith(stem + "_") and cand.name.endswith("." + rest):
                code = cand.name[len(stem) + 1:].split(".", 1)[0]
                if code != "cn" and "-" not in code:
                    cues = verify_vtt._read_cues(cand)
                    if not verify_vtt.check_alignment([(s, e) for s, e, _ in cues],
                                                      [(s, e) for s, e, _ in verify_vtt._read_cues(path)]):
                        src = cues
                        break
        rep = verify_vtt._repeat_warning(*verify_vtt.adjacent_repeats(verify_vtt._read_cues(path), src))
    out = [f"<div class='as-panel'><div class='{'as-warn' if errors else 'as-ok'}'>"
           f"{'格式有错' if errors else '格式正确'}：{_esc(path.name)}（{n} 条）</div>"]
    out += [f"<div class='as-warn'>✘ {_esc(e)}</div>" for e in errors]
    out += [f"<div class='as-key'>! {_esc(w)}</div>" for w in warns + ([rep] if rep else [])]
    return "".join(out) + "</div>"


# ---------------------------------------------------------------------------
# “开始生成”按钮（放在模块级，方便不开浏览器直接测试）
# ---------------------------------------------------------------------------
N_BASIC = 17          # 基本设置控件的个数（顺序见 build_ui 的 basic_inputs）


def run_job_impl(base_cfg: dict, vals):
    """生成器：先不断返回运行中的状态，最后返回结果。每次返回 9 项，对应
    [状态, 日志, 字幕文件, 选择框, 播放提示, 表格, 音频播放器, 视频播放器, 字幕→原文件 对照表]。"""
    import gradio as gr
    (f_list, path_text, rec, outd, mode_v, src_v, tgt_v, asr_v, mt_v, prec_v, mono_v, gran_v,
     vad_v, vad_thr_v, sep_v, swap_v, gloss_v) = vals[:N_BASIC]
    adv_vals = vals[N_BASIC:]
    skip8 = (gr.skip(),) * 8
    if _CURRENT["job"] is not None and not _CURRENT["job"].done.is_set():
        yield ("<div class='as-warn'>上一次运行还没结束</div>",) + skip8
        return
    uploads = [str(p) for p in (f_list or [])]
    typed = [p.strip().strip('"').strip("'") for p in (path_text or "").splitlines() if p.strip()]
    missing = [p for p in typed if not Path(p).exists()]
    if missing:
        yield (f"<div class='as-warn'>路径不存在：{_esc('；'.join(missing))}</div>",) + skip8
        return
    inputs = uploads + typed
    if not inputs:
        yield ("<div class='as-warn'>请先拖入文件，或者填写本机路径</div>",) + skip8
        return
    argv = ["run", *inputs, "--mode", mode_v, "--src-lang", (src_v or "auto").strip(),
            "--tgt-lang", (tgt_v or "zh").strip(), "--asr", asr_v, "--mt", mt_v, "--mt-precision", prec_v,
            "--mono-display", mono_v, "--mt-granularity", gran_v]
    argv += ["--vad-threshold", str(vad_thr_v)] if vad_v else ["--no-vad"]
    if rec:
        argv.append("--recursive")
    if sep_v:
        argv.append("--separate-vocals")
    if swap_v:
        argv.append("--swap-lines")
    if gloss_v:
        argv += ["--glossary", str(gloss_v)]
    if outd and outd.strip():
        argv += ["--output-dir", outd.strip().strip('"')]
    elif uploads:
        argv += ["--output-dir", str(UPLOAD_OUT)]
    cfg = copy.deepcopy(base_cfg)
    try:
        args = build_parser(cfg).parse_args(argv)
    except SystemExit:
        yield ("<div class='as-warn'>参数不对，请检查设置</div>",) + skip8
        return
    for (_g, path, _l, _i), v in zip(ADVANCED, adv_vals):
        if v is None:
            continue
        _set(cfg, path, v)
        for p2 in _MIRROR.get(path, []):
            _set(cfg, p2, v)

    job = _Job()
    _CURRENT["job"] = job
    console.reset()
    threading.Thread(target=_worker, args=(job, args, cfg), daemon=True).start()
    while not job.done.wait(0.8):
        yield (_running_html(job), "\n".join(job.lines)) + (gr.skip(),) * 7
    # 结束：列出生成的字幕
    outs, mmap = [], {}
    if job.result is not None:
        for r in job.result.reports:
            for p in r.outputs:
                outs.append(str(p))
                mmap[str(p)] = str(r.file)
    choices = [(Path(p).name, p) for p in outs]
    first = next((p for p in outs if "_cn." in Path(p).name), outs[0] if outs else None)
    rows = _vtt_rows(first) if first else []
    a_upd, v_upd, info = _player_updates(first, mmap)
    yield (_final_html(job), "\n".join(job.lines), _downloadable(outs) or None,
           gr.update(choices=choices, value=first), info, rows, a_upd, v_upd, mmap)


def _display_vtt(vtt_path) -> str:
    """网页播放器用的字幕副本：每条一直显示到下一条开始，不在结束时间一到就消失。
    最后一条、本来就和下一条首尾相接的不变；有重叠时不缩短。写在 .cache/gradio/display，
    只给播放器用——下载的字幕文件和表格里的时间都不变。读不出字幕时返回原文件。"""
    import verify_vtt
    src = Path(vtt_path)
    cues = verify_vtt._read_cues(src)
    if not cues:
        return str(src)
    blocks = []
    for i, (s, e, t) in enumerate(cues):
        if i + 1 < len(cues):
            e = max(e, cues[i + 1][0])
        blocks.append(f"{i + 1}\n{_ts(s)} --> {_ts(e)}\n{t}\n")
    d = GRADIO_TMP / "display"
    d.mkdir(parents=True, exist_ok=True)
    dst = d / (hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:16] + ".vtt")
    dst.write_text("WEBVTT\n\n" + "\n".join(blocks), encoding="utf-8", newline="\n")
    return str(dst)


def _player_updates(vtt_path, mmap):
    """选中的字幕对应的原文件放进播放器（字幕用 _display_vtt 的显示副本）。太大的文件不放。"""
    import gradio as gr
    from . import media
    hide = gr.update(visible=False, value=None)
    src = mmap.get(vtt_path) if vtt_path else None
    if not src or not Path(src).exists():
        return hide, hide, ""
    p = Path(src)
    size_mb = p.stat().st_size / 2 ** 20
    if size_mb > MAX_PLAYER_MB:
        return hide, hide, (f"<div class='as-dim'>原文件 {size_mb:.0f}MB，太大，不在网页里播放；"
                            f"请用播放器打开原文件和字幕。</div>")
    try:
        play = _servable(p)
        subs = _display_vtt(vtt_path)
    except OSError as e:
        console.warn(f"网页播放器准备不了（{e}），请用播放器打开原文件和字幕")
        return hide, hide, f"<div class='as-warn'>网页播放器准备不了：{_esc(str(e))}</div>"
    if p.suffix.lower() in media.VIDEO_EXTS:
        return hide, gr.update(visible=True, value=play, subtitles=subs), ""
    return gr.update(visible=True, value=play, subtitles=subs), hide, ""


def _servable(path) -> str:
    """Gradio 只肯把当前目录（项目目录）、系统临时目录和它自己缓存里的文件交给网页，别处的报 InvalidPathError。
    “填本机路径”处理项目外的文件时，原文件和写在它旁边的字幕都在别处：先复制进本次的缓存目录
    再交给网页（Gradio 对允许的文件本来也会复制一份进缓存）。复制品跟其他缓存一起清理。"""
    p = Path(path).resolve()
    for root in (GRADIO_TMP, Path(os.getcwd()), Path(tempfile.gettempdir())):
        r = Path(root).resolve()
        if p == r or r in p.parents:
            return str(p)
    dst = GRADIO_TMP / "served" / hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16] / p.name
    st = p.stat()
    if not (dst.exists() and dst.stat().st_size == st.st_size and dst.stat().st_mtime >= st.st_mtime):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    return str(dst)


def _downloadable(paths: list[str]) -> list[str]:
    """下载列表用的路径（见 _servable）。放不进去的只提示，文件本身已经写好了。"""
    out = []
    for p in paths:
        try:
            out.append(_servable(p))
        except OSError as e:
            console.warn(f"{Path(p).name} 放不进网页的下载列表（{e}）；文件已经保存在 {p}")
    return out


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k32.OpenProcess(0x1000, False, pid)                  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() == 5                  # 拒绝访问 = 进程在（别的用户的）；其他错误 = 不存在
    try:
        code = wintypes.DWORD()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _clean_old_sessions(root: Path = GRADIO_ROOT, keep: Path = GRADIO_TMP,
                        legacy_age: float = 86400) -> tuple[int, int]:
    """删掉已经退出的界面进程留下的 Gradio 缓存，返回 (条目数, 字节数)。
    - s<进程号> 目录：那个进程已经不在了就整个删。按 Ctrl+C 退出时 Gradio 自己会删本次的文件，
      直接关窗口、崩溃时它来不及，留下的就是这些。
    - 其他条目（不在 s<进程号> 目录里的）：超过 legacy_age 秒才删（可能是还开着的其他界面的）。"""
    if not root.exists():
        return 0, 0
    n = size = 0
    now = time.time()
    keep = keep.resolve()
    for item in root.iterdir():
        try:
            if item.resolve() == keep:
                continue
            m = re.fullmatch(r"s(\d+)", item.name)
            if m and item.is_dir():
                if _pid_alive(int(m.group(1))):
                    continue
            elif now - item.stat().st_mtime <= legacy_age:
                continue
            files = list(item.rglob("*")) if item.is_dir() else [item]
            b = sum(f.stat().st_size for f in files if f.is_file())
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            n, size = n + 1, size + b
        except OSError as e:
            log.debug("清理 %s 失败：%s", item, e)
    return n, size


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
def _theme(gr):
    sys_fonts = ["system-ui", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]
    return gr.themes.Default(font=sys_fonts, font_mono=["Consolas", "Menlo", "monospace"])


def build_ui():
    import gradio as gr
    from .asr.factory import ASR_CHOICES
    from .mt.factory import MT_CHOICES
    from . import media

    base_cfg = load_config()
    d = base_cfg.get("defaults", {})
    tcfg = base_cfg.get("translation", {})
    acfg = base_cfg.get("asr", {})
    exts = sorted(media.MEDIA_EXTS)

    with gr.Blocks(title="Autoscripter 字幕生成", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("## Autoscripter 字幕生成\n本地离线：音频/视频 → 语音识别 → 翻译 → .vtt 字幕。"
                    "和命令行 `python main.py run` 是同一套流程，设置的含义见 README。")
        with gr.Tab("生成字幕"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 输入")
                    files = gr.File(label="把音频/视频拖到这里（可以多个）", file_count="multiple",
                                    file_types=exts, type="filepath")
                    paths = gr.Textbox(label="或者填本机路径（文件或文件夹；多个就每行一个）", lines=2,
                                       placeholder=r"例如 D:\media\作品A  或  D:\media\01.mp3")
                    recursive = gr.Checkbox(label="文件夹里的子文件夹也处理", value=False)
                    out_dir = gr.Textbox(label="字幕输出目录（可留空）",
                                         placeholder="留空：填路径的文件写在原文件旁边；拖进来的文件写到 项目\\output")
                with gr.Column(scale=1):
                    gr.Markdown("### 2. 基本设置")
                    mode = gr.Radio(MODES, value=d.get("mode", "bilingual"), label="输出哪些字幕")
                    with gr.Row():
                        src_lang = gr.Dropdown(SRC_LANGS, value=d.get("src_lang", "auto"), allow_custom_value=True,
                                               label="源语种", info="知道是什么语言就直接选，比自动识别可靠")
                        tgt_lang = gr.Dropdown(TGT_LANGS, value=d.get("tgt_lang", "zh"), allow_custom_value=True,
                                               label="目标语种")
                    with gr.Row():
                        asr = gr.Dropdown(ASR_CHOICES, value=d.get("asr", "whisper"), label="语音识别模型")
                        mt = gr.Dropdown(MT_CHOICES, value=d.get("mt", "hymt2"), label="翻译模型")
                        precision = gr.Dropdown(PREC, value=tcfg.get("precision", "auto"), label="翻译模型精度")
                    mono = gr.Radio(MONO, value=tcfg.get("mono_display", "sentence"), label="单语译文的排版")
                    granularity = gr.Radio([("整句翻译（推荐）", "sentence"), ("旧：逐条翻译", "cue")],
                                           value=tcfg.get("granularity", "sentence"), label="翻译粒度")
                    with gr.Row():
                        vad = gr.Checkbox(label="VAD 预过滤（音量很小的人声可能被整段丢掉）", value=bool(acfg.get("vad_filter", False)))
                        vad_thr = gr.Slider(0.05, 0.9, value=float(acfg.get("vad_threshold", 0.2)), step=0.05,
                                            label="VAD 阈值（越低越灵敏）")
                    with gr.Row():
                        sep = gr.Checkbox(label="先分离人声（只适合背景音乐很响的音频；小音量人声会被分坏）", value=False)
                        swap = gr.Checkbox(label="双语字幕上下行互换", value=False)
                    glossary = gr.File(label="术语表（可选：每行 原文=译文，或 JSON）", file_count="single",
                                       file_types=[".txt", ".json", ".tsv"], type="filepath")
            adv_inputs = []
            with gr.Accordion("高级参数（config.yaml 里的超参数；只对这一次运行生效，不会改 config.yaml）", open=False):
                groups: dict[str, list] = {}
                for g, path, label, info in ADVANCED:
                    groups.setdefault(g, []).append((path, label, info))
                for g, items in groups.items():
                    gr.Markdown(f"**{g}**")
                    with gr.Row():
                        for path, label, info in items:
                            v = _get(base_cfg, path)
                            if isinstance(v, bool):
                                c = gr.Checkbox(value=v, label=label, info=info or None)
                            elif isinstance(v, int):
                                c = gr.Number(value=v, label=label, info=info or None, precision=0, min_width=160)
                            else:
                                c = gr.Number(value=v, label=label, info=info or None, min_width=160)
                            adv_inputs.append(c)
            with gr.Row():
                start = gr.Button("开始生成", variant="primary", scale=3)
                stop = gr.Button("停止", variant="stop", scale=1)
            gr.Markdown("### 3. 运行情况")
            status = gr.HTML("<div class='as-dim'>还没开始</div>")
            logbox = gr.Textbox(label="运行日志", lines=12, max_lines=12, autoscroll=True, interactive=False)
            gr.Markdown("### 4. 生成的字幕")
            out_files = gr.File(label="字幕文件（已经自动保存，位置见上面的汇总；点文件名可以另存一份）",
                                file_count="multiple", interactive=False)
            pick = gr.Dropdown(choices=[], label="选择要查看的字幕", interactive=True)
            pick_info = gr.HTML()
            table = gr.Dataframe(headers=["#", "开始", "结束", "字幕"], datatype=["number", "str", "str", "str"],
                                 interactive=False, wrap=True, max_height=480, show_search="search",
                                 column_widths=["6%", "14%", "14%", "66%"])
            audio = gr.Audio(label="边听边看字幕（每条显示到下一条开始）", type="filepath", interactive=False,
                             visible=False)
            video = gr.Video(label="边看视频边看字幕（每条显示到下一条开始）", interactive=False, visible=False)
            media_map = gr.State({})

        with gr.Tab("查看已有字幕"):
            gr.Markdown("打开任意 .vtt：表格查看 + 格式校验 + 质量警告（同 `python verify_vtt.py`）。"
                        "中文单语文件的“相邻两条相同”会按同目录里的原文文件分成“说话人重复”和“可疑”两类。")
            with gr.Row():
                v_file = gr.File(label="拖入 .vtt", file_types=[".vtt"], type="filepath", scale=1)
                v_path = gr.Textbox(label="或者填 .vtt 的路径", scale=2)
            v_btn = gr.Button("打开")
            v_info = gr.HTML()
            v_table = gr.Dataframe(headers=["#", "开始", "结束", "字幕"], datatype=["number", "str", "str", "str"],
                                   interactive=False, wrap=True, max_height=600, show_search="search",
                                   column_widths=["6%", "14%", "14%", "66%"])

        basic_inputs = [files, paths, recursive, out_dir, mode, src_lang, tgt_lang, asr, mt, precision, mono,
                        granularity, vad, vad_thr, sep, swap, glossary]

        def run_job(*vals):
            yield from run_job_impl(base_cfg, vals)

        def on_pick(vtt_path, mmap):
            if not vtt_path:
                return [], gr.skip(), gr.skip(), ""
            a_upd, v_upd, info = _player_updates(vtt_path, mmap or {})
            return _vtt_rows(vtt_path), a_upd, v_upd, info

        def on_stop():
            job = _CURRENT["job"]
            if job is None or job.done.is_set():
                return "<div class='as-dim'>现在没有在运行</div>"
            job.stop = True
            job.sink("warn", "收到停止请求：当前这一句翻译完（或当前文件识别完）后停止")
            return _running_html(job)

        def open_vtt(f, p):
            target = f or (p or "").strip().strip('"')
            if not target:
                return "<div class='as-warn'>请先拖入 .vtt 或填写路径</div>", []
            path = Path(target)
            if not path.exists():
                return f"<div class='as-warn'>文件不存在：{_esc(target)}</div>", []
            return _vtt_check_html(path), _vtt_rows(path)

        start.click(run_job, inputs=basic_inputs + adv_inputs,
                    outputs=[status, logbox, out_files, pick, pick_info, table, audio, video, media_map],
                    concurrency_limit=1, api_visibility="private", show_progress="hidden")
        stop.click(on_stop, outputs=[status], queue=False, api_visibility="private")
        pick.change(on_pick, inputs=[pick, media_map], outputs=[table, audio, video, pick_info],
                    api_visibility="private", show_progress="hidden")
        v_btn.click(open_vtt, inputs=[v_file, v_path], outputs=[v_info, v_table], api_visibility="private")
        v_file.change(open_vtt, inputs=[v_file, v_path], outputs=[v_info, v_table], api_visibility="private")
    return demo


def launch(port: int = 7860, open_browser: bool = True) -> int:
    import gradio as gr
    n, size = _clean_old_sessions()
    if n:
        console.info(f"清掉了以前的界面没来得及删的缓存：{n} 项，{size / 2 ** 20:.1f}MB（.cache/gradio）")
    GRADIO_TMP.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    console.info(f"图形界面：http://127.0.0.1:{port}  （只在本机可访问；关闭这个窗口或按 Ctrl+C 退出）")
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, share=False,
                theme=_theme(gr), css=CSS, footer_links=[], show_error=True, quiet=True)
    return 0

"""命令行入口与整体编排（图形界面 subtitle/gui.py 也调用这里的 run_pipeline，走同一条代码路径）。"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import console

log = logging.getLogger("subtitle")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "model")
# 语种识别的把握低于这个值时红字提示
LOW_LANG_PROB = 0.5


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (PROJECT_ROOT / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    md = cfg.get("model_dir")
    if md and not Path(md).is_absolute():          # 相对路径按项目目录算，与当前所在目录无关
        cfg["model_dir"] = str(PROJECT_ROOT / md)
    return cfg


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # 降低第三方库噪音
    for noisy in ("urllib3", "huggingface_hub", "filelock", "httpx", "httpcore", "gradio", "uvicorn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# download 子命令
# ---------------------------------------------------------------------------
def cmd_download(args, cfg) -> int:
    from . import models
    model_dir = Path(args.model_dir)
    models.configure_hf_env(model_dir)

    to_get: list[str] = []
    if args.all_recommended:
        to_get += models.RECOMMENDED_ASR + models.RECOMMENDED_MT
    if args.asr:
        to_get.append(args.asr)
    if args.mt:
        to_get.append(args.mt)
    if not to_get:
        print("未指定要下载的模型。用 --asr / --mt / --all-recommended。")
        return 2

    # 去重保序
    seen = set()
    to_get = [x for x in to_get if not (x in seen or seen.add(x))]
    print(f"将下载: {to_get} -> {model_dir}")
    models.ensure_models(to_get, model_dir, force=args.force)
    print("全部下载完成。")
    return 0


# ---------------------------------------------------------------------------
# run 子命令
# ---------------------------------------------------------------------------
class Cancelled(Exception):
    """图形界面里点了“停止”。"""


@dataclass
class FileReport:
    """一个文件的处理结果（运行结束的汇总、图形界面都用它）。"""
    file: Path
    status: str = "pending"            # ok / failed / pending（没处理到：被停止或前面出错）
    error: str = ""
    asr_backend: str = ""
    cached: bool = False
    language: str = ""
    lang_prob: float | None = None
    lang_forced: bool | None = None
    segments: int = 0
    dropped: int | None = None
    duration: float = 0.0
    src_cues: int = 0
    mono_cues: int = 0
    units: int = 0
    unit_stats: dict = field(default_factory=dict)
    layout_stats: dict = field(default_factory=dict)
    skipped: str = ""
    outputs: list = field(default_factory=list)
    warnings: list = field(default_factory=list)      # 红字：内容可能有错或缺失
    notes: list = field(default_factory=list)         # 黄字：格式上的小偏差，看的时候基本不受影响


@dataclass
class RunResult:
    reports: list
    mt_requested: str = ""
    mt_used: str = ""
    mt_plan: str = ""
    mt_fallback: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.reports) and all(r.status == "ok" for r in self.reports)


def _decide_backend_for_lang(lang: str, cfg: dict, overrides: dict) -> str:
    """lang_adapt 开启时，按适配表 + 命令行覆盖选择 ASR 后端。"""
    base = (lang or "").lower().split("-")[0]
    amap = cfg.get("lang_adapt_map", {})
    # 命令行覆盖
    if base == "ja" and overrides.get("ja_asr"):
        return overrides["ja_asr"]
    if base == "zh" and overrides.get("zh_asr"):
        return overrides["zh_asr"]
    if base == "ko" and overrides.get("ko_asr"):
        return overrides["ko_asr"]
    return amap.get(base, amap.get("fallback", "whisper"))


def _plan_files(files, args, cfg):
    """返回 [(file, backend, src_lang_or_None)]。"""
    plan = []
    overrides = {"ja_asr": args.ja_asr, "zh_asr": args.zh_asr, "ko_asr": args.ko_asr}

    if not args.lang_adapt:
        for f in files:
            src = None if args.src_lang == "auto" else args.src_lang
            plan.append((f, args.asr, src))
        return plan

    # 开启适配：逐文件检测/选择
    from .asr.detect import LanguageDetector
    from . import media
    detector = None
    try:
        for f in files:
            if args.src_lang != "auto":
                lang = args.src_lang
                log.info("[%s] 用户指定源语言 %s，跳过检测", f.name, lang)
            else:
                wav = None
                try:
                    if detector is None:
                        detector = LanguageDetector(Path(args.model_dir), cfg)
                    wav = media.decode_to_wav(f)
                    lang, prob = detector.detect(wav)
                except Exception as e:
                    console.warn(f"{f.name}: 语种检测失败（{type(e).__name__}: {e}），改用 {args.asr} 并让它自己识别语种")
                    plan.append((f, args.asr, None))
                    continue
                finally:
                    media.remove_intermediate(wav)
                log.info("[%s] 检测到语种 %s (p=%.2f)", f.name, lang, prob)
                if prob < LOW_LANG_PROB:
                    console.warn(f"{f.name}: 语种检测把握不大（{lang}，把握 {prob:.0%}），"
                                 f"开头可能是静音/音效；结果不对时请用 --src-lang 指定语种")
            backend = _decide_backend_for_lang(lang, cfg, overrides)
            console.key(f"[适配] {f.name}: 语种 {lang}，选用识别后端 {backend}")
            plan.append((f, backend, lang if args.src_lang == "auto" else args.src_lang))
    finally:
        if detector is not None:
            detector.unload()
    return plan


def apply_overrides(args, cfg) -> None:
    """把命令行参数写进 cfg（命令行优先于 config.yaml）。"""
    acfg = cfg.setdefault("asr", {})
    if getattr(args, "vad", False) or getattr(args, "vad_threshold", None) is not None:
        acfg["vad_filter"] = True                     # 给了 --vad-threshold 也视为要开 VAD
    if getattr(args, "no_vad", False):
        acfg["vad_filter"] = False
    if getattr(args, "vad_threshold", None) is not None:
        acfg["vad_threshold"] = args.vad_threshold

    tcfg = cfg.setdefault("translation", {})
    if getattr(args, "mt_granularity", None):
        tcfg["granularity"] = args.mt_granularity
    if getattr(args, "mono_display", None):
        tcfg["mono_display"] = args.mono_display
    if getattr(args, "mt_precision", None):
        tcfg["precision"] = args.mt_precision

    # 术语表：翻译时只注入当前句子里出现的词条
    if getattr(args, "glossary", None):
        pairs = _load_glossary(Path(args.glossary))
        tcfg["glossary_pairs"] = pairs
        tcfg["glossary_text"] = "; ".join(f"{a} -> {b}" for a, b in pairs) if pairs else "(none)"


def _fmt_dur(sec: float) -> str:
    sec = int(round(sec or 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def _note_transcript(rep: FileReport, tr, src, args) -> None:
    """记下转写结果的关键信息，并醒目地显示（语种不可靠、和目标语种相同、转写为空时红字）。"""
    rep.language = tr.language
    rep.lang_prob = tr.language_probability
    rep.lang_forced = bool(src) if tr.language_forced is None else tr.language_forced
    rep.segments = len(tr.segments)
    rep.dropped = tr.dropped
    rep.duration = tr.duration
    if rep.lang_forced:
        lang_txt = f"语种 {tr.language}（指定）"
    elif rep.lang_prob is not None:
        lang_txt = f"识别语种 {tr.language}（把握 {rep.lang_prob:.0%}）"
    else:
        lang_txt = f"识别语种 {tr.language}（旧缓存，没有记录把握）"
    dropped = f"，质量过滤丢掉 {tr.dropped} 段" if tr.dropped else ""
    console.key(f"[{rep.file.name}] {lang_txt} · {len(tr.segments)} 段{dropped} · 时长 {_fmt_dur(tr.duration)}"
                f"{' · 用的是转写缓存' if rep.cached else ''}")
    if not rep.lang_forced and rep.lang_prob is not None and rep.lang_prob < LOW_LANG_PROB:
        w = (f"语种识别把握不大（{tr.language}，{rep.lang_prob:.0%}），整段转写可能是错的语种；"
             f"请用 --src-lang 指定语种（如 --src-lang ja）")
        rep.warnings.append(w)
        console.warn(f"{rep.file.name}: {w}")
    need_translation = args.mode in ("bilingual", "mono-tgt")
    if need_translation and _base(_base_lang(tr.language, args)) == _base(args.tgt_lang) and tr.segments:
        w = (f"源语种（{_base_lang(tr.language, args)}）和目标语种（{args.tgt_lang}）相同，这个文件不会被翻译"
             f"{'；如果是识别错了，请用 --src-lang 指定' if not rep.lang_forced else ''}")
        rep.warnings.append(w)
        console.warn(f"{rep.file.name}: {w}")


def _note_result(rep: FileReport, res, args) -> None:
    """记下断句/翻译的统计。内容可能有错或缺失的红字提示，格式上的小偏差黄字提示。"""
    rep.src_cues = len(res.src_cues)
    rep.mono_cues = len(res.tgt_mono)
    rep.units = len(res.units or [])
    rep.unit_stats = dict(res.unit_stats or {})
    rep.layout_stats = dict(res.layout_stats or {})
    rep.skipped = res.skipped if args.mode in ("bilingual", "mono-tgt") else ""
    us, ls = rep.unit_stats, rep.layout_stats
    probs, notes = [], []
    if us.get("source"):
        probs.append(f"{us['source']} 句没翻译出来（保留了原文）")
    if us.get("error"):
        probs.append(f"翻译时出错 {us['error']} 次")
    if ls.get("mismatch") and not ls.get("repeat"):      # 旧的 repeat 方式本来就会重复，不算
        probs.append(f"{ls['mismatch']} 句的单语字幕拼起来不等于译文（重复显示或丢字，是程序的 bug，请反馈）")
    if us.get("cue-fallback"):
        notes.append(f"{us['cue-fallback']} 句整句翻译不合格，改成了逐条翻译")
    if ls.get("hold"):
        notes.append(f"{ls['hold']} 条是“延续显示”（译文太短切不开，后一条继续显示同一段文字）")
    if ls.get("over"):
        notes.append(f"{ls['over']} 条单语字幕超出每条字数上限（一句话只对应一条原文、或译文比原文条容得下的长）")
    if ls.get("bi_repeat") and args.mode == "bilingual":
        notes.append(f"双语文件里 {ls['bi_repeat']} 条重复显示整句（译文太短，切不成原文那么多条）")
    for p in probs:
        rep.warnings.append(p)
        console.warn(f"{rep.file.name}: {p}")
    for p in notes:
        rep.notes.append(p)
        console.key(f"{rep.file.name}: {p}")


def _write_outputs(f: Path, tr, res, args, cfg) -> list[Path]:
    """写出一个文件的字幕，返回写出的路径。"""
    from . import vtt
    out_dir = Path(args.output_dir) if args.output_dir else f.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    src_lang = _base_lang(tr.language, args)
    src_code = vtt.display_code(src_lang, cfg)
    tgt_code = vtt.display_code(args.tgt_lang, cfg)

    written = []
    if args.mode in ("bilingual", "mono-src"):
        p = out_dir / vtt.output_name(f, src_code)
        vtt.write_mono(res.src_cues, p)
        written.append(p)
    if args.mode in ("bilingual", "mono-tgt"):
        p = out_dir / vtt.output_name(f, tgt_code)
        vtt.write_mono(res.tgt_mono, p)
        written.append(p)
    if args.mode == "bilingual":
        combo = f"{src_code}-{tgt_code}"
        p = out_dir / vtt.output_name(f, combo)
        vtt.write_bilingual(res.src_cues, res.tgt_bilingual, p, swap=args.swap_lines)
        written.append(p)
    for p in written:
        console.info(f"  写出: {p}")
    return written


def _fail(rep: FileReport, msg: str) -> None:
    rep.status = "failed"
    rep.error = msg
    console.error(f"{rep.file.name}: {msg}")


def _list_inputs(args) -> list[Path]:
    from . import media
    roots = args.input if isinstance(args.input, (list, tuple)) else [args.input]
    files, seen = [], set()
    for r in roots:
        if not Path(r).exists():
            console.warn(f"路径不存在，跳过：{r}")
            continue
        if Path(r).is_file() and not media.is_media_file(Path(r)):
            console.warn(f"不是支持的音频/视频格式，跳过：{r}")
            continue
        for f in media.iter_media_files(Path(r), recursive=args.recursive):
            k = str(f.resolve()).lower()
            if k not in seen:
                seen.add(k)
                files.append(f)
    return files


def run_pipeline(args, cfg, progress=None, should_stop=None) -> RunResult:
    """处理全部输入文件。一个文件出错不影响其他文件；每个文件翻译完立刻写出。

    progress(阶段, 已完成, 总数, 说明) 和 should_stop() 可选（图形界面用）。"""
    from . import models, media
    from .asr.factory import build_asr
    from .cache import TranscriptCache, asr_param_tag
    from .vram import release_cuda

    t_start = time.time()
    model_dir = Path(args.model_dir)
    models.configure_hf_env(model_dir)
    apply_overrides(args, cfg)

    def stop_check():
        if should_stop is not None and should_stop():
            raise Cancelled()

    def prog(stage, done, total, desc=""):
        if progress is not None:
            progress(stage, done, total, desc)
        stop_check()

    files = _list_inputs(args)
    result = RunResult(reports=[FileReport(file=f) for f in files])
    if not files:
        console.error(f"未找到可处理的媒体文件: {args.input}")
        return result
    reports = {r.file: r for r in result.reports}
    need_translation = args.mode in ("bilingual", "mono-tgt")
    acfg = cfg.get("asr", {})
    src_txt = (f"{args.src_lang}（指定）" if args.src_lang != "auto"
               else "自动识别（建议用 --src-lang 指定，如 --src-lang ja）")
    console.key(f"待处理 {len(files)} 个文件 · 输出 {args.mode} · 源语种 {src_txt} · 目标语种 {args.tgt_lang}")
    console.key(f"识别 {args.asr}{'（按语种适配）' if args.lang_adapt else ''} · VAD {'开' if acfg.get('vad_filter') else '关'}"
                f" · 人声分离 {'开' if getattr(args, 'separate_vocals', False) else '关'}"
                + (f" · 翻译 {args.mt}" if need_translation else ""))

    cache = TranscriptCache(PROJECT_ROOT / ".cache")
    try:
        # 1) 规划每个文件的后端/语言；按后端分组，减少模型装卸
        plan = _plan_files(files, args, cfg)
        groups: dict[str, list] = defaultdict(list)
        for (f, backend, src) in plan:
            groups[backend].append((f, src))

        # 缓存标签：VAD / 人声分离设置 + 与默认值不同的 ASR 参数（改变这些会重新转写，而不是复用旧结果）
        vad_tag = f"vad{acfg.get('vad_threshold', 0.2)}" if acfg.get("vad_filter", False) else "novad"
        if getattr(args, "separate_vocals", False):
            vad_tag += "_voc"

        transcripts: dict[Path, object] = {}
        done = 0
        # ===== ASR 阶段：逐后端加载、转写全部、卸载 =====
        for backend_name, items in groups.items():
            ptag = asr_param_tag(acfg, backend_name)
            cache_tag = vad_tag + (f"_{ptag}" if ptag else "")
            if ptag:
                console.key(f"识别参数和默认值不同，转写缓存标签 {cache_tag}（会重新转写，不复用默认参数的结果）")
            log.info("=== ASR 后端 %s：%d 个文件 ===", backend_name, len(items))
            backend = None
            try:
                for (f, src) in items:
                    rep = reports[f]
                    rep.asr_backend = backend_name
                    prog("识别", done, len(files), f.name)
                    try:
                        tr = cache.load(f, backend_name, src, cache_tag)
                        if tr is not None:
                            rep.cached = True
                        else:
                            temp = []
                            try:
                                wav = media.decode_to_wav(f)
                                temp.append(wav)
                                if getattr(args, "separate_vocals", False):
                                    vocals = media.separate_vocals(wav, model_dir)
                                    wav = media.decode_to_wav(vocals)  # 转 16k 单声道人声
                                    temp += [vocals, wav]
                                if backend is None:
                                    backend = build_asr(backend_name, model_dir, cfg)
                                    backend.load()
                                tr = _with_retry(lambda: backend.transcribe(wav, src), what=f"转写 {f.name}")
                            finally:
                                media.remove_intermediate(*temp)
                            if tr.segments:
                                cache.save(f, tr, src, cache_tag)
                        _note_transcript(rep, tr, src, args)
                        if not tr.segments:
                            # 不缓存空结果：否则以后每次都会直接读到“0 条”（曾因语种被误判为 nn 而发生）
                            _fail(rep, f"转写结果为空（语种={tr.language}），没有写入缓存。若语种判断有误，请用 --src-lang 指定；"
                                       f"若开了 --vad，请去掉（音量很小的人声会被 VAD 整段丢掉）")
                        else:
                            transcripts[f] = tr
                    except Cancelled:
                        raise
                    except Exception as e:
                        log.debug("ASR 失败", exc_info=True)
                        _fail(rep, f"语音识别失败（{type(e).__name__}: {e}）")
                    done += 1
            finally:
                if backend is not None:
                    backend.unload()
                release_cuda()

        # 2) 断句 + 翻译（ASR 已全部卸载，再加载 MT），每个文件完成后立刻写出。见 pipeline.segment_and_translate
        from .pipeline import segment_and_translate
        todo = [f for f in files if f in transcripts]
        # 识别出的语种就是目标语种的文件不需要翻译（已红字提示）；全是这种文件时不加载翻译模型
        to_translate = [f for f in todo if _base(_base_lang(transcripts[f].language, args)) != _base(args.tgt_lang)]
        mt = None
        try:
            if need_translation and to_translate:
                release_cuda()
                from .mt.factory import build_mt_with_fallback
                result.mt_requested = args.mt
                try:
                    mt = build_mt_with_fallback(args.mt, model_dir, cfg)
                    mt.load()
                except Exception as e:
                    log.debug("MT 加载失败", exc_info=True)
                    for f in to_translate:
                        _fail(reports[f], f"翻译模型加载失败（{type(e).__name__}: {e}）")
                    todo = [f for f in todo if f not in to_translate]
                    mt = None
                if mt is not None:
                    result.mt_used = getattr(mt, "name", args.mt)
                    result.mt_fallback = getattr(mt, "fallback", "") or ""
                    plan_ = getattr(mt, "load_plan", None)
                    result.mt_plan = plan_["reason"] if isinstance(plan_, dict) else ""
            for i, f in enumerate(todo):
                rep = reports[f]
                tr = transcripts[f]
                prog("翻译" if need_translation else "断句", i, len(todo), f.name)
                src_lang = _base_lang(tr.language, args)
                log.info("断句/翻译 [%s] (%s->%s)", f.name, src_lang, args.tgt_lang)
                try:
                    res = segment_and_translate(
                        tr, mt if need_translation else None, cfg, src_lang, args.tgt_lang,
                        progress=(lambda k, n, _f=f: prog("翻译句子", k, n, _f.name))
                        if (progress or should_stop) else None)
                    _note_result(rep, res, args)
                    rep.outputs = _write_outputs(f, tr, res, args, cfg)
                    rep.status = "ok"
                except Cancelled:
                    raise
                except Exception as e:
                    log.debug("断句/翻译失败", exc_info=True)
                    _fail(rep, f"断句/翻译失败（{type(e).__name__}: {e}），这个文件没有写出字幕")
        finally:
            if mt is not None:
                mt.unload()
            release_cuda()
    except Cancelled:
        console.warn("已停止：后面的文件没有处理")
    result.seconds = time.time() - t_start
    return result


def print_summary(result: RunResult, args) -> None:
    """运行结束的汇总：每个文件一行，有问题的项目红字。"""
    console.info("")
    console.key("==================== 汇总 ====================")
    if result.mt_used:
        used = result.mt_used + (f"（{result.mt_plan}）" if result.mt_plan else "")
        if result.mt_fallback:
            console.warn(f"翻译模型：要求 {result.mt_requested}，实际用的是 {used}——{result.mt_fallback}")
        else:
            console.key(f"翻译模型：{used}")
    for r in result.reports:
        if r.status == "ok":
            us = r.unit_stats
            if args.mode == "mono-src" or r.skipped:
                tr_txt = f"原文 {r.src_cues} 条（没有翻译）"
            elif r.units:
                tr_txt = (f"原文 {r.src_cues} 条 → 译文 {r.mono_cues} 条，{r.units} 句"
                          f"（整句翻译 {us.get('sentence', 0) + us.get('no-context', 0)}）")
            else:
                tr_txt = f"原文 {r.src_cues} 条 → 译文 {r.mono_cues} 条（逐条翻译）"
            lang = (f"{r.language}" + (f" {r.lang_prob:.0%}" if r.lang_prob is not None and not r.lang_forced else "")
                    + ("（指定）" if r.lang_forced else ""))
            line = f"[完成] {r.file.name} · {lang} · {tr_txt}"
            (console.warn if r.warnings else console.ok)(line if not r.warnings else line + f" · {len(r.warnings)} 项需要注意")
            for w in r.warnings:
                console.warn(f"    - {w}")
            for w in r.notes:
                console.key(f"    - {w}")
        elif r.status == "failed":
            console.error(f"[失败] {r.file.name}：{r.error}")
        else:
            console.warn(f"[未处理] {r.file.name}")
    n_ok = sum(r.status == "ok" for r in result.reports)
    n = len(result.reports)
    msg = f"{n_ok}/{n} 个文件完成，用时 {_fmt_dur(result.seconds)}"
    if n_ok == n and n and not any(r.warnings for r in result.reports):
        console.ok(f"全部完成：{msg}")
    elif n_ok == n and n:
        console.warn(f"全部完成，但有需要注意的地方（见上面红字）：{msg}")
    else:
        console.error(f"有文件没有完成：{msg}")


def cmd_run(args, cfg) -> int:
    console.reset()
    result = run_pipeline(args, cfg)
    if not result.reports:
        return 2
    print_summary(result, args)
    return 0 if result.ok else 1


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _with_retry(fn, what: str, retries: int = 2):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            log.warning("%s 失败(第%d次): %s", what, attempt + 1, e)
            time.sleep(1.0)
    raise last


def _load_glossary(path: Path) -> list[tuple[str, str]]:
    """读术语表 -> [(原文, 译文)]。支持 JSON {日:中} 或每行 '日=中'/'日\\t中'。"""
    import json
    text = path.read_text(encoding="utf-8").strip()
    pairs: list[tuple[str, str]] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            pairs = [(str(k), str(v)) for k, v in data.items()]
    except Exception:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "=" if "=" in line else ("\t" if "\t" in line else None)
            if sep:
                a, b = line.split(sep, 1)
                pairs.append((a.strip(), b.strip()))
    log.info("已加载术语表 %d 条", len(pairs))
    return pairs


def _base(lang: str) -> str:
    return (lang or "").lower().split("-")[0]


def _base_lang(detected: str, args) -> str:
    if args.src_lang != "auto":
        return args.src_lang
    return detected or "unknown"


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser(cfg: dict) -> argparse.ArgumentParser:
    d = cfg.get("defaults", {})
    from .asr.factory import ASR_CHOICES
    from .mt.factory import MT_CHOICES
    from .vram import PRECISIONS

    p = argparse.ArgumentParser(
        prog="main.py",
        description="本地离线音/视频 语音识别 + 翻译 + 字幕生成",
    )
    p.add_argument("--verbose", action="store_true", help="调试日志")
    sub = p.add_subparsers(dest="command", required=True)

    # download
    pd = sub.add_parser("download", help="预下载模型")
    pd.add_argument("--asr", help="要下载的 ASR 模型别名")
    pd.add_argument("--mt", help="要下载的 MT 模型别名")
    pd.add_argument("--all-recommended", action="store_true", help="下载推荐组合")
    pd.add_argument("--force", action="store_true", help="强制重新下载")
    pd.add_argument("--model-dir", default=cfg.get("model_dir", DEFAULT_MODEL_DIR))

    # run
    pr = sub.add_parser("run", help="处理文件/文件夹生成字幕")
    pr.add_argument("input", nargs="+", help="输入文件或文件夹（可以给多个）")
    pr.add_argument("--recursive", action="store_true", help="文件夹递归")
    pr.add_argument("--mode", choices=["bilingual", "mono-src", "mono-tgt"],
                    default=d.get("mode", "bilingual"))
    pr.add_argument("--src-lang", default=d.get("src_lang", "auto"),
                    help="源语种（ja/en/ko/zh…）；auto=让 Whisper 自己识别（识别结果和把握会醒目显示）")
    pr.add_argument("--tgt-lang", default=d.get("tgt_lang", "zh"))
    pr.add_argument("--asr", choices=ASR_CHOICES, default=d.get("asr", "whisper"))
    pr.add_argument("--mt", choices=MT_CHOICES, default=d.get("mt", "hymt2"))
    pr.add_argument("--mt-precision", choices=PRECISIONS, default=None,
                    help="翻译模型精度：auto=按显存自动选（默认，>=18GB 全精度、>=10GB 8-bit、否则 4-bit）；"
                         "full / int8 / nf4 = 强制")
    adapt = pr.add_mutually_exclusive_group()
    adapt.add_argument("--lang-adapt", dest="lang_adapt", action="store_true")
    adapt.add_argument("--no-lang-adapt", dest="lang_adapt", action="store_false")
    pr.set_defaults(lang_adapt=d.get("lang_adapt", False))
    pr.add_argument("--ja-asr", choices=ASR_CHOICES, help="日语覆盖后端")
    pr.add_argument("--zh-asr", choices=ASR_CHOICES, help="中文覆盖后端")
    pr.add_argument("--ko-asr", choices=ASR_CHOICES, help="韩语覆盖后端")
    pr.add_argument("--vad", action="store_true",
                    help="打开 VAD 预过滤（默认关闭）。长时间无人声、幻听多的音频可以试；"
                         "人声音量很小的音频不要开，语音会被当成静音整段丢掉")
    pr.add_argument("--no-vad", action="store_true",
                    help="关闭 VAD 过滤（现在已是默认，保留此参数是为了兼容旧命令）")
    pr.add_argument("--vad-threshold", type=float, default=None,
                    help="VAD 阈值(0~1)，越低越灵敏，默认 0.2；给了这个参数等于同时打开 --vad")
    pr.add_argument("--separate-vocals", action="store_true",
                    help="先用 Demucs 分离人声再识别（背景音乐/音效很响时用；人声音量很小的音频不要用，会把人声分坏）")
    pr.add_argument("--glossary", default=None,
                    help="术语表文件（每行 '日文=中文' 或 JSON），翻译时强制遵守专名/术语")
    pr.add_argument("--mt-granularity", choices=["sentence", "cue"], default=None,
                    help="翻译粒度：sentence=整句翻译再分配回各条（默认）；cue=旧的逐条翻译")
    pr.add_argument("--mono-display", choices=["sentence", "repeat", "split"], default=None,
                    help="单语译文：sentence=译文自己排版，一句话尽量一条，条数可以比原文少（默认）；"
                         "repeat=条数同原文，短句在其每一条都显示整句；split=条数同原文，按比例切分")
    pr.add_argument("--output-dir", default=None)
    pr.add_argument("--swap-lines", action="store_true", help="双语调换上下行")
    pr.add_argument("--model-dir", default=cfg.get("model_dir", DEFAULT_MODEL_DIR))

    # gui
    pg = sub.add_parser("gui", help="打开图形界面（浏览器里操作，只在本机 127.0.0.1 上监听）")
    pg.add_argument("--port", type=int, default=7860)
    pg.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    return p


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))

    if args.command == "download":
        return cmd_download(args, cfg)
    if args.command == "run":
        return cmd_run(args, cfg)
    if args.command == "gui":
        from .gui import launch
        return launch(port=args.port, open_browser=not args.no_browser)
    parser.print_help()
    return 1

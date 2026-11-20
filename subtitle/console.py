"""终端里的醒目提示，并把它们记下来供最后的汇总和图形界面显示。

  warn / error  红色：降级、回退、没翻译出来、处理失败等“结果可能不是你以为的那样”的情况
  key           黄色：关键信息（识别出的语种和把握、实际用的模型和精度、每个文件的条数）
  ok            绿色：完成
  info          不加颜色

颜色只在终端里用：输出被重定向到文件/管道、或设置了环境变量 NO_COLOR 时不加颜色（FORCE_COLOR 可强制加）。
"""
from __future__ import annotations

import os
import sys
import threading

_RESET = "\x1b[0m"
_STYLE = {"warn": "\x1b[1;91m", "error": "\x1b[1;91m", "key": "\x1b[1;93m", "ok": "\x1b[92m", "info": ""}
_PREFIX = {"warn": "[警告] ", "error": "[错误] ", "key": "", "ok": "", "info": ""}

_lock = threading.Lock()
_sinks: list = []          # 额外的接收者 f(level, text)，图形界面用
_events: list = []         # 本次运行记下的 (level, text)
_color: bool | None = None


def use_color() -> bool:
    global _color
    if _color is None:
        if os.environ.get("NO_COLOR"):
            _color = False
        elif os.environ.get("FORCE_COLOR"):
            _color = True
        else:
            try:
                _color = bool(sys.stdout.isatty())
            except Exception:
                _color = False
        if _color and os.name == "nt":
            try:                                  # 旧版 Windows 控制台需要打开 ANSI 转义支持
                import colorama
                colorama.just_fix_windows_console()
            except Exception:
                pass
    return _color


def paint(level: str, text: str) -> str:
    style = _STYLE.get(level, "")
    return f"{style}{text}{_RESET}" if style and use_color() else text


def emit(level: str, text: str) -> None:
    with _lock:
        _events.append((level, text))
        sinks = list(_sinks)
    try:
        print(paint(level, _PREFIX.get(level, "") + text), flush=True)
    except Exception:
        pass
    for s in sinks:
        try:
            s(level, text)
        except Exception:
            pass


def warn(text: str) -> None:
    emit("warn", text)


def error(text: str) -> None:
    emit("error", text)


def key(text: str) -> None:
    emit("key", text)


def ok(text: str) -> None:
    emit("ok", text)


def info(text: str) -> None:
    emit("info", text)


def add_sink(fn) -> None:
    with _lock:
        _sinks.append(fn)


def remove_sink(fn) -> None:
    with _lock:
        if fn in _sinks:
            _sinks.remove(fn)


def events() -> list:
    with _lock:
        return list(_events)


def reset() -> None:
    with _lock:
        _events.clear()

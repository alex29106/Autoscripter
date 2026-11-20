"""Autoscripter 入口。

用法:
  python main.py download --all-recommended
  python main.py download --asr whisper --mt qwen3b
  python main.py run <文件或文件夹> [更多文件或文件夹…] [选项]
  python main.py gui            # 图形界面（浏览器里操作，只在本机 127.0.0.1 上监听）

详见 README.md。
"""
from __future__ import annotations

import sys

from subtitle.cli import main

if __name__ == "__main__":
    sys.exit(main())

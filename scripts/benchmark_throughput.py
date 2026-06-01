#!/usr/bin/env python3
"""兼容入口：复用 benchmark_input_ksanal.py 的完整实现。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "benchmark"

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_input_ksanal import args_config, main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    import uvloop  # noqa: E402

    uvloop.install()
    args = args_config()
    main(args)

"""Minimal, dependency-free timing harness shared by the benchmark scripts."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def time_call(fn: Callable[[], Any], *, repeats: int = 7, warmup: int = 2) -> float:
    """Median wall-clock time (seconds) of ``repeats`` timed calls to ``fn``,
    after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_environment() -> None:
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | pandas {pd.__version__}")
    print(f"Platform: {platform.platform()}")
    print()


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(str(row[c]).ljust(widths[c]) for c in columns))

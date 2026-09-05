"""Benchmark: signal turnover, naive vs. vectorised.

Run with:

    python -m benchmarks.benchmark_portfolio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks._timing import format_seconds, print_environment, print_table, time_call
from benchmarks.naive import naive_signal_turnover
from tsfeat.portfolio import signal_turnover

SHAPES = [(500, 100), (2_500, 500), (10_000, 1_000)]


def _weights(n_dates: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.dirichlet(np.ones(n_assets), size=n_dates))


def benchmark_turnover() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_dates, n_assets in SHAPES:
        weights = _weights(n_dates, n_assets)
        naive_time = time_call(lambda weights=weights: naive_signal_turnover(weights), repeats=3)
        vec_time = time_call(lambda weights=weights: signal_turnover(weights))
        rows.append(
            {
                "shape (dates x assets)": f"{n_dates}x{n_assets}",
                "naive": format_seconds(naive_time),
                "tsfeat": format_seconds(vec_time),
                "speedup": f"{naive_time / vec_time:.1f}x",
            }
        )
    return rows


def main() -> None:
    print_environment()
    print("Signal turnover")
    print_table(benchmark_turnover())


if __name__ == "__main__":
    main()

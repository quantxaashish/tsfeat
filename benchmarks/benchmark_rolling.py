"""Benchmark: rolling z-score and forward returns, naive vs. vectorised.

Run with:

    python -m benchmarks.benchmark_rolling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks._timing import format_seconds, print_environment, print_table, time_call
from benchmarks.naive import naive_forward_returns, naive_rolling_zscore
from tsfeat.returns import forward_returns
from tsfeat.rolling import rolling_zscore

# The naive O(T*window) implementation becomes impractically slow at large
# T, so it is only run at sizes where it finishes in a reasonable time. The
# vectorised implementation is run at every size to show its scaling.
ZSCORE_SIZES_NAIVE_AND_VECTORISED = [10_000, 100_000]
ZSCORE_SIZES_VECTORISED_ONLY = [1_000_000]
WINDOW = 20

FORWARD_RETURN_SIZES_NAIVE_AND_VECTORISED = [10_000, 100_000]
FORWARD_RETURN_SIZES_VECTORISED_ONLY = [1_000_000]
HORIZONS = [1, 5, 10, 20]


def _synthetic_prices(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=n)))


def benchmark_rolling_zscore() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in ZSCORE_SIZES_NAIVE_AND_VECTORISED:
        prices = _synthetic_prices(n)
        naive_time = time_call(
            lambda prices=prices: naive_rolling_zscore(prices, window=WINDOW), repeats=3
        )
        vec_time = time_call(lambda prices=prices: rolling_zscore(prices, window=WINDOW))
        rows.append(
            {
                "rows": n,
                "naive": format_seconds(naive_time),
                "tsfeat": format_seconds(vec_time),
                "speedup": f"{naive_time / vec_time:.1f}x",
            }
        )
    for n in ZSCORE_SIZES_VECTORISED_ONLY:
        prices = _synthetic_prices(n)
        vec_time = time_call(lambda prices=prices: rolling_zscore(prices, window=WINDOW))
        rows.append(
            {
                "rows": n,
                "naive": "skipped (O(T*W) too slow)",
                "tsfeat": format_seconds(vec_time),
                "speedup": "n/a",
            }
        )
    return rows


def benchmark_forward_returns() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in FORWARD_RETURN_SIZES_NAIVE_AND_VECTORISED:
        prices = _synthetic_prices(n)
        naive_time = time_call(
            lambda prices=prices: naive_forward_returns(prices, horizons=HORIZONS), repeats=3
        )
        vec_time = time_call(lambda prices=prices: forward_returns(prices, horizons=HORIZONS))
        rows.append(
            {
                "rows": n,
                "naive": format_seconds(naive_time),
                "tsfeat": format_seconds(vec_time),
                "speedup": f"{naive_time / vec_time:.1f}x",
            }
        )
    for n in FORWARD_RETURN_SIZES_VECTORISED_ONLY:
        prices = _synthetic_prices(n)
        vec_time = time_call(lambda prices=prices: forward_returns(prices, horizons=HORIZONS))
        rows.append(
            {
                "rows": n,
                "naive": "skipped (O(T*H) too slow)",
                "tsfeat": format_seconds(vec_time),
                "speedup": "n/a",
            }
        )
    return rows


def main() -> None:
    print_environment()
    print(f"Rolling z-score (window={WINDOW})")
    print_table(benchmark_rolling_zscore())
    print()
    print(f"Forward returns (horizons={HORIZONS})")
    print_table(benchmark_forward_returns())


if __name__ == "__main__":
    main()

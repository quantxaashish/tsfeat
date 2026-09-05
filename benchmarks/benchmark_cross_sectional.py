"""Benchmark: cross-sectional rank and demean, naive vs. vectorised.

Run with:

    python -m benchmarks.benchmark_cross_sectional
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks._timing import format_seconds, print_environment, print_table, time_call
from benchmarks.naive import naive_cross_sectional_demean, naive_cross_sectional_rank
from tsfeat.cross_sectional import cross_sectional_demean, cross_sectional_rank

# naive_cross_sectional_rank is O(T * N^2) (pairwise comparisons per date),
# so it is only run at modest sizes. demean is O(T * N) and tolerates a
# larger naive run.
RANK_SHAPES = [(250, 50), (500, 100)]
RANK_SHAPE_VECTORISED_ONLY = (2_500, 500)

DEMEAN_SHAPES = [(500, 100), (2_500, 500)]


def _panel(n_dates: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_dates, n_assets)))


def benchmark_rank() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_dates, n_assets in RANK_SHAPES:
        panel = _panel(n_dates, n_assets)
        naive_time = time_call(lambda panel=panel: naive_cross_sectional_rank(panel), repeats=3)
        vec_time = time_call(lambda panel=panel: cross_sectional_rank(panel))
        rows.append(
            {
                "shape (dates x assets)": f"{n_dates}x{n_assets}",
                "naive": format_seconds(naive_time),
                "tsfeat": format_seconds(vec_time),
                "speedup": f"{naive_time / vec_time:.1f}x",
            }
        )
    n_dates, n_assets = RANK_SHAPE_VECTORISED_ONLY
    panel = _panel(n_dates, n_assets)
    vec_time = time_call(lambda panel=panel: cross_sectional_rank(panel))
    rows.append(
        {
            "shape (dates x assets)": f"{n_dates}x{n_assets}",
            "naive": "skipped (O(T*N^2) too slow)",
            "tsfeat": format_seconds(vec_time),
            "speedup": "n/a",
        }
    )
    return rows


def benchmark_demean() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_dates, n_assets in DEMEAN_SHAPES:
        panel = _panel(n_dates, n_assets)
        naive_time = time_call(lambda panel=panel: naive_cross_sectional_demean(panel), repeats=3)
        vec_time = time_call(lambda panel=panel: cross_sectional_demean(panel))
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
    print("Cross-sectional rank")
    print_table(benchmark_rank())
    print()
    print("Cross-sectional demean")
    print_table(benchmark_demean())


if __name__ == "__main__":
    main()

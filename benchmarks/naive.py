"""Naive, deliberately row-by-row reference implementations.

These exist **only** for benchmarking and independent correctness
comparison against ``tsfeat``'s vectorised implementations. They contain
exactly the kind of Python-level per-observation iteration that the
production package (``src/tsfeat``) is required never to contain.

``tsfeat`` must never import anything from this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def naive_rolling_zscore(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    """Row-by-row rolling z-score. O(T * window): recomputes the mean and
    standard deviation from scratch for every window position."""
    effective_min_periods = window if min_periods is None else min_periods
    arr = values.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        window_vals = arr[start : i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < effective_min_periods or len(valid) <= ddof:
            continue
        mean = valid.mean()
        std = valid.std(ddof=ddof)
        if std == 0:
            continue
        out[i] = (arr[i] - mean) / std
    return pd.Series(out, index=values.index)


def naive_forward_returns(
    prices: pd.Series,
    horizons: Sequence[int],
    log: bool = False,
) -> pd.DataFrame:
    """Row-by-row forward returns. O(T * len(horizons)): for every origin
    date, looks up the price ``h`` observations ahead one at a time."""
    arr = prices.to_numpy(dtype=float)
    n = len(arr)
    data = {}
    for h in horizons:
        out = np.full(n, np.nan)
        for i in range(n):
            j = i + h
            if j < n:
                out[i] = np.log(arr[j] / arr[i]) if log else arr[j] / arr[i] - 1
        data[f"fwd_{h}"] = out
    return pd.DataFrame(data, index=prices.index)


def naive_cross_sectional_rank(
    panel: pd.DataFrame,
    pct: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """Row-by-row, pair-by-pair cross-sectional rank. O(T * N^2): for every
    date, every asset is compared against every other asset in a Python
    double loop to compute its (tie-averaged) rank."""
    values = panel.to_numpy(dtype=float)
    n_dates, n_assets = values.shape
    out = np.full_like(values, np.nan)
    for t in range(n_dates):
        row = values[t]
        valid_idx = [j for j in range(n_assets) if not np.isnan(row[j])]
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue
        for j in valid_idx:
            less = 0
            equal = 0
            for k in valid_idx:
                if row[k] < row[j]:
                    less += 1
                elif row[k] == row[j]:
                    equal += 1
            rank = less + (equal + 1) / 2.0
            if not ascending:
                rank = n_valid - rank + 1
            out[t, j] = rank / n_valid if pct else rank
    return pd.DataFrame(out, index=panel.index, columns=panel.columns)


def naive_cross_sectional_demean(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row cross-sectional demean. O(T * N): computes each date's
    mean and subtracts it via an explicit Python loop over assets."""
    values = panel.to_numpy(dtype=float)
    n_dates, n_assets = values.shape
    out = np.full_like(values, np.nan)
    for t in range(n_dates):
        row = values[t]
        valid = [x for x in row if not np.isnan(x)]
        if len(valid) == 0:
            continue
        mean = sum(valid) / len(valid)
        for j in range(n_assets):
            if not np.isnan(row[j]):
                out[t, j] = row[j] - mean
    return pd.DataFrame(out, index=panel.index, columns=panel.columns)


def naive_signal_turnover(weights: pd.DataFrame, one_way: bool = True) -> pd.Series:
    """Row-by-row turnover. O(T * N): walks consecutive rebalance dates one
    at a time, summing absolute weight changes asset by asset."""
    values = weights.to_numpy(dtype=float)
    n_dates, n_assets = values.shape
    out = np.full(n_dates, np.nan)
    prev: list[float] | None = None
    for t in range(n_dates):
        row = values[t]
        row_filled = [0.0 if np.isnan(x) else float(x) for x in row]
        if prev is not None:
            total = 0.0
            for j in range(n_assets):
                total += abs(row_filled[j] - prev[j])
            out[t] = total * 0.5 if one_way else total
        prev = row_filled
    return pd.Series(out, index=weights.index)

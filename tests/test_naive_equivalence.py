"""Confirms the naive benchmark references actually compute the same thing
as the production implementations, so that benchmark speedups reflect
vectorisation only, not two different algorithms (see benchmark-fairness
requirements)."""

import numpy as np
import pandas as pd

from benchmarks.naive import (
    naive_cross_sectional_demean,
    naive_cross_sectional_rank,
    naive_forward_returns,
    naive_rolling_zscore,
    naive_signal_turnover,
)
from tsfeat.cross_sectional import cross_sectional_demean, cross_sectional_rank
from tsfeat.portfolio import signal_turnover
from tsfeat.returns import forward_returns
from tsfeat.rolling import rolling_zscore


def test_naive_rolling_zscore_matches_production() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    s.iloc[5] = np.nan
    expected = rolling_zscore(s, window=10, min_periods=5)
    actual = naive_rolling_zscore(s, window=10, min_periods=5)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_naive_forward_returns_matches_production() -> None:
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=100)))
    expected = forward_returns(prices, horizons=[1, 5, 10])
    actual = naive_forward_returns(prices, horizons=[1, 5, 10])
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_naive_cross_sectional_rank_matches_production() -> None:
    rng = np.random.default_rng(2)
    panel = pd.DataFrame(rng.normal(size=(50, 8)))
    panel.iloc[3, 2] = np.nan
    expected = cross_sectional_rank(panel, pct=True)
    actual = naive_cross_sectional_rank(panel, pct=True)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_naive_cross_sectional_demean_matches_production() -> None:
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(rng.normal(size=(50, 8)))
    panel.iloc[3, 2] = np.nan
    expected = cross_sectional_demean(panel)
    actual = naive_cross_sectional_demean(panel)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_naive_signal_turnover_matches_production() -> None:
    rng = np.random.default_rng(4)
    weights = pd.DataFrame(rng.dirichlet(np.ones(6), size=30))
    expected = signal_turnover(weights)
    actual = naive_signal_turnover(weights)
    pd.testing.assert_series_equal(actual, expected, check_names=False)

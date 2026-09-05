"""End-to-end research pipeline demonstrating how tsfeat's primitives
compose in a typical factor-research workflow:

    Yahoo Finance prices
            |
    daily returns
            |
    20-day rolling z-score signal
            |
    cross-sectional rank
            |
    demean -> portfolio weights
            |
    forward returns
            |
    IC decay
            |
    turnover analysis

This is a plumbing demonstration, not a trading strategy: the "signal" used
here (short-horizon price z-score) is arbitrary, has not been evaluated for
economic merit, and this script makes no claim that it is profitable.

Requires internet access and is never imported by the test suite. Run with:

    python examples/research_pipeline.py
"""

from __future__ import annotations

import pandas as pd
from yahoo_finance_demo import UNIVERSE, download_adjusted_close

import tsfeat as tf


def build_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """A short-horizon price z-score, used purely to exercise the API."""
    return tf.rolling_zscore(prices, window=20, min_periods=20)


def rank_based_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank the signal, demean it, and scale it to a
    unit-leverage long/short weight vector per date."""
    ranked = tf.cross_sectional_rank(signal, pct=True)
    demeaned = tf.cross_sectional_demean(ranked)
    gross_leverage = demeaned.abs().sum(axis=1)
    return demeaned.div(gross_leverage, axis=0).where(gross_leverage > 0)


def main() -> None:
    prices = download_adjusted_close(UNIVERSE)
    returns = prices.pct_change()
    print(f"Universe: {list(prices.columns)}")
    print(f"History: {prices.index[0].date()} to {prices.index[-1].date()}")

    signal = build_signal(prices)
    weights = rank_based_weights(signal)

    print("\nMost recent portfolio weights (rank-based, demeaned, unit gross leverage):")
    print(weights.iloc[-1].round(4))

    horizons = [1, 5, 10, 20]
    fwd = tf.forward_returns(prices, horizons=horizons)

    ic = tf.ic_decay(signal, fwd, method="spearman", min_assets=3)
    print("\nIC decay (Spearman rank IC, signal vs. forward returns by horizon):")
    print(ic.round(4))

    turnover = tf.signal_turnover(weights.dropna(how="all"))
    print("\nTurnover of the rank-based portfolio (one-way convention):")
    print(turnover.describe().round(4))

    returns_vol = tf.ew_volatility(returns, half_life=20, min_periods=20, annualization=252)
    print("\nMost recent annualised EW volatility by asset:")
    print(returns_vol.iloc[-1].round(4))


if __name__ == "__main__":
    main()

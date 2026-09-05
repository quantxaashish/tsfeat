"""Minimal example: fetch real prices with yfinance and run a few tsfeat
primitives over them.

This script requires internet access and is never imported by the test
suite (tests use deterministic local fixtures only). Run with:

    python examples/yahoo_finance_demo.py

Note on yfinance's output shape: ``yf.download`` with more than one ticker
returns a DataFrame with MultiIndex columns of ``(field, ticker)``, e.g.
``("Close", "AAPL")``. With a single ticker in the list it still returns
that same MultiIndex shape (as opposed to passing a bare string ticker,
which returns plain columns) — passing a list consistently is what keeps
the downstream code below identical regardless of universe size.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

import tsfeat as tf

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]


def download_adjusted_close(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices as a wide date x ticker DataFrame."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"]
    # yfinance can return columns in a different order than requested, and
    # may drop a ticker entirely if it failed to fetch — reindex explicitly
    # rather than assuming positional alignment with `tickers`.
    return prices.reindex(columns=tickers)


def main() -> None:
    prices = download_adjusted_close(UNIVERSE)
    print(f"Downloaded prices: {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(prices.tail(3))

    returns = prices.pct_change()

    zscore = tf.rolling_zscore(prices, window=20, min_periods=10)
    print("\nRolling 20-day z-score of price level (last date):")
    print(zscore.iloc[-1])

    vol = tf.ew_volatility(returns, half_life=20, min_periods=10, annualization=252)
    print("\nAnnualised EW volatility, half-life=20 (last date):")
    print(vol.iloc[-1])

    fwd = tf.forward_returns(prices, horizons=[1, 5, 10, 20])
    print("\n5-day forward return, most recent date with a defined value:")
    fwd_5 = fwd["fwd_5"].dropna(how="all")
    print(fwd_5.iloc[-1])


if __name__ == "__main__":
    main()

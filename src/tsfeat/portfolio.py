"""Portfolio/signal weight-vector primitives."""

from __future__ import annotations

import pandas as pd

from tsfeat.validation import check_no_duplicate_columns

__all__ = ["signal_turnover"]


def signal_turnover(weights: pd.DataFrame, *, one_way: bool = True) -> pd.Series:
    """Turnover between consecutive rows (rebalance dates) of a weight panel.

    Parameters
    ----------
    weights : DataFrame
        Rows are rebalance dates, columns are assets, values are portfolio
        weights (long-only or long-short).
    one_way : bool, default True
        If ``True`` (default), turnover for a rebalance is
        ``0.5 * sum(abs(w_t - w_{t-1}))`` — the standard "one-way
        turnover" convention, which reports a full round-trip (sell
        everything, buy an entirely different set of names) as ``1.0``
        (100% of the book turned over) rather than ``2.0``. If ``False``,
        the raw, undivided ``sum(abs(w_t - w_{t-1}))`` is returned instead.

    Returns
    -------
    Series
        Indexed like ``weights`` (same index, one row shorter conceptually
        — the first date has no prior weights to compare against and is
        reported as ``NaN``, not ``0``, since "no turnover" and "no prior
        portfolio to compare against" are different things).

    Notes
    -----
    **Universe changes**: if an asset appears in one rebalance's columns
    but not the other (a genuine change in the investable universe rather
    than a `NaN` observation), align by reindexing to the union of columns
    across both dates and treating a missing weight as ``0`` (not held).
    This is deliberate: dropping the asset entirely would silently
    understate turnover exactly at the moment a position is fully entered
    or exited, which is the opposite of what a turnover metric should do.
    ``NaN`` weights already present in the input are likewise treated as
    ``0`` for the purposes of this difference, on the same "not held"
    reasoning.

    Vectorised via ``weights.diff().abs().sum(axis=1)`` — a single
    row-wise difference and column-wise sum, no per-asset or per-date
    Python loop.
    """
    check_no_duplicate_columns(weights, "weights")
    filled = weights.fillna(0.0)
    turnover = filled.diff().abs().sum(axis=1)
    turnover.iloc[0] = float("nan")
    if one_way:
        turnover = turnover * 0.5
    return turnover

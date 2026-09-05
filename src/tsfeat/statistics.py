"""Price-series and signal-quality statistics: VWAP, drawdown, IC decay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tsfeat.validation import (
    check_method,
    check_monotonic_index,
    check_no_duplicate_columns,
    check_positive_int,
)

__all__ = ["vwap", "cumulative_vwap", "DrawdownStats", "max_drawdown", "ic_decay"]

_REQUIRED_TRADE_COLUMNS = ("price", "volume")


def _validate_trades(trades: pd.DataFrame, *, require_timestamp: bool = False) -> None:
    required = _REQUIRED_TRADE_COLUMNS + (("timestamp",) if require_timestamp else ())
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"trades is missing required column(s): {missing}")


def vwap(
    trades: pd.DataFrame,
    *,
    group: str | pd.Series | None = None,
) -> float | pd.Series:
    """Volume-weighted average price, optionally grouped (e.g. by session).

    ``VWAP = sum(price * volume) / sum(volume)``

    Parameters
    ----------
    trades : DataFrame
        Must contain ``price`` and ``volume`` columns. One row per trade;
        several trades sharing an identical timestamp are simply several
        rows and are handled correctly since this is a plain aggregate
        (order within a group never matters for a sum).
    group : str or Series, optional
        If ``None`` (default), a single VWAP is computed over every row.
        If a column name or an array-like of the same length as
        ``trades``, VWAP is computed separately per distinct group value
        (e.g. pass a session or date column/label to get one VWAP per
        session).

    Returns
    -------
    float or Series
        A single ``float`` when ``group`` is ``None``; a ``Series``
        indexed by group value otherwise.

    Notes
    -----
    **Duplicate timestamps**: this function does not use timestamps at
    all — it aggregates whichever rows are given to it — so multiple
    trades sharing a timestamp contribute additively to both the
    numerator and denominator, which is the correct VWAP semantics
    (VWAP does not care about intra-timestamp ordering, only about total
    traded notional and volume). See :func:`cumulative_vwap` for the
    running/intraday case, where duplicate-timestamp ordering could
    otherwise matter.

    **Missing data policy**: rows with a ``NaN`` price or ``NaN`` volume
    are excluded before aggregation (treated as missing trade records).

    **Zero volume**: if total volume for a group (or for the whole input
    when ``group`` is ``None``) is zero, the result for that group is
    ``NaN`` rather than raising or silently returning ``inf``.
    """
    _validate_trades(trades)
    valid = trades.dropna(subset=list(_REQUIRED_TRADE_COLUMNS))
    notional = valid["price"] * valid["volume"]

    if group is None:
        total_volume = valid["volume"].sum()
        if total_volume == 0:
            return float("nan")
        return float(notional.sum() / total_volume)

    key = trades[group] if isinstance(group, str) else group
    key = key.loc[valid.index]
    total_notional = notional.groupby(key).sum()
    total_volume = valid["volume"].groupby(key).sum()
    result = total_notional / total_volume
    return result.where(total_volume != 0)


def cumulative_vwap(trades: pd.DataFrame) -> pd.Series:
    """Running (intraday) VWAP, with same-timestamp trades merged first.

    Parameters
    ----------
    trades : DataFrame
        Must contain ``timestamp``, ``price``, and ``volume`` columns.
        Need not be pre-sorted.

    Returns
    -------
    Series
        Indexed by unique, sorted ``timestamp``. The value at each
        timestamp is the VWAP of all trades at or before that timestamp.

    Notes
    -----
    **Duplicate timestamps — the core engineering challenge here**:
    several trades can share an identical timestamp, and there is no
    reliable way to recover the true intra-timestamp arrival order from
    the data (row order in the input is not assumed to be meaningful).
    Rather than computing a cumulative sum row-by-row — which would make
    the running VWAP depend on arbitrary input row order whenever
    timestamps tie — trades sharing a timestamp are first aggregated
    (summed notional, summed volume) into a single node with
    ``groupby("timestamp")``, and the cumulative sum is taken over those
    per-timestamp aggregates. The running VWAP therefore updates once per
    unique timestamp, never depends on same-timestamp row order, and
    involves no Python-level iteration over trades — the grouping and
    cumulative sum are both vectorised pandas operations.

    Missing prices/volumes/timestamps are dropped first, using the same
    policy as :func:`vwap`. A timestamp whose cumulative volume is still
    zero (e.g. every trade so far had zero volume) reports ``NaN``.
    """
    _validate_trades(trades, require_timestamp=True)
    valid = trades.dropna(subset=["timestamp", *_REQUIRED_TRADE_COLUMNS])
    valid = valid.assign(notional=valid["price"] * valid["volume"])
    per_timestamp = valid.groupby("timestamp", sort=True)[["notional", "volume"]].sum()

    cum_notional = per_timestamp["notional"].cumsum()
    cum_volume = per_timestamp["volume"].cumsum()
    result = cum_notional / cum_volume
    return result.where(cum_volume != 0)


@dataclass(frozen=True)
class DrawdownStats:
    """Summary statistics for a single maximum-drawdown episode.

    Attributes
    ----------
    max_drawdown : float
        The most negative value of ``wealth / running_max(wealth) - 1``
        over the series. Always ``<= 0``.
    peak_date : Any
        Index label of the most recent new high-water mark reached at or
        before ``trough_date``.
    trough_date : Any
        Index label at which ``max_drawdown`` occurs.
    recovery_date : Any
        Index label of the first point after ``trough_date`` at which
        wealth returns to (or exceeds) the pre-drawdown peak level, or
        ``NaT`` if the series never recovers.
    max_duration : Any
        The longest span, over the whole series, between any point and
        the most recent new high-water mark at or before that point. This
        is an ``int`` (number of observations) for a non-datetime index,
        or a ``pandas.Timedelta`` for a ``DatetimeIndex``. It equals
        peak-to-recovery duration for an episode that recovers, and
        peak-to-end-of-series (right-censored) for one that does not —
        both are computed by the same definition, applied uniformly.
    """

    max_drawdown: float
    peak_date: Any
    trough_date: Any
    recovery_date: Any
    max_duration: Any


def max_drawdown(wealth: pd.Series) -> DrawdownStats:
    """Maximum drawdown and its duration from a wealth/equity-curve series.

    Drawdown at ``t`` is defined against the running high-water mark:
    ``drawdown(t) = wealth(t) / running_max(wealth)(t) - 1``, which is
    always ``<= 0``.

    Parameters
    ----------
    wealth : Series
        Cumulative wealth / equity curve / price level series, indexed by
        time (monotonically increasing, no duplicate labels).

    Returns
    -------
    DrawdownStats

    Raises
    ------
    ValueError
        If ``wealth`` is empty, or its index is not monotonically
        increasing / contains duplicate labels.

    Notes
    -----
    Entirely vectorised: ``running_max = wealth.cummax()``; the
    "duration since last high" series used for ``max_duration`` is
    computed with a single ``numpy`` running-max over integer positions
    (``np.maximum.accumulate``), not a per-observation Python loop.
    ``NaN`` observations propagate ``NaN`` into ``running_max`` /
    ``drawdown`` at their own position only, and are ignored (via
    ``skipna``) when carrying the running max forward to later
    observations.
    """
    if len(wealth) == 0:
        raise ValueError("wealth must not be empty")
    check_monotonic_index(wealth.index, "wealth index")

    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])

    peak_value = running_max.loc[trough_date]
    pre_trough = wealth[wealth.index <= trough_date]
    peak_date = pre_trough[pre_trough == peak_value].index[-1]

    post_trough = wealth[wealth.index >= trough_date]
    recovered = post_trough[(post_trough.index > trough_date) & (post_trough >= peak_value)]
    recovery_date = recovered.index[0] if len(recovered) > 0 else pd.NaT

    is_new_high = (wealth == running_max).to_numpy()
    positions = np.arange(len(wealth))
    high_positions = np.where(is_new_high, positions, -1)
    last_high_position = np.maximum.accumulate(high_positions)
    duration_periods = positions - last_high_position

    index = wealth.index
    if isinstance(index, pd.DatetimeIndex):
        elapsed = index - index[last_high_position]
        max_duration: Any = elapsed.max()
    else:
        max_duration = int(duration_periods.max())

    return DrawdownStats(
        max_drawdown=max_dd,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        max_duration=max_duration,
    )


def ic_decay(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    min_assets: int = 3,
) -> pd.DataFrame:
    """Cross-sectional Information Coefficient, summarised per horizon.

    For each date, computes the cross-sectional correlation between
    ``signal`` and each horizon's forward return, then summarises that
    per-date IC series across time for each horizon.

    Parameters
    ----------
    signal : DataFrame
        Wide-format panel (rows are dates, columns are assets). Must
        already be aligned at time ``t`` — i.e. ``signal.loc[t]`` must be
        computable purely from information available at or before ``t``.
        This function has no way to verify that and trusts the caller;
        forward returns are outcomes and are expected to use future
        prices, but the signal must not.
    forward_returns : DataFrame
        The output of :func:`tsfeat.forward_returns` called on a
        multi-asset (``DataFrame``) input: ``MultiIndex`` columns of
        ``(horizon_label, asset)``, e.g. ``("fwd_5", "AAPL")``.
    method : {"pearson", "spearman"}, default "spearman"
        ``"spearman"`` ranks both ``signal`` and each horizon's forward
        returns cross-sectionally (within each date) before computing a
        Pearson correlation on the ranks — the standard definition of
        rank IC. ``"pearson"`` correlates the raw values directly.
    min_assets : int, default 3
        Minimum number of assets with both a valid signal and a valid
        forward return on a given date for that date's IC to be counted;
        dates with fewer valid pairs are excluded (their IC is set to
        ``NaN`` and they do not contribute to ``n_observations``).

    Returns
    -------
    DataFrame
        Indexed by horizon label (e.g. ``"fwd_1"``, ``"fwd_5"``), with
        columns ``mean_ic``, ``std_ic``, ``ic_ir``, ``n_observations``.
        ``ic_ir = mean_ic / std_ic`` (using the sample, ``ddof=1``,
        standard deviation of the per-date IC series) — the standard
        "IC information ratio" convention, analogous to a Sharpe ratio
        computed on the IC time series rather than on returns.
        ``n_observations`` counts dates with a defined (non-NaN) IC, not
        the number of assets.

    Raises
    ------
    ValueError
        If ``method`` is not ``"pearson"`` or ``"spearman"``, or
        ``min_assets`` is not a positive integer.

    Notes
    -----
    Cross-sectional correlation per date is computed with
    ``DataFrame.corrwith(other, axis=1)``, which correlates matching rows
    of two aligned wide frames using pairwise-complete observations —
    this is vectorised across the whole date axis at once (mean, std, and
    covariance are computed for every row simultaneously), not a
    per-date Python loop. The only Python-level loop in this function
    iterates over the (small, fixed) set of horizons, which this
    package's design explicitly allows.

    A cross-sectionally constant signal on a given date has zero
    variance, so its correlation on that date is undefined and reported
    as ``NaN`` by ``corrwith`` itself — no special-casing is required.
    """
    check_method(method, ("pearson", "spearman"))
    check_positive_int(min_assets, "min_assets")
    check_no_duplicate_columns(signal, "signal")

    horizon_labels = list(dict.fromkeys(forward_returns.columns.get_level_values(0)))
    signal_for_corr = signal.rank(axis=1) if method == "spearman" else signal

    rows: list[dict[str, Any]] = []
    for label in horizon_labels:
        fwd = forward_returns[label].reindex(columns=signal.columns)
        fwd_for_corr = fwd.rank(axis=1) if method == "spearman" else fwd

        n_valid = (signal_for_corr.notna() & fwd_for_corr.notna()).sum(axis=1)
        ic_by_date = signal_for_corr.corrwith(fwd_for_corr, axis=1)
        ic_by_date = ic_by_date.where(n_valid >= min_assets)

        mean_ic = float(ic_by_date.mean())
        std_ic = float(ic_by_date.std(ddof=1))
        ic_ir = mean_ic / std_ic if std_ic not in (0.0,) and not np.isnan(std_ic) else float("nan")
        rows.append(
            {
                "horizon": label,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "n_observations": int(ic_by_date.notna().sum()),
            }
        )

    return pd.DataFrame(rows).set_index("horizon")

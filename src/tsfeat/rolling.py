"""Trailing-window time-series primitives.

Every function in this module is backward-looking only: the value reported
at index ``t`` depends solely on observations at or before ``t``. No future
observation ever contributes to a value reported for the past.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

from tsfeat.validation import (
    check_ddof,
    check_half_life,
    check_min_periods,
    check_monotonic_index,
    check_no_duplicate_columns,
    check_positive_int,
    check_window,
)

SeriesOrFrame = TypeVar("SeriesOrFrame", pd.Series, pd.DataFrame)

__all__ = ["rolling_zscore", "ew_volatility", "rolling_correlation"]


def rolling_zscore(
    values: SeriesOrFrame,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
) -> SeriesOrFrame:
    """Rolling z-score of ``values`` with a minimum-periods guard.

    Computes ``(x_t - rolling_mean_t) / rolling_std_t`` using a trailing
    window of length ``window`` ending at (and including) ``t``.

    Parameters
    ----------
    values : Series or DataFrame
        Input observations, indexed by time. A DataFrame is treated as one
        independent series per column.
    window : int
        Trailing window length in observations.
    min_periods : int, optional
        Minimum number of non-NaN observations within the window required to
        produce a value. Defaults to ``window`` (no partial-window output).
    ddof : int, default 1
        Delta degrees of freedom used for the rolling standard deviation.
        The default of 1 (sample standard deviation) is appropriate because
        a rolling window is a finite sample, not the full population.

    Returns
    -------
    Series or DataFrame
        Same shape and index as ``values``. The first ``min_periods - 1``
        observations (per column) are ``NaN``.

    Raises
    ------
    ValueError
        If ``window`` is not positive, ``min_periods`` exceeds ``window``,
        or ``ddof`` is negative.

    Notes
    -----
    **Zero-variance windows**: whenever the rolling standard deviation is
    exactly zero (all observations in the window are identical), the
    z-score is defined as ``NaN`` rather than ``0`` or ``inf``. This is
    enforced explicitly rather than left to fall out of ``0 / 0`` floating
    point arithmetic, so that floating-point noise in the numerator can
    never silently produce ``+/-inf``.

    **No lookahead**: this is a direct wrapper around
    ``pandas.Series.rolling`` / ``pandas.DataFrame.rolling``, which only
    ever look backward from the current position.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> rolling_zscore(s, window=3, min_periods=3).round(3).tolist()
    [nan, nan, 1.0, 1.0, 1.0]
    """
    check_window(window)
    check_min_periods(min_periods, window)
    check_ddof(ddof)
    if isinstance(values, pd.DataFrame):
        check_no_duplicate_columns(values, "values")

    effective_min_periods = window if min_periods is None else min_periods
    roll = values.rolling(window=window, min_periods=effective_min_periods)
    mean = roll.mean()
    std = roll.std(ddof=ddof)

    z = (values - mean) / std
    return z.where(std != 0)


def ew_volatility(
    returns: SeriesOrFrame,
    half_life: float,
    *,
    min_periods: int | None = None,
    annualization: float | None = None,
) -> SeriesOrFrame:
    """Exponentially weighted volatility with a half-life parameterisation.

    Parameters
    ----------
    returns : Series or DataFrame
        Return observations, indexed by time.
    half_life : float
        The number of periods over which the weight of an observation
        decays to half its initial value. Internally converted to a decay
        factor ``alpha = 1 - 0.5 ** (1 / half_life)``, which is exactly the
        conversion ``pandas.DataFrame.ewm(halflife=...)`` performs.
    min_periods : int, optional
        Minimum number of observations before a value is produced. Defaults
        to ``0`` (pandas' own default), meaning the estimate is reported as
        soon as it is numerically defined.
    annualization : float, optional
        If given, the result is multiplied by ``sqrt(annualization)``
        (e.g. pass ``252`` to annualise daily volatility).

    Returns
    -------
    Series or DataFrame
        Same shape and index as ``returns``.

    Raises
    ------
    ValueError
        If ``half_life`` is not positive, or ``annualization`` is not
        positive when given.

    Notes
    -----
    **Pandas semantics used**: this calls
    ``returns.ewm(halflife=half_life, adjust=False).std()`` directly. That
    is a recursive, exponentially-weighted standard deviation computed by
    pandas over the raw return observations themselves — it is *not* the
    RiskMetrics-style recursion applied to squared returns
    (``sigma_t^2 = (1-alpha) sigma_{t-1}^2 + alpha r_t^2``), and this
    function does not reimplement that equation separately. The two are
    closely related but not numerically identical, since pandas' estimator
    also applies its own bias correction based on the accumulated sum of
    weights.

    ``adjust=False`` is used deliberately: it makes the estimate a genuine
    recursive filter (today's value depends only on yesterday's estimate
    and today's observation), rather than ``adjust=True``'s behaviour of
    re-weighting the *entire* history-to-date every time a new observation
    arrives. Both are equally lookahead-safe; ``adjust=False`` is simply
    the more standard convention for a live-updating volatility estimate.

    For a constant return series with enough observations to satisfy
    ``min_periods``, every deviation from the (also constant) exponentially
    weighted mean is exactly zero, so the reported volatility is ``0.0``,
    not ``NaN``.

    Examples
    --------
    >>> import pandas as pd
    >>> r = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    >>> ew_volatility(r, half_life=3, min_periods=2).round(6).tolist()[-1]
    0.0
    """
    check_half_life(half_life)
    if min_periods is not None:
        check_positive_int(min_periods, "min_periods")
    if annualization is not None and annualization <= 0:
        raise ValueError(f"annualization must be positive, got {annualization}")
    if isinstance(returns, pd.DataFrame):
        check_no_duplicate_columns(returns, "returns")

    vol = returns.ewm(halflife=half_life, adjust=False, min_periods=min_periods or 0).std()
    if annualization is not None:
        vol = vol * np.sqrt(annualization)
    return vol


def rolling_correlation(
    returns: pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling pairwise correlation across N asset return series.

    Parameters
    ----------
    returns : DataFrame
        Wide-format return panel: rows are dates, columns are assets.
    window : int
        Trailing window length in observations.
    min_periods : int, optional
        Minimum number of overlapping non-NaN observations required within
        the window for a pair to produce a correlation. Defaults to
        ``window``.

    Returns
    -------
    DataFrame
        A ``MultiIndex`` frame indexed by ``(date, asset)`` with one column
        per asset, i.e. ``result.loc[date]`` is the ``N x N`` correlation
        matrix for that date. This is pandas' native rolling-correlation
        output shape, kept as-is because it is directly usable for
        research (e.g. ``result.loc[some_date, "AAPL"]`` gives AAPL's
        correlation with every other asset on that date) without an extra
        reshape step.

    Raises
    ------
    ValueError
        If ``window`` is not positive, ``min_periods`` exceeds ``window``,
        the columns contain duplicates, or the index is not monotonic
        and duplicate-free.

    Notes
    -----
    This is a thin wrapper over
    ``pandas.DataFrame.rolling(window, min_periods).corr()``. The
    pairwise computation itself is performed by pandas' compiled
    (Cython) implementation rather than by any Python-level loop in this
    package; there is no per-date or per-pair Python iteration here.
    Correlations are symmetric by construction and bounded in
    ``[-1, 1]`` up to floating-point tolerance; the diagonal is ``1.0``
    wherever a self-correlation is defined.
    """
    check_window(window)
    check_min_periods(min_periods, window)
    check_no_duplicate_columns(returns, "returns")
    check_monotonic_index(returns.index, "returns index")

    effective_min_periods = window if min_periods is None else min_periods
    return returns.rolling(window=window, min_periods=effective_min_periods).corr()

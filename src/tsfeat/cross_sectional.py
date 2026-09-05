"""Cross-sectional (same-date, across-asset) transforms.

Every function here operates independently, row by row, across a wide
``date x asset`` panel. No date ever borrows information from another date:
each row's statistics (mean, rank, quantile) are computed only from the
other columns in that same row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tsfeat.validation import check_no_duplicate_columns, check_quantiles

__all__ = ["cross_sectional_rank", "cross_sectional_demean", "grouped_winsorize"]


def cross_sectional_rank(
    panel: pd.DataFrame,
    *,
    pct: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """Rank assets independently within each date (row).

    Parameters
    ----------
    panel : DataFrame
        Wide-format panel: rows are dates, columns are assets.
    pct : bool, default True
        If ``True``, ranks are normalised to ``(0, 1]`` percentile ranks.
        If ``False``, ranks are integer-valued position ranks (``1..n``
        per row, with ties handled per the ``average`` method below).
    ascending : bool, default True
        If ``True`` (default), the smallest value in a row receives the
        lowest rank. Pass ``False`` to rank largest-first.

    Returns
    -------
    DataFrame
        Same shape and index as ``panel``. A ``NaN`` input is never
        assigned a rank and remains ``NaN`` in the output; it is also
        excluded when ranking the other values in that row.

    Notes
    -----
    Ranking is computed with ``panel.rank(axis=1, method="average")``:
    ties within a row receive the average of the ranks they would
    otherwise occupy — deterministic and symmetric, rather than
    order-of-appearance ("first") tie-breaking. This is a pure ``axis=1``
    pandas operation: no per-date Python loop and no cross-date
    information sharing, since each row's rank is computed independently
    of every other row.
    """
    check_no_duplicate_columns(panel, "panel")
    return panel.rank(axis=1, pct=pct, ascending=ascending, method="average")


def cross_sectional_demean(panel: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional (same-date) mean from every observation.

    ``demeaned(asset, date) = signal(asset, date) - mean_across_assets(date)``

    Parameters
    ----------
    panel : DataFrame
        Wide-format panel: rows are dates, columns are assets.

    Returns
    -------
    DataFrame
        Same shape and index as ``panel``. ``NaN`` values are excluded
        from the row mean and remain ``NaN`` in the output; for any row
        with at least one valid observation, the mean of the non-NaN
        outputs in that row is (up to floating-point tolerance) zero.

    Notes
    -----
    Vectorised via ``panel.sub(panel.mean(axis=1), axis=0)`` — one row
    mean per date, broadcast back across that row's columns. No
    information from one date's cross-section ever affects another
    date's demeaned values.
    """
    check_no_duplicate_columns(panel, "panel")
    row_mean = panel.mean(axis=1)
    return panel.sub(row_mean, axis=0)


def grouped_winsorize(
    values: pd.Series,
    groups: pd.Series | np.ndarray[Any, Any],
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorize ``values`` using group-specific quantile clip bounds.

    Within each group (e.g. all assets on a given date, or a
    ``(date, sector)`` composite key), values below the group's ``lower``
    quantile are clipped up to that quantile, and values above the
    group's ``upper`` quantile are clipped down to it.

    Parameters
    ----------
    values : Series
        Observations to winsorize.
    groups : Series or ndarray
        Group label per observation, same length and index alignment as
        ``values``. Pass a date column for date-level winsorization, a
        sector column for sector-level, or a combined key (e.g.
        ``date.astype(str) + "_" + sector``) for ``(date, sector)``.
    lower : float, default 0.01
        Lower quantile, in ``[0, 1]``.
    upper : float, default 0.99
        Upper quantile, in ``[0, 1]``. Must exceed ``lower``.

    Returns
    -------
    Series
        Same shape and index as ``values``.

    Raises
    ------
    ValueError
        If ``lower``/``upper`` are outside ``[0, 1]``, or ``lower >= upper``.

    Notes
    -----
    Quantiles are computed per group with
    ``groupby(groups).transform(...)`` — a vectorised group reduction, not
    a Python loop over group labels. ``NaN`` values are excluded from the
    quantile computation (pandas' default) and remain ``NaN`` after
    clipping. A group of size 1 is a no-op: its lower and upper quantile
    both equal its single value, so clipping leaves it unchanged.
    """
    check_quantiles(lower, upper)
    grouped = values.groupby(groups)
    lower_bound = grouped.transform(lambda s: s.quantile(lower))
    upper_bound = grouped.transform(lambda s: s.quantile(upper))
    return values.clip(lower=lower_bound, upper=upper_bound)

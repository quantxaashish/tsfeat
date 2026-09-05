"""Forward-return construction.

This is the module most directly responsible for lookahead-bias safety:
a forward return reported at index ``t`` is the *outcome* observed between
``t`` and ``t + h``, and is meant to be evaluated against a signal that was
itself only a function of information available at or before ``t``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, cast, overload

import numpy as np
import pandas as pd

from tsfeat.validation import check_horizons, check_no_duplicate_columns

SeriesOrFrame = TypeVar("SeriesOrFrame", pd.Series, pd.DataFrame)

__all__ = ["forward_returns"]


@overload
def forward_returns(
    prices: pd.Series, horizons: Sequence[int], *, log: bool = False
) -> pd.DataFrame: ...
@overload
def forward_returns(
    prices: pd.DataFrame, horizons: Sequence[int], *, log: bool = False
) -> pd.DataFrame: ...


def forward_returns(
    prices: pd.Series | pd.DataFrame,
    horizons: Sequence[int],
    *,
    log: bool = False,
) -> pd.DataFrame:
    """Forward returns at one or more horizons, aligned at the origin date.

    For horizon ``h``, the value reported at index ``t`` is the return
    realised from ``t`` to ``t + h``:

    ``fwd_h(t) = price(t + h) / price(t) - 1``    (or ``log(price(t+h) / price(t))``
    if ``log=True``)

    It is indexed at ``t`` because it represents the future outcome
    *associated with* information available at ``t`` — it is not itself
    computed from any future information.

    Parameters
    ----------
    prices : Series or DataFrame
        Price levels indexed by time. A DataFrame is treated as one
        independent asset per column.
    horizons : sequence of int
        Forward horizons in observations, e.g. ``[1, 5, 10, 20]``.
    log : bool, default False
        If ``True``, compute log returns (``log(p_{t+h} / p_t)``) instead
        of simple returns.

    Returns
    -------
    DataFrame
        For a ``Series`` input: columns named ``fwd_{h}`` for each horizon.
        For a ``DataFrame`` input: ``MultiIndex`` columns
        ``(fwd_{h}, asset)``, so ``result["fwd_5"]`` recovers the
        horizon-5 forward return for every asset. In both cases the index
        is identical to ``prices.index``.

    Raises
    ------
    ValueError
        If ``horizons`` is empty or contains a non-positive value.

    Notes
    -----
    **Lookahead safety**: this function's entire purpose is to compute an
    *outcome*, so it is expected and correct that ``fwd_h(t)`` depends on
    ``price(t+h)``. What must never happen — and is asserted by this
    package's test suite — is a sign or shift error that would let
    ``fwd_h(t)`` leak into the *signal* at ``t`` itself; keep signal
    construction and forward-return construction in separate variables.

    The implementation is ``prices.shift(-h) / prices - 1``: shifting by a
    *negative* amount pulls a future row backward to align with the
    current one. The final ``h`` rows of ``fwd_h`` are ``NaN`` because no
    ``t + h`` observation exists for them — they are left missing, never
    backfilled or wrapped around.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.Series([100.0, 105.0, 110.0, 121.0])
    >>> forward_returns(prices, horizons=[1, 2]).round(4)["fwd_1"].tolist()
    [0.05, 0.0476, 0.1, nan]
    """
    check_horizons(horizons)
    if isinstance(prices, pd.DataFrame):
        check_no_duplicate_columns(prices, "prices")

    def _forward_return(h: int) -> pd.Series | pd.DataFrame:
        future = prices.shift(-h)
        ratio = future / prices
        if log:
            return cast(
                "pd.Series | pd.DataFrame",
                np.log(ratio),
            )
        return ratio - 1

    if isinstance(prices, pd.Series):
        data = {f"fwd_{h}": _forward_return(h) for h in horizons}
        return pd.DataFrame(data, index=prices.index)

    columns = pd.MultiIndex.from_product(
        [[f"fwd_{h}" for h in horizons], prices.columns],
        names=["horizon", "asset"],
    )
    blocks = [_forward_return(h) for h in horizons]
    result = pd.concat(blocks, axis=1, keys=[f"fwd_{h}" for h in horizons])
    result.columns = columns
    return result

"""Internal input-validation helpers.

These are intentionally not exported from :mod:`tsfeat`. They exist to make
every public function fail fast and loudly on malformed input rather than
silently repairing it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def check_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int | np.integer) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def check_window(window: int) -> None:
    check_positive_int(window, "window")


def check_min_periods(min_periods: int | None, window: int) -> None:
    if min_periods is None:
        return
    check_positive_int(min_periods, "min_periods")
    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) cannot exceed window ({window})")


def check_half_life(half_life: float) -> None:
    if not isinstance(half_life, int | float | np.integer | np.floating) or isinstance(
        half_life, bool
    ):
        raise TypeError(f"half_life must be numeric, got {type(half_life).__name__}")
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")


def check_ddof(ddof: int) -> None:
    if not isinstance(ddof, int | np.integer) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative integer, got {ddof}")


def check_quantiles(lower: float, upper: float) -> None:
    for name, value in (("lower", lower), ("upper", upper)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be within [0, 1], got {value}")
    if lower >= upper:
        raise ValueError(f"lower quantile ({lower}) must be < upper quantile ({upper})")


def check_horizons(horizons: Sequence[int]) -> None:
    if len(horizons) == 0:
        raise ValueError("horizons must not be empty")
    for h in horizons:
        check_positive_int(h, "each horizon")


def check_monotonic_index(index: pd.Index, name: str = "index") -> None:
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be monotonically increasing")
    if index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate labels")


def check_no_duplicate_columns(frame: pd.DataFrame, name: str = "frame") -> None:
    if frame.columns.has_duplicates:
        dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"{name} has duplicate columns: {dupes}")


def check_method(method: str, valid: Sequence[str]) -> None:
    if method not in valid:
        raise ValueError(f"method must be one of {list(valid)}, got {method!r}")

import pandas as pd
import pytest

from tsfeat.validation import (
    check_half_life,
    check_horizons,
    check_method,
    check_monotonic_index,
    check_no_duplicate_columns,
    check_positive_int,
    check_quantiles,
)


def test_check_positive_int_wrong_type_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        check_positive_int(3.5, "window")  # type: ignore[arg-type]


def test_check_positive_int_bool_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        check_positive_int(True, "window")


def test_check_half_life_wrong_type_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        check_half_life("3")  # type: ignore[arg-type]


def test_check_monotonic_index_duplicates_raises() -> None:
    with pytest.raises(ValueError):
        check_monotonic_index(pd.Index([1, 1, 2]))


def test_check_monotonic_index_non_monotonic_raises() -> None:
    with pytest.raises(ValueError):
        check_monotonic_index(pd.Index([2, 1, 3]))


def test_check_monotonic_index_valid_passes() -> None:
    check_monotonic_index(pd.Index([1, 2, 3]))


def test_check_no_duplicate_columns_passes_for_unique() -> None:
    check_no_duplicate_columns(pd.DataFrame({"A": [1], "B": [2]}))


def test_check_method_invalid_raises() -> None:
    with pytest.raises(ValueError):
        check_method("kendall", ("pearson", "spearman"))


def test_check_horizons_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        check_horizons([1, 0])


def test_check_quantiles_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        check_quantiles(0.0, 1.5)

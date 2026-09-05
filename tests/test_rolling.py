import numpy as np
import pandas as pd
import pytest

from tsfeat.rolling import ew_volatility, rolling_correlation, rolling_zscore


class TestRollingZscore:
    def test_reference_formula(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        window = 3
        result = rolling_zscore(s, window=window, min_periods=window)
        expected = (s - s.rolling(window).mean()) / s.rolling(window).std(ddof=1)
        pd.testing.assert_series_equal(result, expected)

    def test_min_periods_guard_produces_nan(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_zscore(s, window=3, min_periods=3)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2:].notna().all()

    def test_default_min_periods_equals_window(self) -> None:
        s = pd.Series(np.arange(10, dtype=float))
        result = rolling_zscore(s, window=4)
        assert result.iloc[:3].isna().all()
        assert result.iloc[3:].notna().all()

    def test_constant_series_is_nan_not_inf(self) -> None:
        s = pd.Series([5.0] * 10)
        result = rolling_zscore(s, window=3, min_periods=3)
        assert result.iloc[2:].isna().all()
        assert not np.isinf(result.dropna()).any()

    def test_window_one(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        result = rolling_zscore(s, window=1, min_periods=1)
        # std of a single observation is undefined (NaN) under ddof=1
        assert result.isna().all()

    def test_missing_observations_are_skipped_within_window(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        result = rolling_zscore(s, window=3, min_periods=2)
        assert not result.isna().all()

    def test_extreme_values_finite(self) -> None:
        s = pd.Series([1e300, -1e300, 1e300, -1e300, 1e300])
        result = rolling_zscore(s, window=3, min_periods=3)
        assert np.isfinite(result.dropna()).all()

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5)
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        result = rolling_zscore(s, window=3, min_periods=3)
        pd.testing.assert_index_equal(result.index, idx)

    def test_dataframe_support(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2, 3, 4, 5], "B": [5.0, 4, 3, 2, 1]})
        result = rolling_zscore(df, window=3, min_periods=3)
        assert list(result.columns) == ["A", "B"]
        assert result["A"].iloc[2:].notna().all()

    @pytest.mark.parametrize("bad_window", [0, -1])
    def test_invalid_window_raises(self, bad_window: int) -> None:
        with pytest.raises(ValueError):
            rolling_zscore(pd.Series([1.0, 2.0]), window=bad_window)

    def test_min_periods_exceeding_window_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_zscore(pd.Series([1.0, 2.0, 3.0]), window=2, min_periods=3)

    def test_negative_ddof_raises(self) -> None:
        with pytest.raises(ValueError):
            rolling_zscore(pd.Series([1.0, 2.0, 3.0]), window=2, ddof=-1)


class TestEwVolatility:
    def test_matches_pandas_ewm_directly(self) -> None:
        r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
        result = ew_volatility(r, half_life=3, min_periods=2)
        expected = r.ewm(halflife=3, adjust=False, min_periods=2).std()
        pd.testing.assert_series_equal(result, expected)

    def test_constant_series_gives_zero_not_nan(self) -> None:
        r = pd.Series([0.01] * 6)
        result = ew_volatility(r, half_life=3, min_periods=2)
        assert (result.dropna() == 0.0).all()
        assert result.notna().sum() == 5

    def test_annualization_scales_by_sqrt(self) -> None:
        r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
        raw = ew_volatility(r, half_life=3, min_periods=2)
        annualized = ew_volatility(r, half_life=3, min_periods=2, annualization=252)
        pd.testing.assert_series_equal(annualized, raw * np.sqrt(252))

    def test_invalid_half_life_raises(self) -> None:
        with pytest.raises(ValueError):
            ew_volatility(pd.Series([0.01, 0.02]), half_life=0)

    def test_invalid_annualization_raises(self) -> None:
        with pytest.raises(ValueError):
            ew_volatility(pd.Series([0.01, 0.02]), half_life=3, annualization=-1)

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6)
        r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01], index=idx)
        result = ew_volatility(r, half_life=3)
        pd.testing.assert_index_equal(result.index, idx)

    def test_dataframe_support(self) -> None:
        df = pd.DataFrame({"A": [0.01, -0.02, 0.015], "B": [0.02, 0.01, -0.01]})
        result = ew_volatility(df, half_life=3, min_periods=2)
        assert list(result.columns) == ["A", "B"]

    def test_dataframe_duplicate_columns_raise(self) -> None:
        df = pd.DataFrame([[0.01, 0.02]], columns=["A", "A"])
        with pytest.raises(ValueError):
            ew_volatility(df, half_life=3)


class TestRollingCorrelation:
    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(50, 3)), columns=["A", "B", "C"])
        result = rolling_correlation(df, window=10, min_periods=10)
        last = result.loc[df.index[-1]]
        for col in df.columns:
            assert last.loc[col, col] == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["A", "B", "C"])
        result = rolling_correlation(df, window=8, min_periods=8)
        last_date = df.index[-1]
        block = result.loc[last_date]
        np.testing.assert_allclose(block.to_numpy(), block.to_numpy().T, atol=1e-10)

    def test_bounded(self) -> None:
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("ABCD"))
        result = rolling_correlation(df, window=10, min_periods=10)
        values = result.to_numpy()
        finite = values[np.isfinite(values)]
        assert (finite >= -1 - 1e-8).all()
        assert (finite <= 1 + 1e-8).all()

    def test_duplicate_columns_raise(self) -> None:
        df = pd.DataFrame([[1.0, 2.0]], columns=["A", "A"])
        with pytest.raises(ValueError):
            rolling_correlation(df, window=1)

    def test_non_monotonic_index_raises(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=[1, 0])
        with pytest.raises(ValueError):
            rolling_correlation(df, window=1)

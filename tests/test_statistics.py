import numpy as np
import pandas as pd
import pytest

from tsfeat.returns import forward_returns
from tsfeat.statistics import DrawdownStats, cumulative_vwap, ic_decay, max_drawdown, vwap


class TestVwap:
    def test_hand_calculated_single_group(self) -> None:
        trades = pd.DataFrame({"price": [10.0, 20.0], "volume": [100.0, 100.0]})
        result = vwap(trades)
        assert result == pytest.approx(15.0)

    def test_duplicate_timestamps_aggregate_correctly(self) -> None:
        # three trades at the SAME timestamp plus one at another; grouped
        # VWAP by timestamp should sum notional/volume within each group.
        trades = pd.DataFrame(
            {
                "timestamp": [1, 1, 1, 2],
                "price": [10.0, 11.0, 9.0, 50.0],
                "volume": [100.0, 50.0, 50.0, 10.0],
            }
        )
        result = vwap(trades, group="timestamp")
        expected_ts1 = (10 * 100 + 11 * 50 + 9 * 50) / (100 + 50 + 50)
        assert result.loc[1] == pytest.approx(expected_ts1)
        assert result.loc[2] == pytest.approx(50.0)

    def test_zero_volume_group_is_nan(self) -> None:
        trades = pd.DataFrame({"timestamp": [1, 1], "price": [10.0, 11.0], "volume": [0.0, 0.0]})
        result = vwap(trades, group="timestamp")
        assert np.isnan(result.loc[1])

    def test_zero_volume_overall_is_nan(self) -> None:
        trades = pd.DataFrame({"price": [10.0, 20.0], "volume": [0.0, 0.0]})
        result = vwap(trades)
        assert np.isnan(result)

    def test_missing_price_or_volume_excluded(self) -> None:
        trades = pd.DataFrame({"price": [10.0, np.nan, 30.0], "volume": [100.0, 100.0, 100.0]})
        result = vwap(trades)
        assert result == pytest.approx((10 * 100 + 30 * 100) / 200)

    def test_missing_required_column_raises(self) -> None:
        trades = pd.DataFrame({"price": [10.0]})
        with pytest.raises(ValueError):
            vwap(trades)


class TestCumulativeVwap:
    def test_duplicate_timestamps_merge_before_cumsum(self) -> None:
        trades = pd.DataFrame(
            {
                "timestamp": [1, 1, 2],
                "price": [10.0, 20.0, 30.0],
                "volume": [100.0, 100.0, 100.0],
            }
        )
        result = cumulative_vwap(trades)
        # one unique timestamp of value 1 combining both trades
        assert len(result) == 2
        assert result.loc[1] == pytest.approx(15.0)
        expected_ts2 = (10 * 100 + 20 * 100 + 30 * 100) / 300
        assert result.loc[2] == pytest.approx(expected_ts2)

    def test_row_order_does_not_affect_tied_timestamp_result(self) -> None:
        trades_a = pd.DataFrame(
            {"timestamp": [1, 1], "price": [10.0, 20.0], "volume": [100.0, 200.0]}
        )
        trades_b = trades_a.iloc[::-1].reset_index(drop=True)
        result_a = cumulative_vwap(trades_a)
        result_b = cumulative_vwap(trades_b)
        pd.testing.assert_series_equal(result_a, result_b)

    def test_unsorted_input_is_sorted_chronologically(self) -> None:
        trades = pd.DataFrame(
            {"timestamp": [3, 1, 2], "price": [30.0, 10.0, 20.0], "volume": [1.0, 1.0, 1.0]}
        )
        result = cumulative_vwap(trades)
        assert list(result.index) == [1, 2, 3]

    def test_missing_timestamp_column_raises(self) -> None:
        trades = pd.DataFrame({"price": [10.0], "volume": [1.0]})
        with pytest.raises(ValueError):
            cumulative_vwap(trades)


class TestMaxDrawdown:
    def test_monotonically_rising_series(self) -> None:
        wealth = pd.Series([100.0, 110.0, 120.0, 130.0])
        result = max_drawdown(wealth)
        assert result.max_drawdown == pytest.approx(0.0)
        assert result.max_duration == 0

    def test_monotonically_falling_series_unrecovered(self) -> None:
        wealth = pd.Series([100.0, 90.0, 80.0, 70.0])
        result = max_drawdown(wealth)
        assert result.max_drawdown == pytest.approx(0.7 - 1.0)
        assert pd.isna(result.recovery_date)
        assert result.trough_date == wealth.index[-1]
        assert result.peak_date == wealth.index[0]

    def test_hand_calculated_drawdown(self) -> None:
        wealth = pd.Series([100.0, 120.0, 90.0, 96.0, 130.0])
        result = max_drawdown(wealth)
        # peak at 120 (index 1), trough at 90 (index 2): 90/120 - 1 = -0.25
        assert result.max_drawdown == pytest.approx(-0.25)
        assert result.peak_date == 1
        assert result.trough_date == 2
        # recovers at index 4 (130 >= 120)
        assert result.recovery_date == 4

    def test_max_drawdown_never_positive(self) -> None:
        rng = np.random.default_rng(0)
        wealth = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=200)))
        result = max_drawdown(wealth)
        assert result.max_drawdown <= 0

    def test_datetime_index_duration_is_timedelta(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        wealth = pd.Series([100.0, 120.0, 90.0, 96.0, 130.0], index=idx)
        result = max_drawdown(wealth)
        assert isinstance(result.max_duration, pd.Timedelta)

    def test_missing_data_does_not_crash(self) -> None:
        wealth = pd.Series([100.0, np.nan, 90.0, 110.0])
        result = max_drawdown(wealth)
        assert isinstance(result, DrawdownStats)

    def test_non_monotonic_index_raises(self) -> None:
        wealth = pd.Series([100.0, 90.0], index=[1, 0])
        with pytest.raises(ValueError):
            max_drawdown(wealth)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError):
            max_drawdown(pd.Series([], dtype=float))


class TestIcDecay:
    def _panel(self, n_dates: int, n_assets: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.normal(size=(n_dates, n_assets)),
            columns=[f"A{i}" for i in range(n_assets)],
        )

    def test_perfect_signal_gives_ic_near_one(self) -> None:
        n_dates, n_assets = 60, 8
        fwd_vals = self._panel(n_dates, n_assets, seed=1)
        fwd = pd.concat({"fwd_1": fwd_vals}, axis=1)
        signal = fwd_vals.copy()  # signal exactly matches future return rank
        result = ic_decay(signal, fwd, method="spearman", min_assets=3)
        assert result.loc["fwd_1", "mean_ic"] == pytest.approx(1.0, abs=1e-8)

    def test_reversed_signal_gives_ic_near_negative_one(self) -> None:
        n_dates, n_assets = 60, 8
        fwd_vals = self._panel(n_dates, n_assets, seed=2)
        fwd = pd.concat({"fwd_1": fwd_vals}, axis=1)
        signal = -fwd_vals
        result = ic_decay(signal, fwd, method="spearman", min_assets=3)
        assert result.loc["fwd_1", "mean_ic"] == pytest.approx(-1.0, abs=1e-8)

    def test_constant_signal_gives_nan_ic(self) -> None:
        n_dates, n_assets = 20, 6
        fwd_vals = self._panel(n_dates, n_assets, seed=3)
        fwd = pd.concat({"fwd_1": fwd_vals}, axis=1)
        signal = pd.DataFrame(np.ones((n_dates, n_assets)), columns=fwd_vals.columns)
        result = ic_decay(signal, fwd, method="spearman", min_assets=3)
        assert np.isnan(result.loc["fwd_1", "mean_ic"])

    def test_insufficient_assets_gives_nan(self) -> None:
        signal = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
        fwd_vals = pd.DataFrame({"A": [0.1, 0.2], "B": [0.2, 0.1]})
        fwd = pd.concat({"fwd_1": fwd_vals}, axis=1)
        result = ic_decay(signal, fwd, method="spearman", min_assets=3)
        assert np.isnan(result.loc["fwd_1", "mean_ic"])
        assert result.loc["fwd_1", "n_observations"] == 0

    def test_multiple_horizons_produce_multiple_rows(self) -> None:
        prices = pd.DataFrame(
            100 + np.cumsum(self._panel(50, 5, seed=4).to_numpy(), axis=0),
            columns=[f"A{i}" for i in range(5)],
        )
        fwd = forward_returns(prices, horizons=[1, 5, 10])
        signal = self._panel(50, 5, seed=5)
        result = ic_decay(signal, fwd, method="spearman")
        assert list(result.index) == ["fwd_1", "fwd_5", "fwd_10"]

    def test_invalid_method_raises(self) -> None:
        signal = pd.DataFrame({"A": [1.0], "B": [2.0]})
        fwd = pd.concat({"fwd_1": pd.DataFrame({"A": [0.1], "B": [0.2]})}, axis=1)
        with pytest.raises(ValueError):
            ic_decay(signal, fwd, method="kendall")

import numpy as np
import pandas as pd
import pytest

from tsfeat.returns import forward_returns


class TestForwardReturnsHandComputed:
    def test_single_horizon_hand_calculated(self) -> None:
        # 100 -> 105 -> 110 -> 121
        prices = pd.Series([100.0, 105.0, 110.0, 121.0])
        result = forward_returns(prices, horizons=[1])

        expected = [0.05, 110 / 105 - 1, 121 / 110 - 1, np.nan]
        np.testing.assert_allclose(result["fwd_1"].to_numpy()[:-1], expected[:-1])
        assert np.isnan(result["fwd_1"].iloc[-1])

    def test_multi_horizon_hand_calculated(self) -> None:
        prices = pd.Series([100.0, 105.0, 110.0, 121.0])
        result = forward_returns(prices, horizons=[1, 2])

        np.testing.assert_allclose(result["fwd_1"].iloc[0], 0.05)
        np.testing.assert_allclose(result["fwd_2"].iloc[0], 110 / 100 - 1)
        assert result["fwd_2"].iloc[2:].isna().all()

    def test_log_returns_hand_calculated(self) -> None:
        prices = pd.Series([100.0, 110.0])
        result = forward_returns(prices, horizons=[1], log=True)
        np.testing.assert_allclose(result["fwd_1"].iloc[0], np.log(110 / 100))

    def test_last_h_rows_are_nan(self) -> None:
        prices = pd.Series(np.arange(1, 21, dtype=float))
        result = forward_returns(prices, horizons=[5])
        assert result["fwd_5"].iloc[-5:].isna().all()
        assert result["fwd_5"].iloc[:-5].notna().all()

    def test_constant_prices_give_zero_return(self) -> None:
        prices = pd.Series([50.0] * 10)
        result = forward_returns(prices, horizons=[1, 3])
        assert (result["fwd_1"].dropna() == 0.0).all()
        assert (result["fwd_3"].dropna() == 0.0).all()


class TestForwardReturnsShiftDirection:
    """Dedicated regression test for the single most dangerous bug class:
    an accidental sign flip in the shift direction, which would silently
    turn a forward-looking outcome into a backward-looking (leaked) one.
    """

    def test_shift_direction_is_forward_not_backward(self) -> None:
        # Strictly increasing prices: a *forward* return must be positive.
        # A backward-shift bug would instead report the *trailing* return,
        # which for this monotonically increasing series would also be
        # positive but at the WRONG index, and would fail to go NaN at the
        # correct (final) end of the series.
        prices = pd.Series([10.0, 20.0, 40.0, 80.0, 160.0])
        result = forward_returns(prices, horizons=[1])

        # correct forward semantics: fwd_1(t=0) uses price(1)/price(0)
        assert result["fwd_1"].iloc[0] == pytest.approx(20 / 10 - 1)
        # last row must be NaN (no t+1 available), not filled by wraparound
        assert np.isnan(result["fwd_1"].iloc[-1])
        # first row must NOT be NaN (a backward-shift bug would make the
        # *first* row NaN instead of the last)
        assert result["fwd_1"].notna().iloc[0]

    def test_no_leakage_into_earlier_index(self) -> None:
        # Construct a series where only the LAST value is an outlier.
        # A shift-direction bug would smear that outlier into fwd_1 at an
        # earlier index than it should legitimately appear.
        prices = pd.Series([100.0, 100.0, 100.0, 100.0, 1_000_000.0])
        result = forward_returns(prices, horizons=[1])
        # the outlier may only affect fwd_1 at index 3 (100 -> 1e6)
        assert result["fwd_1"].iloc[3] > 100
        # index 0, 1 must be unaffected (flat 100 -> 100 => 0 return)
        assert result["fwd_1"].iloc[0] == pytest.approx(0.0)
        assert result["fwd_1"].iloc[1] == pytest.approx(0.0)


class TestForwardReturnsMultiAsset:
    def test_multiindex_columns(self) -> None:
        prices = pd.DataFrame({"AAPL": [100.0, 105.0, 110.0], "MSFT": [200.0, 190.0, 210.0]})
        result = forward_returns(prices, horizons=[1])
        assert isinstance(result.columns, pd.MultiIndex)
        assert set(result.columns.get_level_values(0)) == {"fwd_1"}
        assert set(result.columns.get_level_values(1)) == {"AAPL", "MSFT"}
        np.testing.assert_allclose(result[("fwd_1", "AAPL")].iloc[0], 0.05)

    def test_duplicate_columns_raise(self) -> None:
        prices = pd.DataFrame([[1.0, 2.0]], columns=["A", "A"])
        with pytest.raises(ValueError):
            forward_returns(prices, horizons=[1])


class TestForwardReturnsValidation:
    def test_empty_horizons_raises(self) -> None:
        with pytest.raises(ValueError):
            forward_returns(pd.Series([1.0, 2.0]), horizons=[])

    @pytest.mark.parametrize("bad_horizon", [0, -1])
    def test_non_positive_horizon_raises(self, bad_horizon: int) -> None:
        with pytest.raises(ValueError):
            forward_returns(pd.Series([1.0, 2.0]), horizons=[bad_horizon])

    def test_horizon_longer_than_series_is_all_nan_not_error(self) -> None:
        prices = pd.Series([1.0, 2.0, 3.0])
        result = forward_returns(prices, horizons=[10])
        assert result["fwd_10"].isna().all()

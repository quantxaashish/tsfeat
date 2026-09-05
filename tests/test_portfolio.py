import numpy as np
import pandas as pd
import pytest

from tsfeat.portfolio import signal_turnover


class TestSignalTurnover:
    def test_identical_portfolios_zero_turnover(self) -> None:
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
        result = signal_turnover(w)
        assert np.isnan(result.iloc[0])
        assert (result.iloc[1:] == 0.0).all()

    def test_completely_changed_portfolio_known_turnover(self) -> None:
        # from 100% A to 100% B: |1-0| + |0-1| = 2, one-way = 1.0
        w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
        result = signal_turnover(w, one_way=True)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_raw_convention_doubles_one_way(self) -> None:
        w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
        one_way = signal_turnover(w, one_way=True)
        raw = signal_turnover(w, one_way=False)
        assert raw.iloc[1] == pytest.approx(2 * one_way.iloc[1])

    def test_long_only_normalized_weights(self) -> None:
        w = pd.DataFrame({"A": [0.6, 0.3], "B": [0.4, 0.7]})
        result = signal_turnover(w)
        # 0.5 * (|0.3-0.6| + |0.7-0.4|) = 0.5 * 0.6 = 0.3
        assert result.iloc[1] == pytest.approx(0.3)

    def test_long_short_portfolio(self) -> None:
        w = pd.DataFrame({"A": [1.0, -1.0], "B": [-1.0, 1.0]})
        result = signal_turnover(w, one_way=True)
        # 0.5 * (|-1-1| + |1-(-1)|) = 0.5*4 = 2.0
        assert result.iloc[1] == pytest.approx(2.0)

    def test_entering_and_exiting_positions_treated_as_zero(self) -> None:
        w = pd.DataFrame({"A": [1.0, 0.0], "B": [np.nan, 1.0]})
        result = signal_turnover(w)
        # A: |0-1|=1 ; B: NaN treated as 0 held, then 1: |1-0|=1
        # one-way = 0.5*(1+1) = 1.0
        assert result.iloc[1] == pytest.approx(1.0)

    def test_first_row_is_nan(self) -> None:
        w = pd.DataFrame({"A": [1.0, 0.5, 0.2], "B": [0.0, 0.5, 0.8]})
        result = signal_turnover(w)
        assert np.isnan(result.iloc[0])

    def test_duplicate_columns_raise(self) -> None:
        w = pd.DataFrame([[1.0, 2.0]], columns=["A", "A"])
        with pytest.raises(ValueError):
            signal_turnover(w)

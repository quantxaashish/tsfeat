"""Property-based tests (Hypothesis) for invariants stronger than fixed examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tsfeat.cross_sectional import cross_sectional_demean, cross_sectional_rank
from tsfeat.portfolio import signal_turnover
from tsfeat.returns import forward_returns
from tsfeat.rolling import rolling_correlation, rolling_zscore
from tsfeat.statistics import max_drawdown

finite_floats = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
).filter(lambda x: x == 0.0 or abs(x) >= 1e-3)


def _price_series(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=n)))


@st.composite
def series_and_affine(draw: st.DrawFn) -> tuple[pd.Series, float, float]:
    n = draw(st.integers(min_value=15, max_value=60))
    values = draw(arrays(dtype=np.float64, shape=n, elements=finite_floats))
    a = draw(st.floats(min_value=0.1, max_value=50, allow_nan=False))
    b = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
    return pd.Series(values), a, b


class TestRollingZscoreProperties:
    @given(series_and_affine())
    @settings(max_examples=50)
    def test_invariant_under_positive_affine_transform(
        self, data: tuple[pd.Series, float, float]
    ) -> None:
        s, a, b = data
        window = 5
        z1 = rolling_zscore(s, window=window, min_periods=window)
        z2 = rolling_zscore(a * s + b, window=window, min_periods=window)
        # compare only where both sides are finite (non-degenerate windows)
        both_finite = np.isfinite(z1.to_numpy()) & np.isfinite(z2.to_numpy())
        assume(both_finite.any())
        # A generous tolerance is deliberate: rolling variance computed over
        # a window with wide within-window dynamic range is subject to
        # ordinary floating-point catastrophic cancellation (this affects
        # any sum-of-squares-based variance algorithm, pandas' included),
        # which the affine transform can amplify. That is a floating-point
        # property of the underlying arithmetic, not a correctness bug in
        # this wrapper.
        np.testing.assert_allclose(
            z1.to_numpy()[both_finite], z2.to_numpy()[both_finite], atol=1e-4, rtol=1e-4
        )


class TestCrossSectionalDemeanProperties:
    @given(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=50)
    def test_row_mean_is_zero_whenever_defined(
        self, n_dates: int, n_assets: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        panel = pd.DataFrame(rng.normal(size=(n_dates, n_assets)))
        result = cross_sectional_demean(panel)
        row_means = result.mean(axis=1)
        np.testing.assert_allclose(row_means.to_numpy(), 0.0, atol=1e-8)


class TestCorrelationProperties:
    @given(
        st.integers(min_value=15, max_value=40),
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=30)
    def test_symmetric_and_bounded(self, n_dates: int, n_assets: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        cols = [f"A{i}" for i in range(n_assets)]
        panel = pd.DataFrame(rng.normal(size=(n_dates, n_assets)), columns=cols)
        window = min(10, n_dates)
        result = rolling_correlation(panel, window=window, min_periods=window)
        last_date = panel.index[-1]
        block = result.loc[last_date]
        values = block.to_numpy()
        finite = values[np.isfinite(values)]
        assert (finite >= -1 - 1e-8).all()
        assert (finite <= 1 + 1e-8).all()
        np.testing.assert_allclose(values, values.T, atol=1e-8, equal_nan=True)


class TestDrawdownProperties:
    @given(
        st.integers(min_value=2, max_value=100),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=50)
    def test_drawdown_never_positive(self, n: int, seed: int) -> None:
        wealth = _price_series(n, seed)
        result = max_drawdown(wealth)
        assert result.max_drawdown <= 1e-12


class TestForwardReturnsProperties:
    @given(
        st.integers(min_value=2, max_value=50),
        st.integers(min_value=1, max_value=10),
        st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_constant_price_gives_zero_return_and_trailing_nan(
        self, n: int, horizon: int, level: float
    ) -> None:
        prices = pd.Series([level] * n)
        result = forward_returns(prices, horizons=[horizon])
        col = result[f"fwd_{horizon}"]
        if horizon < n:
            np.testing.assert_allclose(col.iloc[: n - horizon].to_numpy(), 0.0, atol=1e-9)
        assert col.iloc[max(n - horizon, 0) :].isna().all()


class TestTurnoverProperties:
    @given(
        st.integers(min_value=2, max_value=20),
        st.integers(min_value=2, max_value=6),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=50)
    def test_unchanged_weights_have_zero_turnover(
        self, n_dates: int, n_assets: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        row = rng.dirichlet(np.ones(n_assets))
        weights = pd.DataFrame(np.tile(row, (n_dates, 1)))
        result = signal_turnover(weights)
        assert (result.iloc[1:].abs() < 1e-10).all()


class TestRankProperties:
    @given(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=50)
    def test_monotonic_transform_preserves_rank_when_no_ties(
        self, n_dates: int, n_assets: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        panel = pd.DataFrame(rng.normal(size=(n_dates, n_assets)))
        sorted_rows = np.sort(panel.to_numpy(), axis=1)
        assume(bool(np.all(np.diff(sorted_rows, axis=1) != 0)))
        ranks_before = cross_sectional_rank(panel, pct=False)
        transformed = panel * 2.5 + 3.0
        ranks_after = cross_sectional_rank(transformed, pct=False)
        pd.testing.assert_frame_equal(ranks_before, ranks_after)

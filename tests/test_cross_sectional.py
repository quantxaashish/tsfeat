import numpy as np
import pandas as pd
import pytest

from tsfeat.cross_sectional import cross_sectional_demean, cross_sectional_rank, grouped_winsorize


class TestCrossSectionalRank:
    def test_hand_calculated_no_ties(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [2.0]})
        result = cross_sectional_rank(panel, pct=False)
        assert result.loc[0, "A"] == 1.0
        assert result.loc[0, "C"] == 2.0
        assert result.loc[0, "B"] == 3.0

    def test_pct_rank_hand_calculated(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]})
        result = cross_sectional_rank(panel, pct=True)
        np.testing.assert_allclose(result.loc[0].to_numpy(), [0.25, 0.5, 0.75, 1.0])

    def test_ties_use_average_method(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [3.0]})
        result = cross_sectional_rank(panel, pct=False)
        assert result.loc[0, "A"] == result.loc[0, "B"] == 1.5
        assert result.loc[0, "C"] == 3.0

    def test_no_cross_date_leakage(self) -> None:
        # date 0 has a huge outlier; date 1 has completely different values.
        # Ranking must never let date 0's outlier influence date 1's ranks.
        panel = pd.DataFrame({"A": [1e9, 1.0], "B": [1.0, 2.0], "C": [2.0, 3.0]})
        result = cross_sectional_rank(panel, pct=False)
        assert result.loc[1].tolist() == [1.0, 2.0, 3.0]

    def test_missing_values_preserved_and_excluded(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [np.nan], "C": [3.0]})
        result = cross_sectional_rank(panel, pct=False)
        assert np.isnan(result.loc[0, "B"])
        assert result.loc[0, "A"] == 1.0
        assert result.loc[0, "C"] == 2.0

    def test_descending(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        result = cross_sectional_rank(panel, pct=False, ascending=False)
        assert result.loc[0, "C"] == 1.0
        assert result.loc[0, "A"] == 3.0

    def test_monotonic_transform_preserves_rank_no_ties(self) -> None:
        rng = np.random.default_rng(0)
        panel = pd.DataFrame(rng.normal(size=(5, 6)))
        ranks_before = cross_sectional_rank(panel, pct=False)
        transformed = panel * 3.0 + 7.0
        ranks_after = cross_sectional_rank(transformed, pct=False)
        pd.testing.assert_frame_equal(ranks_before, ranks_after)


class TestCrossSectionalDemean:
    def test_mean_is_zero_per_row(self) -> None:
        panel = pd.DataFrame({"A": [1.0, 4.0], "B": [2.0, 5.0], "C": [3.0, 9.0]})
        result = cross_sectional_demean(panel)
        np.testing.assert_allclose(result.sum(axis=1).to_numpy(), [0.0, 0.0], atol=1e-10)

    def test_hand_calculated(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        result = cross_sectional_demean(panel)
        np.testing.assert_allclose(result.loc[0].to_numpy(), [-1.0, 0.0, 1.0])

    def test_missing_values_excluded_from_mean(self) -> None:
        panel = pd.DataFrame({"A": [1.0], "B": [np.nan], "C": [3.0]})
        result = cross_sectional_demean(panel)
        assert np.isnan(result.loc[0, "B"])
        np.testing.assert_allclose(result.loc[0, ["A", "C"]].to_numpy(), [-1.0, 1.0])

    def test_no_cross_date_leakage(self) -> None:
        panel = pd.DataFrame({"A": [1000.0, 1.0], "B": [2000.0, 2.0], "C": [3000.0, 3.0]})
        result = cross_sectional_demean(panel)
        np.testing.assert_allclose(result.loc[1].to_numpy(), [-1.0, 0.0, 1.0])


class TestGroupedWinsorize:
    def test_clips_within_group(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0, 100.0, -100.0])
        groups = pd.Series(["g1", "g1", "g1", "g1", "g1"])
        result = grouped_winsorize(values, groups, lower=0.2, upper=0.8)
        assert result.max() < 100.0
        assert result.min() > -100.0

    def test_group_of_size_one_is_noop(self) -> None:
        values = pd.Series([42.0])
        groups = pd.Series(["only"])
        result = grouped_winsorize(values, groups, lower=0.01, upper=0.99)
        assert result.iloc[0] == 42.0

    def test_sector_neutral_example(self) -> None:
        # two sectors, each with an outlier that should only affect its own sector
        values = pd.Series([1.0, 2.0, 3.0, 1000.0, 10.0, 20.0, 30.0, -1000.0])
        sectors = pd.Series(["tech"] * 4 + ["energy"] * 4)
        result = grouped_winsorize(values, sectors, lower=0.1, upper=0.9)
        tech_result = result[sectors == "tech"]
        energy_result = result[sectors == "energy"]
        assert tech_result.max() < 1000.0
        assert energy_result.min() > -1000.0

    def test_nan_preserved(self) -> None:
        values = pd.Series([1.0, np.nan, 3.0, 100.0])
        groups = pd.Series(["g1"] * 4)
        result = grouped_winsorize(values, groups, lower=0.1, upper=0.9)
        assert np.isnan(result.iloc[1])

    def test_invalid_quantiles_raise(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])
        groups = pd.Series(["g1", "g1", "g1"])
        with pytest.raises(ValueError):
            grouped_winsorize(values, groups, lower=0.9, upper=0.1)

    def test_out_of_range_quantiles_raise(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])
        groups = pd.Series(["g1", "g1", "g1"])
        with pytest.raises(ValueError):
            grouped_winsorize(values, groups, lower=-0.1, upper=0.9)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.factors.scoring import (
    calibrate_factor_weights,
    composite_score,
    cross_sectional_zscore,
)


def _panel(n_dates: int, n_symbols: int, rng: np.random.Generator, scale: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    symbols = [f"S{i}" for i in range(n_symbols)]
    return pd.DataFrame(rng.normal(0, scale, (n_dates, n_symbols)), index=dates, columns=symbols)


class TestCrossSectionalZscore:
    def test_each_row_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        panel = _panel(20, 15, rng, scale=5.0) + 100  # offset + rescaled, should still standardize per-row
        z = cross_sectional_zscore(panel)
        row_means = z.mean(axis=1)
        row_stds = z.std(axis=1)
        assert np.allclose(row_means, 0, atol=1e-9)
        assert np.allclose(row_stds, 1, atol=1e-9)

    def test_zero_cross_sectional_variance_gives_nan_not_a_crash(self):
        dates = pd.bdate_range("2019-01-01", periods=3)
        panel = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}, index=dates)
        z = cross_sectional_zscore(panel)  # should not raise
        assert z.isna().all(axis=None)


class TestCalibrateFactorWeights:
    def test_positive_ic_factor_gets_the_dominant_weight(self):
        rng = np.random.default_rng(1)
        n_dates, n_symbols = 100, 40
        forward_returns = _panel(n_dates, n_symbols, rng, scale=0.05)

        # factor_A: genuinely, strongly predictive of forward returns
        factor_a = forward_returns * 3.0 + _panel(n_dates, n_symbols, rng, scale=0.02)
        # factor_b: pure independent noise, true IC = 0
        factor_b = _panel(n_dates, n_symbols, rng, scale=1.0)
        # factor_c: genuinely, strongly ANTI-predictive
        factor_c = -forward_returns * 3.0 + _panel(n_dates, n_symbols, rng, scale=0.02)

        result = calibrate_factor_weights(
            {"a": factor_a, "b": factor_b, "c": factor_c}, forward_returns
        )
        assert result.ic_by_factor["a"] > 0.3
        assert result.ic_by_factor["c"] < -0.3
        assert result.weights["c"] == 0.0
        assert result.weights["a"] > 0.8  # dominates -- b's true IC is 0, may get a small noisy weight
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_all_negative_ic_gives_all_zero_weights(self):
        rng = np.random.default_rng(2)
        n_dates, n_symbols = 60, 30
        forward_returns = _panel(n_dates, n_symbols, rng, scale=0.05)
        factor_bad_1 = -forward_returns + _panel(n_dates, n_symbols, rng, scale=0.01)
        factor_bad_2 = -forward_returns * 2 + _panel(n_dates, n_symbols, rng, scale=0.01)

        result = calibrate_factor_weights(
            {"bad1": factor_bad_1, "bad2": factor_bad_2}, forward_returns
        )
        assert result.weights == {"bad1": 0.0, "bad2": 0.0}

    def test_weights_sum_to_one_when_any_factor_is_included(self):
        rng = np.random.default_rng(3)
        n_dates, n_symbols = 80, 25
        forward_returns = _panel(n_dates, n_symbols, rng, scale=0.05)
        factor_good = forward_returns * 2 + _panel(n_dates, n_symbols, rng, scale=0.03)
        factor_noise = _panel(n_dates, n_symbols, rng, scale=1.0)

        result = calibrate_factor_weights({"good": factor_good, "noise": factor_noise}, forward_returns)
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestCompositeScore:
    def test_matches_hand_calculated_weighted_sum(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        panel_a = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]}, index=dates)
        panel_b = pd.DataFrame({"X": [10.0, 20.0], "Y": [30.0, 40.0]}, index=dates)

        score = composite_score({"a": panel_a, "b": panel_b}, {"a": 0.7, "b": 0.3})
        expected_x0 = 0.7 * 1.0 + 0.3 * 10.0
        assert score.loc[dates[0], "X"] == pytest.approx(expected_x0)

    def test_zero_weight_factor_is_excluded_entirely(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        panel_a = pd.DataFrame({"X": [1.0, 2.0]}, index=dates)
        panel_b = pd.DataFrame({"X": [999.0, 999.0]}, index=dates)  # would blow up the score if included

        score = composite_score({"a": panel_a, "b": panel_b}, {"a": 1.0, "b": 0.0})
        pd.testing.assert_series_equal(score["X"], panel_a["X"], check_names=False)

    def test_missing_symbol_in_one_factor_does_not_nan_poison(self):
        dates = pd.bdate_range("2020-01-01", periods=1)
        panel_a = pd.DataFrame({"X": [1.0], "Y": [2.0]}, index=dates)
        panel_b = pd.DataFrame({"X": [5.0]}, index=dates)  # Y missing entirely from this factor

        score = composite_score({"a": panel_a, "b": panel_b}, {"a": 0.5, "b": 0.5})
        assert score.loc[dates[0], "X"] == pytest.approx(0.5 * 1.0 + 0.5 * 5.0)
        assert score.loc[dates[0], "Y"] == pytest.approx(0.5 * 2.0)  # b's missing value treated as 0, not NaN

    def test_raises_when_every_weight_is_zero(self):
        dates = pd.bdate_range("2020-01-01", periods=1)
        panel_a = pd.DataFrame({"X": [1.0]}, index=dates)
        with pytest.raises(ValueError):
            composite_score({"a": panel_a}, {"a": 0.0})

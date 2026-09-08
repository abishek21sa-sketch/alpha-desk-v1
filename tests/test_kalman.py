from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from alpha_desk.strategies.pairs.kalman import kalman_hedge_ratio


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


class TestKalmanHedgeRatio:
    def test_rejects_invalid_delta(self):
        x = pd.Series(np.arange(20.0))
        with pytest.raises(ValueError):
            kalman_hedge_ratio(x, x, delta=1.5)
        with pytest.raises(ValueError):
            kalman_hedge_ratio(x, x, delta=0.0)

    def test_rejects_too_few_observations(self):
        x = pd.Series(np.arange(5.0))
        with pytest.raises(ValueError):
            kalman_hedge_ratio(x, x, delta=0.01)

    def test_tracks_slowly_drifting_beta_better_than_static_ols(self):
        # x is a real price-like non-stationary series; the TRUE beta drifts
        # slowly and smoothly over the sample (a regime change, not a jump).
        # A single whole-sample OLS beta is necessarily one fixed number --
        # the Kalman filter's time-varying beta should track the true
        # trajectory with much lower error.
        rng = np.random.default_rng(0)
        n = 1000
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
        true_beta = 1.0 + 0.6 * np.sin(2 * np.pi * np.arange(n) / 500)
        noise = rng.normal(0, 0.5, n)
        y = pd.Series(true_beta * x.to_numpy() + noise, index=_dates(n))

        result = kalman_hedge_ratio(x, y, delta=1e-3)

        # give the filter a burn-in period to converge from its diffuse prior
        burn_in = 100
        kf_beta = result.beta.to_numpy()[burn_in:]
        true_beta_eval = true_beta[burn_in:]
        kf_rmse = np.sqrt(np.mean((kf_beta - true_beta_eval) ** 2))

        static_ols = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()
        static_beta = static_ols.params[1]
        static_rmse = np.sqrt(np.mean((static_beta - true_beta_eval) ** 2))

        assert kf_rmse < 0.5 * static_rmse

    def test_recovers_constant_beta_reasonably_when_truly_constant(self):
        rng = np.random.default_rng(1)
        n = 500
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
        true_beta = 2.0
        y = pd.Series(true_beta * x.to_numpy() + rng.normal(0, 0.3, n), index=_dates(n))

        result = kalman_hedge_ratio(x, y, delta=1e-4)
        # late-sample beta should have converged close to the true constant
        assert result.beta.iloc[-50:].mean() == pytest.approx(true_beta, rel=0.1)

    def test_is_causal_truncated_history_gives_identical_early_output(self):
        # the defining property a filter (vs. a smoother) must have: output
        # at time t must be IDENTICAL whether or not the series continues
        # past t. Running the filter on a truncated series and comparing to
        # the first N outputs of the full run is an exact equality check,
        # not a statistical one.
        rng = np.random.default_rng(2)
        n = 300
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
        y = pd.Series(1.3 * x.to_numpy() + rng.normal(0, 0.4, n), index=_dates(n))

        full = kalman_hedge_ratio(x, y, delta=1e-3)
        truncated = kalman_hedge_ratio(x.iloc[:150], y.iloc[:150], delta=1e-3)

        pd.testing.assert_series_equal(
            full.beta.iloc[:150], truncated.beta, check_names=False
        )
        pd.testing.assert_series_equal(
            full.z_score.iloc[:150], truncated.z_score, check_names=False
        )

    def test_smaller_delta_gives_smoother_less_reactive_beta(self):
        rng = np.random.default_rng(3)
        n = 500
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_dates(n))
        y = pd.Series(1.5 * x.to_numpy() + rng.normal(0, 1.0, n), index=_dates(n))

        smooth = kalman_hedge_ratio(x, y, delta=1e-6)
        reactive = kalman_hedge_ratio(x, y, delta=1e-2)

        smooth_variation = smooth.beta.diff().dropna().abs().mean()
        reactive_variation = reactive.beta.diff().dropna().abs().mean()
        assert smooth_variation < reactive_variation

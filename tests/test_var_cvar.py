from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from alpha_desk.risk.var_cvar import historical_var_cvar


class TestHistoricalVarCvar:
    def test_matches_known_normal_distribution_quantile(self):
        rng = np.random.default_rng(0)
        n = 200_000
        sigma = 0.02
        returns = rng.normal(0, sigma, n)
        result = historical_var_cvar(returns, confidence=0.95)
        expected_var = -stats.norm.ppf(0.05, scale=sigma)  # ~1.645 * sigma
        assert result.var == pytest.approx(expected_var, rel=0.03)

    def test_cvar_is_always_at_least_as_large_as_var(self):
        rng = np.random.default_rng(1)
        for seed_returns in [
            rng.normal(0, 0.02, 5000),
            rng.standard_t(df=3, size=5000) * 0.01,  # fat-tailed
            rng.exponential(0.01, 5000) - 0.02,  # skewed
        ]:
            result = historical_var_cvar(seed_returns, confidence=0.95)
            assert result.cvar >= result.var

    def test_higher_confidence_gives_higher_var(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.02, 5000)
        var_95 = historical_var_cvar(returns, confidence=0.95).var
        var_99 = historical_var_cvar(returns, confidence=0.99).var
        assert var_99 > var_95

    def test_rejects_bad_confidence(self):
        returns = np.random.default_rng(3).normal(0, 0.02, 100)
        with pytest.raises(ValueError):
            historical_var_cvar(returns, confidence=1.5)
        with pytest.raises(ValueError):
            historical_var_cvar(returns, confidence=0.0)

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError):
            historical_var_cvar(np.array([0.01, -0.02, 0.005]), confidence=0.95)

    def test_fat_tailed_distribution_gives_larger_cvar_var_gap_than_normal(self):
        # the whole point of CVaR over VaR is capturing tail severity --
        # a fat-tailed distribution should show a bigger CVaR-VaR gap
        # (relative to VaR) than a normal one with the same VaR level.
        rng = np.random.default_rng(4)
        normal_returns = rng.normal(0, 0.02, 20000)
        fat_returns = rng.standard_t(df=2.5, size=20000) * 0.01

        normal_result = historical_var_cvar(normal_returns, confidence=0.99)
        fat_result = historical_var_cvar(fat_returns, confidence=0.99)

        normal_gap = (normal_result.cvar - normal_result.var) / normal_result.var
        fat_gap = (fat_result.cvar - fat_result.var) / fat_result.var
        assert fat_gap > normal_gap

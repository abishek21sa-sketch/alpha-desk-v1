from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from alpha_desk.validation.performance import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


class TestSharpeRatio:
    def test_annualization_scales_by_sqrt_periods(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.02, 500)
        daily = sharpe_ratio(returns)
        annual = sharpe_ratio(returns, periods_per_year=252)
        assert annual == pytest.approx(daily * np.sqrt(252))


class TestProbabilisticSharpeRatio:
    def test_psr_at_benchmark_equals_own_sharpe_is_half(self):
        # P(true SR > SR_hat | observed SR_hat) must be exactly 0.5,
        # regardless of skew/kurtosis/T -- the Z-score numerator is 0.
        rng = np.random.default_rng(1)
        returns = rng.standard_t(df=4, size=300) * 0.01  # fat-tailed, non-normal
        sr_hat = sharpe_ratio(returns)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=sr_hat)
        assert psr == pytest.approx(0.5, abs=1e-9)

    def test_matches_closed_form_under_normality(self):
        # skew=0, kurtosis=3 (normal) collapses the PSR formula to
        # Phi((SR_hat - SR*) * sqrt(T-1)) -- verify our general formula
        # reduces to this closed form when fed genuinely normal returns
        # with skew/kurtosis forced to their population values.
        rng = np.random.default_rng(2)
        n = 5000  # large n so sample skew/kurtosis converge close to 0/3
        returns = rng.normal(0.0008, 0.01, n)
        sr_hat = sharpe_ratio(returns)
        benchmark = 0.0
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=benchmark)
        expected = stats.norm.cdf((sr_hat - benchmark) * np.sqrt(n - 1))
        assert psr == pytest.approx(expected, abs=0.02)

    def test_more_observations_increase_confidence_when_sr_positive(self):
        rng = np.random.default_rng(3)
        short = rng.normal(0.001, 0.02, 60)
        long_sample = np.concatenate([short] * 20)  # same SR, more "observations"
        psr_short = probabilistic_sharpe_ratio(short, benchmark_sr=0.0)
        psr_long = probabilistic_sharpe_ratio(long_sample, benchmark_sr=0.0)
        assert psr_long > psr_short

    def test_negative_skew_reduces_confidence_for_positive_sharpe(self):
        rng = np.random.default_rng(4)
        n = 2000
        symmetric = rng.normal(0.001, 0.02, n)
        # left-skewed: occasional large negative shocks (crash risk), same
        # rough mean/std order of magnitude.
        skewed = rng.normal(0.0015, 0.015, n) - (rng.exponential(0.03, n) * (rng.random(n) < 0.05))

        psr_symmetric = probabilistic_sharpe_ratio(symmetric, benchmark_sr=0.0)
        psr_skewed = probabilistic_sharpe_ratio(skewed, benchmark_sr=0.0)
        assert stats.skew(skewed) < -0.1  # confirm the synthetic sample is actually left-skewed
        assert psr_skewed < psr_symmetric


class TestDeflatedSharpeRatio:
    def test_requires_at_least_two_trials(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0.001, 0.02, 200)
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(returns, trial_sharpe_ratios=np.array([0.5]))

    def test_more_trials_makes_the_same_result_less_significant(self):
        # the multiple-testing correction's whole point: report the SAME
        # winning strategy's returns, but say it came from a sweep of many
        # more candidates -- DSR must go down.
        rng = np.random.default_rng(6)
        winner_returns = rng.normal(0.0012, 0.02, 400)

        few_trials = rng.normal(0.0, 0.05, 5)
        many_trials = rng.normal(0.0, 0.05, 500)
        # ensure the winner's own SR is included and is the max, as it should be
        winner_sr = sharpe_ratio(winner_returns)
        few_trials[0] = winner_sr
        many_trials[0] = winner_sr

        dsr_few = deflated_sharpe_ratio(winner_returns, few_trials)
        dsr_many = deflated_sharpe_ratio(winner_returns, many_trials)

        assert dsr_many.expected_max_sr_null > dsr_few.expected_max_sr_null
        assert dsr_many.dsr < dsr_few.dsr

    def test_dsr_never_exceeds_undeflated_psr(self):
        # deflating against "best of N" can only raise the bar relative to
        # a fixed benchmark of 0 -- DSR should never be more favorable than
        # PSR(0).
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.02, 300)
        trial_srs = rng.normal(0.0, 0.3, 50)
        trial_srs[0] = sharpe_ratio(returns)

        dsr_result = deflated_sharpe_ratio(returns, trial_srs)
        psr_zero = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        assert dsr_result.dsr <= psr_zero + 1e-9

    def test_zero_variance_trials_falls_back_to_psr_zero(self):
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.0, 0.008] * 20)
        identical_trials = np.full(10, sharpe_ratio(returns))
        result = deflated_sharpe_ratio(returns, identical_trials)
        assert result.expected_max_sr_null == 0.0
        assert result.dsr == pytest.approx(probabilistic_sharpe_ratio(returns, 0.0))

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.pairs import cointegration as coint_mod
from alpha_desk.strategies.pairs.cointegration import (
    estimate_half_life,
    screen_universe_for_pairs,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _simulate_ou(theta: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """A genuine discretized OU process: x_t = x_{t-1} + theta*(0 - x_{t-1}) + noise.
    Used both to test half-life estimation directly and to build a
    synthetic cointegrated pair whose residual has real, controllable
    persistence (NOT i.i.d. noise -- i.i.d. noise has near-zero half-life by
    construction, which would make a "detects cointegration" test pass for
    the wrong reason).
    """
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (0 - x[t - 1]) + rng.normal(0, sigma)
    return x


class TestHalfLife:
    def test_recovers_known_half_life_from_synthetic_ou_process(self):
        rng = np.random.default_rng(0)
        theta, n = 0.05, 3000
        true_half_life = np.log(2) / theta
        spread = _simulate_ou(theta, sigma=1.0, n=n, rng=rng)
        estimated = estimate_half_life(pd.Series(spread))
        assert estimated == pytest.approx(true_half_life, rel=0.25)

    def test_random_walk_half_life_is_far_longer_than_genuine_mean_reversion(self):
        # A pure random walk has NO true mean reversion, but a finite-sample
        # AR(1) regression on it is well known to be biased toward a
        # slightly negative coefficient (the classic Dickey-Fuller finite-
        # sample bias) -- so it will rarely return a literal np.inf. The
        # correct, honest check is comparative: random-walk half-lives
        # should be far longer (weaker mean reversion) than a genuinely
        # mean-reverting series', not "always exactly infinite."
        rw_half_lives = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            rw = np.cumsum(rng.normal(0, 1, 500))
            rw_half_lives.append(estimate_half_life(pd.Series(rw)))

        rng = np.random.default_rng(99)
        genuine_ou = _simulate_ou(theta=0.05, sigma=1.0, n=500, rng=rng)
        ou_half_life = estimate_half_life(pd.Series(genuine_ou))

        median_rw_half_life = np.median(rw_half_lives)
        assert median_rw_half_life > 5 * ou_half_life

    def test_non_mean_reverting_beta_gives_infinite_half_life(self):
        # direct unit check of the beta>=0 guard clause itself, independent
        # of any finite-sample-bias question above. Uses a mildly EXPLOSIVE
        # (not merely random-walk) process, spread_t = 1.01*spread_{t-1} +
        # noise, so the regression beta is robustly positive across noise
        # draws -- a perfectly deterministic trend is numerically degenerate
        # (near-zero residual variance) and produces a beta that is only
        # negative by floating-point noise, which isn't what this guard
        # clause is meant to catch.
        rng = np.random.default_rng(7)
        n = 300
        explosive = np.zeros(n)
        explosive[0] = 1.0
        for t in range(1, n):
            explosive[t] = 1.01 * explosive[t - 1] + rng.normal(0, 0.1)
        assert estimate_half_life(pd.Series(explosive)) == float("inf")


class TestPairCointegration:
    def test_detects_synthetically_cointegrated_pair_with_realistic_half_life(self):
        rng = np.random.default_rng(2)
        n = 1000
        common_trend = np.cumsum(rng.normal(0, 0.01, n))  # shared I(1) stochastic trend
        true_beta = 1.5
        # the residual is a genuine OU process (real persistence, controllable
        # half-life), NOT i.i.d. noise -- i.i.d. noise would give a near-zero
        # half-life regardless of cointegration, which would make this test
        # pass for the wrong reason (see _simulate_ou's docstring).
        true_theta = 0.05  # implies half-life = ln(2)/0.05 ~= 13.9 obs
        residual = _simulate_ou(true_theta, sigma=0.02, n=n, rng=rng)

        log_b = 4.0 + common_trend + rng.normal(0, 0.005, n)
        log_a = 3.0 + true_beta * common_trend + residual

        price_a = pd.Series(np.exp(log_a), index=_dates(n))
        price_b = pd.Series(np.exp(log_b), index=_dates(n))

        result = coint_mod.test_pair_cointegration(price_a, price_b, "A", "B")
        assert result.pvalue < 0.05
        assert result.hedge_ratio == pytest.approx(true_beta, rel=0.15)
        assert 5 < result.half_life_days < 40  # true half-life ~13.9, generous band
        assert result.is_tradeable

    def test_independent_random_walks_are_usually_not_cointegrated(self):
        # Engle-Granger at 5% significance has ~5% false-positive rate by
        # construction -- check the false-positive rate across many
        # independent pairs is roughly nominal, not wildly inflated (which
        # would indicate a real bug, e.g. an off-by-one in the regression).
        false_positives = 0
        n_trials = 40
        for seed in range(n_trials):
            rng = np.random.default_rng(1000 + seed)
            n = 500
            price_a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4), index=_dates(n))
            price_b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4), index=_dates(n))
            result = coint_mod.test_pair_cointegration(price_a, price_b, "A", "B")
            if result.pvalue < 0.05:
                false_positives += 1
        assert false_positives / n_trials < 0.20  # generous slack over the 5% nominal rate

    def test_raises_on_insufficient_overlap(self):
        price_a = pd.Series(np.exp(np.arange(30) * 0.001 + 4), index=_dates(30))
        price_b = pd.Series(np.exp(np.arange(30) * 0.001 + 4), index=_dates(30))
        with pytest.raises(ValueError):
            coint_mod.test_pair_cointegration(price_a, price_b, "A", "B")


class TestScreenUniverse:
    def test_screens_all_pairs_and_sorts_by_pvalue(self):
        rng = np.random.default_rng(3)
        n = 400
        prices = {
            sym: pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4), index=_dates(n))
            for sym in ["X", "Y", "Z", "W"]
        }
        results = screen_universe_for_pairs(prices)
        assert len(results) == 6  # C(4, 2)
        pvalues = [r.pvalue for r in results]
        assert pvalues == sorted(pvalues)

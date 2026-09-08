from __future__ import annotations

import numpy as np
import pytest

from alpha_desk.validation.overfitting import probability_of_backtest_overfitting


class TestPBO:
    def test_rejects_odd_n_subsamples(self):
        returns = np.zeros((100, 3))
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(returns, n_subsamples=5)

    def test_rejects_single_strategy(self):
        returns = np.zeros((100, 1))
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(returns)

    def test_pure_noise_gives_pbo_near_one_half(self):
        # no strategy has real skill -- which in-sample strategy "wins" is
        # pure chance, so its OOS rank should be uniform over {1..N},
        # putting PBO at 0.5 in expectation. A SINGLE matrix draw is much
        # too noisy to test directly, though: the 252 combinations for one
        # matrix reuse overlapping time blocks, so they're far from
        # independent -- empirically (see scripts/ dev notes) a single draw
        # at T=1000/N=20/S=10 has stdev ~0.17 around 0.5, so a tight
        # single-seed bound is flaky by construction, not a sign of a bug.
        # Averaging 20 independent matrix draws shrinks stderr to ~0.038,
        # which a real directional bug would blow through easily.
        pbos = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            returns = rng.normal(0.0, 0.02, size=(1000, 20))
            pbos.append(probability_of_backtest_overfitting(returns, n_subsamples=10).pbo)
        mean_pbo = np.mean(pbos)
        assert 0.35 < mean_pbo < 0.65, f"mean PBO over 20 draws = {mean_pbo}, expected ~0.5"

        result = probability_of_backtest_overfitting(
            np.random.default_rng(0).normal(0.0, 0.02, size=(1000, 20)), n_subsamples=10
        )
        assert result.n_combinations == 252  # C(10, 5)

    def test_one_genuinely_skilled_strategy_gives_low_pbo(self):
        # strategy 0 has a real, consistent edge baked in across the whole
        # timeline -- it should reliably be the IS winner AND stay a strong
        # OOS performer, so PBO should be low.
        rng = np.random.default_rng(43)
        t, n = 1000, 20
        returns = rng.normal(0.0, 0.02, size=(t, n))
        returns[:, 0] += 0.003  # strategy 0's real edge, present in every period
        result = probability_of_backtest_overfitting(returns, n_subsamples=10)
        # empirically (20 seeds, dev-time check) this scenario tops out
        # around 0.24 -- 0.3 leaves real margin against single-seed noise
        # while still being far below the ~0.5 pure-noise baseline.
        assert result.pbo < 0.3

    def test_skilled_strategy_pbo_is_lower_than_noise_pbo(self):
        # the comparative claim is the one that actually matters -- run both
        # scenarios off the same noise draw so the comparison isn't an
        # artifact of a lucky seed.
        rng = np.random.default_rng(44)
        t, n = 1000, 16
        base = rng.normal(0.0, 0.02, size=(t, n))

        noise_only = base.copy()
        skilled = base.copy()
        skilled[:, 0] += 0.0025

        pbo_noise = probability_of_backtest_overfitting(noise_only, n_subsamples=8).pbo
        pbo_skilled = probability_of_backtest_overfitting(skilled, n_subsamples=8).pbo
        assert pbo_skilled < pbo_noise

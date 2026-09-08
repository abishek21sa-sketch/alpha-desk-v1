from __future__ import annotations

import numpy as np
import pytest

from alpha_desk.strategies.microstructure.simulation import simulate_market_making_session


class TestSimulateMarketMakingSession:
    def test_starts_flat_with_zero_pnl(self):
        result = simulate_market_making_session(
            initial_mid=100.0, horizon=1.0, n_steps=500, sigma=0.02,
            gamma=0.1, kappa=1.5, arrival_intensity=50.0, seed=0,
        )
        assert result.inventory[0] == 0.0
        assert result.cash[0] == 0.0
        assert result.pnl[0] == 0.0

    def test_deterministic_with_fixed_seed(self):
        kwargs = dict(initial_mid=100.0, horizon=1.0, n_steps=500, sigma=0.02,
                      gamma=0.1, kappa=1.5, arrival_intensity=50.0, seed=42)
        r1 = simulate_market_making_session(**kwargs)
        r2 = simulate_market_making_session(**kwargs)
        np.testing.assert_array_equal(r1.pnl, r2.pnl)
        np.testing.assert_array_equal(r1.inventory, r2.inventory)

    def test_reasonable_arrival_intensity_produces_fills(self):
        result = simulate_market_making_session(
            initial_mid=100.0, horizon=1.0, n_steps=1000, sigma=0.02,
            gamma=0.1, kappa=1.5, arrival_intensity=100.0, seed=1,
        )
        assert (result.n_buy_fills + result.n_sell_fills) > 0

    def test_higher_arrival_intensity_gives_more_fills_on_average(self):
        def total_fills(intensity, seed):
            r = simulate_market_making_session(
                initial_mid=100.0, horizon=1.0, n_steps=1000, sigma=0.02,
                gamma=0.1, kappa=1.5, arrival_intensity=intensity, seed=seed,
            )
            return r.n_buy_fills + r.n_sell_fills

        low = np.mean([total_fills(20.0, s) for s in range(15)])
        high = np.mean([total_fills(200.0, s) for s in range(15)])
        assert high > low

    def test_inventory_limit_is_respected(self):
        result = simulate_market_making_session(
            initial_mid=100.0, horizon=2.0, n_steps=2000, sigma=0.05,
            gamma=0.05, kappa=1.0, arrival_intensity=300.0, inventory_limit=5.0, seed=2,
        )
        # can overshoot by at most 1 unit within the step that crosses the
        # limit (the check happens before that step's fill, not after)
        assert result.inventory.max() <= 6.0
        assert result.inventory.min() >= -6.0

    def test_higher_risk_aversion_keeps_inventory_closer_to_zero_on_average(self):
        def mean_abs_inventory(gamma, seed):
            r = simulate_market_making_session(
                initial_mid=100.0, horizon=1.0, n_steps=1000, sigma=0.03,
                gamma=gamma, kappa=1.0, arrival_intensity=150.0, seed=seed,
            )
            return np.mean(np.abs(r.inventory))

        low_gamma_avg = np.mean([mean_abs_inventory(0.01, s) for s in range(15)])
        high_gamma_avg = np.mean([mean_abs_inventory(2.0, s) for s in range(15)])
        assert high_gamma_avg < low_gamma_avg

    def test_oversaturated_intensity_is_clipped_not_left_invalid(self):
        # regression test for a real calibration bug: arrival_intensity*dt
        # can exceed 1 (an invalid "probability") if intensity is large
        # relative to the step size. Before clipping, this silently
        # saturated fills at exactly 2*n_steps regardless of how much
        # further intensity increased -- verified here directly: a wildly
        # oversaturated intensity must fill EVERY step on both sides
        # (2*n_steps total), not crash or behave inconsistently.
        n_steps = 200
        result = simulate_market_making_session(
            initial_mid=100.0, horizon=1.0, n_steps=n_steps, sigma=0.02,
            gamma=0.001, kappa=1.0, arrival_intensity=1_000_000.0, seed=3,
        )
        assert result.n_buy_fills == n_steps
        assert result.n_sell_fills == n_steps

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            simulate_market_making_session(
                initial_mid=100.0, horizon=1.0, n_steps=0, sigma=0.02,
                gamma=0.1, kappa=1.5, arrival_intensity=50.0,
            )
        with pytest.raises(ValueError):
            simulate_market_making_session(
                initial_mid=100.0, horizon=1.0, n_steps=100, sigma=-0.1,
                gamma=0.1, kappa=1.5, arrival_intensity=50.0,
            )

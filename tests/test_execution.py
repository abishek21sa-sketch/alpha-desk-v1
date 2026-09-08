from __future__ import annotations

import numpy as np
import pytest

from alpha_desk.risk.execution import almgren_chriss_trajectory


class TestAlmgrenChrissTrajectory:
    def test_starts_at_full_position_and_ends_near_zero(self):
        result = almgren_chriss_trajectory(
            shares=100_000, horizon=5, n_intervals=20, volatility=0.5,
            temporary_impact=0.01, permanent_impact=0.001, risk_aversion=1e-6,
        )
        assert result.holdings[0] == pytest.approx(100_000)
        assert result.holdings[-1] == pytest.approx(0.0, abs=1e-6)

    def test_trades_sum_to_total_shares(self):
        result = almgren_chriss_trajectory(
            shares=50_000, horizon=3, n_intervals=15, volatility=0.3,
            temporary_impact=0.02, permanent_impact=0.002, risk_aversion=1e-4,
        )
        assert result.trades.sum() == pytest.approx(50_000)

    def test_zero_risk_aversion_gives_exactly_uniform_trading(self):
        shares, horizon, n = 100_000, 5, 20
        result = almgren_chriss_trajectory(
            shares=shares, horizon=horizon, n_intervals=n, volatility=0.5,
            temporary_impact=0.01, permanent_impact=0.001, risk_aversion=0.0,
        )
        expected_holdings = shares * (1 - np.arange(n + 1) / n)
        np.testing.assert_allclose(result.holdings, expected_holdings, atol=1e-6)
        # uniform trading -> every interval trades the same amount
        assert np.allclose(result.trades, result.trades[0])

    def test_higher_risk_aversion_front_loads_the_trajectory(self):
        kwargs = dict(shares=100_000, horizon=5, n_intervals=20, volatility=0.5,
                       temporary_impact=0.01, permanent_impact=0.001)
        low = almgren_chriss_trajectory(risk_aversion=1e-8, **kwargs)
        high = almgren_chriss_trajectory(risk_aversion=1e-3, **kwargs)
        midpoint = 10
        # more risk-averse -> sell faster/earlier -> less remaining at the midpoint
        assert high.holdings[midpoint] < low.holdings[midpoint]

    def test_higher_risk_aversion_increases_cost_but_decreases_variance(self):
        # the actual point of Almgren-Chriss: there is no free lunch --
        # trading faster to cut price risk costs more in market impact.
        kwargs = dict(shares=100_000, horizon=5, n_intervals=20, volatility=0.5,
                       temporary_impact=0.01, permanent_impact=0.001)
        low = almgren_chriss_trajectory(risk_aversion=1e-8, **kwargs)
        high = almgren_chriss_trajectory(risk_aversion=1e-3, **kwargs)
        assert high.expected_cost > low.expected_cost
        assert high.cost_variance < low.cost_variance

    def test_rejects_invalid_inputs(self):
        base = dict(horizon=5, n_intervals=20, volatility=0.5, temporary_impact=0.01,
                    permanent_impact=0.001, risk_aversion=1e-4)
        with pytest.raises(ValueError):
            almgren_chriss_trajectory(shares=-100, **base)
        with pytest.raises(ValueError):
            almgren_chriss_trajectory(shares=100, **{**base, "n_intervals": 0})
        with pytest.raises(ValueError):
            almgren_chriss_trajectory(shares=100, **{**base, "risk_aversion": -1})

    def test_rejects_degenerate_impact_parameters(self):
        # temporary_impact too small relative to permanent_impact*tau/2
        with pytest.raises(ValueError):
            almgren_chriss_trajectory(
                shares=100_000, horizon=5, n_intervals=2, volatility=0.5,
                temporary_impact=0.0001, permanent_impact=1.0, risk_aversion=1e-4,
            )

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.momentum.backtest import run_tsmom_backtest
from alpha_desk.validation.costs import TransactionCostModel


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2012-01-01", periods=n)


def _trending_universe(n_symbols: int = 6, n_days: int = 1500, seed: int = 0):
    """Each asset alternates between multi-month persistent trends (up or
    down) -- real, sustained direction a trend-following signal should
    profit from regardless of which way any single trend runs.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n_days)
    prices, volumes = {}, {}
    for i in range(n_symbols):
        sym = f"A{i}"
        segment_len = 120
        drift = np.zeros(n_days)
        direction = 1
        pos = 0
        while pos < n_days:
            end = min(pos + segment_len, n_days)
            drift[pos:end] = direction * 0.0015
            direction *= -1
            pos = end
        r = drift + rng.normal(0, 0.01, n_days)
        prices[sym] = pd.Series(100 * np.cumprod(1 + r), index=idx)
        volumes[sym] = pd.Series(rng.uniform(3e5, 1.5e6, n_days), index=idx)
    return prices, volumes


def _whipsaw_universe(n_symbols: int = 6, n_days: int = 1500, seed: int = 0):
    """Pure mean-reverting noise, no persistent trend at all -- the
    condition trend-following is known to lose money in (whipsaws)."""
    rng = np.random.default_rng(seed)
    idx = _dates(n_days)
    prices, volumes = {}, {}
    for i in range(n_symbols):
        sym = f"A{i}"
        r = rng.normal(0, 0.01, n_days)
        prices[sym] = pd.Series(100 * np.cumprod(1 + r), index=idx)
        volumes[sym] = pd.Series(rng.uniform(3e5, 1.5e6, n_days), index=idx)
    return prices, volumes


class TestRunTsmomBacktest:
    def test_raises_on_too_little_history(self):
        prices, volumes = _trending_universe(n_symbols=3, n_days=200)
        with pytest.raises(ValueError):
            run_tsmom_backtest(prices, volumes)

    def test_raises_on_single_asset(self):
        prices, volumes = _trending_universe(n_symbols=1, n_days=1500)
        with pytest.raises(ValueError):
            run_tsmom_backtest(prices, volumes)

    def test_persistent_trends_produce_positive_sharpe(self):
        prices, volumes = _trending_universe(seed=1)
        result = run_tsmom_backtest(
            prices, volumes,
            cost_model=TransactionCostModel(commission_bps=0.0, half_spread_bps=1.0, impact_coefficient=0.05),
        )
        assert result.n_rebalances > 20
        assert result.net_sharpe_annualized > 0.3

    def test_whipsaw_universe_gives_materially_worse_sharpe_than_trending(self):
        trending_prices, trending_volumes = _trending_universe(seed=2)
        whipsaw_prices, whipsaw_volumes = _whipsaw_universe(seed=2)

        trending_result = run_tsmom_backtest(trending_prices, trending_volumes)
        whipsaw_result = run_tsmom_backtest(whipsaw_prices, whipsaw_volumes)

        assert whipsaw_result.net_sharpe_annualized < trending_result.net_sharpe_annualized

    def test_zero_cost_model_gives_net_equal_to_gross(self):
        prices, volumes = _trending_universe(seed=3)
        zero_cost = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=0.0)
        result = run_tsmom_backtest(prices, volumes, cost_model=zero_cost)
        assert result.net_sharpe_annualized == pytest.approx(result.gross_sharpe_annualized, abs=1e-9)
        assert result.cost_drag_annualized == pytest.approx(0.0, abs=1e-9)

    def test_higher_costs_reduce_net_without_changing_gross(self):
        prices, volumes = _trending_universe(seed=4)
        cheap = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.5, impact_coefficient=0.02)
        expensive = TransactionCostModel(commission_bps=0.0, half_spread_bps=15.0, impact_coefficient=0.8)

        cheap_result = run_tsmom_backtest(prices, volumes, cost_model=cheap)
        expensive_result = run_tsmom_backtest(prices, volumes, cost_model=expensive)

        assert expensive_result.net_sharpe_annualized < cheap_result.net_sharpe_annualized
        assert cheap_result.gross_sharpe_annualized == pytest.approx(
            expensive_result.gross_sharpe_annualized, abs=1e-9
        )

    def test_per_asset_sharpe_reported_for_every_asset(self):
        prices, volumes = _trending_universe(n_symbols=5, seed=5)
        result = run_tsmom_backtest(prices, volumes)
        assert len(result.per_asset_net_sharpe) == 5

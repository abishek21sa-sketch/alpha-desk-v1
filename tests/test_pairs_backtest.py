from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.pairs.backtest import backtest_pair
from alpha_desk.validation.costs import TransactionCostModel


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def _synthetic_mean_reverting_pair(n: int = 1500, theta: float = 0.03, seed: int = 0):
    """Two log-price series with a real, controllable common trend and a
    genuinely mean-reverting (OU) residual -- same construction technique as
    test_cointegration.py, so a strategy that's supposed to exploit mean
    reversion has real mean reversion to exploit.

    The OU residual is added to log_b (the response/Y side of
    kalman_hedge_ratio(x=log_a, y=log_b)), NOT log_a (the regressor/X side).
    This matters: adding it to the regressor makes X itself noisy, which is
    a classical errors-in-variables setup -- OLS/Kalman regression of a
    clean Y on a noisy X is attenuation-biased toward zero, which was
    discovered the hard way here (the first version of this test put the
    residual on log_a and the recovered beta converged to ~0.16 against a
    true value of 1.2 -- not a kalman.py bug, a test-construction bug; see
    docs/PAIRS_TRADING.md).
    """
    rng = np.random.default_rng(seed)
    common_trend = np.cumsum(rng.normal(0, 0.008, n))
    residual = np.zeros(n)
    for t in range(1, n):
        residual[t] = residual[t - 1] + theta * (0 - residual[t - 1]) + rng.normal(0, 0.02)

    true_beta = 1.2
    log_a = 3.5 + common_trend + rng.normal(0, 0.003, n)  # clean regressor
    log_b = 4.0 + true_beta * common_trend + residual  # response carries the mean-reverting residual

    idx = _dates(n)
    price_a = pd.Series(np.exp(log_a), index=idx)
    price_b = pd.Series(np.exp(log_b), index=idx)
    volume_a = pd.Series(rng.uniform(5e5, 2e6, n), index=idx)
    volume_b = pd.Series(rng.uniform(5e5, 2e6, n), index=idx)
    return price_a, price_b, volume_a, volume_b


class TestBacktestPair:
    def test_raises_on_insufficient_history(self):
        idx = _dates(100)
        s = pd.Series(np.ones(100), index=idx)
        with pytest.raises(ValueError):
            backtest_pair(s, s, s, s, "A", "B")

    def test_genuine_mean_reversion_produces_positive_net_sharpe(self):
        price_a, price_b, vol_a, vol_b = _synthetic_mean_reverting_pair(seed=0)
        result = backtest_pair(
            price_a, price_b, vol_a, vol_b, "A", "B",
            cost_model=TransactionCostModel(commission_bps=0.0, half_spread_bps=1.0, impact_coefficient=0.05),
        )
        assert result.n_trades > 0
        assert result.net_sharpe_annualized > 0.5

    def test_zero_cost_model_gives_net_equal_to_gross(self):
        price_a, price_b, vol_a, vol_b = _synthetic_mean_reverting_pair(seed=1)
        zero_cost = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=0.0)
        result = backtest_pair(price_a, price_b, vol_a, vol_b, "A", "B", cost_model=zero_cost)
        assert result.net_sharpe_annualized == pytest.approx(result.gross_sharpe_annualized, abs=1e-9)
        assert result.cost_drag_annualized == pytest.approx(0.0, abs=1e-9)

    def test_higher_costs_strictly_reduce_net_sharpe(self):
        price_a, price_b, vol_a, vol_b = _synthetic_mean_reverting_pair(seed=2)
        cheap = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.5, impact_coefficient=0.02)
        expensive = TransactionCostModel(commission_bps=0.0, half_spread_bps=10.0, impact_coefficient=0.5)

        cheap_result = backtest_pair(price_a, price_b, vol_a, vol_b, "A", "B", cost_model=cheap)
        expensive_result = backtest_pair(price_a, price_b, vol_a, vol_b, "A", "B", cost_model=expensive)

        assert expensive_result.net_sharpe_annualized < cheap_result.net_sharpe_annualized
        # gross should be identical -- costs must not leak into the gross calc
        assert cheap_result.gross_sharpe_annualized == pytest.approx(
            expensive_result.gross_sharpe_annualized, abs=1e-9
        )

    def test_causal_truncated_history_gives_identical_early_returns(self):
        price_a, price_b, vol_a, vol_b = _synthetic_mean_reverting_pair(n=800, seed=3)
        full = backtest_pair(price_a, price_b, vol_a, vol_b, "A", "B")
        cutoff = 500
        truncated = backtest_pair(
            price_a.iloc[:cutoff], price_b.iloc[:cutoff],
            vol_a.iloc[:cutoff], vol_b.iloc[:cutoff], "A", "B",
        )
        common_idx = full.returns.index.intersection(truncated.returns.index)
        assert len(common_idx) > 100
        pd.testing.assert_series_equal(
            full.returns.loc[common_idx], truncated.returns.loc[common_idx], check_names=False
        )

    def test_uncorrelated_random_walks_do_not_reliably_profit(self):
        # two INDEPENDENT random walks (no real relationship at all) --
        # the strategy might get lucky on any one seed, but should not show
        # a reliable, large positive Sharpe across many independent draws
        # the way the genuine mean-reversion case does.
        sharpes = []
        for seed in range(15):
            rng = np.random.default_rng(500 + seed)
            n = 1000
            idx = _dates(n)
            price_a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4), index=idx)
            price_b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4), index=idx)
            volume = pd.Series(rng.uniform(5e5, 2e6, n), index=idx)
            result = backtest_pair(price_a, price_b, volume, volume, "A", "B")
            sharpes.append(result.net_sharpe_annualized)
        assert np.mean(sharpes) < 0.5  # nowhere near the >0.5 genuine-edge case above

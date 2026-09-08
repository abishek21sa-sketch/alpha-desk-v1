from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.pead.backtest import run_pead_backtest
from alpha_desk.validation.costs import TransactionCostModel


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _universe_with_events(n_symbols=8, n_days=1500, n_events_per_symbol=15, seed=0, genuine_pead=True):
    rng = np.random.default_rng(seed)
    idx = _dates(n_days)
    prices, volumes, events = {}, {}, {}
    for i in range(n_symbols):
        sym = f"A{i}"
        log_ret = rng.normal(0, 0.01, n_days)
        event_positions = rng.choice(np.arange(100, n_days - 100), size=n_events_per_symbol, replace=False)
        event_positions.sort()
        for pos in event_positions:
            surprise = rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.06)
            log_ret[pos] += surprise  # announcement-day jump
            if genuine_pead:
                # continued drift in the SAME direction over the next 21 days
                drift_per_day = surprise * 0.02
                log_ret[pos + 1 : pos + 22] += drift_per_day
            # else: no continuation at all, pure noise after the jump
        prices[sym] = pd.Series(100 * np.exp(np.cumsum(log_ret)), index=idx)
        volumes[sym] = pd.Series(rng.uniform(3e5, 1.5e6, n_days), index=idx)
        events[sym] = [idx[p] for p in event_positions]
    return prices, volumes, events


class TestRunPeadBacktest:
    def test_raises_on_no_matching_symbols(self):
        prices, volumes, _ = _universe_with_events(n_symbols=2, n_days=1500)
        with pytest.raises(ValueError):
            run_pead_backtest(prices, volumes, event_dates_by_symbol={})

    def test_genuine_pead_effect_is_significant_and_profitable(self):
        prices, volumes, events = _universe_with_events(seed=1, genuine_pead=True)
        result = run_pead_backtest(
            prices, volumes, events,
            cost_model=TransactionCostModel(commission_bps=0.0, half_spread_bps=1.0, impact_coefficient=0.05),
        )
        assert result.significance.drift_spread > 0
        assert result.significance.p_value < 0.05
        assert result.net_sharpe_annualized > 0.0

    def test_no_pead_effect_gives_no_significant_spread(self):
        prices, volumes, events = _universe_with_events(seed=2, genuine_pead=False)
        result = run_pead_backtest(prices, volumes, events)
        assert result.significance.p_value > 0.05

    def test_zero_cost_model_gives_net_equal_to_gross(self):
        prices, volumes, events = _universe_with_events(seed=3, genuine_pead=True)
        zero_cost = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=0.0)
        result = run_pead_backtest(prices, volumes, events, cost_model=zero_cost)
        assert result.net_sharpe_annualized == pytest.approx(result.gross_sharpe_annualized, abs=1e-9)
        assert result.cost_drag_annualized == pytest.approx(0.0, abs=1e-9)

    def test_higher_costs_reduce_net_without_changing_gross(self):
        prices, volumes, events = _universe_with_events(seed=4, genuine_pead=True)
        cheap = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.5, impact_coefficient=0.02)
        expensive = TransactionCostModel(commission_bps=0.0, half_spread_bps=20.0, impact_coefficient=1.0)

        cheap_result = run_pead_backtest(prices, volumes, events, cost_model=cheap)
        expensive_result = run_pead_backtest(prices, volumes, events, cost_model=expensive)

        assert expensive_result.net_sharpe_annualized < cheap_result.net_sharpe_annualized
        assert cheap_result.gross_sharpe_annualized == pytest.approx(
            expensive_result.gross_sharpe_annualized, abs=1e-9
        )

    def test_does_not_double_count_the_announcement_jump_as_drift(self):
        # regression test for a real bug: the first version of this backtest
        # credited each trade's first "return day" with a return ending ON
        # entry_date -- but entry_date IS the last day of the announcement
        # window used to pick the trade's direction, so that return was
        # literally re-scoring half of the same jump, not real subsequent
        # drift. Constructed here as directly as possible: 4 isolated
        # events (2 positive-jump symbols, 2 negative-jump symbols -- enough
        # for pead_significance's >=2-per-bucket requirement), each with a
        # one-time jump followed by EXACTLY FLAT prices for the rest of the
        # holding period. A correct backtest must show ~zero P&L on every
        # one of these trades, since there is nothing for any of them to
        # capture after the jump.
        n = 300
        idx = _dates(n)
        event_pos = 150
        prices, volumes, events = {}, {}, {}
        for i, jump in enumerate([0.05, 0.04, -0.05, -0.04]):
            sym = f"X{i}"
            log_price = np.zeros(n)
            log_price[event_pos:] = jump  # one-time jump, dead flat forever after
            prices[sym] = pd.Series(100 * np.exp(log_price), index=idx)
            volumes[sym] = pd.Series(1e6, index=idx)
            events[sym] = [idx[event_pos]]

        result = run_pead_backtest(
            prices, volumes, events,
            drift_days=20,
            announcement_threshold=0.01,
            cost_model=TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=0.0),
        )
        assert result.n_events_traded == 4
        # completely flat post-announcement prices -> the whole portfolio's
        # gross return series must be all zeros (nothing for any trade to earn or lose).
        assert result.returns.abs().max() < 1e-9

    def test_raises_when_threshold_excludes_all_events(self):
        prices, volumes, events = _universe_with_events(seed=5, genuine_pead=True)
        with pytest.raises(ValueError):
            run_pead_backtest(prices, volumes, events, announcement_threshold=0.99)

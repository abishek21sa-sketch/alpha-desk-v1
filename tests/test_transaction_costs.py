from __future__ import annotations

import pytest

from alpha_desk.validation.costs import TransactionCostModel


class TestTransactionCostModel:
    def test_zero_trade_value_is_just_commission_plus_spread(self):
        model = TransactionCostModel(commission_bps=1.0, half_spread_bps=2.5)
        cost = model.cost_bps(trade_value=0.0, adv_dollar_volume=1e8, daily_vol=0.02)
        assert cost == pytest.approx(3.5)

    def test_impact_grows_with_participation_rate(self):
        model = TransactionCostModel()
        small = model.cost_bps(trade_value=1e4, adv_dollar_volume=1e8, daily_vol=0.02)
        large = model.cost_bps(trade_value=1e7, adv_dollar_volume=1e8, daily_vol=0.02)
        assert large > small

    def test_impact_follows_square_root_not_linear_scaling(self):
        # quadrupling trade size should roughly double the IMPACT component
        # (sqrt law), not quadruple it -- isolate impact by stripping the
        # flat commission+spread term first.
        model = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0)
        adv, vol = 1e8, 0.02
        base_impact = model.cost_bps(trade_value=1e6, adv_dollar_volume=adv, daily_vol=vol)
        quad_impact = model.cost_bps(trade_value=4e6, adv_dollar_volume=adv, daily_vol=vol)
        assert quad_impact == pytest.approx(2 * base_impact, rel=1e-9)

    def test_higher_volatility_increases_impact(self):
        model = TransactionCostModel()
        low_vol = model.cost_bps(trade_value=1e6, adv_dollar_volume=1e8, daily_vol=0.01)
        high_vol = model.cost_bps(trade_value=1e6, adv_dollar_volume=1e8, daily_vol=0.05)
        assert high_vol > low_vol

    def test_more_liquidity_reduces_impact(self):
        model = TransactionCostModel()
        illiquid = model.cost_bps(trade_value=1e6, adv_dollar_volume=1e7, daily_vol=0.02)
        liquid = model.cost_bps(trade_value=1e6, adv_dollar_volume=1e10, daily_vol=0.02)
        assert illiquid > liquid

    def test_round_trip_is_exactly_double_one_way(self):
        model = TransactionCostModel()
        one_way = model.cost_bps(trade_value=5e5, adv_dollar_volume=1e8, daily_vol=0.02)
        round_trip = model.round_trip_cost_bps(trade_value=5e5, adv_dollar_volume=1e8, daily_vol=0.02)
        assert round_trip == pytest.approx(2 * one_way)

    def test_cost_dollars_consistent_with_cost_bps(self):
        model = TransactionCostModel()
        trade_value = 2.5e6
        bps = model.cost_bps(trade_value, adv_dollar_volume=1e8, daily_vol=0.02)
        dollars = model.cost_dollars(trade_value, adv_dollar_volume=1e8, daily_vol=0.02)
        assert dollars == pytest.approx(trade_value * bps / 10_000)

    def test_rejects_invalid_inputs(self):
        model = TransactionCostModel()
        with pytest.raises(ValueError):
            model.cost_bps(trade_value=-1, adv_dollar_volume=1e8, daily_vol=0.02)
        with pytest.raises(ValueError):
            model.cost_bps(trade_value=1e6, adv_dollar_volume=0, daily_vol=0.02)

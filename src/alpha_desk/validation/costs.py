"""A realistic-enough transaction cost model: commission + half-spread +
square-root market impact.

The square-root law (impact ~ sigma * sqrt(participation_rate)) is a widely
documented empirical regularity in market-impact literature (see Almgren,
Thum, Hauptmann & Li, "Direct Estimation of Equity Market Impact" (2005);
also the Kyle/Obizhaeva-Wang line of work) -- not derived here from first
principles, and `impact_coefficient` is an illustrative default, NOT
calibrated against this project's own fill data. A production desk
calibrates this coefficient per-name from its own execution history; until
that calibration exists, treat impact_bps as directionally correct
(bigger trades relative to liquidity cost more, and cost grows sub-linearly
in trade size) rather than a precise cost forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 0.0
    half_spread_bps: float = 2.5
    impact_coefficient: float = 0.1  # dimensionless; see module docstring

    def cost_bps(
        self, trade_value: float, adv_dollar_volume: float, daily_vol: float
    ) -> float:
        """Round-trip-agnostic ONE-WAY cost in basis points of trade value.

        trade_value: dollar value of this single trade (buy or sell leg)
        adv_dollar_volume: average daily dollar volume for the instrument
        daily_vol: daily return volatility (e.g. close-to-close stdev), as
            a decimal (0.02 = 2%/day), NOT annualized
        """
        if trade_value < 0 or adv_dollar_volume <= 0 or daily_vol < 0:
            raise ValueError("trade_value/daily_vol must be >= 0 and adv_dollar_volume > 0")
        participation = trade_value / adv_dollar_volume
        impact_bps = self.impact_coefficient * daily_vol * np.sqrt(participation) * 10_000
        return self.commission_bps + self.half_spread_bps + impact_bps

    def cost_dollars(
        self, trade_value: float, adv_dollar_volume: float, daily_vol: float
    ) -> float:
        return trade_value * self.cost_bps(trade_value, adv_dollar_volume, daily_vol) / 10_000

    def round_trip_cost_bps(
        self, trade_value: float, adv_dollar_volume: float, daily_vol: float
    ) -> float:
        """Entry + exit, each paying the one-way cost once."""
        return 2 * self.cost_bps(trade_value, adv_dollar_volume, daily_vol)

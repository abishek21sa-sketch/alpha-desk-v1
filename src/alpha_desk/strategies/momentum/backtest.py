"""Cost-aware TSMOM backtest across the 10-ETF proxy universe.

Like Phase 3 (pairs), this is a FIXED rule -- lookback, vol-target, and
leverage cap are pre-specified hyperparameters, not fitted from data -- so
there is no purged walk-forward CV to run (nothing is trained) and no
DSR/PBO correction to apply (one pre-specified configuration is run, not
the winner of a sweep). See strategies/pairs/backtest.py's module docstring
for the fuller version of this reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.strategies.factors.backtest import build_forward_returns, build_rebalance_dates
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import sharpe_ratio

from .signals import tsmom_position

REBALANCE_FREQ_DAYS = 21


@dataclass
class TSMOMBacktestResult:
    returns: pd.Series  # per-rebalance-period NET portfolio returns (equal notional per asset)
    per_asset_net_sharpe: dict[str, float]
    n_assets: int
    n_rebalances: int
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float


def run_tsmom_backtest(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    lookback_days: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.10,
    max_leverage: float = 3.0,
    capital_per_asset: float = 100_000.0,
    cost_model: TransactionCostModel | None = None,
) -> TSMOMBacktestResult:
    cost_model = cost_model or TransactionCostModel()
    symbols = sorted(prices)
    if len(symbols) < 2:
        raise ValueError("need at least 2 assets for a TSMOM portfolio")

    common_days = prices[symbols[0]].index
    for sym in symbols[1:]:
        common_days = common_days.intersection(prices[sym].index)
    common_days = common_days.sort_values()

    rebalance_dates = build_rebalance_dates(common_days, REBALANCE_FREQ_DAYS)
    min_history = lookback_days + vol_window + 30
    if len(common_days) < min_history:
        raise ValueError(f"need at least {min_history} trading days of common history, got {len(common_days)}")

    forward_returns = build_forward_returns(prices, rebalance_dates)
    total_capital = capital_per_asset * len(symbols)

    period_gross: dict[pd.Timestamp, float] = {t: 0.0 for t in rebalance_dates}
    period_net: dict[pd.Timestamp, float] = {t: 0.0 for t in rebalance_dates}
    per_asset_returns: dict[str, pd.Series] = {}

    for sym in symbols:
        price = prices[sym]
        position = tsmom_position(price, lookback_days, vol_window, target_vol, max_leverage)
        position_lagged = position.reindex(rebalance_dates).shift(1).fillna(0.0)
        position_change = position_lagged.diff().fillna(position_lagged.iloc[0] if len(position_lagged) else 0.0)

        log_ret = np.log(price / price.shift(1))
        daily_vol = causal_rolling_vol(log_ret).reindex(rebalance_dates)
        adv = causal_rolling_adv(price, volumes[sym]).reindex(rebalance_dates)

        asset_net_returns = {}
        for t in rebalance_dates:
            gross_r = float(position_lagged.loc[t] * forward_returns.loc[t, sym]) if pd.notna(forward_returns.loc[t, sym]) else 0.0
            gross_pnl = gross_r * capital_per_asset

            d_pos = abs(position_change.loc[t])
            cost = 0.0
            if d_pos > 1e-9:
                v, a = daily_vol.loc[t], adv.loc[t]
                if pd.notna(v) and pd.notna(a) and a > 0:
                    trade_value = capital_per_asset * d_pos
                    cost = cost_model.cost_dollars(trade_value, a, v)

            net_pnl = gross_pnl - cost
            period_gross[t] += gross_pnl
            period_net[t] += net_pnl
            asset_net_returns[t] = net_pnl / capital_per_asset

        per_asset_returns[sym] = pd.Series(asset_net_returns)

    gross_returns = pd.Series(period_gross) / total_capital
    net_returns = pd.Series(period_net) / total_capital

    periods_per_year = 252 / REBALANCE_FREQ_DAYS
    gross_sharpe = sharpe_ratio(gross_returns.dropna().to_numpy(), periods_per_year=periods_per_year)
    net_sharpe = sharpe_ratio(net_returns.dropna().to_numpy(), periods_per_year=periods_per_year)

    per_asset_sharpe = {
        sym: sharpe_ratio(r.dropna().to_numpy(), periods_per_year=periods_per_year)
        for sym, r in per_asset_returns.items()
        if r.notna().sum() > 2 and r.std() > 0
    }

    return TSMOMBacktestResult(
        returns=net_returns,
        per_asset_net_sharpe=per_asset_sharpe,
        n_assets=len(symbols),
        n_rebalances=len(net_returns),
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
    )

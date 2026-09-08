"""Calendar-time portfolio backtest (Fama, 1998): trade only events whose
announcement reaction exceeds a threshold (skip noise-level 8-Ks), enter
the direction of the announcement, hold `drift_days`, and aggregate every
symbol's concurrently open trades into one daily portfolio return series.

Fixed-rule strategy (threshold/drift_days/direction rule are all
pre-specified, nothing fitted) -- same reasoning as Phase 3/6 for skipping
purged walk-forward CV and DSR/PBO correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import sharpe_ratio

from .events import compute_event_outcomes, pead_significance, PeadSignificanceResult


@dataclass
class PeadBacktestResult:
    returns: pd.Series  # daily NET calendar-time portfolio returns
    significance: PeadSignificanceResult
    n_events_total: int
    n_events_traded: int
    max_concurrent_trades: int
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float


def run_pead_backtest(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    event_dates_by_symbol: dict[str, list],
    drift_days: int = 21,
    announcement_threshold: float = 0.02,
    capital_per_trade: float = 50_000.0,
    cost_model: TransactionCostModel | None = None,
) -> PeadBacktestResult:
    cost_model = cost_model or TransactionCostModel()
    symbols = sorted(s for s in event_dates_by_symbol if s in prices)
    if not symbols:
        raise ValueError("no symbols with both price data and event dates")

    all_outcomes = []
    trades = []  # (symbol, entry_date, exit_date, direction)
    for sym in symbols:
        outcomes = compute_event_outcomes(prices[sym], event_dates_by_symbol[sym], drift_days)
        if outcomes.empty:
            continue
        outcomes = outcomes.copy()
        outcomes["symbol"] = sym
        all_outcomes.append(outcomes)
        traded = outcomes[outcomes["announcement_return"].abs() > announcement_threshold]
        for event_date, row in traded.iterrows():
            direction = 1.0 if row["announcement_return"] > 0 else -1.0
            trades.append((sym, row["entry_date"], row["exit_date"], direction))

    if not all_outcomes:
        raise ValueError("no events produced a valid outcome window for any symbol")
    pooled_outcomes = pd.concat(all_outcomes)
    significance = pead_significance(pooled_outcomes)

    if not trades:
        raise ValueError(
            f"no events exceeded announcement_threshold={announcement_threshold} -- nothing to trade"
        )

    common_days = prices[symbols[0]].index
    for sym in symbols[1:]:
        common_days = common_days.union(prices[sym].index)
    common_days = common_days.sort_values()

    daily_vol, adv = {}, {}
    for sym in symbols:
        log_ret = np.log(prices[sym] / prices[sym].shift(1))
        daily_vol[sym] = causal_rolling_vol(log_ret)
        adv[sym] = causal_rolling_adv(prices[sym], volumes[sym])

    # A REAL BUG lived here in the first version of this function: it built
    # a single global list of "active days" (any day ANY trade was open)
    # and computed each day's return as price[day]/price[prev_GLOBAL_day],
    # rather than each trade's own day-over-day return. That mixed up whose
    # return belongs to whom, but the fatal instance of it was on a trade's
    # very first charged day: entry_date is literally the LAST day of the
    # 2-day announcement window (idx[pos+1]) used to pick the trade's
    # direction, so crediting a return ENDING on entry_date re-scored part
    # of the same jump that decided the trade's direction -- a
    # self-fulfilling, look-ahead-tainted "return" that had nothing to do
    # with actual drift. It produced a gross Sharpe of 2.42 on a universe
    # whose own event study showed p=0.62 (no real relationship at all) --
    # a huge, internally inconsistent red flag caught specifically by
    # cross-checking the "tradeable" result against the "is this even a
    # real effect" result, not by any single test.
    #
    # Fixed by computing each trade's return series directly via its own
    # price slice: `.pct_change()` on prices[entry:exit] naturally starts
    # its first non-NaN value at entry+1, never crediting a return that
    # ends AT entry -- entry is the state the position starts FROM, not a
    # return day itself.
    trade_pnl_by_date: dict[pd.Timestamp, float] = {}
    concurrent_count_by_date: dict[pd.Timestamp, int] = {}
    entry_cost_by_day: dict[pd.Timestamp, float] = {}
    exit_cost_by_day: dict[pd.Timestamp, float] = {}

    for sym, entry, exit_, direction in trades:
        sym_prices = prices[sym].loc[entry:exit_]
        daily_ret = sym_prices.pct_change().dropna()
        for d, r in daily_ret.items():
            trade_pnl_by_date[d] = trade_pnl_by_date.get(d, 0.0) + direction * capital_per_trade * r

        window = common_days[(common_days >= entry) & (common_days <= exit_)]
        for d in window:
            concurrent_count_by_date[d] = concurrent_count_by_date.get(d, 0) + 1

        for day, cost_bucket in ((entry, entry_cost_by_day), (exit_, exit_cost_by_day)):
            v, a = daily_vol[sym].get(day, np.nan), adv[sym].get(day, np.nan)
            if pd.notna(v) and pd.notna(a) and a > 0:
                cost_bucket[day] = cost_bucket.get(day, 0.0) + cost_model.cost_dollars(
                    capital_per_trade, a, v
                )

    max_concurrent = max(concurrent_count_by_date.values(), default=0)
    total_capital = max(max_concurrent, 1) * capital_per_trade

    all_dates = sorted(set(trade_pnl_by_date) | set(entry_cost_by_day) | set(exit_cost_by_day))
    gross_returns, net_returns = {}, {}
    for day in all_dates:
        gross_pnl = trade_pnl_by_date.get(day, 0.0)
        day_cost = entry_cost_by_day.get(day, 0.0) + exit_cost_by_day.get(day, 0.0)
        gross_returns[day] = gross_pnl / total_capital
        net_returns[day] = (gross_pnl - day_cost) / total_capital

    gross_series = pd.Series(gross_returns).sort_index()
    net_series = pd.Series(net_returns).sort_index()

    gross_sharpe = sharpe_ratio(gross_series.to_numpy(), periods_per_year=252)
    net_sharpe = sharpe_ratio(net_series.to_numpy(), periods_per_year=252)

    return PeadBacktestResult(
        returns=net_series,
        significance=significance,
        n_events_total=len(pooled_outcomes),
        n_events_traded=len(trades),
        max_concurrent_trades=max_concurrent,
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
    )

"""Ties factor construction + IC calibration + portfolio construction into
a walk-forward, cost-aware, out-of-sample backtest.

This is the first strategy in the platform with an actually FITTED
parameter -- the factor weights, calibrated from training-period ICs -- so
this is where Phase 2's purged walk-forward CV does real work (Phase 3's
pairs strategy had no fitted parameters and deliberately did NOT use it;
see strategies/pairs/backtest.py's module docstring for why forcing it in
there would have been theater). Here, skipping it would be a real bug:
without a train/test split, the "predictive" weights would be calibrated on
the same dates they're evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import sharpe_ratio
from alpha_desk.validation.splits import purged_walk_forward_splits

from .factors import low_volatility, momentum_12_1, quality_roe, value_earnings_yield
from .portfolio import construct_long_short
from .scoring import FactorWeights, calibrate_factor_weights, composite_score, cross_sectional_zscore

REBALANCE_FREQ_DAYS = 21  # ~monthly


def build_rebalance_dates(trading_days: pd.DatetimeIndex, freq_days: int = REBALANCE_FREQ_DAYS) -> pd.DatetimeIndex:
    return trading_days[::freq_days]


def build_factor_panels(
    prices: dict[str, pd.Series],
    fundamentals: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Raw (not yet cross-sectionally standardized) factor panels, one
    (rebalance_date x symbol) DataFrame per factor.
    """
    symbols = sorted(prices)
    momentum, low_vol, value, quality = {}, {}, {}, {}
    for sym in symbols:
        price = prices[sym]
        momentum[sym] = momentum_12_1(price).reindex(rebalance_dates)
        low_vol[sym] = low_volatility(price).reindex(rebalance_dates)
        value[sym] = value_earnings_yield(fundamentals[sym], price, rebalance_dates)
        quality[sym] = quality_roe(fundamentals[sym], rebalance_dates)
    return {
        "momentum": pd.DataFrame(momentum),
        "low_vol": pd.DataFrame(low_vol),
        "value": pd.DataFrame(value),
        "quality": pd.DataFrame(quality),
    }


def build_forward_returns(prices: dict[str, pd.Series], rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """forward_returns.loc[t, sym] = the return REALIZED from rebalance date
    t to the NEXT rebalance date -- both the IC-calibration label and what a
    portfolio entered at t actually earns by the next rebalance. The final
    rebalance date has no "next" date and is NaN by construction (nothing
    to compare it to yet).
    """
    symbols = sorted(prices)
    out = {}
    for sym in symbols:
        px = prices[sym].reindex(rebalance_dates)
        out[sym] = px.shift(-1) / px - 1.0
    return pd.DataFrame(out)


@dataclass
class FactorBacktestResult:
    returns: pd.Series  # per-rebalance-period NET returns, OOS only, all folds concatenated
    fold_weights: list[FactorWeights]
    n_folds: int
    n_rebalances: int
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float


def run_factor_backtest(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    fundamentals: dict[str, pd.DataFrame],
    quantile: float = 0.2,
    capital: float = 1_000_000.0,
    cost_model: TransactionCostModel | None = None,
    n_splits: int = 5,
    embargo: int = 1,
) -> FactorBacktestResult:
    cost_model = cost_model or TransactionCostModel()
    symbols = sorted(prices)

    common_days = prices[symbols[0]].index
    for sym in symbols[1:]:
        common_days = common_days.intersection(prices[sym].index)
    common_days = common_days.sort_values()

    rebalance_dates = build_rebalance_dates(common_days)
    if len(rebalance_dates) < 30:
        raise ValueError(f"only {len(rebalance_dates)} rebalance dates available, need >= 30")

    raw_panels = build_factor_panels(prices, fundamentals, rebalance_dates)
    z_panels = {name: cross_sectional_zscore(panel) for name, panel in raw_panels.items()}
    forward_returns = build_forward_returns(prices, rebalance_dates)

    # causal per-symbol vol/ADV, reindexed onto rebalance dates -- used to
    # cost each rebalance's trades.
    daily_vol, adv = {}, {}
    for sym in symbols:
        log_ret = np.log(prices[sym] / prices[sym].shift(1))
        daily_vol[sym] = causal_rolling_vol(log_ret).reindex(rebalance_dates)
        adv[sym] = causal_rolling_adv(prices[sym], volumes[sym]).reindex(rebalance_dates)
    daily_vol_df = pd.DataFrame(daily_vol)
    adv_df = pd.DataFrame(adv)

    splits = purged_walk_forward_splits(
        n_samples=len(rebalance_dates), label_horizon=1, n_splits=n_splits, embargo=embargo
    )

    all_returns: list[pd.Series] = []
    all_gross_returns: list[pd.Series] = []
    fold_weights: list[FactorWeights] = []

    for split in splits:
        train_dates = rebalance_dates[split.train_idx]
        test_dates = rebalance_dates[split.test_idx]

        train_panels = {name: panel.loc[train_dates] for name, panel in z_panels.items()}
        train_forward = forward_returns.loc[train_dates]
        weights = calibrate_factor_weights(train_panels, train_forward)
        fold_weights.append(weights)

        if sum(weights.weights.values()) == 0.0:
            # no factor showed positive training-period IC -- the honest
            # move is to sit out this fold (zero position), not force a
            # score out of factors calibrate_factor_weights already
            # rejected.
            fold_returns = pd.Series(0.0, index=test_dates)
            all_returns.append(fold_returns)
            all_gross_returns.append(fold_returns)
            continue

        test_panels = {name: panel.loc[test_dates] for name, panel in z_panels.items()}
        test_scores = composite_score(test_panels, weights.weights)  # whole fold at once, not re-wrapped per date

        prev_weights = pd.Series(0.0, index=symbols)  # each fold starts flat -- no continuity across the purge/embargo gap
        period_net_returns = {}
        period_gross_returns = {}

        for t in test_dates:
            score_t = test_scores.loc[t]
            try:
                target_weights = construct_long_short(score_t, quantile=quantile)
            except ValueError:
                # not enough valid names this period -- hold flat rather than crash the whole backtest
                target_weights = pd.Series(0.0, index=symbols)

            gross_return_t = float((target_weights * forward_returns.loc[t]).sum(skipna=True))

            turnover = (target_weights - prev_weights).abs()
            cost_t = 0.0
            for sym in symbols:
                d_w = turnover.get(sym, 0.0)
                if d_w <= 1e-12:
                    continue
                trade_value = capital * d_w
                v, a = daily_vol_df.loc[t, sym], adv_df.loc[t, sym]
                if pd.isna(v) or pd.isna(a) or a <= 0:
                    continue
                cost_t += cost_model.cost_dollars(trade_value, a, v)

            period_gross_returns[t] = gross_return_t
            period_net_returns[t] = gross_return_t - (cost_t / capital)
            prev_weights = target_weights

        all_returns.append(pd.Series(period_net_returns))
        all_gross_returns.append(pd.Series(period_gross_returns))

    net_returns = pd.concat(all_returns).sort_index()
    gross_returns = pd.concat(all_gross_returns).sort_index()

    periods_per_year = 252 / REBALANCE_FREQ_DAYS
    net_sharpe = sharpe_ratio(net_returns.dropna().to_numpy(), periods_per_year=periods_per_year)
    gross_sharpe = sharpe_ratio(gross_returns.dropna().to_numpy(), periods_per_year=periods_per_year)

    return FactorBacktestResult(
        returns=net_returns,
        fold_weights=fold_weights,
        n_folds=len(splits),
        n_rebalances=len(net_returns),
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
    )

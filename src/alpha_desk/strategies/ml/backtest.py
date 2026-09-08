"""Walk-forward train/predict/trade for the ML classifier -- same purged
split discipline as Phase 4 (this model IS fitted, so the train/test
separation is load-bearing, not optional), same cost model, same
dollar-neutral portfolio construction, but scored first as a CLASSIFIER
(PR-AUC/Brier/log-loss/calibration) before ever being judged as a trading
strategy (Sharpe) -- reporting only the Sharpe would hide whether the
model's probabilities mean anything at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.strategies.factors.backtest import build_rebalance_dates
from alpha_desk.strategies.factors.portfolio import construct_long_short
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import sharpe_ratio
from alpha_desk.validation.splits import purged_walk_forward_splits

from .features import build_feature_table
from .model import ClassifierMetrics, evaluate_classifier, predict_proba, train_classifier

REBALANCE_FREQ_DAYS = 21


@dataclass
class MLBacktestResult:
    returns: pd.Series  # per-rebalance-period NET returns, OOS only, all folds concatenated
    fold_metrics: list[ClassifierMetrics | None]  # None where a fold was skipped (see module docstring)
    pooled_metrics: ClassifierMetrics  # all OOS (label, prediction) pairs pooled -- the stable headline number
    n_folds: int
    n_folds_skipped: int
    n_rebalances: int
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float


def run_ml_backtest(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    fundamentals: dict[str, pd.DataFrame],
    macro: dict[str, pd.Series],
    quantile: float = 0.2,
    capital: float = 1_000_000.0,
    cost_model: TransactionCostModel | None = None,
    n_splits: int = 3,
    embargo: int = 1,
) -> MLBacktestResult:
    cost_model = cost_model or TransactionCostModel()
    symbols = sorted(prices)

    common_days = prices[symbols[0]].index
    for sym in symbols[1:]:
        common_days = common_days.intersection(prices[sym].index)
    common_days = common_days.sort_values()

    rebalance_dates = build_rebalance_dates(common_days, REBALANCE_FREQ_DAYS)
    if len(rebalance_dates) < 30:
        raise ValueError(f"only {len(rebalance_dates)} rebalance dates available, need >= 30")

    features, label, forward_returns = build_feature_table(prices, fundamentals, macro, rebalance_dates)

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
    fold_metrics: list[ClassifierMetrics | None] = []
    pooled_y: list[pd.Series] = []
    pooled_p: list[pd.Series] = []
    n_skipped = 0

    for split in splits:
        train_dates = rebalance_dates[split.train_idx]
        test_dates = rebalance_dates[split.test_idx]

        X_train = features.loc[(train_dates, slice(None)), :]
        y_train = label.loc[X_train.index]
        X_test = features.loc[(test_dates, slice(None)), :]
        y_test = label.loc[X_test.index]

        try:
            trained = train_classifier(X_train, y_train)
        except ValueError:
            n_skipped += 1
            fold_metrics.append(None)
            fold_returns = pd.Series(0.0, index=test_dates)
            all_returns.append(fold_returns)
            all_gross_returns.append(fold_returns)
            continue

        proba = predict_proba(trained, X_test)
        pooled_y.append(y_test)
        pooled_p.append(proba)
        try:
            fold_metrics.append(evaluate_classifier(y_test, proba))
        except ValueError:
            fold_metrics.append(None)

        prev_weights = pd.Series(0.0, index=symbols)
        period_net, period_gross = {}, {}
        for t in test_dates:
            if t not in proba.index.get_level_values("date"):
                continue
            score_t = proba.xs(t, level="date")
            try:
                target_weights = construct_long_short(score_t, quantile=quantile)
            except ValueError:
                target_weights = pd.Series(0.0, index=symbols)
            target_weights = target_weights.reindex(symbols, fill_value=0.0)

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

            period_gross[t] = gross_return_t
            period_net[t] = gross_return_t - (cost_t / capital)
            prev_weights = target_weights

        all_returns.append(pd.Series(period_net))
        all_gross_returns.append(pd.Series(period_gross))

    net_returns = pd.concat(all_returns).sort_index()
    gross_returns = pd.concat(all_gross_returns).sort_index()

    periods_per_year = 252 / REBALANCE_FREQ_DAYS
    net_sharpe = sharpe_ratio(net_returns.dropna().to_numpy(), periods_per_year=periods_per_year)
    gross_sharpe = sharpe_ratio(gross_returns.dropna().to_numpy(), periods_per_year=periods_per_year)

    if not pooled_y:
        raise ValueError("every fold was skipped -- no OOS predictions to evaluate at all")
    pooled_metrics = evaluate_classifier(pd.concat(pooled_y), pd.concat(pooled_p))

    return MLBacktestResult(
        returns=net_returns,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        n_folds=len(splits),
        n_folds_skipped=n_skipped,
        n_rebalances=len(net_returns),
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
    )

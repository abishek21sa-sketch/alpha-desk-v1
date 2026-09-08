"""Wires cointegration screening + Kalman hedge ratio + z-score signals +
Phase 2's transaction cost model into a cost-aware pairs-trading backtest.

Deliberately does NOT run this through Phase 2's purged walk-forward CV:
that tool exists to stop a FITTED model from training on data it will
later be tested on. This strategy has no fitted parameters -- delta,
entry/exit thresholds, and the Kalman filter's R calibration window are all
fixed hyperparameters, and the filter itself is already causal by
construction (see test_kalman.py). Forcing a train/test split onto a rule
with nothing to fit would be theater, not a real safeguard.

The REAL overfitting risk in a pairs strategy lives one level up, at PAIR
SELECTION: screening many candidate pairs and trading whichever looks best
is exactly the "best of N trials" problem Phase 2's Deflated Sharpe Ratio
and Probability of Backtest Overfitting exist to correct for. That
correction is applied where the pairs are actually screened and compared
(`scripts/run_pairs_strategy.py`), not inside this single-pair function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import sharpe_ratio

from .kalman import kalman_hedge_ratio
from .signals import generate_positions

DEFAULT_BURN_IN = 90  # >= kalman's R-calibration window (60) + rolling vol/ADV window (20) + slack


@dataclass
class PairBacktestResult:
    symbol_a: str
    symbol_b: str
    returns: pd.Series  # fractional returns of capital_per_leg, post-burn-in
    n_trades: int
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float  # gross - net, in annualized-Sharpe space


def backtest_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    volume_a: pd.Series,
    volume_b: pd.Series,
    symbol_a: str,
    symbol_b: str,
    kalman_delta: float = 1e-6,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    capital_per_leg: float = 100_000.0,
    cost_model: TransactionCostModel | None = None,
    burn_in: int = DEFAULT_BURN_IN,
) -> PairBacktestResult:
    cost_model = cost_model or TransactionCostModel()

    df = pd.DataFrame(
        {"a": price_a, "b": price_b, "vol_a": volume_a, "vol_b": volume_b}
    ).dropna().sort_index()
    if len(df) < burn_in + 200:
        raise ValueError(
            f"need at least {burn_in + 200} overlapping observations, got {len(df)}"
        )

    log_a = np.log(df["a"])
    log_b = np.log(df["b"])
    r_a = log_a.diff()
    r_b = log_b.diff()

    # kalman_hedge_ratio(x, y) fits y = alpha + beta*x -- here x=log_a,
    # y=log_b, so the spread is (log_b - beta*log_a) and its return is
    # r_b - beta*r_a. Getting this backwards (r_a - beta*r_b) was a real bug
    # caught while building this: it silently trades a linear combination
    # that has nothing to do with the actual cointegrating relationship
    # whenever beta != 1, and degraded a strategy with a real, verified
    # (oracle-checked) edge down to a near-zero Sharpe. See
    # docs/PAIRS_TRADING.md.
    kf = kalman_hedge_ratio(log_a, log_b, delta=kalman_delta)
    # position/beta used for the return realized AT t must only reflect
    # information through t-1 -- both the z-score that triggers the trade
    # decision and the hedge ratio that sizes it are lagged by one bar.
    beta_lagged = kf.beta.shift(1)
    z_lagged = kf.z_score.shift(1)

    positions = generate_positions(z_lagged, entry_threshold, exit_threshold)
    position_change = positions.diff().fillna(positions.iloc[0])

    spread_return = r_b - beta_lagged * r_a
    gross_pnl = positions * capital_per_leg * spread_return

    daily_vol_a = causal_rolling_vol(r_a)
    daily_vol_b = causal_rolling_vol(r_b)
    adv_a = causal_rolling_adv(df["a"], df["vol_a"])
    adv_b = causal_rolling_adv(df["b"], df["vol_b"])

    post_burn_in = df.index[burn_in:]

    cost = pd.Series(0.0, index=df.index)
    n_trades = 0
    traded_bars = position_change[position_change.abs() > 1e-9].index
    for t in traded_bars:
        d_pos = abs(position_change.loc[t])
        beta_at_t = beta_lagged.loc[t] if not pd.isna(beta_lagged.loc[t]) else 1.0
        # b is the primary (1-unit) leg, a is the beta-scaled hedge leg --
        # must match spread_return's r_b - beta*r_a convention above.
        trade_value_b = capital_per_leg * d_pos
        trade_value_a = capital_per_leg * abs(beta_at_t) * d_pos

        va, adva = daily_vol_a.loc[t], adv_a.loc[t]
        vb, advb = daily_vol_b.loc[t], adv_b.loc[t]
        if pd.isna(va) or pd.isna(adva) or pd.isna(vb) or pd.isna(advb) or adva <= 0 or advb <= 0:
            continue  # burn-in period, no reliable cost inputs yet -- charge nothing rather than guess
        cost.loc[t] = cost_model.cost_dollars(trade_value_a, adva, va) + cost_model.cost_dollars(
            trade_value_b, advb, vb
        )
        if t in post_burn_in:
            n_trades += 1  # count only trades that land inside the reported (post-burn-in) return series

    net_pnl = gross_pnl - cost

    net_returns = (net_pnl.loc[post_burn_in] / capital_per_leg).dropna()
    gross_returns = (gross_pnl.loc[post_burn_in] / capital_per_leg).dropna()

    net_sharpe = sharpe_ratio(net_returns.to_numpy(), periods_per_year=252)
    gross_sharpe = sharpe_ratio(gross_returns.to_numpy(), periods_per_year=252)

    return PairBacktestResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        returns=net_returns,
        n_trades=n_trades,
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
    )

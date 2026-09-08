"""Cost-aware daily backtest of the VIX term-structure signal.

Like Phase 3 (pairs) and Phase 6 (TSMOM), this is a FIXED rule (long the
named instrument whenever the term structure matches `long_when`, flat
otherwise) -- no fitted parameters, so no purged walk-forward CV and no
DSR/PBO correction to apply, only the Probabilistic Sharpe Ratio as an
honesty check on the one configuration actually run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_desk.data.rolling import causal_rolling_adv, causal_rolling_vol
from alpha_desk.validation.costs import TransactionCostModel
from alpha_desk.validation.performance import probabilistic_sharpe_ratio, sharpe_ratio

from .signals import directional_position

MIN_HISTORY_DAYS = 100


@dataclass
class VolRelValResult:
    instrument: str
    long_when: str
    returns: pd.Series  # net daily returns on days with valid data
    n_days: int
    pct_days_in_position: float
    gross_sharpe_annualized: float
    net_sharpe_annualized: float
    cost_drag_annualized: float
    probabilistic_sharpe_ratio: float


def run_vol_relval_backtest(
    vix: pd.Series,
    vix3m: pd.Series,
    tradeable_price: pd.Series,
    tradeable_volume: pd.Series,
    instrument: str,
    long_when: str = "contango",
    capital: float = 100_000.0,
    cost_model: TransactionCostModel | None = None,
) -> VolRelValResult:
    cost_model = cost_model or TransactionCostModel()

    common = vix.index.intersection(vix3m.index).intersection(tradeable_price.index).intersection(tradeable_volume.index)
    common = common.sort_values()
    if len(common) < MIN_HISTORY_DAYS:
        raise ValueError(f"need at least {MIN_HISTORY_DAYS} days of common history, got {len(common)}")

    vix = vix.reindex(common)
    vix3m = vix3m.reindex(common)
    price = tradeable_price.reindex(common)
    volume = tradeable_volume.reindex(common)

    signal = directional_position(vix, vix3m, long_when)
    # position HELD on day t was decided using t-1's close (the lag
    # contract every other strategy on this platform uses) -- shift(1),
    # not shift(0), or this would be trading on same-day-close information.
    position_lagged = signal.shift(1).fillna(0.0)
    position_change = position_lagged.diff().fillna(position_lagged.iloc[0] if len(position_lagged) else 0.0)

    daily_return = price.pct_change()
    log_ret = np.log(price / price.shift(1))
    daily_vol = causal_rolling_vol(log_ret)
    adv = causal_rolling_adv(price, volume)

    gross_returns = position_lagged * daily_return

    costs = pd.Series(0.0, index=common)
    for t in common:
        d_pos = abs(position_change.loc[t])
        if d_pos > 1e-9:
            v, a = daily_vol.loc[t], adv.loc[t]
            if pd.notna(v) and pd.notna(a) and a > 0:
                trade_value = capital * d_pos
                costs.loc[t] = cost_model.cost_dollars(trade_value, a, v)

    net_returns = gross_returns - costs / capital

    valid_mask = net_returns.notna() & gross_returns.notna()
    net_valid = net_returns[valid_mask]
    gross_valid = gross_returns[valid_mask]
    if len(net_valid) < MIN_HISTORY_DAYS:
        raise ValueError(f"fewer than {MIN_HISTORY_DAYS} valid return observations after alignment")

    gross_sharpe = sharpe_ratio(gross_valid.to_numpy(), periods_per_year=252)
    net_sharpe = sharpe_ratio(net_valid.to_numpy(), periods_per_year=252)
    psr = probabilistic_sharpe_ratio(net_valid.to_numpy())

    return VolRelValResult(
        instrument=instrument,
        long_when=long_when,
        returns=net_valid,
        n_days=len(net_valid),
        pct_days_in_position=float(signal.reindex(net_valid.index).mean()),
        gross_sharpe_annualized=float(gross_sharpe),
        net_sharpe_annualized=float(net_sharpe),
        cost_drag_annualized=float(gross_sharpe - net_sharpe),
        probabilistic_sharpe_ratio=float(psr),
    )

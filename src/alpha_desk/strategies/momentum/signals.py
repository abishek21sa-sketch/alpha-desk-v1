"""TSMOM signal construction: trailing return sign, vol-targeted sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(price: pd.Series, lookback_days: int = 252) -> pd.Series:
    """Total return over the trailing `lookback_days` ending at each date
    (causal: value at t uses only price[t] and price[t-lookback_days])."""
    return price / price.shift(lookback_days) - 1.0


def realized_vol(price: pd.Series, window: int = 60) -> pd.Series:
    """Annualized realized volatility of daily log returns, trailing `window` days."""
    log_ret = np.log(price / price.shift(1))
    return log_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def tsmom_position(
    price: pd.Series,
    lookback_days: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.10,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Position size AT each date t, using only information through t (the
    caller is responsible for lagging this by one bar before multiplying
    against the return realized from t to t+1 -- same lag contract as
    strategies.pairs.signals.generate_positions).

    `max_leverage` caps |position| to guard against a near-zero realized-vol
    reading (a very quiet stretch) producing an absurdly large vol-targeted
    size -- a real, standard risk control for this construction, not an
    afterthought.
    """
    signal = np.sign(trailing_return(price, lookback_days))
    vol = realized_vol(price, vol_window)
    position = signal * (target_vol / vol)
    return position.clip(lower=-max_leverage, upper=max_leverage)

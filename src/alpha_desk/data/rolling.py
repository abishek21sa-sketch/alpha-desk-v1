"""Causal (backward-looking, self-excluding) rolling estimates used by every
strategy's cost model to size trades -- shared by Phase 3 (pairs) and
Phase 4 (factors) rather than duplicated, since both need the exact same
"what would a trader actually have known at t" volatility/liquidity inputs.
"""

from __future__ import annotations

import pandas as pd


def causal_rolling_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realized volatility of daily log returns, shifted so the
    value AT t reflects only returns strictly BEFORE t (never t's own
    return) -- a cost estimate for trading at t cannot depend on t's own
    outcome.
    """
    return log_returns.rolling(window, min_periods=5).std().shift(1)


def causal_rolling_adv(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Trailing average daily dollar volume, same shift-by-one causality rule."""
    return (price * volume).rolling(window, min_periods=5).mean().shift(1)

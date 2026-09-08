"""Historical (empirical-quantile) VaR and CVaR -- no distributional
assumption (not parametric/Gaussian), since none of this platform's
strategy return series are close to normally distributed (fat tails,
skew from options-like payoffs in the pairs/PEAD event trades).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VarCvarResult:
    confidence: float
    var: float  # a POSITIVE number representing the loss magnitude at this confidence level
    cvar: float  # expected loss magnitude GIVEN a loss beyond VaR (always >= VaR)
    n_obs: int


def historical_var_cvar(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> VarCvarResult:
    """VaR at `confidence` (e.g. 0.95 -> the 5th percentile loss) and CVaR
    (mean of all losses at or beyond that percentile). Both returned as
    POSITIVE numbers (loss magnitudes), the common risk-reporting
    convention, even though the underlying returns are signed.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    values = pd.Series(returns).dropna().to_numpy()
    if len(values) < 20:
        raise ValueError(f"need at least 20 observations for a meaningful empirical quantile, got {len(values)}")

    alpha = 1 - confidence
    var_quantile = np.quantile(values, alpha)
    tail = values[values <= var_quantile]
    cvar_value = tail.mean() if len(tail) > 0 else var_quantile

    return VarCvarResult(
        confidence=confidence,
        var=float(-var_quantile),
        cvar=float(-cvar_value),
        n_obs=len(values),
    )

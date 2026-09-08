"""Dollar-neutral long/short portfolio construction from a cross-sectional
composite score. Deliberately simple (equal-weighted quantile buckets, no
optimizer) -- Phase 4's contribution is the CALIBRATED SCORE, not a
sophisticated portfolio optimizer; MILP/QP-based construction (mean-
variance, CVaR-aware) is a natural later phase, not duplicated here.
"""

from __future__ import annotations

import pandas as pd


def construct_long_short(
    scores: pd.Series, quantile: float = 0.2, min_names_per_side: int = 3
) -> pd.Series:
    """scores: symbol -> composite score for ONE rebalance date (NaN
    entries are dropped -- a symbol with no valid score that period gets no
    position, not an imputed one). Returns symbol -> portfolio weight:
    equal-weighted within each side, long weights summing to +1.0 and short
    weights summing to -1.0 (dollar-neutral, gross exposure 2.0).

    Raises if the resulting side size would fall below `min_names_per_side`
    -- silently trading a 1-2 name "portfolio" because the universe or
    quantile was too small would hide a real concentration-risk problem
    behind a function that still "worked."
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    valid = scores.dropna()
    n_side = int(len(valid) * quantile)
    if n_side < min_names_per_side:
        raise ValueError(
            f"quantile={quantile} on {len(valid)} valid names gives {n_side} per side, "
            f"below min_names_per_side={min_names_per_side}"
        )

    ranked = valid.sort_values(ascending=False)
    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]

    weights = pd.Series(0.0, index=scores.index)
    weights.loc[longs] = 1.0 / n_side
    weights.loc[shorts] = -1.0 / n_side
    return weights

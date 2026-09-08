"""Cross-sectional standardization + Information-Coefficient-calibrated
composite factor score.

Same discipline as airlinesapp's Health Score: component weights are never
hand-picked. Each factor's weight is its own predictive power (Information
Coefficient -- the cross-sectional correlation between the factor's value
today and the ACTUAL forward return that followed), measured on a training
period and normalized so only genuinely positive-IC factors contribute,
summing to 1.0. A factor with negative or ~zero measured IC gets weight
ZERO, not a negative weight -- flipping a factor's sign based on a noisy
training-period correlation would be fitting noise, not skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: index=date, columns=symbol, values=raw factor reading.
    Standardizes WITHIN each date (row) across the cross-section of symbols,
    so factors on wildly different natural scales (an earnings yield vs. a
    momentum return vs. a volatility) become comparable before combining.
    """
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


@dataclass
class FactorWeights:
    weights: dict[str, float]  # factor name -> weight; sums to 1.0 over included factors (0.0 for excluded)
    ic_by_factor: dict[str, float]  # measured information coefficient per factor, for transparency


def calibrate_factor_weights(
    factor_panels: dict[str, pd.DataFrame], forward_returns: pd.DataFrame
) -> FactorWeights:
    """factor_panels: {factor_name: (date x symbol) panel}, already
    cross-sectionally z-scored. forward_returns: (date x symbol) panel
    where forward_returns.loc[t, sym] is the return REALIZED from t to
    t+horizon -- i.e. already the forward-looking label, not a
    to-be-shifted raw return (the caller owns getting that alignment right;
    see backtest.py).

    IC for a factor = the average, across rebalance dates, of that date's
    cross-sectional Pearson correlation between the factor's z-scores and
    forward_returns. Averaging per-date correlations (not pooling all
    date-symbol pairs into one correlation) is deliberate: it weights each
    rebalance date equally regardless of how many symbols had valid data
    that day, and matches how IC is conventionally reported in factor
    research.
    """
    ic_by_factor: dict[str, float] = {}
    for name, panel in factor_panels.items():
        daily_ic = panel.corrwith(forward_returns, axis=1)
        ic_by_factor[name] = float(daily_ic.mean())

    positive_ics = {name: ic for name, ic in ic_by_factor.items() if ic > 0}
    total = sum(positive_ics.values())

    if total <= 0:
        # no factor showed positive predictive power on this training
        # window -- honest answer is "no weights", not an arbitrary
        # fallback (e.g. equal-weighting everything would silently smuggle
        # in the negative/zero-IC factors this calibration exists to screen
        # out).
        weights = {name: 0.0 for name in factor_panels}
    else:
        weights = {name: (positive_ics.get(name, 0.0) / total) for name in factor_panels}

    return FactorWeights(weights=weights, ic_by_factor=ic_by_factor)


def composite_score(factor_panels: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of the (already z-scored) factor panels. Panels may have
    different NaN patterns (e.g. a stock missing a fundamentals filing);
    `fill_value=0` in the running sum means a symbol's missing factor
    contributes neither positively nor negatively that period rather than
    NaN-poisoning its entire composite score.
    """
    result: pd.DataFrame | None = None
    for name, w in weights.items():
        if w == 0.0:
            continue
        term = factor_panels[name] * w
        result = term if result is None else result.add(term, fill_value=0.0)
    if result is None:
        raise ValueError("no factor has nonzero weight -- nothing to score")
    return result

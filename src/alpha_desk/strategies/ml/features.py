"""Builds the (rebalance_date, symbol) feature/label table the classifier
trains on, from Phase 4's factor panels plus macro regime context.

Point-in-time discipline carries over unchanged from Phase 4: momentum/
low-vol are price-derived (no leakage risk beyond the rolling-window
lookback itself), value/quality join fundamentals on `filed`, and macro
series are reindexed with a forward-fill that only ever looks BACKWARD
(pandas' default `ffill` direction) -- a macro print for date d is only
usable from d onward, matching how it actually becomes public knowledge.
"""

from __future__ import annotations

import pandas as pd

from alpha_desk.strategies.factors.backtest import build_factor_panels, build_forward_returns
from alpha_desk.strategies.factors.scoring import cross_sectional_zscore

MACRO_FEATURE_SERIES = ["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2"]


def build_macro_features(macro: dict[str, pd.Series], rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One column per series in MACRO_FEATURE_SERIES, reindexed onto
    rebalance_dates with a backward-only forward-fill (never pulls in a
    value from a date after the one being filled).
    """
    cols = {}
    for series_id in MACRO_FEATURE_SERIES:
        if series_id not in macro:
            continue
        s = macro[series_id].sort_index()
        cols[series_id] = s.reindex(s.index.union(rebalance_dates)).ffill().reindex(rebalance_dates)
    return pd.DataFrame(cols, index=rebalance_dates)


def build_feature_table(
    prices: dict[str, pd.Series],
    fundamentals: dict[str, pd.DataFrame],
    macro: dict[str, pd.Series],
    rebalance_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (features, label, forward_returns):
      - features: (date, symbol) MultiIndex rows, one column per factor
        (z-scored) plus one per macro series (broadcast across symbols)
      - label: 1.0 if that (date, symbol)'s forward return beat the
        CROSS-SECTIONAL MEDIAN that date, else 0.0 -- a relative-
        performance classification target, deliberately market-direction-
        agnostic (a bad day for the whole market still has winners and
        losers by this definition, which is what a long/short book trades)
      - forward_returns: the (date x symbol) panel of raw forward returns,
        needed downstream to turn predictions into P&L
    """
    raw_panels = build_factor_panels(prices, fundamentals, rebalance_dates)
    z_panels = {name: cross_sectional_zscore(panel) for name, panel in raw_panels.items()}
    forward_returns = build_forward_returns(prices, rebalance_dates)
    macro_features = build_macro_features(macro, rebalance_dates)

    median_by_date = forward_returns.median(axis=1)
    label_panel = forward_returns.gt(median_by_date, axis=0).astype(float)
    label_panel = label_panel.where(forward_returns.notna())  # no label where there's no forward return to grade

    frames = []
    for name, panel in z_panels.items():
        frames.append(panel.stack(future_stack=True).rename(name))
    features = pd.concat(frames, axis=1)
    features.index.names = ["date", "symbol"]

    macro_broadcast = macro_features.reindex(features.index.get_level_values("date")).set_axis(
        features.index
    )
    features = pd.concat([features, macro_broadcast], axis=1)

    label = label_panel.stack(future_stack=True)
    label.index.names = ["date", "symbol"]
    label = label.reindex(features.index)

    return features, label, forward_returns

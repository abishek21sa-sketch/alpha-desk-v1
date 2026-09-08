from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.ml.features import build_feature_table, build_macro_features


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


class TestBuildMacroFeatures:
    def test_forward_fill_is_backward_only_never_from_the_future(self):
        # sparse macro observations; rebalance dates fall BETWEEN them.
        macro_dates = pd.to_datetime(["2018-01-01", "2018-03-01", "2018-06-01"])
        vix = pd.Series([15.0, 25.0, 12.0], index=macro_dates)
        rebalance_dates = pd.to_datetime(["2018-02-01", "2018-04-01", "2018-07-01"])

        result = build_macro_features({"VIXCLS": vix}, rebalance_dates)
        # 2018-02-01 is after the 2018-01-01 print (15.0) and before the
        # 2018-03-01 print -- must show 15.0, NOT the later 25.0.
        assert result.loc[pd.Timestamp("2018-02-01"), "VIXCLS"] == 15.0
        assert result.loc[pd.Timestamp("2018-04-01"), "VIXCLS"] == 25.0
        assert result.loc[pd.Timestamp("2018-07-01"), "VIXCLS"] == 12.0

    def test_date_before_first_observation_is_nan_not_backfilled(self):
        macro_dates = pd.to_datetime(["2018-06-01"])
        vix = pd.Series([20.0], index=macro_dates)
        rebalance_dates = pd.to_datetime(["2018-01-01"])
        result = build_macro_features({"VIXCLS": vix}, rebalance_dates)
        assert pd.isna(result.loc[pd.Timestamp("2018-01-01"), "VIXCLS"])

    def test_missing_series_is_silently_skipped(self):
        rebalance_dates = pd.to_datetime(["2018-01-01"])
        result = build_macro_features({}, rebalance_dates)
        assert list(result.columns) == []


class TestBuildFeatureTable:
    def _minimal_universe(self, n_days: int = 400):
        rng = np.random.default_rng(0)
        idx = _dates(n_days)
        prices, fundamentals = {}, {}
        # 3 symbols with clearly different, constant drift -> clearly
        # different forward-return ranks most periods
        drifts = {"HI": 0.002, "MID": 0.0, "LO": -0.002}
        for sym, drift in drifts.items():
            r = drift + rng.normal(0, 0.01, n_days)
            prices[sym] = pd.Series(100 * np.cumprod(1 + r), index=idx)
            rows = [
                {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2017-12-31", "filed": "2018-02-01", "val": 1e5},
                {"concept": "CommonStockSharesOutstanding", "form": "10-K", "end_date": "2017-12-31", "filed": "2018-02-01", "val": 1e4},
                {"concept": "StockholdersEquity", "form": "10-K", "end_date": "2017-12-31", "filed": "2018-02-01", "val": 5e5},
            ]
            df = pd.DataFrame(rows)
            df["end_date"] = pd.to_datetime(df["end_date"])
            df["filed"] = pd.to_datetime(df["filed"])
            fundamentals[sym] = df
        return prices, fundamentals

    def test_shape_is_multiindex_date_symbol(self):
        prices, fundamentals = self._minimal_universe()
        rebalance_dates = prices["HI"].index[::21]
        features, label, forward_returns = build_feature_table(prices, fundamentals, {}, rebalance_dates)
        assert isinstance(features.index, pd.MultiIndex)
        assert features.index.names == ["date", "symbol"]
        assert set(features.index.get_level_values("symbol")) == {"HI", "MID", "LO"}

    def test_label_matches_above_below_cross_sectional_median(self):
        # hand-build a forward_returns-equivalent scenario directly: 3
        # symbols, one date, clearly ranked returns -- HI must be labeled 1
        # (above median), LO must be labeled 0 (below median).
        prices, fundamentals = self._minimal_universe(n_days=400)
        rebalance_dates = prices["HI"].index[::21]
        features, label, forward_returns = build_feature_table(prices, fundamentals, {}, rebalance_dates)

        # spot check several dates: HI's drift is highest, LO's is lowest,
        # so HI should be labeled 1 and LO labeled 0 on most dates.
        hi_labels = label.xs("HI", level="symbol").dropna()
        lo_labels = label.xs("LO", level="symbol").dropna()
        assert hi_labels.mean() > 0.7
        assert lo_labels.mean() < 0.3

    def test_last_rebalance_date_has_nan_label(self):
        prices, fundamentals = self._minimal_universe()
        rebalance_dates = prices["HI"].index[::21]
        features, label, forward_returns = build_feature_table(prices, fundamentals, {}, rebalance_dates)
        last_date = rebalance_dates[-1]
        assert label.xs(last_date, level="date").isna().all()

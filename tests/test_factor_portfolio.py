from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.factors.portfolio import construct_long_short


class TestConstructLongShort:
    def test_dollar_neutral_and_correct_gross_exposure(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(20)})
        weights = construct_long_short(scores, quantile=0.2)
        assert weights.sum() == pytest.approx(0.0)
        assert weights.abs().sum() == pytest.approx(2.0)

    def test_top_scores_go_long_bottom_scores_go_short(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(20)})
        weights = construct_long_short(scores, quantile=0.2)
        # top 4 (S16..S19) should be long, bottom 4 (S0..S3) should be short
        assert (weights.loc[["S16", "S17", "S18", "S19"]] > 0).all()
        assert (weights.loc[["S0", "S1", "S2", "S3"]] < 0).all()
        assert (weights.loc[["S8", "S9", "S10", "S11"]] == 0).all()

    def test_equal_weighted_within_each_side(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(20)})
        weights = construct_long_short(scores, quantile=0.2)
        longs = weights[weights > 0]
        shorts = weights[weights < 0]
        assert longs.nunique() == 1
        assert shorts.nunique() == 1
        assert longs.iloc[0] == pytest.approx(1.0 / 4)
        assert shorts.iloc[0] == pytest.approx(-1.0 / 4)

    def test_nan_scores_excluded_and_get_no_position(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(20)})
        scores["S5"] = np.nan
        weights = construct_long_short(scores, quantile=0.2)
        assert weights["S5"] == 0.0
        # ranking must be computed only over the 19 valid names
        assert weights.abs().sum() == pytest.approx(2.0)

    def test_raises_when_side_would_be_too_small(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(5)})
        with pytest.raises(ValueError):
            construct_long_short(scores, quantile=0.2, min_names_per_side=3)

    def test_rejects_invalid_quantile(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(10)})
        with pytest.raises(ValueError):
            construct_long_short(scores, quantile=0.6)
        with pytest.raises(ValueError):
            construct_long_short(scores, quantile=0.0)

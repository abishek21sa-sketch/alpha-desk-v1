from __future__ import annotations

import pandas as pd
import pytest

from alpha_desk.strategies.vol_relval.signals import directional_position, term_structure_ratio


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


class TestTermStructureRatio:
    def test_contango_gives_ratio_above_one(self):
        vix = _series([15.0])
        vix3m = _series([18.0])
        ratio = term_structure_ratio(vix, vix3m)
        assert ratio.iloc[0] > 1.0

    def test_backwardation_gives_ratio_below_one(self):
        vix = _series([35.0])
        vix3m = _series([25.0])
        ratio = term_structure_ratio(vix, vix3m)
        assert ratio.iloc[0] < 1.0


class TestDirectionalPosition:
    def test_long_contango_is_flat_during_backwardation(self):
        vix = _series([15.0, 40.0])
        vix3m = _series([18.0, 25.0])  # day 0: contango, day 1: backwardation
        pos = directional_position(vix, vix3m, long_when="contango")
        assert pos.iloc[0] == 1.0
        assert pos.iloc[1] == 0.0

    def test_long_backwardation_is_the_exact_mirror(self):
        vix = _series([15.0, 40.0])
        vix3m = _series([18.0, 25.0])
        pos_contango = directional_position(vix, vix3m, long_when="contango")
        pos_backwardation = directional_position(vix, vix3m, long_when="backwardation")
        assert (pos_contango + pos_backwardation == 1.0).all()

    def test_rejects_invalid_long_when(self):
        vix = _series([15.0])
        vix3m = _series([18.0])
        with pytest.raises(ValueError):
            directional_position(vix, vix3m, long_when="sideways")

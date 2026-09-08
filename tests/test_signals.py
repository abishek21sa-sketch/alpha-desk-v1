from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.pairs.signals import generate_positions


class TestGeneratePositions:
    def test_rejects_bad_thresholds(self):
        z = pd.Series([0.0, 1.0])
        with pytest.raises(ValueError):
            generate_positions(z, entry_threshold=1.0, exit_threshold=1.0)
        with pytest.raises(ValueError):
            generate_positions(z, entry_threshold=-1.0)

    def test_enters_long_on_deep_negative_zscore(self):
        z = pd.Series([0.0, -1.0, -2.5, -2.5])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [0.0, 0.0, 1.0, 1.0]

    def test_enters_short_on_deep_positive_zscore(self):
        z = pd.Series([0.0, 1.0, 2.5, 2.5])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [0.0, 0.0, -1.0, -1.0]

    def test_exits_when_zscore_reverts(self):
        z = pd.Series([-2.5, -2.5, -0.3, -0.3])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [1.0, 1.0, 0.0, 0.0]

    def test_position_is_sticky_between_entry_and_exit_bands(self):
        # z drifts back toward 0 but stays outside the exit band -- position
        # must be held, not flattened prematurely.
        z = pd.Series([-2.5, -1.8, -1.2, -0.8, -0.6])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_flat_position_does_not_reenter_inside_entry_band(self):
        z = pd.Series([0.0, 1.0, 1.9, 0.5])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [0.0, 0.0, 0.0, 0.0]

    def test_can_flip_directly_from_long_through_flat_to_short(self):
        z = pd.Series([-2.5, -0.3, 2.5])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [1.0, 0.0, -1.0]

    def test_nan_zscore_holds_previous_position(self):
        z = pd.Series([-2.5, -2.5, np.nan, np.nan, -0.3])
        pos = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        assert list(pos) == [1.0, 1.0, 1.0, 1.0, 0.0]

    def test_documented_lag_contract(self):
        # this function's own docstring says positions are decided AT bar t
        # from information through t, and must be shifted forward by the
        # CALLER before being multiplied against a return realized t->t+1.
        # This test pins that contract down mechanically: position[t] must
        # be a deterministic function of z_score[:t+1] only -- changing
        # z_score AFTER index t must not change position[t].
        z = pd.Series([-2.5, -1.0, -0.3, 1.0, 2.5])
        full = generate_positions(z, entry_threshold=2.0, exit_threshold=0.5)
        truncated = generate_positions(z.iloc[:3], entry_threshold=2.0, exit_threshold=0.5)
        pd.testing.assert_series_equal(full.iloc[:3], truncated)

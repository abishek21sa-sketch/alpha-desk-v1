from __future__ import annotations

import numpy as np
import pytest

from alpha_desk.validation.splits import assert_no_leakage, purged_walk_forward_splits


class TestPurgedWalkForwardSplits:
    def test_train_always_precedes_test(self):
        splits = purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=5)
        for s in splits:
            assert s.train_idx.max() < s.test_idx.min()

    def test_no_leakage_with_purge_only(self):
        splits = purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=5, embargo=0)
        assert_no_leakage(splits, label_horizon=5)

    def test_no_leakage_with_embargo(self):
        splits = purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=5, embargo=10)
        assert_no_leakage(splits, label_horizon=5)

    def test_embargo_shrinks_train_set_vs_no_embargo(self):
        no_embargo = purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=5, embargo=0)
        with_embargo = purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=5, embargo=10)
        for a, b in zip(no_embargo, with_embargo):
            assert len(b.train_idx) <= len(a.train_idx)

    def test_zero_label_horizon_still_purges_exact_boundary(self):
        # label_horizon=0 means a sample's label is known immediately (no
        # forward-looking window) -- train should still stop exactly where
        # test starts, no overlap.
        splits = purged_walk_forward_splits(n_samples=100, label_horizon=0, n_splits=4)
        for s in splits:
            assert s.train_idx.max() < s.test_idx.min()
        assert_no_leakage(splits, label_horizon=0)

    def test_folds_are_contiguous_and_cover_the_tail(self):
        n_samples, n_splits = 200, 5
        splits = purged_walk_forward_splits(n_samples=n_samples, label_horizon=0, n_splits=n_splits)
        all_test_idx = np.concatenate([s.test_idx for s in splits])
        # first fold starts after the initial training block, which is
        # fold_size = n_samples // (n_splits + 1) samples (see splits.py)
        fold_size = n_samples // (n_splits + 1)
        assert all_test_idx.min() == fold_size
        assert all_test_idx.max() == n_samples - 1
        # no test index appears in two folds
        assert len(all_test_idx) == len(np.unique(all_test_idx))

    def test_raises_on_degenerate_config(self):
        with pytest.raises(ValueError):
            purged_walk_forward_splits(n_samples=10, label_horizon=5, n_splits=5, embargo=5)

    def test_raises_on_too_few_splits(self):
        with pytest.raises(ValueError):
            purged_walk_forward_splits(n_samples=200, label_horizon=5, n_splits=1)

    def test_synthetic_leakage_scenario_would_be_caught(self):
        # sanity-check the checker itself: hand-build a deliberately leaky
        # split and confirm assert_no_leakage actually raises.
        from alpha_desk.validation.splits import Split

        leaky = [Split(fold=0, train_idx=np.array([0, 1, 2, 8]), test_idx=np.array([10, 11]))]
        # sample 8's label window [8, 8+5]=[8,13] overlaps test start=10
        with pytest.raises(AssertionError):
            assert_no_leakage(leaky, label_horizon=5)

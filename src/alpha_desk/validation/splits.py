"""Purged, embargoed walk-forward cross-validation.

Standard k-fold CV silently leaks information on financial time series
because labels are forward-looking: a sample observed at day t whose label
is "5-day forward return" actually encodes information through day t+5. If
that window overlaps the test period, the model has effectively seen the
future. This is THE most common way a retail/academic backtest cheats
without anyone noticing -- see Lopez de Prado, "Advances in Financial
Machine Learning" (2018), ch. 7.

This module only ever trains on the chronological past of a test fold (no
mid-timeline test blocks with later folds treated as "past") -- the
strictest, most defensible setup for a trading backtest, not the more
general PurgedKFold that also permits test blocks in the middle of the
timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward_splits(
    n_samples: int,
    label_horizon: int,
    n_splits: int = 5,
    embargo: int = 0,
) -> list[Split]:
    """Expanding-window walk-forward splits over `n_samples` chronologically
    ordered observations (index 0 = earliest).

    Each sample i's label depends on data through index i + label_horizon
    (e.g. label_horizon=5 for a 5-day-forward-return label). For each of
    n_splits folds:
      - test block: the next contiguous chunk of the timeline
      - train: everything strictly before the test block's start, MINUS
        * purge -- any training sample i whose label window
          [i, i + label_horizon] extends into the test block
          (i.e. i + label_horizon >= test_start)
        * embargo -- an additional `embargo` samples immediately before the
          purge boundary, removed to absorb serial correlation the label
          window alone doesn't capture (Lopez de Prado's recommended fix
          for autocorrelated returns near a train/test boundary)

    Raises ValueError for degenerate configurations (too few samples for
    the requested split/embargo/horizon combination) rather than silently
    returning an empty or wrong-shaped split.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if label_horizon < 0:
        raise ValueError("label_horizon must be >= 0")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    # n_splits folds carved out of the back of the timeline, each of equal
    # size; everything before the first fold is the (growing) initial
    # training history.
    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_splits={n_splits} "
            f"(need at least {n_splits + 1} samples)"
        )

    splits: list[Split] = []
    for k in range(n_splits):
        test_start = fold_size * (k + 1)
        test_end = n_samples if k == n_splits - 1 else fold_size * (k + 2)
        test_idx = np.arange(test_start, test_end)

        purge_boundary = test_start - label_horizon - embargo
        purge_boundary = max(purge_boundary, 0)
        train_idx = np.arange(0, purge_boundary)

        if len(train_idx) == 0:
            raise ValueError(
                f"fold {k}: purge+embargo consumed the entire training set "
                f"(test_start={test_start}, label_horizon={label_horizon}, "
                f"embargo={embargo}) -- reduce n_splits, label_horizon, or embargo"
            )
        if len(test_idx) == 0:
            raise ValueError(f"fold {k}: empty test set")

        splits.append(Split(fold=k, train_idx=train_idx, test_idx=test_idx))

    return splits


def assert_no_leakage(splits: list[Split], label_horizon: int) -> None:
    """Verification helper (also used directly by tests): raises AssertionError
    if any split's train set contains a sample whose label window overlaps
    that split's test block, or if train/test indices overlap directly.
    """
    for s in splits:
        if len(s.train_idx) == 0 or len(s.test_idx) == 0:
            continue
        test_start = s.test_idx.min()
        overlap = s.train_idx[s.train_idx + label_horizon >= test_start]
        assert len(overlap) == 0, (
            f"fold {s.fold}: {len(overlap)} training sample(s) have label "
            f"windows overlapping the test block -- leakage"
        )
        assert len(np.intersect1d(s.train_idx, s.test_idx)) == 0, (
            f"fold {s.fold}: train/test index overlap"
        )

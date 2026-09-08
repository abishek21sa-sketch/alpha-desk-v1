"""Gradient-boosted classifier + probability calibration + honesty-first
evaluation metrics.

Deliberately NOT a linear model: `HistGradientBoostingClassifier` is a real
tree ensemble that can learn nonlinear factor interactions (e.g. "momentum
only matters when value is also cheap") a linear score can't represent --
the genuine machine-learning component of this platform, as opposed to
Phases 3-4's deterministic/econometric methods. It also handles missing
values (NaN factor readings from a burned-in rolling window, or a symbol
lacking a fundamentals filing) natively, which matters a lot for this
data's real-world raggedness.

Calibration matters because a tree ensemble's raw scores are not
well-calibrated probabilities out of the box -- `CalibratedClassifierCV`
with internal cross-validation (fit entirely on TRAIN data, never touching
TEST) fixes that before any probability is used to size a trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


@dataclass
class TrainedClassifier:
    calibrated_model: CalibratedClassifierCV
    feature_names: list[str]
    n_train_obs: int


def train_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 0
) -> TrainedClassifier:
    """Drops rows with a missing label (no forward return to grade); leaves
    missing FEATURE values as NaN, since HistGradientBoostingClassifier
    handles them natively rather than needing an imputer that could itself
    leak information across the train/test boundary if fit carelessly.
    """
    valid = y_train.notna()
    X, y = X_train.loc[valid], y_train.loc[valid]
    if len(X) < 50:
        raise ValueError(f"need at least 50 labeled training rows, got {len(X)}")
    if y.nunique() < 2:
        raise ValueError("training labels are single-class -- cannot fit a classifier")

    # A feature with fewer than 2 distinct non-NaN values in THIS training
    # slice crashes sklearn's histogram binner outright (a real bug found
    # against real data: a FRED series -- BAMLH0A0HYM2 -- turned out to
    # only have ~3 years of history via the fetch endpoint, versus decades
    # for every other series, so an early walk-forward fold's training
    # window saw it as all-NaN/near-constant and
    # HistGradientBoostingClassifier's `_find_binning_thresholds` raised
    # "window shape cannot be larger than input array shape"). Dropping
    # such columns for this fold is safer than crashing the whole backtest
    # over one sparse feature, and is a general data-hygiene guard, not a
    # one-off patch for that one series.
    usable_cols = [c for c in X.columns if X[c].nunique(dropna=True) >= 2]
    dropped = set(X.columns) - set(usable_cols)
    if dropped:
        X = X[usable_cols]
    if not usable_cols:
        raise ValueError("every feature is constant or all-NaN in this training window")

    base_model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=100, random_state=random_state
    )
    # cv=3 folds INSIDE the training set only -- the test fold this model
    # will later score is never involved in fitting either the base model
    # or its calibration.
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated.fit(X, y)

    return TrainedClassifier(
        calibrated_model=calibrated, feature_names=list(X.columns), n_train_obs=len(X)
    )


def predict_proba(trained: TrainedClassifier, X: pd.DataFrame) -> pd.Series:
    proba = trained.calibrated_model.predict_proba(X[trained.feature_names])[:, 1]
    return pd.Series(proba, index=X.index)


@dataclass
class ClassifierMetrics:
    pr_auc: float
    brier_score: float
    log_loss: float
    n_obs: int
    base_rate: float  # fraction of positive labels -- context for interpreting PR-AUC
    calibration_bins: pd.DataFrame  # columns: mean_predicted, mean_actual, count


def evaluate_classifier(y_true: pd.Series, y_pred_proba: pd.Series, n_bins: int = 5) -> ClassifierMetrics:
    aligned = pd.DataFrame({"y": y_true, "p": y_pred_proba}).dropna()
    if aligned.empty:
        raise ValueError("no overlapping non-NaN (y_true, y_pred_proba) pairs to evaluate")
    y, p = aligned["y"], aligned["p"]

    if y.nunique() < 2:
        raise ValueError("y_true is single-class over the evaluated rows -- PR-AUC/log-loss undefined")

    try:
        bin_id = pd.qcut(p, q=min(n_bins, p.nunique()), duplicates="drop")
    except ValueError:
        bin_id = pd.Series(0, index=p.index)
    calibration = (
        pd.DataFrame({"y": y, "p": p, "bin": bin_id})
        .groupby("bin", observed=True)
        .agg(mean_predicted=("p", "mean"), mean_actual=("y", "mean"), count=("y", "size"))
        .reset_index(drop=True)
    )

    return ClassifierMetrics(
        pr_auc=float(average_precision_score(y, p)),
        brier_score=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        n_obs=len(aligned),
        base_rate=float(y.mean()),
        calibration_bins=calibration,
    )

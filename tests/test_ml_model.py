from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.ml.model import evaluate_classifier, predict_proba, train_classifier


def _synthetic_classification_data(n: int, seed: int, signal_strength: float = 2.5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n), "noise": rng.normal(0, 1, n)}
    )
    true_logit = signal_strength * X["f1"] - signal_strength * 0.5 * X["f2"]
    true_prob = 1 / (1 + np.exp(-true_logit))
    y = pd.Series((rng.random(n) < true_prob).astype(float))
    return X, y, true_prob


class TestTrainClassifier:
    def test_raises_on_too_few_rows(self):
        X, y, _ = _synthetic_classification_data(20, seed=0)
        with pytest.raises(ValueError):
            train_classifier(X, y)

    def test_raises_on_single_class_labels(self):
        X, y, _ = _synthetic_classification_data(200, seed=0)
        y_constant = pd.Series(1.0, index=y.index)
        with pytest.raises(ValueError):
            train_classifier(X, y_constant)

    def test_drops_nan_labeled_rows_before_fitting(self):
        X, y, _ = _synthetic_classification_data(300, seed=0)
        y_with_nan = y.copy()
        y_with_nan.iloc[:100] = np.nan
        trained = train_classifier(X, y_with_nan)
        assert trained.n_train_obs == 200

    def test_drops_constant_or_all_nan_feature_columns_instead_of_crashing(self):
        # a real bug found against real data: HistGradientBoostingClassifier's
        # histogram binner raises a raw ValueError ("window shape cannot be
        # larger than input array shape") if a feature has fewer than 2
        # distinct non-NaN values in the training slice -- this happened
        # for a FRED series with much shorter history than the rest of the
        # feature set. train_classifier must drop such columns rather than
        # propagate that crash.
        X, y, _ = _synthetic_classification_data(300, seed=7)
        X_bad = X.copy()
        X_bad["all_nan_feature"] = np.nan
        X_bad["constant_feature"] = 1.0
        trained = train_classifier(X_bad, y)
        assert "all_nan_feature" not in trained.feature_names
        assert "constant_feature" not in trained.feature_names
        assert "f1" in trained.feature_names

    def test_raises_when_every_feature_is_degenerate(self):
        y = pd.Series(np.random.default_rng(0).integers(0, 2, 100).astype(float))
        X = pd.DataFrame({"const": [1.0] * 100, "nan_col": [np.nan] * 100})
        with pytest.raises(ValueError):
            train_classifier(X, y)


class TestPredictAndEvaluate:
    def test_genuine_signal_beats_chance_on_held_out_data(self):
        X_train, y_train, _ = _synthetic_classification_data(2000, seed=1)
        X_test, y_test, _ = _synthetic_classification_data(1000, seed=2)  # independent draw

        trained = train_classifier(X_train, y_train)
        proba = predict_proba(trained, X_test)
        metrics = evaluate_classifier(y_test, proba)

        assert metrics.pr_auc > metrics.base_rate + 0.15  # meaningfully better than the naive baseline
        assert metrics.brier_score < 0.24  # a coin-flip-ish model would sit near 0.25
        assert metrics.n_obs == 1000

    def test_model_is_reasonably_calibrated_on_held_out_data(self):
        X_train, y_train, _ = _synthetic_classification_data(3000, seed=3)
        X_test, y_test, _ = _synthetic_classification_data(2000, seed=4)

        trained = train_classifier(X_train, y_train)
        proba = predict_proba(trained, X_test)
        metrics = evaluate_classifier(y_test, proba, n_bins=5)

        # each bin's mean predicted probability should track its mean
        # actual outcome reasonably closely -- the entire point of
        # calibration, checked on held-out data, not the training set.
        diffs = (metrics.calibration_bins["mean_predicted"] - metrics.calibration_bins["mean_actual"]).abs()
        assert diffs.mean() < 0.12

    def test_no_signal_gives_pr_auc_near_base_rate(self):
        rng = np.random.default_rng(5)
        n = 2000
        X_train = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        y_train = pd.Series((rng.random(n) < 0.5).astype(float))  # y independent of X entirely
        X_test = pd.DataFrame({"f1": rng.normal(0, 1, 1000), "f2": rng.normal(0, 1, 1000)})
        y_test = pd.Series((rng.random(1000) < 0.5).astype(float))

        trained = train_classifier(X_train, y_train)
        proba = predict_proba(trained, X_test)
        metrics = evaluate_classifier(y_test, proba)
        assert abs(metrics.pr_auc - metrics.base_rate) < 0.1

    def test_evaluate_raises_on_single_class_ground_truth(self):
        y_true = pd.Series([1.0, 1.0, 1.0])
        proba = pd.Series([0.6, 0.7, 0.55])
        with pytest.raises(ValueError):
            evaluate_classifier(y_true, proba)

    def test_predict_proba_uses_only_trained_feature_names(self):
        X_train, y_train, _ = _synthetic_classification_data(300, seed=6)
        trained = train_classifier(X_train, y_train)
        X_test_extra_cols = X_train.copy()
        X_test_extra_cols["unused_extra_column"] = 999.0
        proba = predict_proba(trained, X_test_extra_cols)
        assert len(proba) == len(X_train)
        assert proba.between(0, 1).all()

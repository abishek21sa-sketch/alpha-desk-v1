# Phase 5 — Machine Learning Return Classifier

Three modules under `src/alpha_desk/strategies/ml/`: point-in-time feature
construction (reusing Phase 4's factor panels), a calibrated gradient-
boosted classifier, and a walk-forward train/predict/trade backtest. This
is the platform's genuinely nonlinear, learned model -- distinct from
Phases 3-4's deterministic/econometric methods (cointegration tests, Kalman
filters, IC-weighted linear scoring).

## Why a classifier, and why gradient boosting specifically

The task is framed as binary classification -- will this stock beat the
cross-sectional median return over the next rebalance period -- rather than
regression, because it's a cleaner, market-direction-agnostic target (a bad
month for the whole market still has relative winners and losers) and
because classification comes with an honest, standard evaluation toolkit
(PR-AUC, Brier score, log-loss, calibration) that a raw regression R²
doesn't give as directly.

`HistGradientBoostingClassifier` (scikit-learn) is a genuine tree ensemble,
not a GLM -- it can learn nonlinear factor interactions (e.g. "momentum
only matters when value is also cheap") that a linear IC-weighted score
structurally cannot represent. It also handles missing feature values
natively, which matters given how ragged real fundamentals/momentum
lookback data actually is. `CalibratedClassifierCV` (sigmoid/Platt scaling,
fit via internal 3-fold CV on the training fold only) turns the ensemble's
raw scores into probabilities that are actually usable for position sizing
before anything is evaluated or traded.

## `features.py` — reuses Phase 4's point-in-time discipline

Factor panels come straight from `strategies.factors.backtest.build_factor_panels`
(same momentum/low-vol/value/quality construction, same `filed`-not-`end_date`
point-in-time contract). Macro context (VIX, yield-curve slope, HY credit
spread) is added via a backward-only forward-fill -- a macro print for date
d is usable from d onward, never before, tested directly with a filing
timeline where the naive alternative would leak.

## `model.py` — evaluated as a classifier before it's judged as a strategy

`ClassifierMetrics` reports PR-AUC, Brier score, log-loss, and calibration
bins -- the same "prove the model actually predicts something, not just
that a strategy built on it makes money" discipline as every other honesty
check in this platform. All 8 model tests passed on synthetic classification
data with a clean, controllable logistic signal, including a genuine
calibration check evaluated on held-out data (not the training set).

## `backtest.py` — a second, harder-to-fix instance of Phase 4's dilution lesson

**The most important finding in this phase.** The first backtest run
against a 24-symbol, ~12-year synthetic universe with a real, oracle-
confirmed momentum signal (verified via direct point-biserial correlation:
momentum showed 0.03-0.12 correlation with the label, unambiguously real,
not zero) produced a pooled PR-AUC of **0.4926** -- statistically
indistinguishable from the 0.5 base rate. The classifier showed *zero*
detectable skill despite training on data that demonstrably contained real
signal.

This is the same "small-sample IC dilution" phenomenon Phase 4 found (see
`docs/FACTOR_STRATEGY.md`), but worse for a nonlinear learner: a
tree ensemble has many more ways to find and fit a spurious split on one
of the three pure-noise factors (value, quality, low-vol, which have no
true relationship to returns in this synthetic universe) than a simple IC
correlation does, especially with a modest sample. Diagnosed methodically,
not patched blindly:

1. Trained on momentum ALONE (dropping the three noise factors entirely) at
   the same data size: PR-AUC recovered to 0.51-0.54. Confirms the signal
   is real and learnable in isolation -- the problem is the noise
   factors crowding it out, not the model or the data being fundamentally
   too weak.
2. More cross-sectional breadth (40 symbols instead of 24) did NOT fix it
   (PR-AUC 0.4903, no better). L2 regularization (5.0, 20.0) and shallower
   trees did NOT fix it either (0.4938, 0.4919). Both are the "obvious"
   fixes for overfitting a small sample, and neither worked.
3. **More HISTORY did**: doubling to ~24 years (n_days=6000) lifted pooled
   PR-AUC to 0.50-0.54 across 5 independent seeds -- a real, if modest and
   somewhat inconsistent, edge (one of five seeds still landed at
   essentially exactly 0.50).

**The honest conclusion, stated plainly**: a nonlinear tree-based learner
needs substantially MORE historical data than a linear IC-weighted score to
reliably separate a weak-but-real signal from a handful of noise
candidates -- and even with roughly double the data that gave the linear
Phase 4 model a strong, reliable edge, this classifier's edge stayed
modest and not perfectly consistent across seeds. This is a genuine,
useful characteristic of applying ML to a small-cross-section (dozens, not
thousands, of names), monthly-rebalanced, weak-signal problem -- not a
defect to hide. A real desk running this kind of model would need either a
much larger, longer-history universe, fewer candidate features, or
purpose-built regularization/feature-selection beyond what was tried here.

## The real result

Running `scripts/run_ml_strategy.py` against the real 40-equity universe hit
a second real bug before producing a number at all: `HistGradientBoostingClassifier`
crashed with `ValueError: window shape cannot be larger than input array shape`
on every single fold. Diagnosed to the real data, not the code: FRED's
`BAMLH0A0HYM2` (high-yield credit spread) turned out to have only ~3 years
of history via the fetch endpoint (795 rows, starting 2023-09-05) versus
decades for every other macro series (VIXCLS alone has 9,568 rows back to
1990) -- a genuine Phase 1 data-coverage gap that only surfaced now, three
phases later, because this was the first time that series was actually fed
into a model with a real training-window slice. `train_classifier` now
drops any feature with fewer than 2 distinct non-NaN values in the current
training window before fitting -- a general robustness fix, not a special
case for this one series (see `model.py`'s fix above).

With that fixed, the real result: **3 folds, 0 skipped, 4,880 pooled
out-of-sample predictions, PR-AUC 0.508** (base rate 0.500) -- essentially
at chance, consistent with the synthetic-universe finding that this
approach needs more history than this real dataset's ~13.7 years provides.
Brier score 0.2499 and log-loss 0.693 both sit almost exactly at their
coin-flip values (0.25 and ln(2)). Calibration bins cluster tightly around
0.48-0.52 predicted vs. actual -- the model IS well-calibrated, it's just
calibrated to "I don't know."

**The instructive part**: trading these near-chance predictions anyway
produced a gross Sharpe of 0.398 (net 0.359, PSR 0.872) -- a Sharpe that
would look perfectly presentable in isolation, from a classifier that
demonstrably has no measurable skill. This is reported specifically
*because* it's a clean illustration of why this platform checks PR-AUC/
Brier/log-loss BEFORE ever looking at a strategy's Sharpe: a decent-looking
backtest number from a model that isn't actually predicting anything is
exactly the failure mode the rest of this platform (Phase 2's DSR/PBO,
Phase 3's honest negative result) is built to catch. Full numbers in
`artifacts/ml_strategy_report.json`.

## What this backtest does NOT do

- **No formal feature selection or interaction analysis** -- the model
  either learns to ignore noise features or doesn't; nothing here inspects
  which splits it's actually making (a SHAP-based feature-importance pass
  is a natural next step, not built here).
- **`impact_coefficient` is still Phase 2's uncalibrated placeholder**,
  same caveat as every other phase.
- **Calibration is only checked on synthetic data**, not on the real
  universe's actual out-of-sample predictions -- see the real run's results
  for what the calibration bins look like there.

## Running it

```bash
pytest tests/test_ml_features.py tests/test_ml_model.py tests/test_ml_backtest.py -v
python scripts/run_ml_strategy.py   # real run against the 40-equity universe
```

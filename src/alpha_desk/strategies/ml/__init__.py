"""Phase 5 — a genuine machine-learning predictive model, distinct from
Phases 3-4's deterministic/econometric methods (cointegration tests, Kalman
filters, IC-weighted linear scoring). This is the "one properly-validated
predictive model" component of the platform's Decision-Center-style spread
of techniques, matching airlinesapp's regularized-logistic-regression slot
but using a genuinely nonlinear, tree-ensemble learner instead of a GLM.

- features.py: builds the (date, symbol) feature/label table from Phase 4's
  factor panels plus macro regime context, with a strict point-in-time
  contract on every column
- model.py: trains + calibrates a gradient-boosted classifier, reports
  PR-AUC/Brier/log-loss -- the same honesty-first metrics airlinesapp uses,
  not just an accuracy number
- backtest.py: walk-forward (Phase 2's purged splits) train/predict/trade,
  cost-aware, turning the classifier's probabilities into a tradeable
  long/short book via Phase 4's portfolio construction
"""

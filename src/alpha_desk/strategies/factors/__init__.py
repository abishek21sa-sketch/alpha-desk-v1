"""Phase 4 — cross-sectional equity factor long/short.

- factors.py: raw factor construction (momentum, low-vol, value, quality),
  all point-in-time correct for the fundamentals-based ones
- scoring.py: cross-sectional standardization + IC-calibrated composite
  score (weights are FIT from data, never hand-picked -- same discipline as
  airlinesapp's Health Score)
- portfolio.py: dollar-neutral long/short portfolio construction from a
  composite score
- backtest.py: the first strategy in this platform with an actually FITTED
  parameter (the factor weights) -- so this is where Phase 2's purged
  walk-forward CV does real work, unlike Phase 3's fixed-rule strategy
"""

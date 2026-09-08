# Phase 6 — Time-Series Momentum (TSMOM)

`src/alpha_desk/strategies/momentum/`: signal construction + a cost-aware
backtest across the platform's 10 liquid ETF proxies. Moskowitz, Ooi &
Pedersen, *Time Series Momentum* (2012): sign of each asset's own trailing
12-month return, sized to a common target volatility (10% annualized),
capped at 3x leverage. Deliberately **not cross-sectional** — each asset's
position depends only on its own history, unlike Phase 4's ranking
approach, which is the actual methodological difference from Phase 4, not
a relabeling of it.

No purged walk-forward CV and no DSR/PBO correction, for the same reasons
as Phase 3: lookback/vol-target/leverage-cap are fixed, pre-specified
hyperparameters (nothing is fitted), and this is the one pre-specified
configuration run, not the winner of a sweep.

Clean build: 13 tests (6 signal-level, 7 backtest-level) all passed on the
first run — including a whipsaw-vs-trending contrast test (pure
mean-reverting noise must score materially worse than persistent,
alternating trends, the textbook failure mode of trend-following) and the
standard zero/higher-cost checks.

## The real result

`scripts/run_momentum_strategy.py` against the real 10-ETF universe (233
monthly rebalances, full common history): **net Sharpe 0.402** (gross
0.415, cost drag only 0.013 — monthly-rebalanced liquid ETFs are cheap to
trade), **Probabilistic Sharpe Ratio 0.937** — close to, but just short of,
a 95% confidence bar. Reported as exactly that: strong and probably real,
not proven.

Per-asset breakdown is itself informative: **8 of 10 assets show a positive
net Sharpe** (QQQ 0.57, SPY 0.41, GLD 0.41, LQD 0.38, HYG 0.30, TLT 0.24,
IEF 0.23, IWM 0.06), while **USO (-0.30) and UUP (-0.12) are the two
laggards** — crude oil and the dollar index are both known for sharp,
short-lived reversals that punish trend-followers (a real, discussable
characteristic of those markets, not a strategy defect).

## What this does NOT do

- **ETF proxies, not real futures** — a documented substitution from Phase 1
  (no free/legal bulk futures data source); ETF roll/carry mechanics differ
  from the actual futures contracts TSMOM was originally validated on.
- **No cross-asset correlation control** — 10 independent per-asset sizing
  decisions summed together, not a portfolio-level risk budget that accounts
  for e.g. TLT/IEF or LQD/HYG moving together.
- **`impact_coefficient` is still Phase 2's uncalibrated placeholder**, same
  caveat as every other phase.

## Running it

```bash
pytest tests/test_momentum_signals.py tests/test_momentum_backtest.py -v
python scripts/run_momentum_strategy.py
```

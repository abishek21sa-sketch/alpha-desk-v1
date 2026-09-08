# Phase 4 — Cross-Sectional Equity Factor Long/Short

Four modules under `src/alpha_desk/strategies/factors/`: point-in-time-correct
factor construction, IC-calibrated composite scoring, dollar-neutral
portfolio construction, and a walk-forward backtest. This is the first
strategy in the platform with an actually FITTED parameter -- unlike
Phase 3's fixed-rule Kalman filter, the factor weights here are learned
from training data, which is exactly why Phase 2's purged walk-forward CV
finally does real work in this phase (see `backtest.py`'s module docstring
for the contrast).

## `factors.py` — four factors, one calibration trap avoided by construction

- **Momentum**: 12-1 month total return (Jegadeesh & Titman 1993) --
  excludes the most recent month deliberately, so short-term reversal
  doesn't contaminate the signal.
- **Low-volatility**: negative trailing 60-day realized vol.
- **Value**: trailing earnings yield (NetIncomeLoss / shares / price).
- **Quality**: return on equity (NetIncomeLoss / StockholdersEquity).

Value and quality both go through `point_in_time_fundamental`, which joins
on `filed` (when EDGAR actually published the number), never `end_date`
(the accounting period it describes) -- Phase 1's data dictionary flagged
this exact trap as "the single most common way a fundamentals-based
backtest silently cheats," and `test_factors.py` includes a filing
constructed specifically to catch a naive end_date join: a 10-K/A
restatement with an EARLY end_date but a LATE filed date, which must NOT
leak into a query dated before its actual filing. All 11 factor tests
passed cleanly on the first run -- no bugs here, credited directly to
having Phase 1's warning in mind while writing this module, not to writing
it carefully in general.

## `scoring.py` — weights are fit, never hand-picked

Same discipline as airlinesapp's Health Score: each factor's weight is its
own Information Coefficient (the average cross-sectional correlation
between the factor and the ACTUAL forward return that followed), measured
on a training window, with only positive-IC factors included and
normalized to sum to 1.0. A factor with negative or ~zero measured IC gets
weight ZERO -- not a negative weight, which would be fitting a noisy
correlation's sign rather than genuine skill.

## `portfolio.py` — deliberately simple

Equal-weighted top/bottom quantile buckets, dollar-neutral. No optimizer.
Phase 4's contribution is the calibrated score; a MILP/QP-based portfolio
optimizer (mean-variance, CVaR-aware) is explicitly a later-phase concern
(the platform's stat-arb-adjacent risk/execution phase), not duplicated
half-built here.

## `backtest.py` — a real finding about calibration data requirements

**Where Phase 2's purged walk-forward CV finally earns its keep**: factor
weights are calibrated on each fold's TRAIN dates only (via
`calibrate_factor_weights`), then applied unchanged to that fold's TEST
dates -- a genuine train/test split, unlike Phase 3 where nothing was
fitted. `label_horizon=1` in the purge call is exactly right at the
rebalance-date granularity: each rebalance's forward-return label depends
on the NEXT rebalance date, so purging removes exactly the one training
sample whose label would otherwise leak from the test block.

**A real, useful finding while testing this, not a bug**: the first version
of `test_genuine_momentum_signal_produces_positive_net_sharpe` used a
24-symbol, ~6.3-year synthetic universe with a genuine, oracle-confirmed
momentum edge (a non-walk-forward, momentum-only-weighted version of the
same data scores Sharpe 0.89) -- but the walk-forward version only scored
net Sharpe 0.118, with momentum's calibrated weight diluted to 12-38% by
three factors (value, quality, low-vol) that have NO true relationship to
returns in this synthetic universe at all. The reason: each fold's training
window (as short as ~12 rebalance dates in the first fold) is too small
for IC estimation to reliably separate a real signal from sampling noise in
the other three factors -- a spurious noise-factor IC can rival or exceed a
real signal's IC when both are measured over only a couple dozen
cross-sectional snapshots. This was confirmed, not assumed: extending the
synthetic history to ~12 years and using fewer/larger walk-forward folds
(n_splits=3 instead of 5) let momentum's weight average 39-96% and net
Sharpe land between 0.29 and 1.29 across 6 independent seeds. **The lesson
generalizes**: multi-factor IC calibration is data-hungry, and a real
allocator running this kind of system needs either a longer history, a
larger universe (more cross-sectional breadth per date directly reduces
per-date IC noise), or fewer walk-forward folds with correspondingly larger
training windows -- not a fix to the code, a property of the statistics.

**What this backtest does NOT do**:
- **Universe screening happens once, up front** (the platform's full
  40-equity universe, not re-selected per fold) -- unlike Phase 3, there is
  no "best of N universes" selection bias here to worry about, since the
  universe isn't a tuned choice.
- **`impact_coefficient` is still Phase 2's uncalibrated placeholder** --
  same caveat as every other phase's cost figures.
- **No sector/beta neutrality constraint** on the long/short book beyond
  dollar-neutrality -- a real factor book usually also targets sector- or
  beta-neutrality to isolate the factor bet from a broad market/sector tilt;
  not implemented here.

## The real result

Running `scripts/run_factor_strategy.py` against the real 40-equity
universe (full common history, 2013-01-02 onward, ~13.7 years, n_splits=3):
**3 folds, 123 monthly rebalances, gross Sharpe 0.285, net Sharpe 0.273**
(cost drag only 0.012 -- monthly rebalancing on liquid large caps is cheap,
as expected). **Probabilistic Sharpe Ratio: 0.798** -- meaning roughly an
80% probability the true per-period Sharpe exceeds zero, which is
encouraging but does not clear a 95% confidence bar. Reported as exactly
that: a genuinely positive but not statistically airtight result, not
rounded up to "it works" or down to "it doesn't."

The factor breakdown across all 3 folds is the more interesting part:
**momentum (weight 0.45-0.58) and value (0.36-0.51) were consistently the
two positive-IC factors**, quality contributed a small residual weight
(0.03-0.13), and **low-volatility had a NEGATIVE information coefficient in
every single fold** (-0.078 to -0.148) -- i.e. on this real universe and
period, low-vol names tended to have LOWER forward returns, the opposite of
the textbook "low-volatility anomaly," and `calibrate_factor_weights`
correctly excluded it (weight 0.0) every time rather than force a tilt
against what the data actually showed. That consistency across
independent, non-overlapping folds -- not just a single lucky measurement
-- is itself a meaningful, discussable finding, and exactly the kind of
thing this platform's calibration discipline is built to surface honestly
rather than paper over.

Unlike Phase 3, no DSR/PBO correction is applied here, deliberately: this
run is one pre-specified configuration, not the winner of a sweep across
many. Full fold-by-fold weights and ICs are in
`artifacts/factor_strategy_report.json`.

## Running it

```bash
pytest tests/test_factors.py tests/test_factor_scoring.py tests/test_factor_portfolio.py tests/test_factor_backtest.py -v
# 33 tests, all synthetic
python scripts/run_factor_strategy.py
# real run against the 40-equity universe -- see the script's own output for this run's actual numbers
```

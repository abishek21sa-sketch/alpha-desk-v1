# Phase 2 — Validation Engine

Four modules under `src/alpha_desk/validation/`, all with citations to the
actual papers they implement (not just "this is standard practice"). This
doc is the honest account of what each one proves, what it doesn't, and the
real bugs the test suite caught while building it — the same discipline as
`docs/DATA_BACKBONE.md`.

## `splits.py` — purged, embargoed walk-forward CV

Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch. 7.
Every training sample used for a given test fold ends chronologically
before that fold starts; a sample whose forward-looking label window would
overlap the test fold is purged, plus an extra embargo buffer for serial
correlation the label window alone doesn't cover.

**What it doesn't do**: this is strictly walk-forward (train always
precedes test) — not the more general mid-timeline `PurgedKFold` from the
same source, which also permits test blocks in the interior of the
timeline with purging on both sides. That's a deliberately narrower, safer
tool: it can never accidentally let a fold "see" data chronologically after
what it's predicting, which the more general form technically can if used
carelessly.

## `performance.py` — Sharpe, PSR, DSR

Bailey & Lopez de Prado, *The Sharpe Ratio Efficient Frontier* (2012) and
*The Deflated Sharpe Ratio* (2014).

- **PSR**: P(true per-period Sharpe exceeds a benchmark), accounting for
  sample skewness/kurtosis, not just assuming Gaussian returns.
- **DSR**: PSR tested against "what Sharpe would the best of N independent
  trials reach by pure luck" instead of a fixed benchmark — the correction
  for having tried many strategies/parameter combinations and reporting
  only the winner.

**Verified, not just implemented**: `tests/test_deflated_sharpe.py` checks
real mathematical properties, not just "it runs" — PSR(SR_hat)==0.5 exactly
regardless of skew/kurtosis/T (a hard identity), PSR converges to its
closed-form Gaussian special case under normal returns, negative skew
provably lowers PSR for a positive Sharpe, and DSR is monotonically harder
to clear as the number of trials grows (checked analytically-motivated,
then stress-tested over 200 random seeds with zero failures before being
trusted).

**What it doesn't do**: `trial_sharpe_ratios`' cross-trial variance is used
as a scale for the "luck budget" — if you only ever report one trial's
Sharpe, there's no way to estimate that variance, and the module falls back
to an un-deflated PSR(0) rather than fabricating a number (see
`test_zero_variance_trials_falls_back_to_psr_zero`). Passing an honest,
complete list of every trial actually run (not a cherry-picked subset) is
the caller's responsibility — DSR cannot detect a dishonest `trial_sharpe_ratios` list.

## `overfitting.py` — Probability of Backtest Overfitting (CSCV)

Bailey, Borwein, Lopez de Prado, Zhu, *The Probability of Backtest
Overfitting* (2017). Splits the timeline into S blocks, tries every way of
using half as "in-sample" and half as "out-of-sample," and asks: how often
does the in-sample-best strategy end up below the out-of-sample median?

**A real bug this caught during development**: the first single-seed test
(`pure noise -> PBO ~ 0.5`) failed at 0.68 against a naive `[0.35, 0.65]`
band. Before assuming a bug, this was checked by Monte Carlo: PBO's
single-matrix sampling stdev is genuinely ~0.17-0.19 at T=1000/N=20/S=10,
because CSCV's 252 combinations reuse overlapping time blocks and are far
from independent draws — a single realization landing at 0.68 is expected
sampling noise, not a defect. Confirmed by averaging 100 independent
matrices (mean 0.508, stderr 0.019 — squarely on 0.5). The fix was to the
*test* (average multiple independent draws, as any honest Monte Carlo
check must), not the implementation. This is exactly the kind of thing
this document exists to record rather than quietly patch and move past.

**What it doesn't do**: uses Sharpe ratio as the IS/OOS performance metric
(the literature's own running example); a different metric (Sortino,
Calmar, ...) would need to be swapped in explicitly, not assumed
equivalent.

## `costs.py` — transaction cost model

Commission + half-spread + a square-root market-impact term (participation
rate scaling), following the widely-documented empirical square-root law
(Almgren, Thum, Hauptmann & Li 2005; Kyle/Obizhaeva-Wang line of work).

**What it doesn't do — the most important caveat in this whole phase**:
`impact_coefficient=0.1` is an illustrative placeholder, not calibrated
against this platform's own fills (there are no fills yet — nothing has
traded). Every later backtest that reports a "net of costs" return is net
of *this uncalibrated estimate*, not a validated cost forecast. Tests check
that the model's costs move in the right *direction* (bigger trades cost
more, illiquid names cost more, cost grows sub-linearly via the sqrt term)
— they cannot and do not validate the *magnitude* is realistic.

## Running it

```bash
pytest tests/test_purged_cv.py tests/test_deflated_sharpe.py tests/test_pbo.py tests/test_transaction_costs.py -v
# 31 tests, all synthetic (no external data dependency) -- run standalone
```

# Phase 3 — Statistical Arbitrage / Pairs Trading

Four modules under `src/alpha_desk/strategies/pairs/`: cointegration
screening, a Kalman-filter dynamic hedge ratio, a z-score entry/exit state
machine, and a cost-aware backtest that ties them together. This was the
buggiest phase so far, on purpose documented in full rather than cleaned up
after the fact — every bug below was caught by a test that expected a real,
checkable property and got a wrong answer, not by inspection.

## `cointegration.py` — Engle-Granger screening + OU half-life

Standard two-step test on log prices (`statsmodels.tsa.stattools.coint`),
plus a half-life estimate from the discretized OU/AR(1) regression
`delta_spread_t = c + beta*spread_{t-1}`, `half_life = -ln(2)/beta`.

**Bugs caught while testing this module:**
- `estimate_half_life` on a pure random walk returned a large-but-finite
  number (135.5), not the `np.inf` the first version of the test assumed.
  This turned out to be correct, expected behavior — the classic
  Dickey-Fuller finite-sample bias means an AR(1) coefficient estimated on
  a true random walk is very slightly negative almost every time, not
  exactly zero. Fixed the *test* to compare against a genuinely
  mean-reverting series' half-life instead of asserting an exact `inf`.
- The first synthetic "cointegrated pair" test built its residual as pure
  i.i.d. noise, which trivially has a near-zero half-life regardless of
  whether the pair is "cointegrated" — the test would have passed even with
  a broken half-life estimator. Fixed by giving the synthetic residual a
  real, controllable OU process (known theta -> known half-life) instead of
  raw noise.
- `test_pair_cointegration` (the module function) was originally imported
  directly into the test file, which pytest then collected AS a test
  itself because its name starts with `test_`. Fixed by importing the
  module and calling `coint_mod.test_pair_cointegration(...)`.

## `kalman.py` — dynamic hedge ratio

Standard 2-state `[alpha, beta]` random-walk Kalman filter (Chan,
*Algorithmic Trading*, ch. 2), with the innovation and its variance doubling
as a ready-made z-score.

**A real look-ahead bug**: the observation-noise variance `R`, when not
given explicitly, was originally estimated from `Var(diff(y))` over the
FULL series -- meaning every single state estimate secretly depended on
data from the end of the series, including points that hadn't happened yet
relative to wherever the filter currently was. Caught by
`test_is_causal_truncated_history_gives_identical_early_output`: running
the filter on a truncated series produced *slightly* different early
output than running it on the full series and slicing -- which is
impossible for a genuinely causal filter. Fixed by calibrating `R` from
only the first `calibration_window` (default 60) observations, so
everything after that window is unambiguously causal, and the calibration
window itself is a small, explicitly documented exception.

**A subtler, more interesting finding — delta must be tuned to the spread's
own half-life**: with the "obviously slow" default `delta=1e-4`, a
synthetic pair with a genuine, strong, ~14-day-half-life mean-reverting
spread never produced a single trade — `|z|` never exceeded ~1.1 across
1500 observations. The cause: at that delta, the Kalman filter's beta was
adaptive enough to track the OU process's own wobble and "explain" it away
as a temporary hedge-ratio shift, leaving almost no innovation for the
z-score to detect. A parameter sweep confirmed the pattern directly:

| delta | frac(days with `\|z\|`>2) |
|---|---|
| 1e-4 | 0.0% |
| 1e-5 | 2.5% |
| 1e-6 | 19.7% |
| 1e-7 | 40.5% |

Changed the default to `delta=1e-6`. There is no universally correct value
— it has to be small relative to the specific pair's mean-reversion
half-life (from `cointegration.py`), or the filter cannibalizes its own
signal. This is a real, known failure mode of Kalman-filter pairs trading,
not specific to this implementation, and is exactly the kind of thing that
would come up in a live conversation about why a "textbook" Kalman pairs
strategy stopped trading.

## `signals.py` — z-score entry/exit

A stateful sticky-position loop (enter at `|z|>=entry_threshold`, exit at
`|z|<=exit_threshold`). Clean on the first pass — 9/9 tests green
immediately, including an exact-equality causality check (truncated input
must reproduce the first N outputs of the full run bit-for-bit, since this
is a pure left-to-right state machine with no floating-point accumulation
sensitive to array length).

## `backtest.py` — the integration bug

**The most consequential bug in this phase**: `kalman_hedge_ratio(x, y)`
fits `y = alpha + beta*x`; called as `kalman_hedge_ratio(log_a, log_b)`,
that means `x=log_a`, `y=log_b`, and the spread is `log_b - beta*log_a`.
The backtest's return calculation had this backwards —
`spread_return = r_a - beta*r_b` instead of `r_b - beta*r_a` — silently
trading a linear combination of the two legs that has nothing to do with
the actual cointegrating relationship whenever `beta != 1`. The trade-value
sizing for transaction costs had the same legs swapped for the same reason.

This one hid behind a SECOND bug for a while: the synthetic test data's
first version put the OU residual on the regressor (`log_a`/x) side, which
is a classical errors-in-variables setup and attenuation-biases any
regression toward zero — so the recovered beta (~0.16 against a true 1.2)
looked "just noisy," not obviously wrong, and masked how badly off the
spread-direction bug actually was. Both had to be fixed before the
"genuine mean reversion produces positive Sharpe" test could pass
honestly: an oracle strategy (true beta, true residual, zero estimation
error) was computed independently to confirm a real edge existed in the
synthetic data (Sharpe 1.58) before trusting that the Kalman-based
version's much lower number (Sharpe ~0.9 after both fixes, vs. ~0.06
before) reflected realistic estimation drag rather than a still-broken
implementation.

**What this backtest does NOT do** (by design, not oversight):
- **No purged walk-forward CV.** This strategy has no fitted parameters —
  delta, thresholds, and the R-calibration window are fixed
  hyperparameters, and the Kalman filter is already causal by construction.
  There is nothing here that could "train on the test set" the way a
  fitted model (Phase 4's factor regression) could. Running Phase 2's CV
  machinery anyway would be theater, not a real safeguard — see the
  module's own docstring.
- **Pair selection uses the full historical sample** (cointegration
  screening isn't restricted to an initial in-sample window). This is a
  genuine, acknowledged form of selection bias: screening hundreds of pairs
  and trading whichever looks best is the "best of N trials" problem, just
  one level up from a single strategy's parameter sweep. This is exactly
  why `scripts/run_pairs_strategy.py` runs every tradeable pair's Sharpe
  through Phase 2's Deflated Sharpe Ratio (against N = number of pairs
  tried) and Probability of Backtest Overfitting — the correction is
  applied at the level where the real overfitting risk actually lives,
  rather than bolted onto a part of the pipeline where it wouldn't mean
  anything.
- **`impact_coefficient` is still Phase 2's uncalibrated placeholder** — see
  `docs/VALIDATION_ENGINE.md`. Nothing in Phase 3 calibrates it either.

## Why bound the lookback window

`scripts/run_pairs_strategy.py` screens only the trailing 6 years, not the
full available history. Two of the 40 equities have data back to 1962
(16,278 rows) -- screening the full history isn't just slow (a single
Engle-Granger test on ~16k observations takes several seconds; at 780 pairs
that's an hour-plus run, discovered the hard way when the first full-history
run was killed after 5+ minutes with zero output), it's methodologically
questionable: a "cointegrating relationship" spanning a name's entire
1962-2026 history says almost nothing about whether it's tradeable today,
across multiple business-model changes, spinoffs, and regime shifts. A
bounded, recent lookback (6 years / ~1500 trading days, matching this
phase's synthetic tests) is both the faster choice and the more defensible
one -- this is not a compromise made for the sake of speed alone.

## The real result — and why it's reported as a negative one

Running `scripts/run_pairs_strategy.py` against the real 40-equity universe
(6-year lookback): 780 pairs screened, **49 passed** p<0.05 and a 1-60 day
half-life. The best pair by net Sharpe was **HON/MS** (Honeywell / Morgan
Stanley), p=0.0055, half-life 30.9 days, 77 trades, **gross Sharpe 0.79, net
Sharpe 0.73** (cost drag only 0.06 -- both are liquid large caps, so this
part is believable).

Then the two Phase 2 tools this whole architecture was built to apply here:

- **Deflated Sharpe Ratio**, correcting for having tried all 49 pairs and
  reported the best one: **DSR = 0.436**. That means there is only a 43.6%
  probability the true Sharpe exceeds what pure luck across 49 trials would
  produce -- nowhere near the 95% a real claim of skill would need.
- **Probability of Backtest Overfitting** across all 49 pairs' return
  series: **0.548** -- statistically indistinguishable from the ~0.5 pure-
  noise baseline this exact metric was shown to produce on synthetic random
  data in `docs/VALIDATION_ENGINE.md`.

**The honest conclusion: this specific 40-name universe, over this 6-year
window, does not contain a pairs-trading edge that survives its own
multiple-testing correction.** HON/MS's 0.73 Sharpe looks like a genuine
result in isolation and is exactly the number a less careful writeup would
report as "the strategy." Reporting DSR and PBO next to it, instead of
just it, is the entire point of Phase 2 existing before Phase 3 was
built -- a pipeline that only reports the flattering number isn't validated,
it's just unaudited. The full 49-pair table and both corrections are saved
to `artifacts/pairs_strategy_report.json` for whoever wants to check this
independently.

This is not a dead end for the platform -- it's a real, defensible finding:
either the universe needs to be larger (49 candidates is a small trial
count for DSR to clear), the lookback/half-life bounds need revisiting, or
(most likely, per the literature this phase is built on) simple pairs
cointegration on daily-bar large-cap equities is a genuinely hard place to
find edge in 2020s markets, since it's also the most obvious, most
back-tested strategy family in retail quant finance. That is itself a
useful, discussable data point.

## Running it

```bash
pytest tests/test_cointegration.py tests/test_kalman.py tests/test_signals.py tests/test_pairs_backtest.py -v
# 28 tests, all synthetic
python scripts/run_pairs_strategy.py
# real run: screens the 40-equity universe, backtests every tradeable pair,
# reports the best pair's Sharpe both raw and deflated for how many pairs
# were tried -- see the script's own output for this run's actual numbers.
```

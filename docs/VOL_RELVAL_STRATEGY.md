# Phase 10 — VIX Term-Structure Relative Value

`src/alpha_desk/strategies/vol_relval/`: a fixed-rule strategy trading the
VIX term structure (spot VIX vs. VIX3M, the CBOE's 3-month volatility
index) via two real, tradeable ETFs — SVXY (short front-month vol) and
VIXY (long front-month vol). 10 tests, all passing, including one that
caught a wrong assumption in the test itself (not the code — see below).

## Why ETFs, not a synthetic futures curve

The classic version of this trade is built on the VIX futures curve
directly. This module deliberately uses SVXY/VIXY instead of reconstructing
a synthetic futures curve: both are real, exchange-listed instruments whose
real historical NAV returns already embed real roll cost and decay — a
backtest on their actual returns is arguably *more* honest than a
synthetic curve reconstruction would be, and needs no assumptions about
futures contract weighting or roll schedules.

**Two real, disclosed data notes**:
- **SVXY changed its exposure from -1x to -0.5x after the February 2018
  "Volpocalypse"** (the regulatory response to the VIX ETN blowup that
  fully liquidated the related XIV note) — a real, permanent structural
  change in what SVXY's returns represent, not a data error. This module
  does not attempt to normalize across it; the backtest below runs on
  SVXY's real, continuous, as-reported returns straight through that event.
- **History is genuinely limited by data availability, not truncated by
  choice**: ^VIX3M only exists back to 2006-07, SVXY only since 2011-10 —
  earlier dates are unavailable at the source, not filled in.

## `signals.py` — the term-structure signal

`term_structure_ratio = VIX3M / VIX`. Above 1.0 is **contango** (the
market prices in more expected vol further out — the historically normal
state, confirmed at **89.0% of days** over the full 2006-2026 real
VIX/VIX3M history pulled by this platform). Below 1.0 is **backwardation**
(near-term vol priced above further-out vol — typically acute stress).
`directional_position(vix, vix3m, long_when=...)` turns this into a binary
long-or-flat signal for either side of the trade.

## `backtest.py` — same lag discipline as every other phase

Position on day *t* is decided using day *t-1*'s close, never day *t*'s own
— the same lag contract Phases 3 and 6 use. One test
(`test_position_uses_only_the_prior_days_signal_not_same_day`) verifies
this directly with an isolated synthetic price jump: a look-ahead bug would
let the strategy dodge or ride the jump based on same-day information,
while the correct lagged behavior must miss it either way, only paying (or
not paying) the transaction cost of a position change around it. **Writing
this test surfaced a wrong assumption in the test itself, not a code bug**:
the first version asserted the jump day's net return would be *exactly*
zero when the position is flat that day — but closing out of a real
position still costs money (the transaction cost model correctly charges a
few basis points for unwinding the prior day's long), so the net return on
a flat day with a position CHANGE the day before is a small negative cost
drag, not exactly zero. Fixed by asserting the return is small (a cost
drag), not literally zero — the code was right, the test's expectation
wasn't.

No fitted parameters here (same reasoning as Phases 3 and 6) — no walk-
forward CV, no DSR/PBO correction, only the Probabilistic Sharpe Ratio as
an honesty check on the one configuration actually run.

## The real report — two sides of the same trade, real divergence

`scripts/run_vol_relval_strategy.py` fetches real VIX/VIX3M/SVXY/VIXY
history live from Yahoo Finance (same "fetch live, don't persist to the
warehouse" precedent as Phase 7's SEC 8-K fetch) and runs the backtest
twice:

| Scenario | Days in position | Gross Sharpe | Net Sharpe | Cost drag | PSR |
|---|---|---|---|---|---|
| **SVXY long, contango** (harvest the roll) | 93.2% | 0.802 | 0.790 | 0.011 | **0.998** |
| **VIXY long, backwardation** (crisis hedge) | 7.7% | 0.110 | 0.094 | 0.015 | 0.647 |

**The harvest side has real, high-confidence edge; the hedge side does
not** — a genuine divergence, not a symmetric result, and reported as such
rather than averaged together into one misleading number. SVXY is in
position 93.2% of the time (close to but not identical to the 89.0%
raw-contango rate, the gap coming from the one-day lag), net Sharpe barely
moves off gross (0.790 vs 0.802 — SVXY is liquid enough that costs barely
matter), and PSR 0.998 says the sample is large and consistent enough
(3,716 days) to be confident the true Sharpe is positive — this is one of
the more robust findings on this platform, consistent with the volatility
risk premium being one of the most well-documented risk premia in
finance. VIXY-long-in-backwardation, by contrast, only triggers on 7.7% of
days, and its Sharpe (0.11, PSR 0.65) is not a compelling standalone
result — backwardation days are, almost by definition, the hardest days to
trade profitably even when correctly identified.

**A real, verifiable finding while sanity-checking the SVXY curve**: the
term structure flipped to backwardation on 2018-02-02, one day *before*
the catastrophic "Volpocalypse" crash of 2018-02-05/02-06 (SVXY -32% and
-83% those two days respectively, the event that fully liquidated the
related XIV note). Because of the strategy's lag, it was still long into
2018-02-02's own -13.2% drop (a real, absorbed loss) — but the
backwardation signal from that day's close correctly forced it flat for
2018-02-05 and 02-06, missing the catastrophic -83% day entirely. Traced
directly in the equity curve: cumulative return falls from +14.93 to
+12.83 on 02-02, then sits flat at +12.82 straight through the crash
rather than following SVXY into its near-total collapse. Not a designed-in
feature — a real consequence of the signal that happened to show up
exactly where it matters most, verified against real dates and real SVXY
prices rather than assumed.

## What this does NOT do

- **Binary long-or-flat, not continuously sized to the magnitude of the
  term-structure spread** — a day one tick into contango gets the same
  full position as a day deep in contango. A magnitude-scaled version is a
  natural extension, not built here.
- **No combined single strategy** — the two scenarios are reported
  separately, not netted into one book; a real allocator would need to
  decide how (or whether) to combine a high-confidence low-Sharpe-uncertainty
  harvest trade with a low-confidence hedge trade, a decision this report
  doesn't make.
- **SVXY's 2018 leverage change is disclosed, not adjusted for** — the
  reported Sharpe blends returns from two structurally different exposure
  levels (-1x pre-Feb-2018, -0.5x after) into one series.

## Running it

```bash
pytest tests/test_vol_relval_signals.py tests/test_vol_relval_backtest.py -v
python scripts/run_vol_relval_strategy.py
```

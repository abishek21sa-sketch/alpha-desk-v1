# Phase 8 — Risk & Execution Control Tower

`src/alpha_desk/risk/`: three modules wrapping every strategy built so far
rather than adding a new one — historical VaR/CVaR, real historical-window
stress replay, and Almgren-Chriss optimal execution. 17 tests, all passed
on the first implementation attempt for `var_cvar.py` (6) and
`execution.py` (7) — `stress.py` (4) also clean.

## `var_cvar.py` — historical, not parametric

Empirical-quantile VaR/CVaR, deliberately not a Gaussian/parametric model
— none of this platform's strategies have return distributions close to
normal (the pairs and PEAD strategies are explicitly event/threshold-driven
with fat tails). Verified against a known closed-form normal-distribution
quantile (matches within 3% at n=200k) and a synthetic check that a fat-
tailed distribution shows a bigger CVaR-VaR gap than a normal one at the
same VaR level — the entire reason CVaR exists over VaR.

## `stress.py` — replay, not a hypothetical shock

Slices each strategy's own REAL, already-realized return series against
two named historical windows (COVID crash, 2022 rate shock) and reports
what actually happened — not a generic "-30% equity shock" applied
uniformly. **2008 is deliberately omitted**: this platform's equity
universe has no common price history before 2013 (see
`docs/DATA_BACKBONE.md`), so a "2008 stress test" would be unreplayable
for most strategies here — left out rather than faked with an unrelated
substitute window.

## `execution.py` — a real calibration bug caught before it shipped

Almgren-Chriss (2000) optimal execution: the trajectory that minimizes a
risk-adjusted trade-off between market-impact cost and price-risk
variance while a large order rests in the market. All 7 tests (position
conservation, the exact risk-neutral -> uniform-trading limiting case,
front-loading under higher risk aversion, and the core cost-vs-variance
trade-off) passed immediately — **but the first real run's illustrative
numbers were absurd**: liquidating just 1% of AAPL's ADV over 5 days came
out costing **3,062 basis points (30.6%)**. The math was right; the input
calibration wasn't -- the temporary/permanent impact coefficients were set
as an arbitrary small multiple of share price (`2.5e-6 * price`) with no
connection to how large the order actually was relative to real trading
volume. Recalibrated to the standard reference point of "impact if trading
at a rate equal to the full ADV" (~10bps temporary, ~5bps permanent, both
standard order-of-magnitude figures for a linear impact model) — the same
order now costs **0.04-0.19bps** in expectation, with price-risk standard
deviation (140-280bps) correctly dominating for an order this small
relative to ADV. That's the right qualitative story: a genuinely small
order's risk is about holding the stock while you sell it, not about
moving the market.

## The real report

`scripts/run_risk_report.py` reconstructs each of the five strategies' real
per-period returns from their own saved equity curves and applies both
VaR/CVaR and the stress replay to each. Headline: **only PEAD and pairs
have enough history to test against the COVID window** (2013+ price data,
but factor/ML/momentum's specific return series don't happen to have
enough rebalance dates landing inside that ~1-month window at their
coarser cadence) — reported as `has_coverage: false` rather than a
fabricated number, exactly the same discipline as every other data gap in
this platform.

## What this does NOT do

- **No portfolio-level aggregation across strategies** — VaR/CVaR/stress
  are computed per-strategy, not for a single blended "the whole platform"
  book (that would need explicit capital allocation weights across
  strategies, a decision this report doesn't make).
- **The Almgren-Chriss example uses one illustrative order size/horizon**
  for one name (AAPL) — not wired into any of the five strategies'
  actual rebalancing trade sizes yet.
- **`impact_coefficient`/eta/gamma remain uncalibrated against this
  platform's own real fills** (there are no real fills — nothing has
  traded) — same caveat as Phase 2's transaction cost model, now doubly
  true after finding a 5-order-of-magnitude calibration error once already.

## Running it

```bash
pytest tests/test_var_cvar.py tests/test_stress.py tests/test_execution.py -v
python scripts/run_risk_report.py
```

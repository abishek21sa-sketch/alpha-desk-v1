# Phase 9 — Market Making (Avellaneda-Stoikov)

`src/alpha_desk/strategies/microstructure/`: a limit-order-book market-making
strategy built on the Avellaneda & Stoikov (2008) closed-form optimal-quoting
model. 19 tests, all passing (11 for the closed-form math in
`market_maker.py`, 8 for the session simulation in `simulation.py`) —
two of those tests exist because a first implementation attempt was wrong.

## What is real and what is simulated — read this before the numbers below

This is the one strategy on this platform that does **not** run on real
market data end-to-end, and that substitution is disclosed explicitly here
rather than left implicit:

- **Real**: the volatility used to drive the simulated price path. Every
  session below is calibrated to AAPL's own real trailing 252-day daily
  return volatility, pulled from this platform's warehouse the same way
  every other phase pulls its inputs — not an assumed or round-number sigma.
- **Simulated**: the order flow itself. The fundamental mid-price follows
  simulated arithmetic Brownian motion (`sigma * sqrt(dt) * Z`), and market
  order arrivals hitting the maker's quotes follow a simulated Poisson
  process with intensity `A * exp(-kappa * distance)` — both standard
  assumptions in the market-making literature (this is exactly the setup
  Avellaneda-Stoikov derive their closed-form solution for), but not
  replayed from an exchange.

**Why not real tick data**: real historical order-book data (IEX's TOPS/DEEP
files) exists and is downloadable, but at 600MB+ *compressed, per day, for
the entire market* (not per-symbol), in a custom binary protocol (IEXTP1)
with no off-the-shelf Python parser — building a parser for a proprietary
binary exchange feed is a multi-hour project on its own before any
market-making logic gets touched. Given the phase budget for this platform,
the honest trade-off was: simulate order flow openly and disclose it, rather
than either (a) quietly present simulated-flow results as if they were
real-market results, or (b) sink the remaining build time into an IEXTP1
parser instead of finishing the platform's other phases. `A`/`kappa` (the
arrival-intensity parameters) are standard illustrative magnitudes from the
market-making literature, not fit to any real fill data — there is no real
fill data on this platform to fit them to.

## `market_maker.py` — the closed-form math

Two pieces:
- **Reservation price**: `mid - inventory * gamma * sigma^2 * time_remaining`
  — the mid-price adjusted for the maker's own inventory risk. A long
  position pulls it below mid (the maker wants to sell down); a short
  position pushes it above mid (wants to buy back). Collapses exactly to
  `mid` at zero inventory or zero time remaining.
- **Optimal spread**: `gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)`
  — the total bid-ask width around the reservation price.

**A real, non-obvious finding while testing this**: the textbook intuition
"more risk-averse -> wider spread" is **not universally true** of this
formula. The spread has two competing terms — the first increases in
gamma (inventory risk), but the second, `(2/gamma)*ln(1+gamma/kappa)`,
*decreases* in gamma over a wide middle range. A numeric scan at
`sigma=0.3, T=1.0, kappa=1.5` shows the spread actually **dips to a minimum
around gamma ~2-5 before rising again** — confirmed directly by scanning the
formula, not assumed from the textbook description. The test
(`test_higher_risk_aversion_widens_spread`) was corrected to compare
`gamma=10` vs `gamma=100`, both safely past that minimum in the regime where
the textbook monotonic result actually holds; a naive test comparing, say,
`gamma=0.1` vs `gamma=5` would have failed and (worse) could have been
"fixed" by weakening the assertion instead of understanding why.

## `simulation.py` — a real overflow bug caught before it shipped

The discrete-time session loop approximates "probability of a fill in this
step" as `arrival_intensity * exp(-kappa * distance) * dt` (the standard
Poisson-thinning approximation), which is **only valid when this product is
much less than 1**. An early development run used `arrival_intensity=2000`
with `dt ~ 0.0026` (390 steps/session), giving `intensity * dt ~ 5.1` — a
"probability" greater than 1, which the naive `rng.random() < p` check
silently treated as "always fires." The tell: comparing several different
oversaturated intensities all produced the *exact same* total fill count
(`2 * n_steps`, i.e. every step filled on both sides), which is impossible
for a family of genuinely different arrival rates and was the signal
something was wrong rather than "the market is just very liquid."

**Fix**: clip the fill probability to `[0, 1]` explicitly
(`min(arrival_intensity * exp(...) * dt, 1.0)`), and pick calibration
parameters so `arrival_intensity * dt` stays comfortably under ~0.1 in real
runs (see below) rather than relying on the clip to paper over a broken
calibration. Locked in with a regression test
(`test_oversaturated_intensity_is_clipped_not_left_invalid`) that deliberately
uses `arrival_intensity=1_000_000` and asserts the result saturates cleanly
at exactly `n_steps` fills per side rather than crashing or silently
misbehaving.

## The real report

`scripts/run_microstructure_strategy.py`:
1. Pulls AAPL's real trailing-252-day daily volatility from the warehouse
   (`1.58%/day` as of the last data refresh) and converts it to a dollar
   volatility using AAPL's real last price (`$319.97` -> `$5.06/day`).
2. Checks its own calibration: `arrival_intensity=30` over `n_steps=390`
   (one session as 1-minute bars) gives `A*dt = 0.077`, under the ~0.1
   validity threshold documented in `simulation.py` — printed and included
   in the report so the calibration is auditable, not asserted.
3. Runs 300 independent simulated sessions each for three risk-aversion
   scenarios (`gamma = 0.01 / 0.05 / 0.20`), reporting mean/std P&L, percent
   of profitable sessions, mean fill count, and mean peak absolute
   inventory per scenario.

Representative result (one real run; stochastic, will vary by seed range
but not by qualitative shape):

| Scenario | gamma | Mean P&L/session | Std P&L | Fills/session | Profitable | Mean max\|inventory\| |
|---|---|---|---|---|---|---|
| Low risk aversion | 0.01 | $21.64 | $10.28 | 21.0 | 98% | 3.6 |
| Moderate risk aversion | 0.05 | $19.13 | $6.95 | 18.9 | 100% | 2.4 |
| High risk aversion | 0.20 | $11.57 | $4.78 | 11.7 | 100% | 1.6 |

This is internally consistent with the theory rather than just "numbers
that came out": higher risk aversion quotes a wider spread, so it fills
*less* often and holds *less* inventory on average — trading away expected
P&L for lower variance and a higher fraction of profitable sessions. It is
not a claim that any of these gamma values is "correct" — gamma is a risk
preference, not something to be fit from data.

## What this does NOT do

- **Not real exchange order flow** — see the disclosure above. This is the
  one phase on this platform where the *strategy signal itself* (order
  arrivals) is simulated rather than observed; only the volatility input is
  real.
- **P&L is per unit (1 share per fill)**, not scaled to any real position
  sizing, capital allocation, or transaction-cost model from the rest of
  this platform (e.g. Phase 8's execution-cost calibration is not applied
  here — this is a single continuously-quoting maker, not a one-shot
  liquidation order, so they are not directly comparable).
- **`arrival_intensity` and `kappa` are illustrative literature values**,
  not fit to any real fill data (none exists on this platform to fit them
  to) — unlike `sigma`, which is real.
- **No adverse-selection or toxic-flow modeling** — the simulated arrival
  process has no informed-trader component; every fill is equally likely to
  be uninformed retail-style flow, which is optimistic relative to a real
  market.

## Running it

```bash
pytest tests/test_market_maker.py tests/test_microstructure_simulation.py -v
python scripts/run_microstructure_strategy.py
```

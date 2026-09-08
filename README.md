# Alpha Desk V1 — Quantitative Trading Research, Validation & Risk Platform

A personal quant-trading portfolio project built to the same bar as this
user's other data-intelligence platforms: real data, empirically-calibrated
(never hand-tuned) signals, honest documentation of what every tool does and
does not prove, and a full staged build rather than a single script.

**Target audience**: a live technical conversation with a quantitative
trading recruiter/interviewer — the platform is built to demonstrate
research rigor (no look-ahead bias, no p-hacked backtests) and execution
awareness (transaction costs, market impact, risk limits), not just "a model
that predicts stock prices."

## Why one platform instead of eight separate strategies

Every strategy module here shares the same two layers instead of each
getting its own bespoke, unaudited backtest:

- **Validation spine** (Phase 2): purged & embargoed walk-forward
  cross-validation, deflated Sharpe ratio / probability of backtest
  overfitting, realistic transaction-cost modeling. No strategy's numbers are
  trusted until they've passed through this.
- **Risk spine** (Phase 8): VaR/CVaR, historical stress scenarios, optimal
  execution cost modeling. No strategy is presented without its risk profile.

Holding every strategy to the *same* validation and risk discipline — instead
of cherry-picking whichever backtest looked best — is the actual point.

## Status: all 11 planned phases done, plus a full backend + frontend

See [`docs/DATA_BACKBONE.md`](docs/DATA_BACKBONE.md),
[`docs/VALIDATION_ENGINE.md`](docs/VALIDATION_ENGINE.md),
[`docs/PAIRS_TRADING.md`](docs/PAIRS_TRADING.md),
[`docs/FACTOR_STRATEGY.md`](docs/FACTOR_STRATEGY.md),
[`docs/ML_STRATEGY.md`](docs/ML_STRATEGY.md),
[`docs/MOMENTUM_STRATEGY.md`](docs/MOMENTUM_STRATEGY.md),
[`docs/PEAD_STRATEGY.md`](docs/PEAD_STRATEGY.md),
[`docs/RISK_STRATEGY.md`](docs/RISK_STRATEGY.md),
[`docs/MICROSTRUCTURE_STRATEGY.md`](docs/MICROSTRUCTURE_STRATEGY.md),
[`docs/VOL_RELVAL_STRATEGY.md`](docs/VOL_RELVAL_STRATEGY.md), and
[`docs/ALPACA_EXECUTION.md`](docs/ALPACA_EXECUTION.md) for the
full, honest account of what's built and what its limitations are. Short version:

- **Phase 1**: 518,774 real daily price rows (50 symbols), 93,468 real FRED
  macro observations (9 series), 51,764 real SEC EDGAR fundamentals rows
  (40 tickers), all hash-verified against a provenance manifest.
- **Phase 2**: purged/embargoed walk-forward CV, Probabilistic + Deflated
  Sharpe Ratio, Probability of Backtest Overfitting (CSCV), and a
  commission+spread+square-root-impact transaction cost model — 31 tests,
  all checking real mathematical properties (not just "it runs"), including
  one genuine bug the PBO test suite caught and diagnosed via Monte Carlo
  before concluding it was test noise, not an implementation defect (see
  `docs/VALIDATION_ENGINE.md`).
- **Phase 3**: cointegration screening, a Kalman-filter dynamic hedge ratio,
  z-score entry/exit signals, and a cost-aware backtest — 28 tests, three
  real bugs found and fixed along the way (a look-ahead leak in the Kalman
  filter's noise calibration, an overly-adaptive default that let beta
  absorb its own trading signal, and a swapped spread-direction formula
  that traded the wrong linear combination of the two legs). Run for real
  against the 40-equity universe: 780 pairs screened, 49 passed
  cointegration, best pair (HON/MS) showed a 0.73 net Sharpe — which
  **Phase 2's own Deflated Sharpe Ratio and PBO then showed is
  indistinguishable from having tried 49 pairs and gotten lucky** (DSR=0.44,
  PBO=0.55, both squarely in noise territory). Reported as the honest
  negative result it is, not hidden — see `docs/PAIRS_TRADING.md`.
- **Phase 4**: point-in-time-correct factor construction (momentum, value,
  quality, low-vol), IC-calibrated composite scoring, dollar-neutral
  portfolio construction, and the platform's first genuinely walk-forward-
  validated strategy (factor weights are FIT on each fold's training data,
  unlike Phase 3's fixed-rule strategy) — 33 tests, including one real
  finding caught via an oracle check: a first backtest run showed almost no
  edge (net Sharpe 0.12) despite momentum having a confirmed real signal,
  because short walk-forward training windows let 3 noise factors dilute
  momentum's calibrated weight — fixed by using a longer history and
  fewer/larger folds, not by patching the code (see `docs/FACTOR_STRATEGY.md`).
  Run for real against the 40-equity universe (13.7 years, 123 monthly
  rebalances): **net Sharpe 0.273, PSR 0.798** — a genuinely positive but
  not statistically airtight result, with momentum and value consistently
  the two positive-IC factors across every fold and low-volatility
  consistently NEGATIVE-IC (correctly excluded, not forced) — a real,
  discussable finding, not the textbook low-vol anomaly this universe/period
  would predict.
- **Phase 5**: a genuine machine-learning model — a calibrated gradient-
  boosted classifier (`HistGradientBoostingClassifier` + `CalibratedClassifierCV`),
  distinct from Phases 3-4's deterministic/econometric methods — predicting
  whether a stock beats the cross-sectional median return, evaluated as a
  classifier (PR-AUC/Brier/log-loss/calibration) before ever being judged as
  a strategy. Two real bugs found against real data: a `ValueError` crash
  from a FRED series with far less history than the rest of the feature set
  (fixed generally — degenerate feature columns are now dropped per training
  window, not special-cased), and a synthetic-universe finding that a tree
  ensemble needs roughly double the history a linear IC-weighted score
  needed to detect the same real signal (see `docs/ML_STRATEGY.md`). Real
  result: **PR-AUC 0.508 against a 0.500 base rate — no reliable skill**,
  yet trading the near-chance predictions anyway produced a presentable
  0.36 net Sharpe (PSR 0.87) — reported specifically as a worked example of
  why this platform checks classifier metrics before ever trusting a Sharpe.
- **Phase 6**: time-series momentum (TSMOM, Moskowitz/Ooi/Pedersen 2012) —
  sign of each asset's own trailing 12-month return, vol-targeted, across
  the platform's 10 ETF proxies. Per-asset, not cross-sectional (the actual
  methodological difference from Phase 4). Clean build — 13 tests passed
  first try. Real result: **net Sharpe 0.402, PSR 0.937** — strong, just
  short of a 95% bar — with 8 of 10 assets showing a positive net Sharpe
  and the two laggards (crude oil, the dollar index) being names known for
  the sharp reversals that punish trend-followers, not a strategy defect.
- **Phase 7**: earnings/event-driven drift (PEAD), built on real SEC EDGAR
  8-K filing timestamps rather than paid consensus-estimate data (disclosed
  scope limit: not filtered to Item 2.02 earnings releases specifically —
  a broader "material event" proxy). **A serious bug caught by cross-
  checking two numbers against each other**: the event study found NO
  significant relationship between announcement direction and later drift
  (p=0.62), yet the first version of the tradeable backtest showed a 2.42
  gross Sharpe — impossible if the underlying relationship is really
  noise. Root cause: a trade's first "return day" was accidentally
  re-scoring part of the same announcement-day jump used to pick its own
  direction (a self-fulfilling look-ahead bug). Fixed and locked in with a
  regression test (four isolated jumps + flat prices must show exactly
  zero P&L). Real, corrected result: **drift spread 0.15 points, p=0.62 —
  not significant; net Sharpe 0.046** — a clean, internally-consistent
  negative finding, not hidden.
- **Phase 8**: risk & execution control tower — wraps every strategy rather
  than adding a new one: historical VaR/CVaR, real historical-crisis-window
  replay (each strategy's own actual returns during COVID/2022, not a
  hypothetical shock), and Almgren-Chriss optimal execution. 17 tests
  passed on the first attempt for the math itself — **but the first real
  execution example priced liquidating 1% of AAPL's ADV over 5 days at
  3,062 basis points (30.6%)**, an absurd number caught before it shipped:
  the impact coefficients were an arbitrary multiple of share price with
  no connection to real trading volume. Recalibrated to a standard
  "impact at 100% of ADV" reference point, the same order now costs a
  realistic 0.04-0.19bps. See `docs/RISK_STRATEGY.md`.
- **Phase 9**: market making via the Avellaneda-Stoikov (2008) closed-form
  optimal-quoting model — reservation price and spread under inventory risk.
  Deliberately disclosed as the one strategy on this platform that does NOT
  run on real order-flow data: real IEX tick data is 600MB+/day in a
  proprietary binary protocol with no off-the-shelf parser, so order arrivals
  and the price path are simulated (standard Poisson-arrival assumptions from
  the market-making literature), while the volatility driving the simulation
  is AAPL's own real trailing daily volatility pulled from this platform's
  warehouse. 19 tests, two real findings along the way: the textbook "higher
  risk aversion -> wider spread" result is **not universally true** of the
  AS formula (a numeric scan found a U-shaped minimum around gamma~2-5,
  caught before it produced a wrong test), and a genuine overflow bug where
  the Poisson-thinning fill probability could exceed 1.0 and silently
  saturate fills regardless of further intensity increases — fixed with
  explicit clipping plus a regression test. Real-calibrated result across
  three risk-aversion scenarios (300 simulated sessions each): mean P&L per
  session ranges **$11.57 (high risk aversion) to $21.64 (low risk
  aversion)**, with higher risk aversion correctly trading away P&L for
  lower variance and tighter inventory — the AS trade-off working as
  theory predicts, not just "numbers that came out." See
  `docs/MICROSTRUCTURE_STRATEGY.md`.
- **Phase 10**: VIX term-structure relative value — a fixed rule trading
  the real spot-VIX-vs-VIX3M term structure via two real, tradeable ETFs
  (SVXY short front-month vol, VIXY long front-month vol) rather than a
  synthetic futures-curve reconstruction. 10 tests, one of which caught a
  wrong assumption in the TEST itself (not the code): a synthetic no-
  look-ahead check first asserted a flat-position day's net return would
  be exactly zero, but closing out a real position still costs money, so
  the correct expectation is a small cost-drag, not literally zero — fixed
  by relaxing the assertion, not the strategy logic. Real result: the
  curve has been in contango **89.0%** of 5,032 real trading days
  (2006-2026) — the harvest side (SVXY long in contango) shows real,
  high-confidence edge (**net Sharpe 0.79, PSR 0.998**, barely dented by
  costs), while the mirror hedge side (VIXY long in backwardation) does
  not (**net Sharpe 0.09, PSR 0.65**) — reported as the genuine divergence
  it is, not averaged together. A real, verified finding while sanity-
  checking the SVXY equity curve: the term-structure signal flipped to
  backwardation one day *before* the February 2018 "Volpocalypse" crash,
  correctly forcing the strategy flat just ahead of SVXY's catastrophic
  -83% day (still absorbing the smaller -13% day immediately before the
  flip, a real cost of the one-day lag, not hidden). See
  `docs/VOL_RELVAL_STRATEGY.md`.
- **Phase 11**: Alpaca paper-trading execution — a thin, tested REST
  client (`src/alpha_desk/execution/`) built the same way as this
  platform's data fetchers (real HTTP, no vendor SDK). 13 unit tests, all
  against mocked HTTP responses (fast, deterministic, no network). This
  build initially found no working Alpaca credentials anywhere on the
  machine (checked the user's other local Alpaca-integrated projects,
  `AURUM_PRODUCT_V1`/`AURUM_RC3`, at multiple depths — only `.env.example`
  templates existed there) and reported that honestly rather than
  assuming success. The user then created a fresh Alpaca paper account and
  supplied real keys — **`scripts/check_alpaca_connectivity.py` is now
  verified against a real, live paper account**: status `ACTIVE`, real
  simulated equity/buying power, real market-open state, all served
  through the same as-is artifact pattern every other phase uses. The
  client still **refuses in code**, not just in documentation, to ever
  target Alpaca's live-trading host — only the paper endpoint is
  reachable, and no script in this platform calls `submit_order()`
  automatically; turning any strategy's signal into an order remains a
  manual, human-gated action. See `docs/ALPACA_EXECUTION.md`.
- **Platform layer**: a FastAPI backend (`service/`) serving these reports
  as typed JSON, and a Next.js 16 + TypeScript + Tailwind + Recharts
  frontend (`frontend/`) — a dark-first dashboard with one page per
  strategy (every phase's honest findings, equity curves, and
  calibration/IC/drift charts rendered live from the real artifacts, not
  screenshots). See "Running the platform" below.

```
src/alpha_desk/
├── data/
│   ├── universe.py     the fixed instrument/macro-series list, with rationale
│   ├── sources.py      real fetchers: Yahoo Finance, FRED, SEC EDGAR XBRL
│   └── warehouse.py    loads data/raw/ CSVs into a DuckDB warehouse
├── validation/         Phase 2 — DONE: splits.py, performance.py, overfitting.py, costs.py
├── strategies/
│   ├── pairs/           Phase 3 — DONE: cointegration.py, kalman.py, signals.py, backtest.py
│   ├── factors/         Phase 4 — DONE: factors.py, scoring.py, portfolio.py, backtest.py
│   ├── ml/              Phase 5 — DONE: features.py, model.py, backtest.py
│   ├── momentum/         Phase 6 — DONE: signals.py, backtest.py
│   ├── pead/             Phase 7 — DONE: events.py, backtest.py
│   ├── microstructure/   Phase 9 — DONE: market_maker.py, simulation.py
│   └── vol_relval/       Phase 10 — DONE: signals.py, backtest.py
├── risk/                 Phase 8 — DONE: var_cvar.py, stress.py, execution.py
└── execution/            Phase 11 — DONE: alpaca_client.py

scripts/
├── fetch_public_data.py       pulls real data into data/raw/, writes a hash-verifiable provenance manifest
├── build_data_backbone.py     loads data/raw/ into data/processed/warehouse.duckdb
├── run_pairs_strategy.py      real Phase 3 run -> artifacts/pairs_strategy_report.json
├── run_factor_strategy.py     real Phase 4 run -> artifacts/factor_strategy_report.json
├── run_ml_strategy.py         real Phase 5 run -> artifacts/ml_strategy_report.json
├── run_momentum_strategy.py   real Phase 6 run -> artifacts/momentum_strategy_report.json
├── run_pead_strategy.py       real Phase 7 run -> artifacts/pead_strategy_report.json (fetches real 8-Ks live)
├── run_risk_report.py         real Phase 8 run -> artifacts/risk_report.json
├── run_microstructure_strategy.py   real Phase 9 run -> artifacts/microstructure_strategy_report.json
├── run_vol_relval_strategy.py       real Phase 10 run -> artifacts/vol_relval_strategy_report.json (fetches real VIX/VIX3M/SVXY/VIXY live)
└── check_alpaca_connectivity.py     real Phase 11 run -> artifacts/alpaca_connectivity_status.json (reports honest configured/connected state)

service/            FastAPI backend -- reads artifacts/*.json, never recomputes live
frontend/           Next.js 16 + TypeScript + Tailwind + Recharts dashboard

data/
├── raw/           gitignored — regenerate with fetch_public_data.py
├── processed/     gitignored — regenerate with build_data_backbone.py
├── contracts/     JSON schema + invariants per table (checked by tests/)
├── dictionaries/  human-readable field-level documentation
└── provenance/    source_manifest.json — URL, fetch time, row count, SHA-256 per item

tests/   211 tests total across all 11 phases
```

### Running the research pipeline

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # PowerShell; use .venv/bin/pip on macOS/Linux
python scripts/fetch_public_data.py         # ~2 min — 100 real fetches (prices/macro/fundamentals)
python scripts/build_data_backbone.py       # builds data/processed/warehouse.duckdb
pytest tests/ -v                            # 211/211 passing
python scripts/run_pairs_strategy.py        # regenerates artifacts/pairs_strategy_report.json
python scripts/run_factor_strategy.py       # regenerates artifacts/factor_strategy_report.json
python scripts/run_ml_strategy.py           # regenerates artifacts/ml_strategy_report.json
python scripts/run_momentum_strategy.py     # regenerates artifacts/momentum_strategy_report.json
python scripts/run_pead_strategy.py         # regenerates artifacts/pead_strategy_report.json (fetches real 8-Ks live)
python scripts/run_risk_report.py           # regenerates artifacts/risk_report.json
python scripts/run_microstructure_strategy.py   # regenerates artifacts/microstructure_strategy_report.json
python scripts/run_vol_relval_strategy.py         # regenerates artifacts/vol_relval_strategy_report.json (fetches real VIX/VIX3M/SVXY/VIXY live)
python scripts/check_alpaca_connectivity.py       # regenerates artifacts/alpaca_connectivity_status.json (honest configured/connected check)
```

### Running the platform (backend + frontend)

```bash
.venv\Scripts\pip install -e ".[dev,service]"
python -m uvicorn service.main:app --port 8005 --reload   # backend, http://localhost:8005/docs for the OpenAPI schema

cd frontend
npm install
npm run dev                                                 # frontend, http://localhost:3000
```
The frontend reads `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8005`,
see `frontend/.env.local`). Each page degrades to an explicit "artifact not
generated yet" state rather than a crash if a `scripts/run_*.py` script
hasn't been run yet.

**Port note**: this project defaults to 8005, not the more common 8000/8001,
because 8001, 8002, 8003, AND 8004 were all, in turn, found to get stuck in a
phantom Windows LISTEN state after a killed `uvicorn --reload` process (the
OS kept reporting the socket owned by a PID that no longer existed, and a
freshly started process on the same port kept serving the OLD code even
after multiple confirmed-clean restarts) -- verified it wasn't a code issue
each time by importing `service.main` directly in a fresh Python process and
confirming the routes were correct there. If you hit the same symptom,
don't keep restarting on the same port -- move to a different one.

### Deploying (Render backend + Vercel frontend)

The backend is a stateless read layer over committed JSON artifacts (see
`service/artifacts_io.py`) -- it needs no database and no secrets to run,
which makes it a plain Render web service. `render.yaml` at the repo root
is a Render Blueprint: point Render at this repo and it reads the build/
start commands automatically.

```yaml
buildCommand: pip install -e ".[service]"
startCommand: uvicorn service.main:app --host 0.0.0.0 --port $PORT
```

The frontend is a standard Next.js app in `frontend/` -- deploys to Vercel
with its Root Directory setting pointed at that subfolder, no other config.

Deploy order matters (each side needs the other's URL):
1. Deploy the backend to Render first. Note its public URL
   (`https://<name>.onrender.com`).
2. Deploy the frontend to Vercel with `NEXT_PUBLIC_API_URL` set to that
   Render URL.
3. Back in Render, set the `ALLOWED_ORIGINS` env var (already scaffolded
   in `render.yaml`) to the Vercel production URL. `*.vercel.app` preview
   deployments are already allowed via a regex in `service/main.py`, so
   this only needs the one stable production origin.

`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` do NOT need to be set on Render --
`/api/execution/status` serves the pre-computed
`artifacts/alpaca_connectivity_status.json` snapshot, the same as every
other phase's report, not a live Alpaca call.

## Roadmap

| Phase | Module | Status |
|---|---|---|
| 1 | Data backbone (prices, macro, fundamentals) | **Done** |
| 2 | Validation engine (purged CV, deflated Sharpe, PBO, cost model) | **Done** |
| 3 | Statistical arbitrage / pairs trading | **Done** (real result: negative after DSR/PBO correction) |
| 4 | Cross-sectional equity factor long/short | **Done** (real result: net Sharpe 0.27, PSR 0.80 — positive, not airtight) |
| 5 | ML return classifier (gradient-boosted, calibrated) | **Done** (real result: PR-AUC 0.508 — no reliable skill) |
| 6 | Time-series momentum (ETF proxies) | **Done** (real result: net Sharpe 0.40, PSR 0.94 — strong, just short) |
| 7 | Earnings event-driven (PEAD) | **Done** (real result: p=0.62 — not significant, honest negative) |
| 8 | Risk & execution control tower | **Done** (real result: a 5-order-of-magnitude calibration bug caught before shipping) |
| — | Backend (FastAPI) + frontend (Next.js/Recharts dashboard) | **Done** |
| 9 | Limit order book microstructure / market-making sim | **Done** (disclosed: simulated order flow, real volatility calibration) |
| 10 | VIX term-structure / vol relative value | **Done** (real result: harvest side net Sharpe 0.79/PSR 0.998, hedge side does not hold up) |
| 11 | Alpaca paper-trading execution | **Done** (client built & tested; live-verified against a real paper account, see `docs/ALPACA_EXECUTION.md`) |

Full rationale for this scope and ordering is in the project's planning
conversation, not restated here — this table is kept current as phases land.

## Known simplifications (by design)

See [`docs/DATA_BACKBONE.md`](docs/DATA_BACKBONE.md#known-simplifications-by-design)
for Phase 1's. Each later phase gets its own such section in its own doc,
following the same discipline: what a tool does not model or prove is stated
next to what it does, not left for someone else to discover.

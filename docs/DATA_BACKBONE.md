# Phase 1 — Data Backbone

Real data, three sources, one DuckDB warehouse. This document is the honest
account of what's actually in it and what isn't -- read this before trusting
anything a later strategy/validation module says.

## What's in the warehouse right now

Run `python scripts/fetch_public_data.py` then `python scripts/build_data_backbone.py`
to reproduce from scratch. As of the last run:

- **518,774 price rows** across 50 symbols (40 equities + 10 ETF proxies), full available history, split/dividend-adjusted.
- **93,468 macro observation rows** across 9 FRED series (yield curve, fed funds, VIX, HY credit spread, CPI, unemployment).
- **51,764 fundamentals rows** across 40 tickers, 8 XBRL concepts each, every historical filing (not just the latest).
- **100/100 fetch items succeeded**, hash-verified against `data/provenance/source_manifest.json` (`tests/test_data_backbone.py::TestProvenanceManifest`).

**A gap found three phases later, in Phase 5**: `BAMLH0A0HYM2` (ICE BofA
high-yield credit spread) fetched successfully and passed every Phase 1
test, but FRED's public endpoint only serves ~3 years of history for this
specific series (795 rows from 2023-09-05) versus decades for every other
macro series here (e.g. VIXCLS has 9,568 rows back to 1990) -- confirmed
by re-fetching it directly; not transient. None of Phase 1's tests checked
"does this series' date range look reasonable," only schema/structural
invariants, so this passed silently until Phase 5's ML classifier actually
trained a model on a window where this series was effectively all-NaN and
crashed on it (see `docs/ML_STRATEGY.md`). Fixed defensively downstream
(features with too little history are dropped before model fitting) rather
than by re-litigating Phase 1's already-shipped tests -- but worth knowing
if you use `BAMLH0A0HYM2` for anything: it is real, correct data, just far
shorter than its row count in the fetch manifest might suggest at a glance.

## Sources and why each was picked

| Source | What it provides | Authority level |
|---|---|---|
| SEC EDGAR (`data.sec.gov`) | Fundamentals (XBRL), insider/institutional data in later phases | **Government-authoritative** — the same standing BTS On-Time Performance data has in the airlinesapp project. Every number here is what the company itself filed. |
| FRED (`fred.stlouisfed.org`) | Yield curve, fed funds, VIX, credit spreads, CPI, unemployment | **Government-authoritative** (Federal Reserve Bank of St. Louis). |
| Yahoo Finance (via `yfinance`) | Daily OHLCV prices | **Not authoritative in the same sense.** This is an unofficial, undocumented public endpoint, not a licensed market-data vendor (Bloomberg, Polygon, Tiingo, a broker's own feed). It is real, unmodified market data — not synthetic — but it carries none of a vendor's SLA, and Yahoo has changed/broken this endpoint's shape before without notice. A production desk would use a licensed feed here; this project uses Yahoo because it requires no signup/API key and is what's actually reproducible by anyone re-running this repo. Documented here rather than glossed over.

## Known simplifications (by design)

- **Equity universe is today's constituents, not point-in-time.** The 40-name
  list in `alpha_desk.data.universe.EQUITY_UNIVERSE` is a hand-picked,
  sector-diverse set of *currently* large, liquid names — not a reconstructed
  historical index membership. Any backtest over this universe has a mild
  survivorship bias: a name that was delisted, went bankrupt, or was acquired
  during the sample window simply isn't in the universe at all. A real
  point-in-time membership dataset (e.g. a licensed S&P 500 historical
  constituents file) would remove this; none was freely available, so this is
  flagged rather than hidden.
- **No raw futures data.** Time-series-momentum work (later phase) uses liquid
  ETF proxies (`TLT`, `GLD`, `USO`, `UUP`, `HYG`, `LQD`, ...) instead of actual
  futures contracts, because there is no free/legal bulk-download source for
  continuous futures series (Nasdaq's old free CHRIS/Quandl futures dataset
  was discontinued; commercial alternatives are paywalled). ETF proxies track
  the same risk premia with different roll/carry mechanics — a documented
  substitution, not the real instrument.
- **XBRL concept coverage is a fixed 8-tag allowlist** (`alpha_desk.data.sources.XBRL_CONCEPTS`),
  not the full company-facts payload (which has hundreds of filer-specific
  tags per company). A concept a filer tagged under a nonstandard name is
  silently absent from `fundamentals`, not backfilled or estimated.
- **`filed` vs `end_date` matters and is not automatically enforced downstream.**
  The warehouse stores both; every invariant test checks `filed >= end_date`,
  but it is each *strategy* module's own responsibility to join fundamentals
  on `filed` (point-in-time knowledge), not `end_date` (period the number
  describes) — see `data/dictionaries/DATA_DICTIONARY.md`. Getting this wrong
  is the single most common way a fundamentals-based backtest silently cheats
  with look-ahead bias.
- **No FX/multi-currency handling** — everything here is USD-denominated,
  US-listed. Not a limitation that matters for this universe, but worth
  stating rather than assuming.
- **FRED series can and do have missing observations** (`value` is `NaN` for
  holidays, not-yet-released prints, or discontinued windows) — `macro`'s
  contract states this explicitly; no forward-fill or interpolation is
  applied at the warehouse layer, so any consumer that needs a dense daily
  series has to make that choice itself, deliberately, rather than inheriting
  an invisible one from the data layer.

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/fetch_public_data.py       # ~2 min: 50 price series (1 batch call) + 9 FRED series + 40 SEC filers (rate-limited)
python scripts/build_data_backbone.py     # loads data/raw/ into data/processed/warehouse.duckdb
pytest tests/test_data_backbone.py -v     # 11 tests: schema/invariant checks + hash-verified provenance
```

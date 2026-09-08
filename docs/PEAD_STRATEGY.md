# Phase 7 — Earnings/Event-Driven Drift (PEAD)

`src/alpha_desk/strategies/pead/`: an event study (does announcement-day
direction predict subsequent drift?) plus a calendar-time tradeable
portfolio (Fama, 1998), built on SEC EDGAR 8-K filing timestamps rather
than paid consensus-estimate (SUE) data.

## The real scope limitation, disclosed up front

SEC's `submissions` API returns every 8-K a filer submitted with its date,
but **no item-level detail** (which 8-Ks are Item 2.02 "Results of
Operations," i.e. actual earnings releases, versus M&A, executive changes,
or other material events) without fetching and parsing each individual
filing's document — not done here. So this is a **broader "material
corporate event" drift study, not a pure isolate-earnings PEAD** — a
disclosed proxy, not a hidden one, following the same discipline as every
other data substitution in this platform (Stooq→Yahoo, futures→ETFs).
"Surprise direction" is likewise proxied by the sign of the 2-day
announcement-window return itself (a legitimate, literature-precedented
substitute for SUE when consensus-estimate data isn't accessible), not a
comparison to an analyst consensus number.

## A real, serious bug — caught by cross-checking two numbers against each other

The first real run produced a result that should never have been reported
without investigation: the event study found **no significant relationship**
between announcement direction and subsequent drift (p=0.62, spread
≈0.0015 — statistically nothing) — and yet the "tradeable" backtest of
exactly that (nonexistent) relationship showed a **gross Sharpe of 2.42**.
Those two numbers cannot both be true: if direction doesn't predict drift,
trading on direction cannot produce a Sharpe that high. That internal
contradiction was the signal to stop and debug rather than write down an
exciting number.

**Root cause**: `entry_date` for a trade is defined as the *last* day of
the 2-day announcement window used to pick the trade's own direction. The
first version of the calendar-time aggregator computed each day's return
as `price[day] / price[some_previous_day]` using a globally-shared "most
recent active day" rather than each trade's own price history — and on a
trade's very first charged day, that computation ended up re-scoring part
of the *same announcement jump* that had just been used to decide whether
to go long or short. Every new trade got a free, self-fulfilling first-day
"win" that had nothing to do with real subsequent drift.

**Fixed** by computing each trade's return series directly from its own
`prices[symbol].loc[entry:exit].pct_change()` — which naturally starts its
first non-NaN value at `entry + 1`, never crediting a return that *ends*
at `entry`. A direct regression test locks this in: four isolated
announcement jumps followed by exactly flat prices must show **zero**
portfolio P&L, since there's nothing left to capture after the jump — the
buggy version would have failed this test outright.

After the fix, the two numbers agree, as they must: no real effect, no
real edge.

## The real result

`scripts/run_pead_strategy.py` against the real 40-equity universe: 3,036
real 8-K filings fetched, **3,005 valid event windows, 1,438 traded** above
a 2% announcement-reaction threshold.

- **Event study**: mean 21-day drift after a positive announcement: +1.70%;
  after a negative one: +1.54%. Spread: **0.15 percentage points**, Welch
  t-test **p = 0.62** — not remotely significant. Both buckets drift up by
  a similar amount, consistent with a generally rising market over the
  sample rather than any event-specific continuation effect.
- **Trading it anyway**: gross Sharpe **0.097**, net Sharpe **0.046** (cost
  drag 0.051 — on an already-negligible gross edge, costs eat roughly half
  of it). A clean, honest non-result, reported as exactly that.

This is a genuine negative finding on this specific proxy (broad 8-Ks, not
isolated earnings releases) and universe — it does not rule out real PEAD
on a properly item-filtered earnings-only event set, which would need
either scraping individual filing documents or a paid item-classified feed.

## Running it

```bash
pytest tests/test_pead_events.py tests/test_pead_backtest.py -v
python scripts/run_pead_strategy.py
```

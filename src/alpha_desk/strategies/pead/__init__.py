"""Phase 7 — earnings/event-driven drift (PEAD-style), using SEC EDGAR 8-K
filing timestamps as the event set rather than paid consensus-estimate
(SUE) data. See docs/PEAD_STRATEGY.md for the honest scope limitation this
implies (not filtered to Item 2.02 earnings releases specifically -- SEC's
submissions API has no item-level detail without fetching each filing).

- events.py: per-event announcement return + subsequent drift, plus the
  actual statistical test of whether drift continues in the announcement's
  direction (the PEAD finding itself)
- backtest.py: a calendar-time tradeable portfolio built from events whose
  announcement reaction exceeds a threshold
"""

"""Phase 10 — VIX term-structure relative value.

The VIX term structure (spot VIX vs. VIX3M, the CBOE's 3-month volatility
index) is in CONTANGO (VIX3M > VIX) roughly 80-85% of trading days
historically -- the market prices in more expected volatility further out,
a persistent term premium. Front-month VIX futures (and the ETNs/ETFs that
hold them, like VXX/VIXY) decay toward spot as they roll down this curve,
which is the standard "short-vol carry" trade: be short front-month vol
exposure while the curve is in contango, harvesting the roll-down.

This module trades that signal using SVXY (ProShares Short VIX Short-Term
Futures ETF -- literally holds the short side of this trade already) and
VIXY (ProShares VIX Short-Term Futures ETF -- the long side), rather than
reconstructing a synthetic VIX futures curve from scratch. Both are real,
tradeable, exchange-listed instruments with real historical NAV returns
that already embed real roll cost/decay -- arguably a MORE honest backtest
input than a synthetic futures-curve reconstruction would be.

Two real, disclosed data-quality notes:
- SVXY changed its exposure from -1x to -0.5x after the February 2018
  "Volpocalypse" (a regulatory response to the VIX ETN blowup that liquidated
  the original XIV) -- this is a real, permanent structural change in what
  SVXY's returns represent, not a data error, and this module does not
  attempt to normalize across it.
- VIX3M (ticker ^VIX3M) only has history back to 2006-07, and SVXY only
  since 2011-10 -- earlier dates are simply unavailable, not backfilled.

- signals.py: the contango/backwardation signal itself
- backtest.py: a cost-aware daily backtest, run twice (SVXY-long-in-contango,
  VIXY-long-in-backwardation) to see which side of the trade, if either,
  survives real transaction costs
"""

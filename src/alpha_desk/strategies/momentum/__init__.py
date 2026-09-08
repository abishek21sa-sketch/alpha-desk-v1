"""Phase 6 — time-series momentum (TSMOM), on the platform's 10 liquid ETF
proxies (no raw futures data exists free/legally in bulk -- see
docs/DATA_BACKBONE.md).

Moskowitz, Ooi & Pedersen, "Time Series Momentum" (2012): sign of each
asset's own trailing 12-month return, sized to a common target volatility.
Deliberately NOT cross-sectional (Phase 4's ranking approach) -- each
asset's position depends only on its own history, which is what makes this
a genuinely different technique from Phase 4, not a rename of it.
"""

"""The fixed instrument and macro-series universe Phase 1 pulls real data for.

Equity list is a hand-picked, sector-diverse set of currently large, liquid
US-listed names plus a small cross-asset ETF proxy set (rates, credit, gold,
oil, dollar) used by later momentum/regime modules. This is NOT a
point-in-time index membership snapshot -- it is today's constituents,
selected for liquidity and sector spread, not reconstructed historically.
That means any backtest run over this universe carries a mild survivorship
bias (delisted/acquired/bankrupt names from the sample period are absent).
Documented here rather than discovered later; see docs/DATA_BACKBONE.md.
"""

from __future__ import annotations

# (ticker, stooq_suffix, sector) -- sector is for factor-model grouping later.
EQUITY_UNIVERSE: list[tuple[str, str, str]] = [
    ("AAPL", "us", "Technology"),
    ("MSFT", "us", "Technology"),
    ("NVDA", "us", "Technology"),
    ("GOOGL", "us", "Technology"),
    ("META", "us", "Technology"),
    ("AMZN", "us", "Consumer Discretionary"),
    ("AVGO", "us", "Technology"),
    ("ORCL", "us", "Technology"),
    ("CRM", "us", "Technology"),
    ("ADBE", "us", "Technology"),
    ("JPM", "us", "Financials"),
    ("BAC", "us", "Financials"),
    ("GS", "us", "Financials"),
    ("MS", "us", "Financials"),
    ("WFC", "us", "Financials"),
    ("UNH", "us", "Health Care"),
    ("JNJ", "us", "Health Care"),
    ("LLY", "us", "Health Care"),
    ("PFE", "us", "Health Care"),
    ("ABBV", "us", "Health Care"),
    ("WMT", "us", "Consumer Staples"),
    ("PG", "us", "Consumer Staples"),
    ("KO", "us", "Consumer Staples"),
    ("PEP", "us", "Consumer Staples"),
    ("MCD", "us", "Consumer Discretionary"),
    ("NKE", "us", "Consumer Discretionary"),
    ("HD", "us", "Consumer Discretionary"),
    ("CAT", "us", "Industrials"),
    ("BA", "us", "Industrials"),
    ("GE", "us", "Industrials"),
    ("UPS", "us", "Industrials"),
    ("HON", "us", "Industrials"),
    ("XOM", "us", "Energy"),
    ("CVX", "us", "Energy"),
    ("COP", "us", "Energy"),
    ("DIS", "us", "Communication Services"),
    ("NFLX", "us", "Communication Services"),
    ("T", "us", "Communication Services"),
    ("VZ", "us", "Communication Services"),
    ("NEE", "us", "Utilities"),
]

# Liquid ETF proxies for cross-asset momentum/regime work (Phase 4/7). Used
# as a documented substitute for raw futures data, which has no free/legal
# bulk-download source -- see docs/DATA_BACKBONE.md's known-simplifications
# section for why.
ETF_UNIVERSE: list[tuple[str, str, str]] = [
    ("SPY", "us", "US Equity"),
    ("QQQ", "us", "US Equity (Tech)"),
    ("IWM", "us", "US Equity (Small Cap)"),
    ("TLT", "us", "Long-Term Treasuries"),
    ("IEF", "us", "Intermediate Treasuries"),
    ("GLD", "us", "Gold"),
    ("USO", "us", "Crude Oil"),
    ("UUP", "us", "US Dollar Index"),
    ("HYG", "us", "High Yield Credit"),
    ("LQD", "us", "Investment Grade Credit"),
]

# FRED series (all fetchable from the public fredgraph.csv endpoint, no API
# key required). id -> human label.
FRED_SERIES: dict[str, str] = {
    "DGS3MO": "3-Month Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "DFF": "Effective Federal Funds Rate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS",
    "CPIAUCSL": "CPI, All Urban Consumers (SA)",
    "UNRATE": "Civilian Unemployment Rate",
}


def all_price_symbols() -> list[tuple[str, str, str]]:
    return EQUITY_UNIVERSE + ETF_UNIVERSE

"""Raw factor construction. Every function returns a plain time-indexed
pd.Series for ONE symbol -- scoring.py stacks these into a (date, symbol)
panel and cross-sectionally standardizes them.

The two fundamentals-based factors (value, quality) are POINT-IN-TIME
correct by construction: they join on `filed` (when the number actually
became public via EDGAR), never `end_date` (the accounting period the
number describes) -- see data/dictionaries/DATA_DICTIONARY.md's warning
about exactly this look-ahead trap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(price: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 month momentum (Jegadeesh & Titman 1993): total return from
    t-lookback to t-skip, deliberately excluding the most recent `skip`
    trading days (short-term reversal is a well-documented DIFFERENT effect
    from momentum, and including the most recent month is the standard
    mistake that contaminates a momentum factor with reversal noise).
    """
    return price.shift(skip) / price.shift(lookback) - 1.0


def low_volatility(price: pd.Series, window: int = 60) -> pd.Series:
    """Negative trailing realized volatility of daily log returns, so a
    HIGHER score consistently means "more attractive" across all factors
    (low-vol names score high, matching value/quality/momentum's convention).
    """
    log_returns = np.log(price / price.shift(1))
    return -log_returns.rolling(window, min_periods=window // 2).std()


def point_in_time_fundamental(
    fundamentals: pd.DataFrame, concept: str, as_of_dates: pd.DatetimeIndex, form: str = "10-K"
) -> pd.Series:
    """The most recently FILED value of `concept` (restricted to annual
    `form` filings, default 10-K, to avoid mixing quarterly and annual
    scales) as of each date in `as_of_dates`, for ONE ticker's fundamentals
    slice. Uses `filed`, never `end_date` -- a value isn't usable until it
    was actually public.

    Implemented with merge_asof (backward direction) rather than a Python
    loop per date: correct AND avoids an O(n_dates * n_filings) scan.
    """
    filings = (
        fundamentals[(fundamentals["concept"] == concept) & (fundamentals["form"] == form)]
        .sort_values("filed")
        .drop_duplicates(subset="filed", keep="last")
    )
    if filings.empty:
        return pd.Series(np.nan, index=as_of_dates)

    left = pd.DataFrame({"as_of": as_of_dates}).sort_values("as_of")
    merged = pd.merge_asof(
        left, filings[["filed", "val"]], left_on="as_of", right_on="filed", direction="backward"
    )
    return pd.Series(merged["val"].to_numpy(), index=merged["as_of"]).reindex(as_of_dates)


def value_earnings_yield(
    fundamentals: pd.DataFrame, price: pd.Series, as_of_dates: pd.DatetimeIndex
) -> pd.Series:
    """Trailing earnings yield: most recent annual NetIncomeLoss / shares
    outstanding, divided by price on each rebalance date -- an E/P-style
    value signal (higher = cheaper = more attractive, consistent sign
    convention with the other factors).
    """
    net_income = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of_dates)
    shares = point_in_time_fundamental(fundamentals, "CommonStockSharesOutstanding", as_of_dates)
    eps = net_income / shares
    px = price.reindex(as_of_dates)
    return eps / px


def quality_roe(fundamentals: pd.DataFrame, as_of_dates: pd.DatetimeIndex) -> pd.Series:
    """Return on equity: most recent annual NetIncomeLoss / StockholdersEquity."""
    net_income = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of_dates)
    equity = point_in_time_fundamental(fundamentals, "StockholdersEquity", as_of_dates)
    return net_income / equity

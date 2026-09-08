"""The VIX term-structure signal: contango vs. backwardation."""

from __future__ import annotations

import pandas as pd


def term_structure_ratio(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """VIX3M / VIX. > 1 means contango (the historically normal state);
    < 1 means backwardation (typically acute market stress)."""
    return vix3m / vix


def directional_position(vix: pd.Series, vix3m: pd.Series, long_when: str) -> pd.Series:
    """1.0 on days matching `long_when` ("contango" or "backwardation"), 0.0
    otherwise -- a binary long-or-flat signal, using each date's OWN close
    (the caller lags this by one day before applying it to the next day's
    return, the same lag contract as every other strategy on this platform).
    """
    if long_when not in ("contango", "backwardation"):
        raise ValueError("long_when must be 'contango' or 'backwardation'")
    ratio = term_structure_ratio(vix, vix3m)
    is_contango = ratio > 1.0
    return (is_contango if long_when == "contango" else ~is_contango).astype(float)

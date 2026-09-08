"""Thin, honest fetchers for the three real data sources Phase 1 uses.

No source here is faked or simulated -- every function makes a real HTTP
request to a real public provider and returns exactly what it sent back
(reshaped into a tidy DataFrame, nothing invented). Network/parsing failures
raise rather than silently returning empty data, so a partial fetch is
never mistaken for a complete one -- callers (scripts/fetch_public_data.py)
decide how to record that in provenance.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests
import yfinance as yf

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC explicitly requires a descriptive User-Agent identifying the requester
# (see https://www.sec.gov/os/webmaster-faq#developers) -- this is the
# project's own outbound identification, not user PII sent to an unrelated
# service.
SEC_USER_AGENT = "AlphaDeskV1 research project (contact: abishek21sa@hotmail.com)"

REQUEST_TIMEOUT_S = 20
SEC_RATE_LIMIT_SLEEP_S = 0.3


def fetch_yfinance_history(symbols: list[str], period: str = "max") -> dict[str, pd.DataFrame]:
    """Split-and-dividend-adjusted daily OHLCV for a batch of Yahoo Finance
    symbols, one HTTP round-trip for the whole batch. Yahoo's public chart
    endpoint is unofficial/undocumented (not a licensed vendor feed) -- see
    docs/DATA_BACKBONE.md's known-simplifications section; it is nonetheless
    real, unmodified market data, not synthetic.

    Returns {symbol: DataFrame} with columns [symbol, date, open, high, low,
    close, volume]; a symbol Yahoo returned nothing for is simply absent from
    the dict (caller records that as a failure, not silently skips it).
    """
    raw = yf.download(
        tickers=symbols,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            continue
        sub = raw[symbol].dropna(how="all")
        if sub.empty:
            continue
        sub = sub.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        sub["symbol"] = symbol
        sub["date"] = pd.to_datetime(sub["date"])
        out[symbol] = sub[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values("date")
    return out


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Full history for one FRED series via the public fredgraph.csv endpoint."""
    url = FRED_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    text = resp.text.strip()
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    # FRED's fredgraph.csv has used both "DATE" and "observation_date" as the
    # date column name across its history -- accept either.
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or series_id not in df.columns:
        raise ValueError(f"fred schema mismatch for {series_id}: got {list(df.columns)}")
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"])
    # FRED marks missing observations (holidays/not-yet-released) with "."
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    return df[["series_id", "date", "value"]].sort_values("date")


def fetch_sec_ticker_to_cik() -> dict[str, int]:
    """SEC's canonical ticker -> CIK mapping (one JSON file, all US filers)."""
    resp = requests.get(
        SEC_TICKERS_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=REQUEST_TIMEOUT_S
    )
    resp.raise_for_status()
    raw = resp.json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def fetch_sec_company_facts(cik: int) -> dict:
    """Raw XBRL company-facts JSON for one filer (every tagged concept, every filing)."""
    url = SEC_COMPANY_FACTS_URL.format(cik=cik)
    resp = requests.get(
        url, headers={"User-Agent": SEC_USER_AGENT}, timeout=REQUEST_TIMEOUT_S
    )
    resp.raise_for_status()
    return resp.json()


def fetch_sec_submissions(cik: int) -> dict:
    """Raw filing-history JSON for one filer -- form type, filing date, and
    accession number for (by default) the filer's most recent ~1000
    filings, under `["filings"]["recent"]"` as parallel arrays. Filers with
    more history have older filings paginated into separate documents
    referenced under `["filings"]["files"]`, not fetched here -- 1000
    filings comfortably covers this project's lookback window for every
    ticker in the universe.
    """
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def extract_8k_filing_dates(submissions_json: dict) -> list[str]:
    """ISO date strings for every 8-K in the submissions payload's 'recent'
    filings block. NOT filtered to a specific 8-K item number (e.g. Item
    2.02, "Results of Operations") -- SEC's submissions API doesn't expose
    item-level detail without fetching each filing's own document, so this
    is every 8-K a filer submitted (M&A, executive changes, etc. included,
    not just earnings releases) -- see docs/PEAD_STRATEGY.md for why this
    is disclosed as a real scope limitation, not hidden.
    """
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    return [d for f, d in zip(forms, dates) if f == "8-K"]


# The fixed set of us-gaap XBRL concepts Phase 1 extracts per filer -- the
# raw companyfacts JSON has hundreds of tags per company; this is the subset
# the later fundamentals/factor work actually uses.
XBRL_CONCEPTS: list[str] = [
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "EarningsPerShareDiluted",
    "CommonStockSharesOutstanding",
    "OperatingIncomeLoss",
]


def extract_xbrl_concepts(facts_json: dict, concepts: list[str] = XBRL_CONCEPTS) -> pd.DataFrame:
    """Flatten selected us-gaap concepts out of a companyfacts payload into one tidy table."""
    rows = []
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            continue
        for unit, observations in node.get("units", {}).items():
            for obs in observations:
                rows.append(
                    {
                        "concept": concept,
                        "unit": unit,
                        "end_date": obs.get("end"),
                        "start_date": obs.get("start"),
                        "val": obs.get("val"),
                        "fy": obs.get("fy"),
                        "fp": obs.get("fp"),
                        "form": obs.get("form"),
                        "filed": obs.get("filed"),
                        "accn": obs.get("accn"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df.sort_values(["concept", "end_date"])


def polite_sleep() -> None:
    time.sleep(SEC_RATE_LIMIT_SLEEP_S)

"""Loads data/raw/ CSVs into a single DuckDB warehouse at
data/processed/warehouse.duckdb, with one table per dataset. Pure
load-and-shape -- no computed signals live here; those belong to
alpha_desk.validation / alpha_desk.strategies once built.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = PROCESSED / "warehouse.duckdb"


def _load_prices() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "prices").glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


def _load_macro() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "macro").glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["series_id", "date", "value"])
    return pd.concat(frames, ignore_index=True)


def _load_fundamentals() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "fundamentals").glob("*.csv")):
        if path.name.startswith("_"):
            continue
        df = pd.read_csv(path, parse_dates=["end_date", "start_date", "filed"])
        df["ticker"] = path.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_universe_metadata() -> pd.DataFrame:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from alpha_desk.data.universe import ETF_UNIVERSE, EQUITY_UNIVERSE

    rows = [
        {"symbol": t, "sector": s, "asset_class": "equity"} for t, _, s in EQUITY_UNIVERSE
    ] + [{"symbol": t, "sector": s, "asset_class": "etf"} for t, _, s in ETF_UNIVERSE]
    return pd.DataFrame(rows)


def build_warehouse() -> dict:
    """(Re)builds the DuckDB warehouse from whatever is currently in data/raw/.
    Returns a small summary dict (used by scripts/build_data_backbone.py to
    print a report and by tests to assert non-trivial content).
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prices = _load_prices()
    macro = _load_macro()
    fundamentals = _load_fundamentals()
    universe = _load_universe_metadata()

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("DROP TABLE IF EXISTS prices")
        con.execute("DROP TABLE IF EXISTS macro")
        con.execute("DROP TABLE IF EXISTS fundamentals")
        con.execute("DROP TABLE IF EXISTS universe")
        con.register("prices_df", prices)
        con.execute("CREATE TABLE prices AS SELECT * FROM prices_df")
        con.register("macro_df", macro)
        con.execute("CREATE TABLE macro AS SELECT * FROM macro_df")
        con.register("fundamentals_df", fundamentals)
        con.execute("CREATE TABLE fundamentals AS SELECT * FROM fundamentals_df")
        con.register("universe_df", universe)
        con.execute("CREATE TABLE universe AS SELECT * FROM universe_df")

        summary = {
            "prices_rows": con.execute("SELECT count(*) FROM prices").fetchone()[0],
            "prices_symbols": con.execute("SELECT count(DISTINCT symbol) FROM prices").fetchone()[0],
            "macro_rows": con.execute("SELECT count(*) FROM macro").fetchone()[0],
            "macro_series": con.execute("SELECT count(DISTINCT series_id) FROM macro").fetchone()[0],
            "fundamentals_rows": con.execute("SELECT count(*) FROM fundamentals").fetchone()[0],
            "fundamentals_tickers": (
                con.execute("SELECT count(DISTINCT ticker) FROM fundamentals").fetchone()[0]
                if len(fundamentals)
                else 0
            ),
        }
    finally:
        con.close()

    (PROCESSED / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)

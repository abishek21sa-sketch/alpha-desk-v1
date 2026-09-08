"""Phase 1: build the DuckDB warehouse from data/raw/ and print a report.
Assumes scripts/fetch_public_data.py has already been run at least once.

Usage:
    python scripts/build_data_backbone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.warehouse import DB_PATH, build_warehouse  # noqa: E402


def main() -> None:
    summary = build_warehouse()
    print(f"Warehouse built at {DB_PATH}")
    print(f"  prices:       {summary['prices_rows']:,} rows across {summary['prices_symbols']} symbols")
    print(f"  macro:        {summary['macro_rows']:,} rows across {summary['macro_series']} series")
    print(f"  fundamentals: {summary['fundamentals_rows']:,} rows across {summary['fundamentals_tickers']} tickers")

    if summary["prices_symbols"] == 0:
        print("\nWARNING: no price data -- run scripts/fetch_public_data.py first.")
        sys.exit(1)


if __name__ == "__main__":
    main()

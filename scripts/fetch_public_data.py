"""Phase 1: pull real data from Stooq (prices), FRED (macro), and SEC EDGAR
(fundamentals) into data/raw/, and write a hash-verifiable provenance
manifest recording exactly what was fetched, from where, and when.

Every fetch is real -- no cached fixtures, no synthetic fallback. A failed
symbol/series is recorded as a failure in the manifest, not silently
dropped, so data/provenance/source_manifest.json is always an honest
account of what this run actually retrieved.

Usage:
    python scripts/fetch_public_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data import sources  # noqa: E402
from alpha_desk.data.universe import (  # noqa: E402
    EQUITY_UNIVERSE,
    FRED_SERIES,
    all_price_symbols,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROVENANCE = ROOT / "data" / "provenance"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(manifest: list[dict], **kwargs) -> None:
    kwargs["fetched_at"] = now_iso()
    manifest.append(kwargs)
    status = kwargs.get("status")
    name = kwargs.get("name")
    print(f"  [{status}] {name}")


def fetch_prices(manifest: list[dict]) -> None:
    out_dir = RAW / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = all_price_symbols()
    tickers = [t for t, _, _ in symbols]
    print(f"Fetching {len(tickers)} price series from Yahoo Finance (one batch request)...")
    try:
        by_symbol = sources.fetch_yfinance_history(tickers, period="max")
    except Exception as exc:  # noqa: BLE001
        record(manifest, dataset="prices", name="_batch", source="yfinance",
               status="failed", error=str(exc))
        return

    for ticker in tickers:
        path = out_dir / f"{ticker}.csv"
        df = by_symbol.get(ticker)
        if df is None or df.empty:
            record(manifest, dataset="prices", name=ticker, source="yfinance",
                   status="failed", error="no data returned for this symbol")
            continue
        df.to_csv(path, index=False)
        record(
            manifest,
            dataset="prices",
            name=ticker,
            source="yfinance",
            source_url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            path=path.relative_to(ROOT).as_posix(),
            rows=len(df),
            date_min=str(df["date"].min().date()) if len(df) else None,
            date_max=str(df["date"].max().date()) if len(df) else None,
            sha256=sha256_of(path),
            status="ok",
        )


def fetch_macro(manifest: list[dict]) -> None:
    out_dir = RAW / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(FRED_SERIES)} macro series from FRED...")
    for series_id, label in FRED_SERIES.items():
        path = out_dir / f"{series_id}.csv"
        try:
            df = sources.fetch_fred_series(series_id)
            df.to_csv(path, index=False)
            record(
                manifest,
                dataset="macro",
                name=series_id,
                label=label,
                source="fred",
                source_url=sources.FRED_URL.format(series_id=series_id),
                path=path.relative_to(ROOT).as_posix(),
                rows=len(df),
                date_min=str(df["date"].min().date()) if len(df) else None,
                date_max=str(df["date"].max().date()) if len(df) else None,
                sha256=sha256_of(path),
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            record(
                manifest,
                dataset="macro",
                name=series_id,
                source="fred",
                status="failed",
                error=str(exc),
            )


def fetch_fundamentals(manifest: list[dict]) -> None:
    out_dir = RAW / "fundamentals"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [t for t, _, _ in EQUITY_UNIVERSE]
    print("Fetching SEC ticker -> CIK map...")
    try:
        ticker_to_cik = sources.fetch_sec_ticker_to_cik()
    except Exception as exc:  # noqa: BLE001
        record(manifest, dataset="fundamentals", name="ticker_map", source="sec_edgar",
               status="failed", error=str(exc))
        return

    cik_map_path = RAW / "fundamentals" / "_ticker_cik_map.json"
    subset = {t: ticker_to_cik.get(t) for t in tickers}
    cik_map_path.write_text(json.dumps(subset, indent=2))
    record(
        manifest,
        dataset="fundamentals",
        name="ticker_cik_map",
        source="sec_edgar",
        source_url=sources.SEC_TICKERS_URL,
        path=cik_map_path.relative_to(ROOT).as_posix(),
        rows=len(subset),
        sha256=sha256_of(cik_map_path),
        status="ok",
    )

    print(f"Fetching SEC company facts for {len(tickers)} tickers "
          f"(rate-limited to {sources.SEC_RATE_LIMIT_SLEEP_S}s/request)...")
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        path = out_dir / f"{ticker}.csv"
        if cik is None:
            record(manifest, dataset="fundamentals", name=ticker, source="sec_edgar",
                   status="failed", error="ticker not found in SEC company_tickers.json")
            continue
        try:
            facts = sources.fetch_sec_company_facts(cik)
            df = sources.extract_xbrl_concepts(facts)
            df.to_csv(path, index=False)
            record(
                manifest,
                dataset="fundamentals",
                name=ticker,
                cik=cik,
                source="sec_edgar",
                source_url=sources.SEC_COMPANY_FACTS_URL.format(cik=cik),
                path=path.relative_to(ROOT).as_posix(),
                rows=len(df),
                concepts_found=sorted(df["concept"].unique().tolist()) if len(df) else [],
                sha256=sha256_of(path),
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            record(manifest, dataset="fundamentals", name=ticker, source="sec_edgar",
                   status="failed", error=str(exc))
        finally:
            sources.polite_sleep()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    fetch_prices(manifest)
    fetch_macro(manifest)
    fetch_fundamentals(manifest)

    ok = sum(1 for m in manifest if m["status"] == "ok")
    failed = sum(1 for m in manifest if m["status"] == "failed")
    summary = {
        "run_at": now_iso(),
        "total_items": len(manifest),
        "ok": ok,
        "failed": failed,
        "items": manifest,
    }
    manifest_path = PROVENANCE / "source_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nDone: {ok} ok, {failed} failed. Manifest -> {manifest_path.relative_to(ROOT)}")
    if failed:
        print("Failed items:")
        for m in manifest:
            if m["status"] == "failed":
                print(f"  - {m['dataset']}/{m['name']}: {m.get('error')}")


if __name__ == "__main__":
    main()

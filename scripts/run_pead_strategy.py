"""Phase 7: fetch real 8-K filing dates from SEC EDGAR for the 40-equity
universe, then run the PEAD calendar-time backtest.

8-K dates are fetched live here (not persisted to the warehouse) -- 40
rate-limited requests take well under a minute, and this is Phase 7's only
consumer of this data so far.

Usage:
    python scripts/run_pead_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data import sources  # noqa: E402
from alpha_desk.data.universe import EQUITY_UNIVERSE  # noqa: E402
from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.strategies.pead.backtest import run_pead_backtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load_prices_volumes():
    con = get_connection(read_only=True)
    tickers = [t for t, _, _ in EQUITY_UNIVERSE]
    df = con.execute(
        "SELECT symbol, date, close, volume FROM prices WHERE symbol IN "
        f"({','.join('?' * len(tickers))}) ORDER BY symbol, date",
        tickers,
    ).df()
    con.close()
    prices, volumes = {}, {}
    for sym, g in df.groupby("symbol"):
        g = g.set_index("date").sort_index()
        prices[sym] = g["close"]
        volumes[sym] = g["volume"]
    return prices, volumes


def fetch_8k_events(tickers: list[str]) -> dict[str, list[str]]:
    print("Fetching SEC ticker -> CIK map...")
    ticker_to_cik = sources.fetch_sec_ticker_to_cik()
    events = {}
    print(f"Fetching 8-K filing history for {len(tickers)} tickers "
          f"(rate-limited to {sources.SEC_RATE_LIMIT_SLEEP_S}s/request)...")
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            print(f"  [skip] {ticker}: not found in SEC ticker map")
            continue
        try:
            submissions = sources.fetch_sec_submissions(cik)
            dates = sources.extract_8k_filing_dates(submissions)
            events[ticker] = dates
            print(f"  [ok] {ticker}: {len(dates)} 8-K filings")
        except Exception as exc:  # noqa: BLE001
            print(f"  [failed] {ticker}: {exc}")
        finally:
            sources.polite_sleep()
    return events


def main() -> None:
    print("Loading real prices/volumes from the warehouse...")
    prices, volumes = load_prices_volumes()
    tickers = [t for t, _, _ in EQUITY_UNIVERSE]

    events = fetch_8k_events(tickers)
    total_events = sum(len(v) for v in events.values())
    print(f"\n{total_events} total 8-K filings across {len(events)} tickers")

    print("\nRunning PEAD calendar-time backtest (21-day drift window, 2% announcement threshold)...")
    result = run_pead_backtest(prices, volumes, events)

    sig = result.significance
    print(f"\n{'=' * 70}")
    print(f"EVENT STUDY ({result.n_events_total} events with valid outcome windows, "
          f"{result.n_events_traded} traded above threshold)")
    print(f"  positive-announcement events: {sig.n_positive_events}, mean 21d drift: {sig.mean_drift_positive:.4f}")
    print(f"  negative-announcement events: {sig.n_negative_events}, mean 21d drift: {sig.mean_drift_negative:.4f}")
    print(f"  drift spread (the PEAD effect): {sig.drift_spread:.4f}")
    print(f"  Welch t-test: t={sig.t_stat:.3f}, p={sig.p_value:.4f}")
    if sig.p_value >= 0.05:
        print("  --> NOT statistically significant at the 5% level. Reported honestly.")
    else:
        print("  --> statistically significant at the 5% level.")

    print(f"\nTRADING IT (max {result.max_concurrent_trades} concurrent positions)")
    print(f"  gross annualized Sharpe: {result.gross_sharpe_annualized:.3f}")
    print(f"  net annualized Sharpe:   {result.net_sharpe_annualized:.3f}")
    print(f"  cost drag (Sharpe pts):  {result.cost_drag_annualized:.3f}")

    equity_curve = (1.0 + result.returns.dropna()).cumprod() - 1.0
    ARTIFACTS.mkdir(exist_ok=True)
    report = {
        "n_events_total": result.n_events_total,
        "n_events_traded": result.n_events_traded,
        "max_concurrent_trades": result.max_concurrent_trades,
        "gross_sharpe": result.gross_sharpe_annualized,
        "net_sharpe": result.net_sharpe_annualized,
        "cost_drag": result.cost_drag_annualized,
        "significance": {
            "n_positive_events": sig.n_positive_events,
            "n_negative_events": sig.n_negative_events,
            "mean_drift_positive": sig.mean_drift_positive,
            "mean_drift_negative": sig.mean_drift_negative,
            "drift_spread": sig.drift_spread,
            "t_stat": sig.t_stat,
            "p_value": sig.p_value,
        },
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "cumulative_return": float(v)}
            for d, v in equity_curve.items()
        ],
    }
    out_path = ARTIFACTS / "pead_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

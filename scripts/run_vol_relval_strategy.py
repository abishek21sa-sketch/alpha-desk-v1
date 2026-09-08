"""Phase 10: fetch real VIX/VIX3M/SVXY/VIXY history live from Yahoo Finance
(not persisted to the warehouse -- this is Phase 10's only consumer, same
precedent as Phase 7's live SEC 8-K fetch), then run the term-structure
relative-value backtest twice: SVXY long-in-contango (the standard
"harvest the roll" trade) and VIXY long-in-backwardation (the mirror
"hedge/momentum" trade), to see which side of the trade, if either,
survives real transaction costs.

Usage:
    python scripts/run_vol_relval_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.sources import fetch_yfinance_history  # noqa: E402
from alpha_desk.strategies.vol_relval.backtest import run_vol_relval_backtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

SYMBOLS = ["^VIX", "^VIX3M", "SVXY", "VIXY"]


def main() -> None:
    print(f"Fetching real daily history for {SYMBOLS} from Yahoo Finance...")
    history = fetch_yfinance_history(SYMBOLS, period="max")
    missing = [s for s in SYMBOLS if s not in history]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no data for {missing}")

    for sym in SYMBOLS:
        df = history[sym]
        print(f"  {sym}: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")

    vix = history["^VIX"].set_index("date")["close"]
    vix3m = history["^VIX3M"].set_index("date")["close"]
    svxy_price = history["SVXY"].set_index("date")["close"]
    svxy_volume = history["SVXY"].set_index("date")["volume"]
    vixy_price = history["VIXY"].set_index("date")["close"]
    vixy_volume = history["VIXY"].set_index("date")["volume"]

    common_ratio_days = vix.index.intersection(vix3m.index)
    contango_ratio = vix3m.reindex(common_ratio_days) / vix.reindex(common_ratio_days)
    pct_contango_all_history = float((contango_ratio > 1.0).mean())
    print(f"\nTerm structure: {pct_contango_all_history:.1%} of days in contango over "
          f"{len(common_ratio_days)} days of common VIX/VIX3M history.")

    print("\n" + "=" * 78)
    print("Scenario 1: long SVXY (short front-month vol) while the curve is in contango")
    result_svxy = run_vol_relval_backtest(
        vix, vix3m, svxy_price, svxy_volume, instrument="SVXY", long_when="contango",
    )
    print(f"  n_days={result_svxy.n_days}  pct_in_position={result_svxy.pct_days_in_position:.1%}  "
          f"gross_sharpe={result_svxy.gross_sharpe_annualized:.3f}  net_sharpe={result_svxy.net_sharpe_annualized:.3f}  "
          f"cost_drag={result_svxy.cost_drag_annualized:.3f}  PSR={result_svxy.probabilistic_sharpe_ratio:.3f}")

    print("\nScenario 2: long VIXY (long front-month vol) while the curve is in backwardation")
    result_vixy = run_vol_relval_backtest(
        vix, vix3m, vixy_price, vixy_volume, instrument="VIXY", long_when="backwardation",
    )
    print(f"  n_days={result_vixy.n_days}  pct_in_position={result_vixy.pct_days_in_position:.1%}  "
          f"gross_sharpe={result_vixy.gross_sharpe_annualized:.3f}  net_sharpe={result_vixy.net_sharpe_annualized:.3f}  "
          f"cost_drag={result_vixy.cost_drag_annualized:.3f}  PSR={result_vixy.probabilistic_sharpe_ratio:.3f}")
    print("=" * 78)

    def result_to_dict(r):
        return {
            "instrument": r.instrument,
            "long_when": r.long_when,
            "n_days": r.n_days,
            "pct_days_in_position": r.pct_days_in_position,
            "gross_sharpe_annualized": r.gross_sharpe_annualized,
            "net_sharpe_annualized": r.net_sharpe_annualized,
            "cost_drag_annualized": r.cost_drag_annualized,
            "probabilistic_sharpe_ratio": r.probabilistic_sharpe_ratio,
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "cumulative_return": c}
                for d, c in (1.0 + r.returns).cumprod().sub(1.0).items()
            ],
        }

    report = {
        "pct_contango_all_history": pct_contango_all_history,
        "n_days_term_structure_history": len(common_ratio_days),
        "scenarios": {
            "svxy_long_in_contango": result_to_dict(result_svxy),
            "vixy_long_in_backwardation": result_to_dict(result_vixy),
        },
    }

    out_path = ARTIFACTS / "vol_relval_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

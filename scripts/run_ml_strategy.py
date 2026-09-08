"""Phase 5: run the calibrated gradient-boosted classifier against the real
40-equity universe, walk-forward, and report it honestly AS A CLASSIFIER
(PR-AUC/Brier/log-loss/calibration) before ever looking at the Sharpe ratio
of trading its predictions.

Usage:
    python scripts/run_ml_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.universe import EQUITY_UNIVERSE, FRED_SERIES  # noqa: E402
from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.strategies.ml.backtest import run_ml_backtest  # noqa: E402
from alpha_desk.validation.performance import probabilistic_sharpe_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
N_SPLITS = 3


def load_universe():
    con = get_connection(read_only=True)
    tickers = [t for t, _, _ in EQUITY_UNIVERSE]

    price_df = con.execute(
        "SELECT symbol, date, close, volume FROM prices WHERE symbol IN "
        f"({','.join('?' * len(tickers))}) ORDER BY symbol, date",
        tickers,
    ).df()
    fund_df = con.execute(
        "SELECT ticker, concept, unit, end_date, start_date, val, fy, fp, form, filed, accn "
        "FROM fundamentals WHERE ticker IN "
        f"({','.join('?' * len(tickers))}) ORDER BY ticker, concept, filed",
        tickers,
    ).df()
    macro_df = con.execute(
        "SELECT series_id, date, value FROM macro WHERE series_id IN "
        f"({','.join('?' * len(FRED_SERIES))}) ORDER BY series_id, date",
        list(FRED_SERIES),
    ).df()
    con.close()

    prices, volumes, fundamentals, macro = {}, {}, {}, {}
    for sym, g in price_df.groupby("symbol"):
        g = g.set_index("date").sort_index()
        prices[sym] = g["close"]
        volumes[sym] = g["volume"]
    for sym, g in fund_df.groupby("ticker"):
        fundamentals[sym] = g.drop(columns="ticker").reset_index(drop=True)
    for series_id, g in macro_df.groupby("series_id"):
        macro[series_id] = g.set_index("date")["value"].sort_index()

    for sym in tickers:
        fundamentals.setdefault(sym, pd.DataFrame(columns=["concept", "form", "end_date", "filed", "val"]))

    return prices, volumes, fundamentals, macro


def main() -> None:
    print("Loading real prices/volumes/fundamentals/macro from the warehouse...")
    prices, volumes, fundamentals, macro = load_universe()
    print(f"  {len(prices)} symbols, {len(macro)} macro series loaded")

    print(f"\nRunning walk-forward ML backtest (n_splits={N_SPLITS})...")
    result = run_ml_backtest(prices, volumes, fundamentals, macro, n_splits=N_SPLITS)

    pm = result.pooled_metrics
    print(f"\n{'=' * 70}")
    print(f"CLASSIFIER PERFORMANCE ({result.n_folds} folds, {result.n_folds_skipped} skipped, "
          f"{pm.n_obs} pooled OOS predictions)")
    print(f"  base rate (fraction beating median): {pm.base_rate:.3f}")
    print(f"  PR-AUC:     {pm.pr_auc:.3f}")
    print(f"  Brier score: {pm.brier_score:.4f}  (0=perfect, 0.25=coin flip)")
    print(f"  log-loss:    {pm.log_loss:.4f}")
    print("  calibration bins (mean predicted vs. mean actual):")
    for _, row in pm.calibration_bins.iterrows():
        print(f"    predicted={row['mean_predicted']:.3f}  actual={row['mean_actual']:.3f}  n={int(row['count'])}")

    psr = probabilistic_sharpe_ratio(result.returns.dropna().to_numpy(), benchmark_sr=0.0)

    print(f"\nTRADING THE PREDICTIONS ({result.n_rebalances} rebalances)")
    print(f"  gross annualized Sharpe: {result.gross_sharpe_annualized:.3f}")
    print(f"  net annualized Sharpe:   {result.net_sharpe_annualized:.3f}")
    print(f"  cost drag (Sharpe pts):  {result.cost_drag_annualized:.3f}")
    print(f"  Probabilistic Sharpe Ratio (P[true per-period Sharpe > 0]): {psr:.3f}")

    if pm.pr_auc <= pm.base_rate + 0.02:
        print("\n  --> PR-AUC is at or near the base rate: this model shows no reliable "
              "out-of-sample skill on this universe/period. Reported honestly.")
        print("  --> The positive Sharpe above, if PSR is also unremarkable, is exactly "
              "the kind of small-sample noise Phase 2's tools exist to catch -- a "
              "near-chance classifier producing a decent-looking backtest Sharpe is a "
              "textbook illustration of why PR-AUC/Brier must be checked BEFORE the Sharpe, "
              "not instead of it.")

    equity_curve = (1.0 + result.returns.dropna()).cumprod() - 1.0

    ARTIFACTS.mkdir(exist_ok=True)
    report = {
        "n_splits": N_SPLITS,
        "n_folds": result.n_folds,
        "n_folds_skipped": result.n_folds_skipped,
        "n_rebalances": result.n_rebalances,
        "gross_sharpe": result.gross_sharpe_annualized,
        "net_sharpe": result.net_sharpe_annualized,
        "cost_drag": result.cost_drag_annualized,
        "probabilistic_sharpe_ratio": psr,
        "pooled_metrics": {
            "pr_auc": pm.pr_auc,
            "brier_score": pm.brier_score,
            "log_loss": pm.log_loss,
            "n_obs": pm.n_obs,
            "base_rate": pm.base_rate,
            "calibration_bins": pm.calibration_bins.to_dict(orient="records"),
        },
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "cumulative_return": float(v)}
            for d, v in equity_curve.items()
        ],
    }
    out_path = ARTIFACTS / "ml_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

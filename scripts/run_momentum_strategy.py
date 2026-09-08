"""Phase 6: run the TSMOM backtest against the real 10-ETF proxy universe.

Usage:
    python scripts/run_momentum_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.universe import ETF_UNIVERSE  # noqa: E402
from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.strategies.momentum.backtest import run_tsmom_backtest  # noqa: E402
from alpha_desk.validation.performance import probabilistic_sharpe_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load_etfs():
    con = get_connection(read_only=True)
    tickers = [t for t, _, _ in ETF_UNIVERSE]
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


def main() -> None:
    print("Loading real ETF proxy prices/volumes from the warehouse...")
    prices, volumes = load_etfs()
    print(f"  {len(prices)} ETFs loaded: {sorted(prices)}")

    print("\nRunning TSMOM backtest (12-month lookback, 10% vol target, monthly rebalance)...")
    result = run_tsmom_backtest(prices, volumes)
    psr = probabilistic_sharpe_ratio(result.returns.dropna().to_numpy(), benchmark_sr=0.0)

    print(f"\n{'=' * 70}")
    print(f"PORTFOLIO RESULT ({result.n_assets} assets, {result.n_rebalances} rebalances)")
    print(f"  gross annualized Sharpe: {result.gross_sharpe_annualized:.3f}")
    print(f"  net annualized Sharpe:   {result.net_sharpe_annualized:.3f}")
    print(f"  cost drag (Sharpe pts):  {result.cost_drag_annualized:.3f}")
    print(f"  Probabilistic Sharpe Ratio (P[true per-period Sharpe > 0]): {psr:.3f}")
    if psr < 0.95:
        print("  --> does not clear a 95% confidence bar. Reported honestly.")

    print("\n  Per-asset net Sharpe:")
    for sym, sr in sorted(result.per_asset_net_sharpe.items(), key=lambda x: -x[1]):
        print(f"    {sym}: {sr:.2f}")

    equity_curve = (1.0 + result.returns.dropna()).cumprod() - 1.0
    ARTIFACTS.mkdir(exist_ok=True)
    report = {
        "n_assets": result.n_assets,
        "n_rebalances": result.n_rebalances,
        "gross_sharpe": result.gross_sharpe_annualized,
        "net_sharpe": result.net_sharpe_annualized,
        "cost_drag": result.cost_drag_annualized,
        "probabilistic_sharpe_ratio": psr,
        "per_asset_sharpe": result.per_asset_net_sharpe,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "cumulative_return": float(v)}
            for d, v in equity_curve.items()
        ],
    }
    out_path = ARTIFACTS / "momentum_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

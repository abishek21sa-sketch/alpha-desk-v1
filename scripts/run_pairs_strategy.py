"""Phase 3: screen the real equity universe for cointegrated pairs, backtest
every tradeable pair through the Kalman + z-score + cost-aware pipeline, and
apply Phase 2's Deflated Sharpe Ratio / PBO across ALL pairs tried -- not
just the winner -- to honestly report whether the best-looking pair is
actually distinguishable from having tried a lot of pairs and gotten lucky.

Usage:
    python scripts/run_pairs_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.universe import EQUITY_UNIVERSE  # noqa: E402
from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.strategies.pairs.backtest import backtest_pair  # noqa: E402
from alpha_desk.strategies.pairs.cointegration import screen_universe_for_pairs  # noqa: E402
from alpha_desk.validation.overfitting import probability_of_backtest_overfitting  # noqa: E402
from alpha_desk.validation.performance import deflated_sharpe_ratio, sharpe_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


LOOKBACK_YEARS = 6  # ~1500 trading days -- see docs/PAIRS_TRADING.md's "Why bound the lookback" note


def load_prices_and_volumes() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    con = get_connection(read_only=True)
    tickers = [t for t, _, _ in EQUITY_UNIVERSE]
    df = con.execute(
        "SELECT symbol, date, close, volume FROM prices WHERE symbol IN "
        f"({','.join('?' * len(tickers))}) "
        f"AND date >= (SELECT max(date) FROM prices) - INTERVAL '{LOOKBACK_YEARS} years' "
        "ORDER BY symbol, date",
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
    print("Loading real prices/volumes from the warehouse...")
    prices, volumes = load_prices_and_volumes()
    print(f"  {len(prices)} symbols loaded")

    print(f"\nScreening {len(prices) * (len(prices) - 1) // 2} pairs for cointegration "
          "(Engle-Granger, full history)...")
    results = screen_universe_for_pairs(prices)
    tradeable = [r for r in results if r.is_tradeable]
    print(f"  {len(tradeable)} pairs pass p<0.05 and 1<=half-life<=60 days "
          f"out of {len(results)} screened")

    if not tradeable:
        print("\nNo tradeable pairs found in this universe/period. Stopping honestly "
              "rather than lowering the bar to manufacture a result.")
        return

    print(f"\nBacktesting all {len(tradeable)} tradeable pairs "
          "(Kalman hedge ratio + z-score signals + real transaction costs)...")
    backtest_results = []
    for r in tradeable:
        try:
            bt = backtest_pair(
                prices[r.symbol_a], prices[r.symbol_b],
                volumes[r.symbol_a], volumes[r.symbol_b],
                r.symbol_a, r.symbol_b,
            )
        except ValueError as exc:
            print(f"  skipped {r.symbol_a}/{r.symbol_b}: {exc}")
            continue
        backtest_results.append((r, bt))
        print(f"  {r.symbol_a}/{r.symbol_b}: p={r.pvalue:.4f} half_life={r.half_life_days:.1f}d "
              f"n_trades={bt.n_trades} net_sharpe={bt.net_sharpe_annualized:.2f}")

    if not backtest_results:
        print("\nEvery tradeable pair failed the backtest's own minimum-history "
              "requirement. Stopping honestly.")
        return

    # rank by NET (cost-adjusted) Sharpe -- the honest number, not gross
    backtest_results.sort(key=lambda x: x[1].net_sharpe_annualized, reverse=True)
    best_coint, best_bt = backtest_results[0]

    trial_sharpes = np.array([bt.net_sharpe_annualized for _, bt in backtest_results])
    # per-period (daily) Sharpe for the DSR calculation -- must match the
    # frequency of best_bt.returns, not the annualized headline number.
    trial_sharpes_daily = trial_sharpes / np.sqrt(252)

    dsr = deflated_sharpe_ratio(best_bt.returns.to_numpy(), trial_sharpes_daily)

    print(f"\n{'=' * 70}")
    print(f"BEST PAIR: {best_bt.symbol_a}/{best_bt.symbol_b}")
    print(f"  cointegration p-value: {best_coint.pvalue:.4f}, half-life: {best_coint.half_life_days:.1f} days")
    print(f"  trades: {best_bt.n_trades}")
    print(f"  gross annualized Sharpe: {best_bt.gross_sharpe_annualized:.3f}")
    print(f"  net annualized Sharpe:   {best_bt.net_sharpe_annualized:.3f}")
    print(f"  cost drag (Sharpe pts):  {best_bt.cost_drag_annualized:.3f}")
    print(f"\n  DEFLATED for having tried {len(backtest_results)} pairs:")
    print(f"    expected max Sharpe under pure luck (per-period): {dsr.expected_max_sr_null:.4f}")
    print(f"    P(true Sharpe > luck-adjusted benchmark): {dsr.dsr:.3f}")
    if dsr.dsr < 0.95:
        print("    --> NOT distinguishable from having tried many pairs and gotten lucky "
              "at a 95% confidence level. Reported honestly, not hidden.")
    else:
        print("    --> survives the multiple-testing correction at 95% confidence.")

    # PBO across all backtested pairs -- needs equal-length aligned return
    # series; use each pair's common (intersected) date range.
    pbo_pbo, pbo_n_combinations = None, None
    if len(backtest_results) >= 2:
        common_idx = backtest_results[0][1].returns.index
        for _, bt in backtest_results[1:]:
            common_idx = common_idx.intersection(bt.returns.index)
        if len(common_idx) >= 20:
            returns_matrix = np.column_stack(
                [bt.returns.reindex(common_idx).fillna(0.0).to_numpy() for _, bt in backtest_results]
            )
            n_sub = 10 if len(common_idx) >= 500 else (4 if len(common_idx) >= 100 else 2)
            if n_sub >= 2 and returns_matrix.shape[1] >= 2:
                try:
                    pbo_result = probability_of_backtest_overfitting(returns_matrix, n_subsamples=n_sub)
                    pbo_pbo, pbo_n_combinations = pbo_result.pbo, pbo_result.n_combinations
                    print(f"\n  Probability of Backtest Overfitting across all {len(backtest_results)} "
                          f"pairs' return series: {pbo_result.pbo:.3f} "
                          f"({pbo_result.n_combinations} combinations)")
                except ValueError as exc:
                    print(f"\n  PBO skipped: {exc}")

    equity_curve = (1.0 + best_bt.returns.fillna(0.0)).cumprod() - 1.0

    ARTIFACTS.mkdir(exist_ok=True)
    report = {
        "n_pairs_screened": len(results),
        "n_tradeable": len(tradeable),
        "n_backtested": len(backtest_results),
        "best_pair": f"{best_bt.symbol_a}/{best_bt.symbol_b}",
        "best_pair_pvalue": best_coint.pvalue,
        "best_pair_half_life_days": best_coint.half_life_days,
        "best_pair_n_trades": best_bt.n_trades,
        "best_pair_gross_sharpe": best_bt.gross_sharpe_annualized,
        "best_pair_net_sharpe": best_bt.net_sharpe_annualized,
        "best_pair_cost_drag": best_bt.cost_drag_annualized,
        "deflated_sharpe_ratio": dsr.dsr,
        "expected_max_sharpe_under_luck": dsr.expected_max_sr_null,
        "pbo": pbo_pbo,
        "pbo_n_combinations": pbo_n_combinations,
        "all_pairs": [
            {
                "pair": f"{r.symbol_a}/{r.symbol_b}",
                "pvalue": r.pvalue,
                "half_life_days": r.half_life_days,
                "n_trades": bt.n_trades,
                "net_sharpe": bt.net_sharpe_annualized,
            }
            for r, bt in backtest_results
        ],
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "cumulative_return": float(v)}
            for d, v in equity_curve.items()
        ],
    }
    out_path = ARTIFACTS / "pairs_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Phase 8: apply VaR/CVaR + historical stress replay to every strategy's
real, already-computed return series (reconstructed from each artifact's
equity curve), plus one representative Almgren-Chriss execution example
sized off real ADV/volatility for a liquid name in the universe.

Usage:
    python scripts/run_risk_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.risk.execution import almgren_chriss_trajectory  # noqa: E402
from alpha_desk.risk.stress import stress_test  # noqa: E402
from alpha_desk.risk.var_cvar import historical_var_cvar  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

STRATEGY_ARTIFACTS = {
    "pairs": "pairs_strategy_report.json",
    "factors": "factor_strategy_report.json",
    "ml": "ml_strategy_report.json",
    "momentum": "momentum_strategy_report.json",
    "pead": "pead_strategy_report.json",
}


def equity_curve_to_returns(equity_curve: list[dict]) -> pd.Series:
    dates = pd.to_datetime([p["date"] for p in equity_curve])
    cum = pd.Series([p["cumulative_return"] for p in equity_curve], index=dates).sort_index()
    equity = 1.0 + cum
    returns = equity.pct_change()
    returns.iloc[0] = cum.iloc[0]  # first period's return, relative to a baseline of 0
    return returns


def load_strategy_returns() -> dict[str, pd.Series]:
    returns_by_strategy = {}
    for name, filename in STRATEGY_ARTIFACTS.items():
        path = ARTIFACTS / filename
        if not path.exists():
            print(f"  [skip] {name}: {filename} not found -- run its script first")
            continue
        report = json.loads(path.read_text())
        returns_by_strategy[name] = equity_curve_to_returns(report["equity_curve"])
    return returns_by_strategy


def main() -> None:
    print("Loading each strategy's real return series from its artifact...")
    returns_by_strategy = load_strategy_returns()

    report = {"var_cvar": {}, "stress": {}}

    print("\n" + "=" * 70)
    print("VAR / CVAR (historical, 95% and 99% confidence)")
    for name, returns in returns_by_strategy.items():
        entry = {}
        for confidence in (0.95, 0.99):
            try:
                result = historical_var_cvar(returns, confidence=confidence)
                entry[f"var_{int(confidence*100)}"] = result.var
                entry[f"cvar_{int(confidence*100)}"] = result.cvar
                print(f"  {name:10s} VaR{int(confidence*100)}={result.var:.4f}  CVaR{int(confidence*100)}={result.cvar:.4f}  (n={result.n_obs})")
            except ValueError as exc:
                print(f"  {name:10s} skipped at {confidence}: {exc}")
        report["var_cvar"][name] = entry

    print("\n" + "=" * 70)
    print("HISTORICAL STRESS REPLAY (each strategy's own real returns during real crises)")
    for name, returns in returns_by_strategy.items():
        results = stress_test(returns)
        report["stress"][name] = [
            {
                "window": r.window_name, "has_coverage": r.has_coverage,
                "cumulative_return": r.cumulative_return, "max_drawdown": r.max_drawdown,
                "n_obs": r.n_obs,
            }
            for r in results
        ]
        for r in results:
            if r.has_coverage:
                print(f"  {name:10s} {r.window_name:20s} cum_return={r.cumulative_return:+.3f}  "
                      f"max_drawdown={r.max_drawdown:.3f}  (n={r.n_obs})")
            else:
                print(f"  {name:10s} {r.window_name:20s} no coverage (history doesn't reach this window)")

    print("\n" + "=" * 70)
    print("ALMGREN-CHRISS OPTIMAL EXECUTION (representative example: liquidating AAPL)")
    con = get_connection(read_only=True)
    px = con.execute("SELECT close, volume FROM prices WHERE symbol='AAPL' ORDER BY date DESC LIMIT 60").df()
    con.close()
    daily_vol = float(np.log(px["close"] / px["close"].shift(-1)).std())  # reversed order, shift(-1) = prior day
    adv_dollars = float((px["close"] * px["volume"]).mean())
    last_price = float(px["close"].iloc[0])
    adv_shares = adv_dollars / last_price
    shares_to_sell = 500_000  # a large, illustrative institutional-size order
    print(f"  AAPL: last price ${last_price:.2f}, ADV ${adv_dollars/1e6:.1f}M, daily vol {daily_vol:.4f}")
    print(f"  Order: {shares_to_sell:,} shares (~{shares_to_sell*last_price/adv_dollars*100:.1f}% of ADV)")

    # Calibrated relative to ADV, not an arbitrary small multiple of price --
    # a first version used price-scaled constants (2.5e-6/2.5e-7 * price)
    # that were off by ~5 orders of magnitude (a 1%-of-ADV order came out
    # costing >3,000bps -- obviously wrong for a genuinely small order).
    # This calibration targets ~10bps temporary / ~5bps permanent impact
    # when trading at a rate equal to the full ADV, a standard order-of-
    # magnitude reference point for linear impact models.
    temporary_impact = 0.0010 * last_price / adv_shares
    permanent_impact = 0.0005 * last_price / adv_shares

    execution_results = {}
    for label, risk_aversion in [("risk_neutral", 0.0), ("moderate", 1e-9), ("risk_averse", 1e-7)]:
        traj = almgren_chriss_trajectory(
            shares=shares_to_sell, horizon=5, n_intervals=10,
            volatility=daily_vol * last_price,  # convert to $/share vol
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            risk_aversion=risk_aversion,
        )
        cost_bps = traj.expected_cost / (shares_to_sell * last_price) * 10_000
        cost_std_bps = np.sqrt(traj.cost_variance) / (shares_to_sell * last_price) * 10_000
        execution_results[label] = {
            "risk_aversion": risk_aversion, "expected_cost_bps": cost_bps, "cost_std_bps": cost_std_bps,
            "holdings_pct": (traj.holdings / shares_to_sell).tolist(),
        }
        print(f"  {label:12s} (lambda={risk_aversion:.0e}): expected cost {cost_bps:.3f}bps, "
              f"cost std {cost_std_bps:.1f}bps")

    report["execution_example"] = {
        "symbol": "AAPL", "last_price": last_price, "adv_dollars": adv_dollars,
        "daily_vol": daily_vol, "shares_to_sell": shares_to_sell, "scenarios": execution_results,
    }

    out_path = ARTIFACTS / "risk_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

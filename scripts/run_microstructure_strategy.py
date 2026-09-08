"""Phase 9: Avellaneda-Stoikov market-making simulation, calibrated to a
real symbol's real daily volatility (not an arbitrary sigma).

Order flow itself (arrival Poisson process, fundamental price path) is
SIMULATED -- see src/alpha_desk/strategies/microstructure/__init__.py and
docs/MICROSTRUCTURE_STRATEGY.md for why, and for the arrival_intensity*dt
validity threshold this script respects.

Usage:
    python scripts/run_microstructure_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.data.warehouse import get_connection  # noqa: E402
from alpha_desk.strategies.microstructure.simulation import (  # noqa: E402
    simulate_market_making_session,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

SYMBOL = "AAPL"
N_STEPS = 390  # one trading session as 1-minute bars
ARRIVAL_INTENSITY = 30.0  # A: base fill intensity at the mid. A*dt = 30/390 = 0.077, comfortably under the ~0.1 validity threshold documented in simulation.py
KAPPA = 1.0
INVENTORY_LIMIT = 50.0
N_SESSIONS = 300  # independent simulated sessions per scenario, for stable mean/std estimates

GAMMA_SCENARIOS = {
    "low_risk_aversion": 0.01,
    "moderate_risk_aversion": 0.05,
    "high_risk_aversion": 0.20,
}


def load_real_volatility(symbol: str) -> dict:
    con = get_connection(read_only=True)
    px = con.execute(
        "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 252", [symbol]
    ).df()
    con.close()
    if len(px) < 30:
        raise RuntimeError(f"not enough price history for {symbol} to calibrate volatility")
    # reversed order (DESC), so shift(-1) is the prior trading day
    daily_returns = px["close"].pct_change(-1).dropna()
    daily_vol = float(daily_returns.std())
    last_price = float(px["close"].iloc[0])
    return {"daily_vol": daily_vol, "last_price": last_price, "n_obs": len(px)}


def run_scenario(gamma: float, sigma_dollar: float, last_price: float) -> dict:
    fills, pnls, max_abs_inventory = [], [], []
    for seed in range(N_SESSIONS):
        result = simulate_market_making_session(
            initial_mid=last_price,
            horizon=1.0,
            n_steps=N_STEPS,
            sigma=sigma_dollar,
            gamma=gamma,
            kappa=KAPPA,
            arrival_intensity=ARRIVAL_INTENSITY,
            inventory_limit=INVENTORY_LIMIT,
            seed=seed,
        )
        fills.append(result.n_buy_fills + result.n_sell_fills)
        pnls.append(result.final_pnl)
        max_abs_inventory.append(float(np.max(np.abs(result.inventory))))

    pnls_arr = np.array(pnls)
    mean_pnl = float(pnls_arr.mean())
    std_pnl = float(pnls_arr.std())
    return {
        "gamma": gamma,
        "n_sessions": N_SESSIONS,
        "mean_fills_per_session": float(np.mean(fills)),
        "mean_pnl_per_session": mean_pnl,
        "std_pnl_per_session": std_pnl,
        "pnl_sharpe_like": mean_pnl / std_pnl if std_pnl > 0 else None,
        "pct_sessions_profitable": float(np.mean(pnls_arr > 0)),
        "mean_max_abs_inventory": float(np.mean(max_abs_inventory)),
        "worst_session_pnl": float(pnls_arr.min()),
        "best_session_pnl": float(pnls_arr.max()),
    }


def main() -> None:
    print(f"Loading real daily volatility for {SYMBOL} from the warehouse...")
    vol_stats = load_real_volatility(SYMBOL)
    daily_vol = vol_stats["daily_vol"]
    last_price = vol_stats["last_price"]
    sigma_dollar = daily_vol * last_price
    print(f"  {SYMBOL}: last price ${last_price:.2f}, real daily vol {daily_vol:.4%} "
          f"(n={vol_stats['n_obs']} sessions) -> sigma=${sigma_dollar:.3f}/day")

    a_dt = ARRIVAL_INTENSITY / N_STEPS
    print(f"\nCalibration check: arrival_intensity*dt = {ARRIVAL_INTENSITY}/{N_STEPS} = {a_dt:.4f} "
          f"({'OK, under 0.1 threshold' if a_dt < 0.1 else 'WARNING: at/above threshold, fills may saturate'})")

    print(f"\nRunning {N_SESSIONS} simulated sessions per scenario "
          f"(kappa={KAPPA}, arrival_intensity={ARRIVAL_INTENSITY}, inventory_limit={INVENTORY_LIMIT})...")
    print("=" * 78)

    scenarios = {}
    for name, gamma in GAMMA_SCENARIOS.items():
        stats = run_scenario(gamma, sigma_dollar, last_price)
        scenarios[name] = stats
        print(f"  {name:24s} (gamma={gamma:5.2f}): "
              f"mean_pnl=${stats['mean_pnl_per_session']:7.3f}  "
              f"std=${stats['std_pnl_per_session']:6.3f}  "
              f"fills={stats['mean_fills_per_session']:5.1f}  "
              f"profitable={stats['pct_sessions_profitable']:.0%}  "
              f"max|inv|={stats['mean_max_abs_inventory']:.1f}")

    print("=" * 78)
    print("\nNote: P&L is in dollars per session for a 1-share-per-fill market maker "
          "on simulated (not exchange-replayed) order flow -- see docs/MICROSTRUCTURE_STRATEGY.md "
          "for the full disclosure of what is real (volatility calibration) vs. simulated (order flow, fills).")

    report = {
        "symbol": SYMBOL,
        "calibration": {
            "daily_vol": daily_vol,
            "last_price": last_price,
            "sigma_dollar_per_day": sigma_dollar,
            "n_steps": N_STEPS,
            "arrival_intensity": ARRIVAL_INTENSITY,
            "arrival_intensity_x_dt": a_dt,
            "kappa": KAPPA,
            "inventory_limit": INVENTORY_LIMIT,
            "n_sessions_per_scenario": N_SESSIONS,
        },
        "scenarios": scenarios,
    }

    out_path = ARTIFACTS / "microstructure_strategy_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nFull report -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

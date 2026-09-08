"""Almgren-Chriss (2000/2001) optimal execution: the trajectory for
liquidating (or acquiring) a large position that minimizes a risk-adjusted
combination of expected market-impact cost and the variance of that cost
from price risk while the order is still resting.

Two knobs, `X0` shares to trade over horizon `T` split into `N` intervals:
more intervals means slower, gentler trading (lower impact cost) but more
time exposed to price risk (higher cost variance) -- `risk_aversion`
(lambda) is what the trader uses to pick a point on that trade-off, not a
free efficiency gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionTrajectory:
    times: np.ndarray  # length N+1, 0..T
    holdings: np.ndarray  # length N+1, shares remaining at each time (holdings[0]=X0, holdings[-1]=0)
    trades: np.ndarray  # length N, shares traded in each interval
    expected_cost: float  # dollars, permanent + temporary impact
    cost_variance: float  # (dollars)^2, from price risk on the remaining position
    kappa: float  # trading-intensity parameter (1/kappa is the trajectory's characteristic timescale)


def almgren_chriss_trajectory(
    shares: float,
    horizon: float,
    n_intervals: int,
    volatility: float,
    temporary_impact: float,
    permanent_impact: float,
    risk_aversion: float,
) -> ExecutionTrajectory:
    """
    shares: X0, total shares to liquidate (positive) over `horizon` (same
        time units as `volatility`, e.g. both in "days").
    volatility: sigma, per-unit-time price volatility (absolute, e.g.
        dollars/share per sqrt(day), not annualized).
    temporary_impact: eta, cost coefficient for TRADING RATE (impact that
        reverts after each trade).
    permanent_impact: gamma, cost coefficient for TOTAL SIZE TRADED (impact
        that persists in the price).
    risk_aversion: lambda, >= 0. At 0 (risk-neutral), the optimal strategy
        degenerates to uniform/linear trading (TWAP) -- verified directly
        in tests, not just asserted in this docstring.
    """
    if shares <= 0:
        raise ValueError("shares must be positive")
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")

    tau = horizon / n_intervals
    eta_tilde = temporary_impact - 0.5 * permanent_impact * tau
    if eta_tilde <= 0:
        raise ValueError(
            "temporary_impact too small relative to permanent_impact*tau/2 -- "
            "the discrete-time model requires eta_tilde > 0"
        )

    times = np.linspace(0, horizon, n_intervals + 1)

    if risk_aversion == 0:
        kappa = 0.0
        holdings = shares * (1 - times / horizon)
    else:
        kappa_bar_sq = risk_aversion * volatility**2 / eta_tilde
        cosh_arg = 0.5 * kappa_bar_sq * tau**2 + 1
        kappa = np.arccosh(cosh_arg) / tau
        holdings = shares * np.sinh(kappa * (horizon - times)) / np.sinh(kappa * horizon)

    trades = -np.diff(holdings)  # shares sold in each interval (positive for a liquidation)

    expected_cost = 0.5 * permanent_impact * shares**2 + eta_tilde * np.sum(trades**2) / tau
    # variance of implementation-shortfall cost from price risk on whatever
    # is still held during each interval (left-Riemann approx of
    # sigma^2 * integral_0^T holdings(t)^2 dt, the standard A-C result).
    cost_variance = float(volatility**2 * np.sum(holdings[:-1] ** 2) * tau)

    return ExecutionTrajectory(
        times=times,
        holdings=holdings,
        trades=trades,
        expected_cost=float(expected_cost),
        cost_variance=cost_variance,
        kappa=float(kappa),
    )

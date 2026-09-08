"""Discrete-time simulation of an Avellaneda-Stoikov market maker over one
trading session. The FUNDAMENTAL PRICE PATH and ORDER ARRIVALS are
SIMULATED (arithmetic Brownian motion for price; Poisson arrivals with
intensity A*exp(-kappa*distance) for fills, both standard in the market-
making literature) -- not replayed from real exchange ticks. See this
package's __init__.py and docs/MICROSTRUCTURE_STRATEGY.md for why, and for
how `sigma` is calibrated to this platform's own real daily volatility data
rather than picked arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market_maker import optimal_quotes


@dataclass
class MarketMakingSessionResult:
    times: np.ndarray
    mid_prices: np.ndarray
    inventory: np.ndarray  # length matches times
    cash: np.ndarray
    pnl: np.ndarray  # mark-to-market: cash + inventory * mid_price
    n_buy_fills: int  # maker bought (a sell market order hit the maker's bid)
    n_sell_fills: int  # maker sold (a buy market order hit the maker's ask)
    final_pnl: float
    pnl_std: float


def simulate_market_making_session(
    initial_mid: float,
    horizon: float,
    n_steps: int,
    sigma: float,
    gamma: float,
    kappa: float,
    arrival_intensity: float,
    inventory_limit: float | None = None,
    seed: int | None = None,
) -> MarketMakingSessionResult:
    """
    sigma: price volatility, same time units as `horizon` (e.g. if horizon
        is in "trading days" and sigma is daily-close-to-close vol, they
        match; the calling script converts this platform's real annualized/
        daily vol into whatever session-length units are being simulated).
    arrival_intensity: A, the base Poisson intensity when quoting AT the
        mid-price (distance=0) -- decays as exp(-kappa*distance) further out.
    inventory_limit: if set, the maker stops quoting on whichever side
        would push |inventory| past this limit (a simple, standard risk
        control -- not part of the closed-form AS solution itself).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    times = np.linspace(0, horizon, n_steps + 1)

    mid = np.empty(n_steps + 1)
    mid[0] = initial_mid
    inventory = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    n_buy_fills = 0
    n_sell_fills = 0

    for i in range(n_steps):
        time_remaining = horizon - times[i]
        bid, ask = optimal_quotes(mid[i], inventory[i], gamma, sigma, time_remaining, kappa)
        bid_distance = max(mid[i] - bid, 0.0)
        ask_distance = max(ask - mid[i], 0.0)

        quote_bid = inventory_limit is None or inventory[i] < inventory_limit
        quote_ask = inventory_limit is None or inventory[i] > -inventory_limit

        # Poisson-thinning approximation P(arrival in dt) ~ intensity*dt is
        # only valid when intensity*dt << 1 -- a real calibration bug here
        # during development used arrival_intensity=2000 with dt~0.0026,
        # giving intensity*dt~5.1, which the naive `< probability` check
        # silently treated as "fills every single step" (capping total
        # fills at exactly 2*n_steps, an unmistakable tell once compared
        # across supposedly-different intensities that all produced the
        # identical fill count). Clipped to [0, 1] defensively here so an
        # invalid calibration degrades to a visibly-saturated 100%-per-step
        # fill rate rather than a silently-wrong "probability" > 1; the
        # actual fix is choosing arrival_intensity so intensity*dt stays
        # comfortably under ~0.1 (see docs/MICROSTRUCTURE_STRATEGY.md).
        p_buy_fill = min(arrival_intensity * np.exp(-kappa * bid_distance) * dt, 1.0) if quote_bid else 0.0
        p_sell_fill = min(arrival_intensity * np.exp(-kappa * ask_distance) * dt, 1.0) if quote_ask else 0.0

        buy_fill = rng.random() < p_buy_fill
        sell_fill = rng.random() < p_sell_fill

        new_inventory = inventory[i]
        new_cash = cash[i]
        if buy_fill:
            new_inventory += 1
            new_cash -= bid
            n_buy_fills += 1
        if sell_fill:
            new_inventory -= 1
            new_cash += ask
            n_sell_fills += 1

        mid[i + 1] = mid[i] + sigma * np.sqrt(dt) * rng.standard_normal()
        inventory[i + 1] = new_inventory
        cash[i + 1] = new_cash

    pnl = cash + inventory * mid

    return MarketMakingSessionResult(
        times=times,
        mid_prices=mid,
        inventory=inventory,
        cash=cash,
        pnl=pnl,
        n_buy_fills=n_buy_fills,
        n_sell_fills=n_sell_fills,
        final_pnl=float(pnl[-1]),
        pnl_std=float(np.std(pnl)),
    )

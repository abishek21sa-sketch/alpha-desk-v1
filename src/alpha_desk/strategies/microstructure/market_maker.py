"""Avellaneda & Stoikov, "High-frequency trading in a limit order book"
(2008) -- the closed-form optimal quotes for a risk-averse market maker
facing a random-walk fundamental price and Poisson order arrivals whose
intensity decays with distance from the mid-price.

Two pieces: a RESERVATION price (the mid-price adjusted for the maker's
current inventory risk -- not the price the maker quotes, the price around
which it centers its quotes) and an optimal total SPREAD around it. Long
inventory pulls the reservation price below mid (the maker wants to sell
down its position); short inventory pushes it above mid.
"""

from __future__ import annotations

import numpy as np


def reservation_price(
    mid_price: float, inventory: float, gamma: float, sigma: float, time_remaining: float
) -> float:
    """mid_price: current fundamental/mid price.
    inventory: maker's current position (positive = long).
    gamma: risk aversion (> 0).
    sigma: price volatility (same time units as `time_remaining`).
    time_remaining: T - t, time left in the trading session.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if time_remaining < 0:
        raise ValueError("time_remaining must be >= 0")
    return mid_price - inventory * gamma * sigma**2 * time_remaining


def optimal_spread(gamma: float, sigma: float, time_remaining: float, kappa: float) -> float:
    """kappa: decay rate of order-arrival intensity with quote distance
    (higher kappa = arrivals fall off faster as the maker quotes further
    from mid -- a "thinner," less patient market).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if time_remaining < 0:
        raise ValueError("time_remaining must be >= 0")
    return gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / kappa)


def optimal_quotes(
    mid_price: float, inventory: float, gamma: float, sigma: float, time_remaining: float, kappa: float
) -> tuple[float, float]:
    """Returns (bid, ask), centered on the reservation price, split evenly
    by the optimal total spread."""
    r = reservation_price(mid_price, inventory, gamma, sigma, time_remaining)
    spread = optimal_spread(gamma, sigma, time_remaining, kappa)
    return r - spread / 2, r + spread / 2

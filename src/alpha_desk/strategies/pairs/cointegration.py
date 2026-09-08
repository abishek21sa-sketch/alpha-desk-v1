"""Engle-Granger pair screening + Ornstein-Uhlenbeck half-life estimation.

Engle & Granger (1987): two I(1) (unit-root, non-stationary) series are
"cointegrated" if some linear combination of them is stationary -- i.e.
there's a hedge ratio that turns two random-walk-like prices into a
mean-reverting spread. `statsmodels.tsa.stattools.coint` implements the
two-step test (Augmented Dickey-Fuller on the regression residual).

Screening on log prices, not raw prices: the cointegrating relationship
log(y) = alpha + beta*log(x) + stationary_residual corresponds to a
constant-elasticity price RATIO being mean-reverting, which is the
economically sensible notion of "these two names trade together" for
stocks at very different price levels -- a raw-price spread would instead
implicitly assume a 1-share-for-1-share hedge is meaningful, which it
isn't.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass
class CointegrationResult:
    symbol_a: str
    symbol_b: str
    pvalue: float
    hedge_ratio: float  # from regressing log(price_a) on log(price_b)
    intercept: float
    half_life_days: float
    n_obs: int

    @property
    def is_tradeable(self) -> bool:
        return self.pvalue < 0.05 and 1 <= self.half_life_days <= 60


def estimate_half_life(spread: pd.Series) -> float:
    """Half-life (in observations) of mean reversion for an assumed
    Ornstein-Uhlenbeck spread, via the discretized AR(1) regression
    delta_spread_t = c + beta * spread_{t-1} + eps_t. beta < 0 required for
    mean reversion; half_life = -ln(2) / beta. Returns +inf if the fitted
    beta is >= 0 (the spread shows no mean-reverting tendency at all).
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged = lagged.loc[delta.index]

    x = sm.add_constant(lagged.to_numpy())
    model = sm.OLS(delta.to_numpy(), x).fit()
    beta = model.params[1]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)


def test_pair_cointegration(
    price_a: pd.Series, price_b: pd.Series, symbol_a: str, symbol_b: str
) -> CointegrationResult:
    """Engle-Granger test on log(price_a) ~ log(price_b), on their common
    (inner-joined) date range. Raises if fewer than 60 overlapping
    observations -- not enough history to trust a cointegration test.
    """
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    if len(df) < 60:
        raise ValueError(
            f"only {len(df)} overlapping observations for {symbol_a}/{symbol_b}, need >= 60"
        )
    log_a = np.log(df["a"])
    log_b = np.log(df["b"])

    score, pvalue, _ = coint(log_a, log_b)

    x = sm.add_constant(log_b.to_numpy())
    ols = sm.OLS(log_a.to_numpy(), x).fit()
    intercept, hedge_ratio = ols.params
    spread = log_a - (intercept + hedge_ratio * log_b)

    return CointegrationResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        pvalue=float(pvalue),
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        half_life_days=estimate_half_life(spread),
        n_obs=len(df),
    )


def screen_universe_for_pairs(
    prices: dict[str, pd.Series], significance: float = 0.05
) -> list[CointegrationResult]:
    """All pairwise combinations of `prices` (symbol -> price Series indexed
    by date), Engle-Granger tested, returned sorted by p-value ascending.
    Does NOT filter to `is_tradeable` -- callers decide what to act on;
    this is a screening report, not a trading decision.
    """
    results: list[CointegrationResult] = []
    symbols = sorted(prices)
    for sym_a, sym_b in combinations(symbols, 2):
        try:
            result = test_pair_cointegration(prices[sym_a], prices[sym_b], sym_a, sym_b)
        except ValueError:
            continue
        results.append(result)
    return sorted(results, key=lambda r: r.pvalue)

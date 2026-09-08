"""Sharpe ratio, Probabilistic Sharpe Ratio (PSR), and Deflated Sharpe Ratio
(DSR) -- Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier"
(2012) and "The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting and Non-Normality" (2014).

A raw backtest Sharpe ratio answers "how good does this look." PSR answers
"how confident should I be that the TRUE Sharpe exceeds some benchmark,
given this many observations and this return distribution's skew/kurtosis."
DSR answers the harder, more honest question: "...given that this is the
best of N strategies I tried" -- the correction that matters most for
anyone who has ever run a parameter sweep and reported only the winner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int | None = None) -> float:
    """Per-period Sharpe ratio, or annualized if periods_per_year is given."""
    returns = np.asarray(returns, dtype=float)
    sr = returns.mean() / returns.std(ddof=1)
    if periods_per_year is not None:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sr: float = 0.0
) -> float:
    """PSR(SR*): P(true per-period Sharpe > benchmark_sr), given the
    observed sample's Sharpe, skewness, (non-excess) kurtosis, and length.

    Uses PER-PERIOD returns and a PER-PERIOD benchmark_sr -- annualizing
    either before calling this changes the answer, since T (sample length)
    must match the frequency of the Sharpe ratio being tested.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        raise ValueError("need at least 2 observations")

    sr_hat = sharpe_ratio(returns)
    skew = stats.skew(returns, bias=False)
    # scipy's kurtosis(..., fisher=True) is EXCESS kurtosis (normal = 0);
    # the PSR formula wants regular kurtosis (normal = 3).
    kurt = stats.kurtosis(returns, fisher=False, bias=False)

    denom = np.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2, 1e-12))
    z = (sr_hat - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


@dataclass
class DeflatedSharpeResult:
    dsr: float
    sr_hat: float
    expected_max_sr_null: float
    n_trials: int
    n_obs: int


def deflated_sharpe_ratio(
    returns: np.ndarray, trial_sharpe_ratios: np.ndarray
) -> DeflatedSharpeResult:
    """DSR: PSR of the winning strategy's returns against the benchmark of
    "what Sharpe ratio would the BEST of N trials reach by pure luck,"
    instead of against a fixed 0.

    `returns` are the winning strategy's own per-period returns (its skew/
    kurtosis/length feed the PSR formula, same as probabilistic_sharpe_ratio).
    `trial_sharpe_ratios` are the per-period Sharpe ratios of ALL N trials
    that were actually run (the winner's own Sharpe should be included --
    its variance across trials is what estimates how much "luck budget" N
    independent attempts have). N=len(trial_sharpe_ratios) must be >= 2.
    """
    trial_sharpe_ratios = np.asarray(trial_sharpe_ratios, dtype=float)
    n_trials = len(trial_sharpe_ratios)
    if n_trials < 2:
        raise ValueError("need at least 2 trial Sharpe ratios to estimate a null")

    sr_variance = trial_sharpe_ratios.var(ddof=1)
    if sr_variance <= 0:
        # every trial produced an identical Sharpe -- no cross-trial variance
        # to estimate a luck budget from; the honest answer is "no correction
        # possible", so fall back to an un-deflated PSR(0) rather than
        # dividing by zero.
        expected_max_sr_null = 0.0
    else:
        sr_std = np.sqrt(sr_variance)
        expected_max_sr_null = sr_std * (
            (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

    dsr = probabilistic_sharpe_ratio(returns, benchmark_sr=expected_max_sr_null)
    return DeflatedSharpeResult(
        dsr=dsr,
        sr_hat=sharpe_ratio(returns),
        expected_max_sr_null=float(expected_max_sr_null),
        n_trials=n_trials,
        n_obs=len(returns),
    )

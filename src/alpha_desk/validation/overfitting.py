"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV) -- Bailey, Borwein, Lopez de Prado, Zhu, "The
Probability of Backtest Overfitting" (2017).

Answers a different question than DSR: not "is this Sharpe ratio
statistically significant," but "if I pick the best-in-sample strategy out
of everything I tried, how often does it turn out to be BELOW MEDIAN
out-of-sample" -- i.e. how much am I actually just selecting noise. PBO near
0.5 means in-sample rank carries ~no out-of-sample information (pure
overfitting); PBO near 0 means the in-sample winner reliably stays a
winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class PBOResult:
    pbo: float
    n_combinations: int
    logits: np.ndarray


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_subsamples: int = 10
) -> PBOResult:
    """returns_matrix: shape (T, N) -- T chronological periods (rows), N
    competing strategy configurations (columns), per-period returns.

    n_subsamples (S, must be even): the timeline is split into S equal
    contiguous blocks. Every way of choosing S/2 blocks as the "IS" set
    (C(S, S/2) combinations) is tried; the complementary S/2 blocks are
    "OOS". For each combination: whichever strategy has the best IS Sharpe
    ratio is noted, then its OOS relative rank is converted to a logit
    lambda = ln(rank / (1 - rank)). PBO = fraction of combinations where
    lambda <= 0 (the IS winner finished at or below the OOS median).
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    t, n_strategies = returns_matrix.shape
    if n_subsamples % 2 != 0:
        raise ValueError("n_subsamples must be even (CSCV splits IS/OOS symmetrically)")
    if n_strategies < 2:
        raise ValueError("need at least 2 strategy configurations to compare")
    if t < n_subsamples:
        raise ValueError(f"need at least {n_subsamples} time periods, got {t}")

    block_bounds = np.array_split(np.arange(t), n_subsamples)
    half = n_subsamples // 2

    logits: list[float] = []
    for is_blocks in combinations(range(n_subsamples), half):
        is_blocks_set = set(is_blocks)
        oos_blocks = [b for b in range(n_subsamples) if b not in is_blocks_set]

        is_idx = np.concatenate([block_bounds[b] for b in is_blocks])
        oos_idx = np.concatenate([block_bounds[b] for b in oos_blocks])

        is_returns = returns_matrix[is_idx]
        oos_returns = returns_matrix[oos_idx]

        is_sharpe = is_returns.mean(axis=0) / is_returns.std(axis=0, ddof=1)
        oos_sharpe = oos_returns.mean(axis=0) / oos_returns.std(axis=0, ddof=1)

        best_is_strategy = int(np.nanargmax(is_sharpe))

        # relative rank of the IS-winner's OOS Sharpe among all N strategies,
        # in (0, 1); 1.0 = best OOS performer, near 0 = worst.
        rank = stats_rank(oos_sharpe, best_is_strategy)
        rank = min(max(rank, 1e-6), 1 - 1e-6)  # avoid +/- inf logits at the extremes
        logits.append(float(np.log(rank / (1 - rank))))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr <= 0))
    return PBOResult(pbo=pbo, n_combinations=len(logits_arr), logits=logits_arr)


def stats_rank(values: np.ndarray, index: int) -> float:
    """Relative rank in (0, 1) of values[index] within values (1.0 = max)."""
    order = np.argsort(np.argsort(values))  # 0 = smallest
    return (order[index] + 1) / len(values)

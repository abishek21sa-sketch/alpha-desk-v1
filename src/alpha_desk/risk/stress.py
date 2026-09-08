"""Historical stress testing by REPLAY, not by hypothetical shock model:
slices a strategy's own real, already-realized return series against named
historical crisis date windows and reports what actually happened, plus
the max drawdown within that window. Deliberately not a generic "-30%
equity shock" assumption applied uniformly to every strategy -- each
strategy's own realized behavior in a real crisis (or the honest absence
of coverage, if its history doesn't reach back that far) is more
informative and harder to game than a made-up shock size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Named historical windows. 2008 GFC is deliberately NOT included: this
# platform's equity universe has no common price history before 2013 (see
# docs/DATA_BACKBONE.md), so any "2008 stress test" here would be
# unreplayable for most strategies -- omitted rather than faked with a
# shorter or unrelated window standing in for it.
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-01", "2022-10-31"),
}


@dataclass
class StressWindowResult:
    window_name: str
    start: str
    end: str
    has_coverage: bool
    cumulative_return: float | None
    max_drawdown: float | None
    n_obs: int


def _max_drawdown(cum_returns: pd.Series) -> float:
    """Max peak-to-trough drawdown of a (1+cumulative_return)-style equity
    curve built from `cum_returns` (a raw cumulative-return series, not yet
    rebased to 1.0)."""
    equity = 1.0 + cum_returns
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def stress_test(returns: pd.Series, windows: dict[str, tuple[str, str]] | None = None) -> list[StressWindowResult]:
    """`returns` must be a date-indexed per-period return series (not
    cumulative). For each named window, slices the series to that date
    range; if there's no overlap at all, reports `has_coverage=False`
    rather than a fabricated zero.
    """
    windows = windows if windows is not None else STRESS_WINDOWS
    results = []
    for name, (start, end) in windows.items():
        window_returns = returns.loc[(returns.index >= start) & (returns.index <= end)]
        if len(window_returns) < 2:
            results.append(
                StressWindowResult(
                    window_name=name, start=start, end=end, has_coverage=False,
                    cumulative_return=None, max_drawdown=None, n_obs=len(window_returns),
                )
            )
            continue

        cum = (1.0 + window_returns).cumprod() - 1.0
        results.append(
            StressWindowResult(
                window_name=name,
                start=start,
                end=end,
                has_coverage=True,
                cumulative_return=float(cum.iloc[-1]),
                max_drawdown=_max_drawdown(cum),
                n_obs=len(window_returns),
            )
        )
    return results

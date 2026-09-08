"""Per-event announcement return + subsequent drift, and the actual
statistical test of the PEAD effect: does a stock's reaction on the
announcement day predict continued drift in the SAME direction over the
following weeks (the anomaly), or is any observed pattern indistinguishable
from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def compute_event_outcomes(
    price: pd.Series, event_dates: list[str] | list[pd.Timestamp], drift_days: int = 21
) -> pd.DataFrame:
    """For each event date, snapped forward to the next available trading
    day (filings often land after-hours or on non-trading days, so the
    market's first chance to react is the next session):
      - announcement_return: return from the trading day BEFORE the
        snapped event day to the day AFTER it (a 2-day window, to catch an
        after-hours reaction landing on either side)
      - drift_return: cumulative return from 2 trading days after the
        event to `drift_days` after it -- the actual post-announcement
        window, deliberately excluding the announcement window itself so
        drift is measured on WHAT HAPPENS NEXT, not the reaction already
        captured in announcement_return

    Events too close to the start or end of `price`'s history to fit both
    windows are silently dropped (nothing to compute there), returned in a
    DataFrame indexed by the snapped event date.
    """
    idx = price.index
    log_price = np.log(price)
    rows = []
    for raw_date in event_dates:
        event_ts = pd.Timestamp(raw_date)
        pos = idx.searchsorted(event_ts, side="left")
        if pos <= 0 or pos + 1 + drift_days >= len(idx):
            continue
        announcement_return = float(np.exp(log_price.iloc[pos + 1] - log_price.iloc[pos - 1]) - 1)
        drift_return = float(
            np.exp(log_price.iloc[pos + 1 + drift_days] - log_price.iloc[pos + 1]) - 1
        )
        rows.append(
            {
                "event_date": idx[pos],
                "entry_date": idx[pos + 1],
                "exit_date": idx[pos + 1 + drift_days],
                "announcement_return": announcement_return,
                "drift_return": drift_return,
            }
        )
    return pd.DataFrame(rows).set_index("event_date") if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "announcement_return", "drift_return"]
    )


@dataclass
class PeadSignificanceResult:
    n_positive_events: int
    n_negative_events: int
    mean_drift_positive: float
    mean_drift_negative: float
    drift_spread: float  # mean_drift_positive - mean_drift_negative -- the PEAD effect size
    t_stat: float
    p_value: float


def pead_significance(outcomes: pd.DataFrame) -> PeadSignificanceResult:
    """Welch's t-test (unequal variance assumed -- no reason the two
    buckets should share variance) on drift_return, positive- vs negative-
    announcement-return events. This is the actual PEAD hypothesis test:
    H0 is that subsequent drift is unrelated to the announcement's
    direction; a significant positive spread is the anomaly.
    """
    positive = outcomes[outcomes["announcement_return"] > 0]
    negative = outcomes[outcomes["announcement_return"] < 0]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError(
            f"need >= 2 events in each bucket, got {len(positive)} positive / {len(negative)} negative"
        )

    t_stat, p_value = stats.ttest_ind(
        positive["drift_return"], negative["drift_return"], equal_var=False
    )
    return PeadSignificanceResult(
        n_positive_events=len(positive),
        n_negative_events=len(negative),
        mean_drift_positive=float(positive["drift_return"].mean()),
        mean_drift_negative=float(negative["drift_return"].mean()),
        drift_spread=float(positive["drift_return"].mean() - negative["drift_return"].mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
    )

"""Z-score entry/exit state machine for a mean-reverting spread.

Standard "Bollinger-band"-style pairs-trading rule: enter when the spread
is stretched far from its model-implied value (|z| > entry_threshold), exit
once it has reverted most of the way back (|z| < exit_threshold). Positions
are STICKY -- once entered, a position is held until the exit condition
fires, not re-evaluated fresh every bar -- which is why this is a stateful
loop, not a vectorized `np.where`.

Position convention: +1 means the spread is BELOW its model value and
expected to rise (long y, short beta*x); -1 means the spread is ABOVE its
model value and expected to fall (short y, long beta*x); 0 is flat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_positions(
    z_score: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """Positions are decided using z_score up to and including bar t, and
    are meant to be applied to the spread's return realized FROM t to t+1
    (i.e. the caller must shift this series forward by one bar before
    multiplying by returns -- this function only encodes the signal, not
    the lag; see backtest.py for where that shift happens and
    test_signals.py::test_documented_lag_contract for why it's tested at
    the boundary rather than assumed).
    """
    if exit_threshold >= entry_threshold:
        raise ValueError("exit_threshold must be < entry_threshold")
    if entry_threshold <= 0 or exit_threshold < 0:
        raise ValueError("thresholds must be >= 0 (exit) / > 0 (entry)")

    z = z_score.to_numpy()
    n = len(z)
    positions = np.zeros(n)
    current = 0.0

    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            positions[t] = current
            continue

        if current == 0.0:
            if zt <= -entry_threshold:
                current = 1.0
            elif zt >= entry_threshold:
                current = -1.0
        elif current == 1.0:
            if zt >= -exit_threshold:
                current = 0.0
        elif current == -1.0:
            if zt <= exit_threshold:
                current = 0.0

        positions[t] = current

    return pd.Series(positions, index=z_score.index, name="position")

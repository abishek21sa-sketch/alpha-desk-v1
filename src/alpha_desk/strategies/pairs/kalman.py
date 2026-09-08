"""Kalman-filter dynamic hedge ratio for a pairs-trading spread.

A static OLS hedge ratio (cointegration.py's `hedge_ratio`) assumes the
relationship between two names is constant for the whole sample. Real
relationships drift (a merger, a index-weight change, a regime shift). This
implements the standard "recursive least squares via Kalman filter"
formulation used in pairs-trading practice (e.g. Ernest Chan, *Algorithmic
Trading* (2013), ch. 2): state = [alpha_t, beta_t], a random walk, observed
through y_t = alpha_t + beta_t * x_t + noise.

The filter's innovation (one-step-ahead prediction error) and its variance
fall out of the recursion for free and are exactly what a z-score signal
needs -- no separate rolling-window z-score calculation required, and
critically, the innovation at time t uses only information through t-1's
state estimate, so it is causal by construction (see `test_kalman.py`'s
causality check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanFilterResult:
    alpha: pd.Series
    beta: pd.Series
    innovation: pd.Series  # e_t = y_t - (alpha_{t|t-1} + beta_{t|t-1} * x_t)
    innovation_std: pd.Series  # sqrt(S_t), for turning innovation into a z-score
    z_score: pd.Series  # innovation / innovation_std


def kalman_hedge_ratio(
    x: pd.Series,
    y: pd.Series,
    delta: float = 1e-6,
    observation_noise_var: float | None = None,
    calibration_window: int = 60,
) -> KalmanFilterResult:
    """Runs the 2-state [alpha, beta] Kalman filter over the common index of
    x and y (inner-joined, chronologically sorted).

    delta: controls process noise Q = delta/(1-delta) * I -- how fast
        alpha/beta are allowed to drift. Smaller delta = smoother, slower-
        adapting hedge ratio; this is Chan's parametrization, chosen for
        interpretability (delta in (0, 1)) over specifying Q directly.

        THIS MATTERS MORE THAN IT LOOKS: if delta is too large relative to
        the spread's own mean-reversion half-life, beta becomes adaptive
        enough to "explain away" the mean-reverting wobble as a temporary
        hedge-ratio shift instead of leaving it as innovation -- which
        destroys the very z-score signal this filter exists to produce.
        Confirmed empirically while building this module: on a synthetic
        pair with a true ~14-day mean-reversion half-life, delta=1e-4
        (a seemingly reasonable "slow" value) let beta drift enough that
        |z| never exceeded ~1.1 across 1500 observations -- no trade ever
        fired. Dropping to delta=1e-6 (this default) recovered a normal
        trading frequency (~3.5% of days with |z|>2). See
        docs/PAIRS_TRADING.md for the full writeup and the delta sweep that
        found this. There is no universal correct delta -- it must be small
        relative to how fast the PAIR's specific spread actually mean-
        reverts, which cointegration.py's half_life_days estimates.
    observation_noise_var: R, the assumed noise variance of y_t given the
        current state. If not given, defaults to the variance of y's first
        difference over ONLY the first `calibration_window` observations --
        deliberately NOT the whole series. Estimating R from the full
        sample (including its tail) would make every state estimate
        secretly depend on future data relative to the point being
        filtered, which defeats the point of a causal filter; see
        test_kalman.py::test_is_causal_truncated_history_gives_identical_early_output,
        which originally failed for exactly this reason before this fix.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")

    df = pd.DataFrame({"x": x, "y": y}).dropna().sort_index()
    if len(df) < 10:
        raise ValueError(f"need at least 10 overlapping observations, got {len(df)}")

    n = len(df)
    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()

    if observation_noise_var is None:
        window = min(calibration_window, n)
        observation_noise_var = float(np.var(np.diff(y_vals[:window]), ddof=1))
        if observation_noise_var <= 0:
            observation_noise_var = 1e-6
    r = observation_noise_var
    q = delta / (1 - delta) * np.eye(2)

    theta = np.zeros(2)  # [alpha, beta], initialized at 0
    p = np.eye(2) * 1.0  # initial state covariance -- diffuse prior

    alphas = np.empty(n)
    betas = np.empty(n)
    innovations = np.empty(n)
    innovation_stds = np.empty(n)

    for t in range(n):
        # predict (random-walk transition: F = I)
        p_pred = p + q

        h = np.array([1.0, x_vals[t]])
        y_pred = h @ theta  # theta here is still theta_{t-1}, i.e. theta_{t|t-1}

        s = h @ p_pred @ h.T + r  # innovation variance (scalar)
        e = y_vals[t] - y_pred  # innovation

        k = p_pred @ h / s  # Kalman gain (2,)
        theta = theta + k * e
        p = p_pred - np.outer(k, h) @ p_pred

        alphas[t] = theta[0]
        betas[t] = theta[1]
        innovations[t] = e
        innovation_stds[t] = np.sqrt(s)

    idx = df.index
    innovation_std_series = pd.Series(innovation_stds, index=idx)
    innovation_series = pd.Series(innovations, index=idx)
    return KalmanFilterResult(
        alpha=pd.Series(alphas, index=idx),
        beta=pd.Series(betas, index=idx),
        innovation=innovation_series,
        innovation_std=innovation_std_series,
        z_score=innovation_series / innovation_std_series,
    )

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.momentum.signals import realized_vol, tsmom_position, trailing_return


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


class TestTrailingReturn:
    def test_matches_hand_calculation(self):
        n = 400
        price = pd.Series(1.0 * (1.001 ** np.arange(n)), index=_dates(n))
        tr = trailing_return(price, lookback_days=252)
        t = 300
        expected = price.iloc[t] / price.iloc[t - 252] - 1.0
        assert tr.iloc[t] == pytest.approx(expected)


class TestRealizedVol:
    def test_recovers_known_annualized_volatility(self):
        rng = np.random.default_rng(0)
        n = 1000
        true_daily_sigma = 0.015
        price = pd.Series(np.exp(np.cumsum(rng.normal(0, true_daily_sigma, n))), index=_dates(n))
        vol = realized_vol(price, window=250)
        expected_annual = true_daily_sigma * np.sqrt(252)
        assert vol.iloc[-1] == pytest.approx(expected_annual, rel=0.1)


class TestTsmomPosition:
    def test_positive_trend_gives_positive_position(self):
        n = 400
        price = pd.Series(1.0 * (1.002 ** np.arange(n)), index=_dates(n))
        pos = tsmom_position(price, lookback_days=252, vol_window=60, target_vol=0.10)
        assert pos.iloc[-1] > 0

    def test_negative_trend_gives_negative_position(self):
        n = 400
        price = pd.Series(1.0 * (0.998 ** np.arange(n)), index=_dates(n))
        pos = tsmom_position(price, lookback_days=252, vol_window=60, target_vol=0.10)
        assert pos.iloc[-1] < 0

    def test_higher_realized_vol_gives_smaller_position_magnitude(self):
        rng = np.random.default_rng(1)
        n = 400
        trend = 0.0005 + rng.normal(0, 0.005, n)  # mild positive drift, low noise
        calm_price = pd.Series(np.exp(np.cumsum(trend)), index=_dates(n))
        wild_price = pd.Series(np.exp(np.cumsum(trend + rng.normal(0, 0.03, n))), index=_dates(n))

        calm_pos = tsmom_position(calm_price, target_vol=0.10, max_leverage=100.0)
        wild_pos = tsmom_position(wild_price, target_vol=0.10, max_leverage=100.0)
        assert abs(calm_pos.iloc[-1]) > abs(wild_pos.iloc[-1])

    def test_leverage_cap_clips_extreme_positions(self):
        n = 400
        # near-flat, ultra-low-vol price -> vol-targeted size would be huge without a cap
        price = pd.Series(100 + np.arange(n) * 0.0001, index=_dates(n))
        pos = tsmom_position(price, target_vol=0.10, max_leverage=3.0)
        assert pos.dropna().abs().max() <= 3.0 + 1e-9

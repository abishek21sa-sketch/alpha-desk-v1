from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.risk.stress import stress_test


class TestStressTest:
    def test_no_coverage_when_series_predates_window(self):
        idx = pd.bdate_range("2005-01-01", periods=100)
        returns = pd.Series(np.zeros(100), index=idx)
        results = stress_test(returns, windows={"covid_crash_2020": ("2020-02-19", "2020-03-23")})
        assert len(results) == 1
        assert results[0].has_coverage is False
        assert results[0].cumulative_return is None

    def test_cumulative_return_matches_hand_calculation(self):
        idx = pd.bdate_range("2020-02-19", periods=10)
        daily_returns = [-0.05, -0.03, 0.01, -0.08, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00]
        returns = pd.Series(daily_returns, index=idx)
        results = stress_test(returns, windows={"covid_crash_2020": ("2020-02-19", "2020-03-23")})
        result = results[0]
        assert result.has_coverage is True

        expected_cum = np.prod([1 + r for r in daily_returns]) - 1
        assert result.cumulative_return == pytest.approx(expected_cum)

    def test_max_drawdown_matches_hand_calculation(self):
        # goes up 10%, then down in two steps totaling a known peak-to-trough drop
        idx = pd.bdate_range("2020-02-19", periods=4)
        returns = pd.Series([0.10, -0.20, -0.10, 0.50], index=idx)
        results = stress_test(returns, windows={"covid_crash_2020": ("2020-02-19", "2020-03-23")})
        result = results[0]
        # equity path (starting from 1.0 pre-window): 1.0 -> 1.10 -> 0.88 -> 0.792 -> 1.188
        # peak is 1.10, trough is 0.792 -> drawdown = 0.792/1.10 - 1
        expected_dd = 0.792 / 1.10 - 1
        assert result.max_drawdown == pytest.approx(expected_dd, rel=1e-6)

    def test_multiple_windows_reported_independently(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 400), index=idx)
        results = stress_test(
            returns,
            windows={
                "covid_crash_2020": ("2020-02-19", "2020-03-23"),
                "rate_shock_2022": ("2022-01-01", "2022-10-31"),
            },
        )
        names = {r.window_name for r in results}
        assert names == {"covid_crash_2020", "rate_shock_2022"}
        covid = next(r for r in results if r.window_name == "covid_crash_2020")
        rate_shock = next(r for r in results if r.window_name == "rate_shock_2022")
        assert covid.has_coverage is True
        assert rate_shock.has_coverage is False  # series ends in early 2021, no 2022 data

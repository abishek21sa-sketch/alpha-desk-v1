from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.pead.events import compute_event_outcomes, pead_significance


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


class TestComputeEventOutcomes:
    def test_matches_hand_calculation(self):
        n = 100
        # deterministic, distinct daily returns so windows are unambiguous
        price = pd.Series(100 * (1.001 ** np.arange(n)), index=_dates(n))
        event_date = price.index[50]
        outcomes = compute_event_outcomes(price, [event_date], drift_days=10)
        assert len(outcomes) == 1
        row = outcomes.iloc[0]

        expected_announcement = price.iloc[51] / price.iloc[49] - 1
        expected_drift = price.iloc[61] / price.iloc[51] - 1
        assert row["announcement_return"] == pytest.approx(expected_announcement)
        assert row["drift_return"] == pytest.approx(expected_drift)

    def test_snaps_weekend_event_to_next_trading_day(self):
        n = 100
        price = pd.Series(100 * (1.001 ** np.arange(n)), index=_dates(n))
        # find an actual trading day in the middle of the series and the
        # Monday that immediately follows it, rather than assuming index
        # arithmetic lands on a particular weekday.
        anchor = price.index[50]
        next_monday = anchor + pd.Timedelta((7 - anchor.weekday()) % 7 or 7, unit="D")
        while next_monday.weekday() != 0:
            next_monday += pd.Timedelta(days=1)
        saturday_before = next_monday - pd.Timedelta(days=2)

        outcomes_direct = compute_event_outcomes(price, [next_monday], drift_days=10)
        outcomes_weekend = compute_event_outcomes(price, [saturday_before], drift_days=10)
        # both should snap to the same next trading day (that Monday)
        assert list(outcomes_direct.index) == list(outcomes_weekend.index)

    def test_events_too_close_to_edges_are_dropped(self):
        n = 50
        price = pd.Series(100 * (1.001 ** np.arange(n)), index=_dates(n))
        near_start = price.index[0]
        near_end = price.index[-1]
        valid = price.index[25]
        outcomes = compute_event_outcomes(price, [near_start, near_end, valid], drift_days=10)
        assert len(outcomes) == 1
        assert outcomes.index[0] == valid

    def test_empty_event_list_gives_empty_frame_with_expected_columns(self):
        price = pd.Series(100 * (1.001 ** np.arange(50)), index=_dates(50))
        outcomes = compute_event_outcomes(price, [], drift_days=10)
        assert len(outcomes) == 0
        assert set(outcomes.columns) == {"entry_date", "exit_date", "announcement_return", "drift_return"}


class TestPeadSignificance:
    def test_detects_genuine_pead_effect(self):
        rng = np.random.default_rng(0)
        n = 200
        announcement = rng.normal(0, 0.03, n)
        # genuine PEAD: drift continues in the SAME direction as the announcement
        drift = 0.5 * announcement + rng.normal(0, 0.02, n)
        outcomes = pd.DataFrame({"announcement_return": announcement, "drift_return": drift})

        result = pead_significance(outcomes)
        assert result.drift_spread > 0
        assert result.p_value < 0.01

    def test_no_relationship_gives_no_significant_spread(self):
        rng = np.random.default_rng(1)
        n = 200
        announcement = rng.normal(0, 0.03, n)
        drift = rng.normal(0, 0.02, n)  # independent of announcement direction
        outcomes = pd.DataFrame({"announcement_return": announcement, "drift_return": drift})

        result = pead_significance(outcomes)
        assert result.p_value > 0.05

    def test_raises_on_too_few_events_in_a_bucket(self):
        outcomes = pd.DataFrame({"announcement_return": [0.01, 0.02, -0.01], "drift_return": [0.01, 0.02, 0.01]})
        with pytest.raises(ValueError):
            pead_significance(outcomes)

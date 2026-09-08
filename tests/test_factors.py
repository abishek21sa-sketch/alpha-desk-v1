from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.factors.factors import (
    low_volatility,
    momentum_12_1,
    point_in_time_fundamental,
    quality_roe,
    value_earnings_yield,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


class TestMomentum:
    def test_matches_hand_calculated_value(self):
        n = 300
        idx = _dates(n)
        # deterministic price path: 1.0 at t=0, growing 0.1%/day
        price = pd.Series(1.0 * (1.001 ** np.arange(n)), index=idx)
        mom = momentum_12_1(price, lookback=252, skip=21)
        t = 260  # safely past lookback
        expected = price.iloc[t - 21] / price.iloc[t - 252] - 1.0
        assert mom.iloc[t] == pytest.approx(expected)

    def test_excludes_most_recent_month_from_the_signal(self):
        # a sharp move in the last `skip` days must NOT change momentum at
        # all -- momentum(t) only depends on price[t-skip] and
        # price[t-lookback], never on price[t-skip+1 .. t].
        n = 300
        idx = _dates(n)
        rng = np.random.default_rng(0)
        base_returns = rng.normal(0.0005, 0.01, n)
        price_a = pd.Series(np.cumprod(1 + base_returns), index=idx)

        price_b = price_a.copy()
        # violently alter only the most recent 10 days (< skip=21)
        price_b.iloc[-10:] = price_b.iloc[-10:] * np.linspace(1.0, 2.5, 10)

        mom_a = momentum_12_1(price_a, lookback=252, skip=21)
        mom_b = momentum_12_1(price_b, lookback=252, skip=21)
        pd.testing.assert_series_equal(mom_a.iloc[:-10], mom_b.iloc[:-10])


class TestLowVolatility:
    def test_recovers_known_volatility_with_correct_sign(self):
        rng = np.random.default_rng(1)
        n = 1000
        true_sigma = 0.02
        log_returns = rng.normal(0, true_sigma, n)
        price = pd.Series(np.exp(np.cumsum(log_returns)), index=_dates(n))

        low_vol = low_volatility(price, window=250)
        # should be NEGATIVE (low-vol convention) and close in magnitude to true_sigma
        assert low_vol.iloc[-1] < 0
        assert abs(low_vol.iloc[-1]) == pytest.approx(true_sigma, rel=0.15)

    def test_higher_realized_vol_gives_lower_score(self):
        rng = np.random.default_rng(2)
        n = 500
        calm = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=_dates(n))
        wild = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.05, n))), index=_dates(n))
        assert low_volatility(calm, window=200).iloc[-1] > low_volatility(wild, window=200).iloc[-1]


class TestPointInTimeFundamental:
    def _fundamentals(self) -> pd.DataFrame:
        # deliberately includes a LATE-filed restatement (10-K/A) with an
        # EARLY end_date, and a non-10-K form that must be ignored by the
        # default form filter -- both are realistic EDGAR shapes.
        rows = [
            {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2019-12-31", "filed": "2020-02-15", "val": 100.0},
            {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2020-12-31", "filed": "2021-02-10", "val": 150.0},
            {"concept": "NetIncomeLoss", "form": "10-K/A", "end_date": "2019-12-31", "filed": "2021-06-01", "val": 999.0},
            {"concept": "NetIncomeLoss", "form": "10-Q", "end_date": "2020-06-30", "filed": "2020-08-01", "val": 40.0},
        ]
        df = pd.DataFrame(rows)
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["filed"] = pd.to_datetime(df["filed"])
        return df

    def test_returns_nan_before_first_filing(self):
        fundamentals = self._fundamentals()
        as_of = pd.DatetimeIndex(["2019-06-01"])
        result = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of)
        assert pd.isna(result.iloc[0])

    def test_returns_value_right_after_its_filed_date(self):
        fundamentals = self._fundamentals()
        as_of = pd.DatetimeIndex(["2020-02-16", "2021-02-11"])
        result = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of)
        assert result.iloc[0] == 100.0
        assert result.iloc[1] == 150.0

    def test_does_not_leak_a_later_filed_value_early(self):
        # as of just BEFORE the 2021-02-10 filing, must still see the 2020
        # value (100.0), even though that 2021 filing's END DATE (2020-12-31)
        # is earlier than a naive end_date-based join might suggest is "due".
        fundamentals = self._fundamentals()
        as_of = pd.DatetimeIndex(["2021-02-09"])
        result = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of)
        assert result.iloc[0] == 100.0

    def test_ignores_non_matching_form_by_default(self):
        # the 10-Q value (40.0, filed 2020-08-01) must never appear when
        # form="10-K" (the default) -- querying just after it must still
        # show the prior 10-K's value, not the 10-Q's.
        fundamentals = self._fundamentals()
        as_of = pd.DatetimeIndex(["2020-08-02"])
        result = point_in_time_fundamental(fundamentals, "NetIncomeLoss", as_of)
        assert result.iloc[0] == 100.0

    def test_missing_concept_returns_all_nan_not_a_crash(self):
        fundamentals = self._fundamentals()
        as_of = pd.DatetimeIndex(["2021-01-01"])
        result = point_in_time_fundamental(fundamentals, "Assets", as_of)
        assert pd.isna(result.iloc[0])


class TestValueAndQuality:
    def test_earnings_yield_matches_hand_calculation(self):
        rows = [
            {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2019-12-31", "filed": "2020-02-15", "val": 1_000_000.0},
            {"concept": "CommonStockSharesOutstanding", "form": "10-K", "end_date": "2019-12-31", "filed": "2020-02-15", "val": 100_000.0},
        ]
        fundamentals = pd.DataFrame(rows)
        fundamentals["end_date"] = pd.to_datetime(fundamentals["end_date"])
        fundamentals["filed"] = pd.to_datetime(fundamentals["filed"])

        as_of = pd.DatetimeIndex(["2020-03-01"])
        price = pd.Series([20.0], index=as_of)

        ey = value_earnings_yield(fundamentals, price, as_of)
        # EPS = 1,000,000 / 100,000 = 10; E/P = 10 / 20 = 0.5
        assert ey.iloc[0] == pytest.approx(0.5)

    def test_roe_matches_hand_calculation(self):
        rows = [
            {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2019-12-31", "filed": "2020-02-15", "val": 2_000_000.0},
            {"concept": "StockholdersEquity", "form": "10-K", "end_date": "2019-12-31", "filed": "2020-02-15", "val": 10_000_000.0},
        ]
        fundamentals = pd.DataFrame(rows)
        fundamentals["end_date"] = pd.to_datetime(fundamentals["end_date"])
        fundamentals["filed"] = pd.to_datetime(fundamentals["filed"])

        as_of = pd.DatetimeIndex(["2020-03-01"])
        roe = quality_roe(fundamentals, as_of)
        assert roe.iloc[0] == pytest.approx(0.2)

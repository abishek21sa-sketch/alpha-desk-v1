from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.vol_relval.backtest import run_vol_relval_backtest
from alpha_desk.validation.costs import TransactionCostModel


def _flat_history(n: int, start_price: float = 100.0) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """n days of a constant price (zero return) and constant huge volume
    (so transaction costs are negligible), all in contango -- a warmup
    stretch so rolling vol/ADV windows are populated before the days under
    test."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    price = pd.Series(start_price, index=idx)
    volume = pd.Series(10_000_000.0, index=idx)
    return idx, price, volume


class TestRunVolRelValBacktest:
    def test_rejects_too_little_history(self):
        idx, price, volume = _flat_history(30)
        vix = pd.Series(15.0, index=idx)
        vix3m = pd.Series(18.0, index=idx)
        with pytest.raises(ValueError):
            run_vol_relval_backtest(vix, vix3m, price, volume, instrument="TEST")

    def test_flat_when_signal_never_matches_produces_zero_returns(self):
        n = 150
        idx, price, volume = _flat_history(n)
        # price actually moves so a non-flat position would show nonzero
        # returns -- but the signal is always contango and we ask for
        # long_when="backwardation", so the strategy should never take a
        # position and every net return must be exactly zero.
        price = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)
        vix = pd.Series(15.0, index=idx)
        vix3m = pd.Series(18.0, index=idx)  # always contango
        result = run_vol_relval_backtest(vix, vix3m, price, volume, instrument="TEST", long_when="backwardation")
        assert (result.returns == 0.0).all()
        assert result.pct_days_in_position == 0.0

    def test_position_uses_only_the_prior_days_signal_not_same_day(self):
        n = 120
        idx, price, volume = _flat_history(n)
        # constant price except one isolated +10% jump on day `jump_idx`.
        jump_idx = 110
        prices = np.full(n, 100.0)
        prices[jump_idx:] = 110.0
        price = pd.Series(prices, index=idx)

        # contango every day EXCEPT the day immediately before the jump
        # (backwardation there) -- if the backtest is using each day's OWN
        # close (a look-ahead bug), it would see backwardation and go flat
        # exactly across the jump anyway by coincidence with a bad lag; the
        # correct, lagged behavior is: signal on day (jump_idx - 1) is
        # backwardation -> position on day (jump_idx - 1) [decided using
        # day (jump_idx - 2)'s close, which was contango] is LONG, capturing
        # nothing since price hasn't moved yet, and position on day
        # jump_idx is FLAT (decided using day (jump_idx-1)'s backwardation
        # close) -- so the jump's return is deliberately missed.
        vix = pd.Series(15.0, index=idx)
        vix3m = pd.Series(18.0, index=idx)
        vix.iloc[jump_idx - 1] = 40.0
        vix3m.iloc[jump_idx - 1] = 25.0  # backwardation on this one day

        result = run_vol_relval_backtest(vix, vix3m, price, volume, instrument="TEST", long_when="contango")
        # the jump's return (day jump_idx) must be excluded from gross P&L:
        # the position on that day was decided using the prior
        # (backwardation) day's close, so it must be flat. Net return that
        # day is a small NEGATIVE cost drag (closing the position out costs
        # money), not literally zero -- but it must be nowhere near the 10%
        # jump a look-ahead bug (or a mistakenly-still-long position) would
        # have captured.
        assert abs(result.returns.loc[idx[jump_idx]]) < 0.005

    def test_positive_drift_while_in_position_gives_positive_gross_sharpe(self):
        n = 150
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # steady positive drift, always in contango, always long -> gross
        # Sharpe must be strongly positive (deterministic drift, tiny cost).
        price = pd.Series(100.0 * (1.002 ** np.arange(n)), index=idx)
        volume = pd.Series(10_000_000.0, index=idx)
        vix = pd.Series(15.0, index=idx)
        vix3m = pd.Series(18.0, index=idx)
        result = run_vol_relval_backtest(vix, vix3m, price, volume, instrument="TEST", long_when="contango")
        assert result.gross_sharpe_annualized > 5.0
        assert result.pct_days_in_position == pytest.approx(1.0)

    def test_net_sharpe_never_exceeds_gross_sharpe(self):
        n = 150
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(0)
        price = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
        volume = pd.Series(5_000_000.0, index=idx)
        # alternate contango/backwardation every ~10 days so there are
        # actual position changes (and therefore actual costs) to compare.
        vix = pd.Series(15.0, index=idx)
        vix3m = pd.Series([18.0 if (i // 10) % 2 == 0 else 12.0 for i in range(n)], index=idx)
        result = run_vol_relval_backtest(
            vix, vix3m, price, volume, instrument="TEST", long_when="contango",
            cost_model=TransactionCostModel(half_spread_bps=5.0),
        )
        assert result.cost_drag_annualized >= 0.0
        assert result.net_sharpe_annualized <= result.gross_sharpe_annualized

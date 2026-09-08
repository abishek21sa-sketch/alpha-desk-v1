from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_desk.strategies.factors.backtest import run_factor_backtest
from alpha_desk.validation.costs import TransactionCostModel


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _flat_fundamentals(seed_val: float) -> pd.DataFrame:
    # minimal, plausible-shaped fundamentals with NO real relationship to
    # returns -- gives value/quality something non-degenerate to compute
    # without injecting any genuine predictive signal via those factors.
    rows = [
        {"concept": "NetIncomeLoss", "form": "10-K", "end_date": "2014-12-31", "filed": "2015-02-15", "val": seed_val},
        {"concept": "CommonStockSharesOutstanding", "form": "10-K", "end_date": "2014-12-31", "filed": "2015-02-15", "val": 1_000_000.0},
        {"concept": "StockholdersEquity", "form": "10-K", "end_date": "2014-12-31", "filed": "2015-02-15", "val": seed_val * 5},
    ]
    df = pd.DataFrame(rows)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def _universe_with_genuine_momentum(n_symbols: int = 24, n_days: int = 3000, seed: int = 0):
    """Each symbol gets a persistent, symbol-specific drift (latent alpha).
    Since the drift doesn't change over time, past momentum (which reflects
    a symbol's own historical drift) is a genuinely predictive, if noisy,
    signal for that symbol's FUTURE return -- the textbook mechanism behind
    a real momentum factor, constructed here with a known, controllable
    true IC rather than hoped for.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n_days)
    prices, volumes, fundamentals = {}, {}, {}
    for i in range(n_symbols):
        sym = f"S{i:02d}"
        alpha = rng.uniform(-0.0004, 0.0012)
        daily_returns = alpha + rng.normal(0, 0.015, n_days)
        prices[sym] = pd.Series(100 * np.cumprod(1 + daily_returns), index=idx)
        volumes[sym] = pd.Series(rng.uniform(3e5, 1.5e6, n_days), index=idx)
        fundamentals[sym] = _flat_fundamentals(seed_val=rng.uniform(1e5, 1e6))
    return prices, volumes, fundamentals


def _universe_pure_noise(n_symbols: int = 24, n_days: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = _dates(n_days)
    prices, volumes, fundamentals = {}, {}, {}
    for i in range(n_symbols):
        sym = f"S{i:02d}"
        daily_returns = rng.normal(0, 0.015, n_days)  # NO persistent per-symbol drift at all
        prices[sym] = pd.Series(100 * np.cumprod(1 + daily_returns), index=idx)
        volumes[sym] = pd.Series(rng.uniform(3e5, 1.5e6, n_days), index=idx)
        fundamentals[sym] = _flat_fundamentals(seed_val=rng.uniform(1e5, 1e6))
    return prices, volumes, fundamentals


class TestRunFactorBacktest:
    def test_raises_on_too_little_history(self):
        prices, volumes, fundamentals = _universe_with_genuine_momentum(n_symbols=10, n_days=200)
        with pytest.raises(ValueError):
            run_factor_backtest(prices, volumes, fundamentals)

    def test_genuine_momentum_signal_produces_positive_net_sharpe(self):
        # NOTE on n_days=3000, n_splits=3: a first version of this test used
        # n_days=1600, n_splits=5 and got net_sharpe=0.118 despite momentum
        # having a real, oracle-confirmed edge (a pure-momentum-weighted,
        # non-walk-forward version of this same universe scores Sharpe
        # 0.89) -- the walk-forward folds' TRAINING windows were too short
        # for IC calibration to reliably tell momentum's real signal apart
        # from the three pure-noise factors (value/quality/low_vol have no
        # true relationship to returns in this synthetic universe), so
        # spurious noise-factor IC diluted momentum's weight down to
        # 0.12-0.38 instead of dominating. Longer history + fewer/larger
        # folds (verified positive across 6 seeds, range 0.29-1.29) fixes
        # this -- see docs/FACTOR_STRATEGY.md for the full writeup. This
        # was a genuine, useful finding about calibration data requirements,
        # not a bug in the backtest itself (confirmed via the oracle check).
        prices, volumes, fundamentals = _universe_with_genuine_momentum(seed=1)
        result = run_factor_backtest(
            prices, volumes, fundamentals, n_splits=3,
            cost_model=TransactionCostModel(commission_bps=0.0, half_spread_bps=1.0, impact_coefficient=0.05),
        )
        assert result.n_rebalances > 20
        assert result.net_sharpe_annualized > 0.2
        # momentum should be the (or among the) dominant calibrated factor(s)
        # in most folds, since it's the only genuinely predictive one here
        momentum_weights = [fw.weights.get("momentum", 0.0) for fw in result.fold_weights]
        assert np.mean(momentum_weights) > 0.3

    def test_pure_noise_universe_does_not_reliably_produce_a_strong_sharpe(self):
        sharpes = []
        for seed in range(8):
            prices, volumes, fundamentals = _universe_pure_noise(seed=seed)
            result = run_factor_backtest(prices, volumes, fundamentals)
            sharpes.append(result.net_sharpe_annualized)
        assert np.nanmean(sharpes) < 0.3  # well below the genuine-signal case above

    def test_zero_cost_model_gives_net_equal_to_gross(self):
        prices, volumes, fundamentals = _universe_with_genuine_momentum(seed=2)
        zero_cost = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=0.0)
        result = run_factor_backtest(prices, volumes, fundamentals, cost_model=zero_cost)
        assert result.net_sharpe_annualized == pytest.approx(result.gross_sharpe_annualized, abs=1e-9)
        assert result.cost_drag_annualized == pytest.approx(0.0, abs=1e-9)

    def test_higher_costs_reduce_net_sharpe_without_changing_gross(self):
        prices, volumes, fundamentals = _universe_with_genuine_momentum(seed=3)
        cheap = TransactionCostModel(commission_bps=0.0, half_spread_bps=0.5, impact_coefficient=0.02)
        expensive = TransactionCostModel(commission_bps=0.0, half_spread_bps=15.0, impact_coefficient=0.8)

        cheap_result = run_factor_backtest(prices, volumes, fundamentals, cost_model=cheap)
        expensive_result = run_factor_backtest(prices, volumes, fundamentals, cost_model=expensive)

        assert expensive_result.net_sharpe_annualized < cheap_result.net_sharpe_annualized
        assert cheap_result.gross_sharpe_annualized == pytest.approx(
            expensive_result.gross_sharpe_annualized, abs=1e-9
        )

    def test_returns_index_has_no_duplicate_dates_across_folds(self):
        prices, volumes, fundamentals = _universe_with_genuine_momentum(seed=4)
        result = run_factor_backtest(prices, volumes, fundamentals)
        assert result.returns.index.is_unique

    def test_fold_weights_recorded_for_every_fold(self):
        prices, volumes, fundamentals = _universe_with_genuine_momentum(seed=5)
        result = run_factor_backtest(prices, volumes, fundamentals, n_splits=5)
        assert result.n_folds == 5
        assert len(result.fold_weights) == 5
        for fw in result.fold_weights:
            total = sum(fw.weights.values())
            assert total == pytest.approx(1.0) or total == 0.0

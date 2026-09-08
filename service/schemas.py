"""Pydantic response models. These mirror exactly what
scripts/run_*_strategy.py write to artifacts/*.json -- the API is a read
layer over pre-computed, honestly-reported research artifacts, not a live
recompute engine (re-screening 780 pairs takes minutes; this is a
deliberate architecture choice, not a missing feature -- see README's
Phase 10 entry for the planned live/paper-trading extension).
"""

from __future__ import annotations

from pydantic import BaseModel


class EquityCurvePoint(BaseModel):
    date: str
    cumulative_return: float


class DataBackboneStatus(BaseModel):
    prices_rows: int
    prices_symbols: int
    macro_rows: int
    macro_series: int
    fundamentals_rows: int
    fundamentals_tickers: int


class PairSummary(BaseModel):
    pair: str
    pvalue: float
    half_life_days: float
    n_trades: int
    net_sharpe: float


class PairsStrategyReport(BaseModel):
    n_pairs_screened: int
    n_tradeable: int
    n_backtested: int
    best_pair: str
    best_pair_pvalue: float
    best_pair_half_life_days: float
    best_pair_n_trades: int
    best_pair_gross_sharpe: float
    best_pair_net_sharpe: float
    best_pair_cost_drag: float
    deflated_sharpe_ratio: float
    expected_max_sharpe_under_luck: float
    pbo: float | None = None
    pbo_n_combinations: int | None = None
    all_pairs: list[PairSummary]
    equity_curve: list[EquityCurvePoint]


class FoldWeights(BaseModel):
    fold: int
    weights: dict[str, float]
    ic_by_factor: dict[str, float]


class FactorStrategyReport(BaseModel):
    n_splits: int
    n_folds: int
    n_rebalances: int
    gross_sharpe: float
    net_sharpe: float
    cost_drag: float
    probabilistic_sharpe_ratio: float
    fold_weights: list[FoldWeights]
    equity_curve: list[EquityCurvePoint]


class CalibrationBin(BaseModel):
    mean_predicted: float
    mean_actual: float
    count: int


class ClassifierMetricsSchema(BaseModel):
    pr_auc: float
    brier_score: float
    log_loss: float
    n_obs: int
    base_rate: float
    calibration_bins: list[CalibrationBin]


class MLStrategyReport(BaseModel):
    n_splits: int
    n_folds: int
    n_folds_skipped: int
    n_rebalances: int
    gross_sharpe: float
    net_sharpe: float
    cost_drag: float
    probabilistic_sharpe_ratio: float
    pooled_metrics: ClassifierMetricsSchema
    equity_curve: list[EquityCurvePoint]


class MomentumStrategyReport(BaseModel):
    n_assets: int
    n_rebalances: int
    gross_sharpe: float
    net_sharpe: float
    cost_drag: float
    probabilistic_sharpe_ratio: float
    per_asset_sharpe: dict[str, float]
    equity_curve: list[EquityCurvePoint]


class PeadSignificance(BaseModel):
    n_positive_events: int
    n_negative_events: int
    mean_drift_positive: float
    mean_drift_negative: float
    drift_spread: float
    t_stat: float
    p_value: float


class PeadStrategyReport(BaseModel):
    n_events_total: int
    n_events_traded: int
    max_concurrent_trades: int
    gross_sharpe: float
    net_sharpe: float
    cost_drag: float
    significance: PeadSignificance
    equity_curve: list[EquityCurvePoint]


class HealthResponse(BaseModel):
    status: str
    phases_available: list[str]

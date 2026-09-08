export interface EquityCurvePoint {
  date: string;
  cumulative_return: number;
}

export interface DataBackboneStatus {
  prices_rows: number;
  prices_symbols: number;
  macro_rows: number;
  macro_series: number;
  fundamentals_rows: number;
  fundamentals_tickers: number;
}

export interface PairSummary {
  pair: string;
  pvalue: number;
  half_life_days: number;
  n_trades: number;
  net_sharpe: number;
}

export interface PairsStrategyReport {
  n_pairs_screened: number;
  n_tradeable: number;
  n_backtested: number;
  best_pair: string;
  best_pair_pvalue: number;
  best_pair_half_life_days: number;
  best_pair_n_trades: number;
  best_pair_gross_sharpe: number;
  best_pair_net_sharpe: number;
  best_pair_cost_drag: number;
  deflated_sharpe_ratio: number;
  expected_max_sharpe_under_luck: number;
  pbo: number | null;
  pbo_n_combinations: number | null;
  all_pairs: PairSummary[];
  equity_curve: EquityCurvePoint[];
}

export interface FoldWeights {
  fold: number;
  weights: Record<string, number>;
  ic_by_factor: Record<string, number>;
}

export interface FactorStrategyReport {
  n_splits: number;
  n_folds: number;
  n_rebalances: number;
  gross_sharpe: number;
  net_sharpe: number;
  cost_drag: number;
  probabilistic_sharpe_ratio: number;
  fold_weights: FoldWeights[];
  equity_curve: EquityCurvePoint[];
}

export interface CalibrationBin {
  mean_predicted: number;
  mean_actual: number;
  count: number;
}

export interface ClassifierMetrics {
  pr_auc: number;
  brier_score: number;
  log_loss: number;
  n_obs: number;
  base_rate: number;
  calibration_bins: CalibrationBin[];
}

export interface MomentumStrategyReport {
  n_assets: number;
  n_rebalances: number;
  gross_sharpe: number;
  net_sharpe: number;
  cost_drag: number;
  probabilistic_sharpe_ratio: number;
  per_asset_sharpe: Record<string, number>;
  equity_curve: EquityCurvePoint[];
}

export interface PeadSignificance {
  n_positive_events: number;
  n_negative_events: number;
  mean_drift_positive: number;
  mean_drift_negative: number;
  drift_spread: number;
  t_stat: number;
  p_value: number;
}

export interface PeadStrategyReport {
  n_events_total: number;
  n_events_traded: number;
  max_concurrent_trades: number;
  gross_sharpe: number;
  net_sharpe: number;
  cost_drag: number;
  significance: PeadSignificance;
  equity_curve: EquityCurvePoint[];
}

export interface VarCvarEntry {
  var_95?: number;
  cvar_95?: number;
  var_99?: number;
  cvar_99?: number;
}

export interface StressWindowEntry {
  window: string;
  has_coverage: boolean;
  cumulative_return: number | null;
  max_drawdown: number | null;
  n_obs: number;
}

export interface ExecutionScenario {
  risk_aversion: number;
  expected_cost_bps: number;
  cost_std_bps: number;
  holdings_pct: number[];
}

export interface RiskReport {
  var_cvar: Record<string, VarCvarEntry>;
  stress: Record<string, StressWindowEntry[]>;
  execution_example: {
    symbol: string;
    last_price: number;
    adv_dollars: number;
    daily_vol: number;
    shares_to_sell: number;
    scenarios: Record<string, ExecutionScenario>;
  };
}

export interface MicrostructureScenario {
  gamma: number;
  n_sessions: number;
  mean_fills_per_session: number;
  mean_pnl_per_session: number;
  std_pnl_per_session: number;
  pnl_sharpe_like: number | null;
  pct_sessions_profitable: number;
  mean_max_abs_inventory: number;
  worst_session_pnl: number;
  best_session_pnl: number;
}

export interface MicrostructureReport {
  symbol: string;
  calibration: {
    daily_vol: number;
    last_price: number;
    sigma_dollar_per_day: number;
    n_steps: number;
    arrival_intensity: number;
    arrival_intensity_x_dt: number;
    kappa: number;
    inventory_limit: number;
    n_sessions_per_scenario: number;
  };
  scenarios: Record<string, MicrostructureScenario>;
}

export interface VolRelValScenario {
  instrument: string;
  long_when: string;
  n_days: number;
  pct_days_in_position: number;
  gross_sharpe_annualized: number;
  net_sharpe_annualized: number;
  cost_drag_annualized: number;
  probabilistic_sharpe_ratio: number;
  equity_curve: EquityCurvePoint[];
}

export interface VolRelValReport {
  pct_contango_all_history: number;
  n_days_term_structure_history: number;
  scenarios: {
    svxy_long_in_contango: VolRelValScenario;
    vixy_long_in_backwardation: VolRelValScenario;
  };
}

export interface AlpacaConnectivityStatus {
  configured: boolean;
  base_url: string;
  connected: boolean;
  message?: string;
  account_status?: string;
  equity?: string;
  buying_power?: string;
  market_open?: boolean;
}

export interface MLStrategyReport {
  n_splits: number;
  n_folds: number;
  n_folds_skipped: number;
  n_rebalances: number;
  gross_sharpe: number;
  net_sharpe: number;
  cost_drag: number;
  probabilistic_sharpe_ratio: number;
  pooled_metrics: ClassifierMetrics;
  equity_curve: EquityCurvePoint[];
}

"""Phase 3 — statistical arbitrage / pairs trading.

- cointegration.py: Engle-Granger pair screening + mean-reversion half-life
- kalman.py: Kalman-filter dynamic hedge ratio (replaces a static OLS beta)
- signals.py: z-score entry/exit state machine
- backtest.py: cost-aware backtest wired through the Phase 2 validation engine
"""

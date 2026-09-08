"""Phase 8 — risk & execution control tower. Wraps every strategy built so
far rather than adding a new one: historical VaR/CVaR, real historical-
window stress replay, and Almgren-Chriss optimal execution.

- var_cvar.py: historical (empirical-quantile) VaR and CVaR
- stress.py: replays each strategy's OWN real return history through named
  historical crisis windows -- not a hypothetical shock model
- execution.py: Almgren-Chriss optimal execution trajectory (impact-vs-
  timing-risk trade-off), for a resting order too large to fill at once
"""

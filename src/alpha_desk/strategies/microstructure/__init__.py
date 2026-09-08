"""Phase 9 — limit-order-book market making, via the Avellaneda-Stoikov
(2008) optimal market-making model.

Deliberately NOT built on raw exchange tick data: real IEX historical
TOPS/DEEP files exist and are downloadable, but at ~600MB+ COMPRESSED PER
DAY for the entire market (not per-symbol), in a custom binary protocol
(IEXTP1) with no off-the-shelf parser -- a multi-hour undertaking on its
own before any strategy work happens. Instead, this module simulates order
flow (Poisson arrivals, standard in the market-making literature) with
intensity/volatility parameters calibrated to this platform's own REAL
daily volatility data, and is explicit about that substitution everywhere
it matters -- see docs/MICROSTRUCTURE_STRATEGY.md.

- market_maker.py: the closed-form Avellaneda-Stoikov reservation price and
  optimal bid/ask spread
- simulation.py: a discrete-time market-making session simulation (real
  calibrated volatility, simulated Poisson order arrivals)
"""

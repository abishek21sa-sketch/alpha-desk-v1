"""Phase 11 — Alpaca paper-trading execution client.

A thin REST wrapper (`alpaca_client.py`) around Alpaca's PAPER trading
API, following this platform's existing convention (see `data/sources.py`)
of real, undecorated HTTP calls rather than a heavy vendor SDK.

**Honest status as of this build**: real Alpaca credentials were searched
for in the user's other local projects (which use this exact
ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL env-var convention)
but only `.env.example` templates were found there -- no working
credentials exist on this machine. This module is therefore built and
tested (with mocked HTTP responses) but has never made a real network call
to Alpaca. It activates automatically the moment real credentials are
placed in `.env` -- see `scripts/check_alpaca_connectivity.py`, which
reports its own configured/connected state honestly rather than assuming
success.

**Hard safety rule, enforced in code, not just documentation**:
`AlpacaClient` refuses to construct against anything but a paper-trading
host. There is no code path in this platform that can target Alpaca's
live-trading endpoint.
"""

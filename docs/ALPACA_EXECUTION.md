# Phase 11 — Alpaca Paper-Trading Execution

`src/alpha_desk/execution/`: a thin REST wrapper around Alpaca's paper-
trading API (`alpaca_client.py`), following this platform's existing
convention (`data/sources.py`) of real, undecorated HTTP calls rather than
a vendor SDK. 13 tests, all passing, all against mocked HTTP responses —
the client's unit tests don't hit the network at all; see "Real connection
status" below for the live account check, which does.

## What this is for

Every strategy on this platform stops at producing a signal and a report.
Phase 11 is the one piece that could, in principle, act on a signal — but
it deliberately still doesn't, automatically. `submit_order()` exists and
is tested, but nothing in this platform's pipeline calls it. Order
submission is a manual, human-gated action, matching
`PROJECT_METADATA.json`'s `"human_authority":
"ALL_TRADE_DECISIONS_HUMAN_GATED"` / `"autonomous_execution": false`
declarations that have held since Phase 1.

## Real connection status: live

This build initially searched the user's other local projects
(`AURUM_PRODUCT_V1` and `AURUM_RC3`, both of which already integrate
Alpaca) for reusable credentials, at multiple depths, exact filenames, and
Windows environment variables (User/Machine/Process scope) — only
`.env.example` templates existed in either project; no working credential
was found there. The user then created a fresh Alpaca paper account and
supplied real keys directly, which now live in a local, gitignored `.env`
(never committed, never sent anywhere but Alpaca's own API).

`scripts/check_alpaca_connectivity.py` made one real, live call each to
`/v2/account` and `/v2/clock` against `https://paper-api.alpaca.markets`
and got back a genuine account snapshot — status `ACTIVE`, real (simulated)
equity and buying power, and the market's actual open/closed state at
call time. This is reported through the exact same code path as the
"not configured" state was before: `artifacts/alpaca_connectivity_status.json`
reflects whatever `check_alpaca_connectivity.py` actually found, with no
special-casing between the two states — the honesty discipline didn't
change, only the real-world fact it's reporting did.

This phase still does the same two things it always did:
1. **Builds and tests the client against mocked HTTP responses** — every
   method (`get_account`, `get_clock`, `list_positions`, `list_orders`,
   `submit_order`) is exercised the way it will actually be called, with
   the real Alpaca REST contract (endpoint paths, header names, JSON
   payload shape) baked into the test expectations. The unit tests
   themselves are still fully mocked (fast, deterministic, no network) —
   the live check is a separate, explicit step (`check_alpaca_connectivity.py`),
   not folded into the test suite.
2. **`scripts/check_alpaca_connectivity.py` reports its own state
   honestly** rather than assuming success: "not configured" if
   `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` are unset, the real account/clock
   response if they're set and valid, or a real auth failure if they're
   set but wrong. Its output artifact reflects exactly one of those three
   states, never a fabricated fourth.

Re-running the script re-verifies the connection and refreshes the
artifact with the account's current live numbers — no code changes
needed, and no credentials are ever written anywhere but the local `.env`.

## A hard safety rule enforced in code, not just documentation

`AlpacaClient.__init__` refuses to construct at all if `base_url` resolves
to Alpaca's LIVE trading host (`api.alpaca.markets`, as opposed to
`paper-api.alpaca.markets`) — this is a real, tested guardrail
(`test_refuses_the_live_trading_host`), not a comment. There is no code
path anywhere in this platform that can target live trading, regardless of
what `ALPACA_BASE_URL` gets set to by mistake.

## What this does NOT do

- **No automated order submission** — `submit_order()` is built and
  tested but never called by any script in this platform. Turning any
  strategy's signal into a real (even paper) order is left as a manual,
  human-initiated action.
- **No portfolio-level position reconciliation** between this platform's
  own strategy reports and whatever a real Alpaca account actually holds
  — that would require a live connection this build doesn't have.

## Running it

```bash
pytest tests/test_alpaca_client.py -v
python scripts/check_alpaca_connectivity.py
```

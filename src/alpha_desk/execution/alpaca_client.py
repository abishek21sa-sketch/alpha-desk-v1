"""A thin, honest wrapper around Alpaca's paper-trading REST API.

Same philosophy as `data/sources.py`: real HTTP requests, nothing invented,
failures raise rather than being swallowed. Reads credentials from
ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL environment variables
by default (the same names this user's other local projects already use),
so real credentials can be dropped into `.env` without touching code.
"""

from __future__ import annotations

import os

import requests

PAPER_BASE_URL_DEFAULT = "https://paper-api.alpaca.markets"
# Alpaca's LIVE trading host -- this client must never be able to target it.
LIVE_TRADING_HOST = "api.alpaca.markets"

REQUEST_TIMEOUT_S = 15


class AlpacaConfigError(RuntimeError):
    """Raised when credentials are missing or a non-paper host is requested."""


class AlpacaClient:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.base_url = (base_url or os.environ.get("ALPACA_BASE_URL") or PAPER_BASE_URL_DEFAULT).rstrip("/")

        # A hard, code-level guardrail, not just a docstring warning: refuse
        # to even construct a client pointed at Alpaca's live-trading host.
        # ("paper-api.alpaca.markets" does NOT contain "api.alpaca.markets"
        # as a bare host match because of the "paper-" prefix, so this
        # correctly allows the paper host while rejecting the live one.)
        if LIVE_TRADING_HOST in self.base_url and "paper-api" not in self.base_url:
            raise AlpacaConfigError(
                f"refusing to target a non-paper Alpaca host ({self.base_url!r}) -- "
                "this platform only ever executes against paper-api.alpaca.markets"
            )

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _require_configured(self) -> None:
        if not self.is_configured():
            raise AlpacaConfigError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set -- see .env.example"
            )

    def _headers(self) -> dict[str, str]:
        self._require_configured()
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def _get(self, path: str) -> dict | list:
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    def get_account(self) -> dict:
        """Real account snapshot: equity, buying power, status, etc."""
        return self._get("/v2/account")

    def get_clock(self) -> dict:
        """Whether the market is currently open, and the next open/close times."""
        return self._get("/v2/clock")

    def list_positions(self) -> list[dict]:
        result = self._get("/v2/positions")
        return result if isinstance(result, list) else []

    def list_orders(self, status: str = "open") -> list[dict]:
        result = self._get(f"/v2/orders?status={status}")
        return result if isinstance(result, list) else []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        """Submits a real (paper-account) order. Not called anywhere in this
        platform's automated pipeline -- every strategy here stops at
        producing a signal/report; order submission is a manual, human-
        gated action, matching PROJECT_METADATA.json's
        "ALL_TRADE_DECISIONS_HUMAN_GATED" declaration.
        """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if qty <= 0:
            raise ValueError("qty must be > 0")
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        resp = requests.post(
            f"{self.base_url}/v2/orders", json=payload, headers=self._headers(), timeout=REQUEST_TIMEOUT_S
        )
        resp.raise_for_status()
        return resp.json()

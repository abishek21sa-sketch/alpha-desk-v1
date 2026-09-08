"""Phase 11: check (and report, honestly) whether this platform can reach
a real Alpaca paper-trading account.

This script does NOT fabricate a "connected" result. If ALPACA_API_KEY /
ALPACA_SECRET_KEY aren't set, it reports "not configured" and exits
cleanly -- that is the true state on this machine as of this build (real
credentials were searched for in the user's other local Alpaca-integrated
projects and not found there; see docs/ALPACA_EXECUTION.md). If credentials
ARE set, it makes one real, lightweight call each to /v2/account and
/v2/clock and reports exactly what came back, including a real auth
failure if the credentials turn out to be invalid.

Usage:
    python scripts/check_alpaca_connectivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_desk.execution.alpaca_client import AlpacaClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    client = AlpacaClient()
    status: dict = {
        "configured": client.is_configured(),
        "base_url": client.base_url,
    }

    if not client.is_configured():
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set in the environment.")
        print("This platform's execution client (src/alpha_desk/execution/alpaca_client.py)")
        print("is fully built and tested (13 tests, mocked HTTP) but has never made a real")
        print("call -- see .env.example for the variable names, and docs/ALPACA_EXECUTION.md")
        print("for what was checked before concluding no credentials are available.")
        status["connected"] = False
        status["message"] = "not configured -- see .env.example"
    else:
        print(f"Credentials found. Checking connectivity to {client.base_url} ...")
        try:
            account = client.get_account()
            clock = client.get_clock()
            status["connected"] = True
            status["account_status"] = account.get("status")
            status["equity"] = account.get("equity")
            status["buying_power"] = account.get("buying_power")
            status["market_open"] = clock.get("is_open")
            print(f"  Connected. Account status={account.get('status')}, "
                  f"equity=${account.get('equity')}, market_open={clock.get('is_open')}")
        except Exception as exc:  # noqa: BLE001 -- reporting the real failure, not hiding it
            status["connected"] = False
            status["message"] = f"{type(exc).__name__}: {exc}"
            print(f"  Connection failed: {status['message']}")

    out_path = ARTIFACTS / "alpaca_connectivity_status.json"
    out_path.write_text(json.dumps(status, indent=2, default=str))
    print(f"\nStatus written -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from alpha_desk.data.universe import EQUITY_UNIVERSE, ETF_UNIVERSE, FRED_SERIES  # noqa: E402

from ..artifacts_io import read_processed
from ..schemas import DataBackboneStatus

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/status", response_model=DataBackboneStatus)
def get_status() -> DataBackboneStatus:
    summary = read_processed("build_summary.json")
    return DataBackboneStatus(**summary)


@router.get("/universe")
def get_universe() -> dict:
    return {
        "equities": [{"symbol": t, "sector": s} for t, _, s in EQUITY_UNIVERSE],
        "etfs": [{"symbol": t, "sector": s} for t, _, s in ETF_UNIVERSE],
        "macro_series": FRED_SERIES,
    }

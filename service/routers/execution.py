from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact

router = APIRouter(prefix="/api/execution", tags=["execution"])


@router.get("/status")
def get_alpaca_status() -> dict:
    # Reports exactly what scripts/check_alpaca_connectivity.py found --
    # "not configured" is a real, honest state here, not an error to hide.
    return read_artifact("alpaca_connectivity_status.json")

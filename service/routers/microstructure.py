from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact

router = APIRouter(prefix="/api/microstructure", tags=["microstructure"])


@router.get("/report")
def get_microstructure_report() -> dict:
    # Same rationale as risk.py: this report's shape (calibration block +
    # per-gamma-scenario stats) is its own shape, not the shared
    # equity-curve/metrics row schema the other strategy reports use --
    # returned as-is from the artifact.
    return read_artifact("microstructure_strategy_report.json")

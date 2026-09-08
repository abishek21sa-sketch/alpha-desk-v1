from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact

router = APIRouter(prefix="/api/vol-relval", tags=["vol_relval"])


@router.get("/report")
def get_vol_relval_report() -> dict:
    # Same rationale as risk.py/microstructure.py: two named scenarios with
    # their own equity curves, not the shared single-strategy row schema --
    # returned as-is from the artifact.
    return read_artifact("vol_relval_strategy_report.json")

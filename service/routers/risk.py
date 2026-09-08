from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/report")
def get_risk_report() -> dict:
    # Not a strict Pydantic response_model: this report's shape (per-
    # strategy VaR/CVaR, per-strategy stress windows with optional
    # coverage, one execution-example block) is genuinely heterogeneous
    # rather than a fixed row schema like the other strategy reports --
    # returned as-is from the artifact, still real, hash-traceable JSON.
    return read_artifact("risk_report.json")

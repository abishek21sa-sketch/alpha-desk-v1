from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact
from ..schemas import PeadStrategyReport

router = APIRouter(prefix="/api/pead", tags=["pead"])


@router.get("/report", response_model=PeadStrategyReport)
def get_pead_report() -> PeadStrategyReport:
    return PeadStrategyReport(**read_artifact("pead_strategy_report.json"))

from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact
from ..schemas import MLStrategyReport

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/report", response_model=MLStrategyReport)
def get_ml_report() -> MLStrategyReport:
    return MLStrategyReport(**read_artifact("ml_strategy_report.json"))

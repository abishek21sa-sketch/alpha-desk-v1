from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact
from ..schemas import FactorStrategyReport

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("/report", response_model=FactorStrategyReport)
def get_factors_report() -> FactorStrategyReport:
    return FactorStrategyReport(**read_artifact("factor_strategy_report.json"))

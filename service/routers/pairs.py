from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact
from ..schemas import PairsStrategyReport

router = APIRouter(prefix="/api/pairs", tags=["pairs"])


@router.get("/report", response_model=PairsStrategyReport)
def get_pairs_report() -> PairsStrategyReport:
    return PairsStrategyReport(**read_artifact("pairs_strategy_report.json"))

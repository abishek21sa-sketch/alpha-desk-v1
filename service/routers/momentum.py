from __future__ import annotations

from fastapi import APIRouter

from ..artifacts_io import read_artifact
from ..schemas import MomentumStrategyReport

router = APIRouter(prefix="/api/momentum", tags=["momentum"])


@router.get("/report", response_model=MomentumStrategyReport)
def get_momentum_report() -> MomentumStrategyReport:
    return MomentumStrategyReport(**read_artifact("momentum_strategy_report.json"))

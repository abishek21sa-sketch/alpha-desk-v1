"""FastAPI service exposing Alpha Desk's research artifacts to the
frontend. A thin, synchronous read layer over pre-computed JSON reports --
see artifacts_io.py's module docstring for why this deliberately isn't a
live-recompute API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import data, execution, factors, microstructure, ml, momentum, pairs, pead, risk, vol_relval
from .schemas import HealthResponse

app = FastAPI(
    title="Alpha Desk API",
    description="Quantitative trading research, validation, and risk platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(pairs.router)
app.include_router(factors.router)
app.include_router(ml.router)
app.include_router(momentum.router)
app.include_router(pead.router)
app.include_router(risk.router)
app.include_router(microstructure.router)
app.include_router(vol_relval.router)
app.include_router(execution.router)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        phases_available=[
            "data_backbone", "validation_engine", "pairs", "factors", "ml",
            "momentum", "pead", "risk", "microstructure", "vol_relval", "execution",
        ],
    )

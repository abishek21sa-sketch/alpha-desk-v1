"""FastAPI service exposing Alpha Desk's research artifacts to the
frontend. A thin, synchronous read layer over pre-computed JSON reports --
see artifacts_io.py's module docstring for why this deliberately isn't a
live-recompute API.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import data, execution, factors, microstructure, ml, momentum, pairs, pead, risk, vol_relval
from .schemas import HealthResponse

app = FastAPI(
    title="Alpha Desk API",
    description="Quantitative trading research, validation, and risk platform.",
    version="0.1.0",
)

# ALLOWED_ORIGINS: comma-separated exact origins (e.g. the deployed Vercel
# URL) -- set as a real env var in production (see render.yaml); defaults
# to the local dev frontend so `uvicorn service.main:app` still works
# out of the box with no configuration.
_default_origins = "http://localhost:3000"
allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Vercel preview deployments get a unique per-branch/PR subdomain
    # (e.g. alpha-desk-v1-git-<branch>-<user>.vercel.app) that can't be
    # listed in advance -- this regex covers any *.vercel.app origin in
    # addition to the exact origins above, so preview deploys work without
    # needing a Render env var update on every branch push.
    allow_origin_regex=r"https://.*\.vercel\.app",
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

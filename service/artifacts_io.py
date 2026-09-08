"""Shared helper for reading pre-computed JSON artifacts off disk. One
place to get the "artifact not generated yet" error message right, rather
than repeating a try/except per router.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"


def read_artifact(filename: str) -> dict:
    path = ARTIFACTS / filename
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{filename} has not been generated yet -- run the corresponding "
                f"scripts/run_*.py script first. This API serves pre-computed "
                f"research artifacts, not live recomputation."
            ),
        )
    return json.loads(path.read_text())


def read_processed(filename: str) -> dict:
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{filename} not found -- run scripts/build_data_backbone.py first.",
        )
    return json.loads(path.read_text())

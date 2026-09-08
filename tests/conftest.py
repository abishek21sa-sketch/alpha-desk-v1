from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from alpha_desk.data.warehouse import DB_PATH

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def warehouse_con():
    if not DB_PATH.exists():
        pytest.skip("warehouse.duckdb not built -- run scripts/build_data_backbone.py first")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def manifest() -> dict:
    path = ROOT / "data" / "provenance" / "source_manifest.json"
    if not path.exists():
        pytest.skip("source_manifest.json missing -- run scripts/fetch_public_data.py first")
    return json.loads(path.read_text())

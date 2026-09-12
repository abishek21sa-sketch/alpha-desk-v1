"""Validates the built warehouse against data/contracts/*.json's invariants,
and cross-checks the provenance manifest's recorded hashes against the raw
files actually on disk (hash-verifiable evidence, not just a claim)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alpha_desk.data.universe import EQUITY_UNIVERSE, ETF_UNIVERSE, FRED_SERIES

ROOT = Path(__file__).resolve().parent.parent


class TestPricesContract:
    def test_expected_symbol_count(self, warehouse_con):
        n = warehouse_con.execute("SELECT count(DISTINCT symbol) FROM prices").fetchone()[0]
        assert n == len(EQUITY_UNIVERSE) + len(ETF_UNIVERSE)

    def test_no_duplicate_symbol_date(self, warehouse_con):
        dupes = warehouse_con.execute(
            "SELECT symbol, date, count(*) c FROM prices GROUP BY 1, 2 HAVING c > 1"
        ).fetchall()
        assert dupes == []

    def test_ohlc_ordering(self, warehouse_con):
        # A tiny float64 epsilon tolerance is deliberate, not a loosened
        # check: split/dividend adjustment multiplies each of O/H/L/C by a
        # slightly different cumulative factor, so two columns that were
        # exactly equal pre-adjustment can land ~1e-14 apart post-adjustment
        # -- confirmed by inspection (violations top out at ~4e-14, i.e.
        # machine-epsilon noise, not a real ordering violation). A real data
        # error would show a violation many orders of magnitude larger than
        # this, which this tolerance would still catch.
        eps = 1e-9
        bad = warehouse_con.execute(
            f"SELECT count(*) FROM prices WHERE high < low - {eps} OR high < open - {eps} "
            f"OR high < close - {eps} OR low > open + {eps} OR low > close + {eps}"
        ).fetchone()[0]
        assert bad == 0

    def test_non_negative_and_finite(self, warehouse_con):
        bad = warehouse_con.execute(
            "SELECT count(*) FROM prices WHERE open < 0 OR high < 0 OR low < 0 OR close < 0 "
            "OR volume < 0 OR isnan(open) OR isnan(close)"
        ).fetchone()[0]
        assert bad == 0

    def test_no_future_dates(self, warehouse_con):
        bad = warehouse_con.execute("SELECT count(*) FROM prices WHERE date > current_date").fetchone()[0]
        assert bad == 0


class TestMacroContract:
    def test_expected_series_count(self, warehouse_con):
        n = warehouse_con.execute("SELECT count(DISTINCT series_id) FROM macro").fetchone()[0]
        assert n == len(FRED_SERIES)

    def test_no_duplicate_series_date(self, warehouse_con):
        dupes = warehouse_con.execute(
            "SELECT series_id, date, count(*) c FROM macro GROUP BY 1, 2 HAVING c > 1"
        ).fetchall()
        assert dupes == []


class TestFundamentalsContract:
    def test_filed_never_before_period_end(self, warehouse_con):
        # instant concepts (e.g. Assets) can have filed == end_date's fiscal
        # year-end filed later; the only hard invariant is filed >= end_date.
        bad = warehouse_con.execute(
            "SELECT count(*) FROM fundamentals WHERE filed < end_date"
        ).fetchone()[0]
        assert bad == 0

    def test_concepts_within_allowlist(self, warehouse_con):
        from alpha_desk.data.sources import XBRL_CONCEPTS

        rows = warehouse_con.execute("SELECT DISTINCT concept FROM fundamentals").fetchall()
        found = {r[0] for r in rows}
        assert found.issubset(set(XBRL_CONCEPTS))


class TestProvenanceManifest:
    def test_manifest_hashes_match_files_on_disk(self, manifest):
        ok_items = [m for m in manifest["items"] if m["status"] == "ok" and "sha256" in m]
        assert len(ok_items) > 0
        # data/raw/ is gitignored (real fetched market data, not committed),
        # so a fresh checkout has the manifest but not the files it
        # describes -- skip rather than fail in that environment, matching
        # warehouse_con's pattern for the not-yet-built warehouse.
        if not (ROOT / ok_items[0]["path"]).exists():
            pytest.skip("data/raw/ not present -- run scripts/fetch_public_data.py first")
        # Spot-check every item rather than sampling -- the files are small
        # enough (data/raw is a few tens of MB total) that this is cheap.
        for item in ok_items:
            path = ROOT / item["path"]
            assert path.exists(), f"manifest references missing file {path}"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == item["sha256"], f"hash mismatch for {item['path']}"

    def test_no_unrecorded_failures_for_core_datasets(self, manifest):
        # prices and macro are load-bearing for every downstream module --
        # a silent partial fetch there should fail the build, not pass quietly.
        failed = [m for m in manifest["items"] if m["status"] == "failed"]
        failed_core = [m for m in failed if m["dataset"] in ("prices", "macro")]
        assert failed_core == [], f"unexpected core-dataset fetch failures: {failed_core}"

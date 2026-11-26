#!/usr/bin/env python3
"""Tests for scripts/itemcheck.py — the per-item difficulty audit."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECK = ROOT / "scripts" / "itemcheck.py"


def report(model: str, mode: str, rows: list[tuple[str, str, float]], blank: bool = False) -> dict:
    return {
        "model": model, "prompt_mode": mode, "items": len(rows), "elapsed_s": 1.0,
        "blank_responses": 0, "summary": [],
        "rows": [
            {"id": rid, "task": task, "ratio": ratio, "points": ratio, "max_points": 1.0,
             "blank": blank, "meta": {}, "checks": [{"label": "some-check", "ok": ratio >= 1.0,
                                                     "points": ratio}]}
            for rid, task, ratio in rows
        ],
    }


class TestItemCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.results = self.tmp / "results"
        self.results.mkdir()

    def write(self, name: str, data: dict) -> None:
        (self.results / name).write_text(json.dumps(data), encoding="utf-8")

    def run_check(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(CHECK), "--results", str(self.results), *extra],
                              capture_output=True, text=True, cwd=ROOT)

    def test_saturated_useful_and_broken(self):
        self.write("a.json", report("m1", "raw", [
            ("i-free", "writer", 1.0), ("i-mid", "writer", 0.5), ("i-hard", "writer", 0.0),
        ]))
        self.write("b.json", report("m1", "doctrine", [
            ("i-free", "writer", 1.0), ("i-mid", "writer", 0.6), ("i-hard", "writer", 0.0),
        ]))
        out = self.run_check().stdout
        self.assertIn("| i-free | writer | 2 | 1.000", out)
        self.assertIn("saturated", out)
        self.assertIn("broken-or-hard", out)
        self.assertIn("saturated=1", out)
        self.assertIn("useful=1", out)

    def test_unstable_when_spread_is_wide(self):
        self.write("a.json", report("m1", "raw", [("i", "writer", 0.1)]))
        self.write("b.json", report("m1", "doctrine", [("i", "writer", 1.0)]))
        self.assertIn("unstable", self.run_check().stdout)

    def test_single_observation_is_thin_not_useful(self):
        self.write("a.json", report("m1", "raw", [("i", "writer", 0.5)]))
        self.assertIn("thin", self.run_check().stdout)

    def test_always_blank_is_called_out(self):
        self.write("a.json", report("m1", "raw", [("i", "writer", 0.0)], blank=True))
        self.write("b.json", report("m1", "doctrine", [("i", "writer", 0.0)], blank=True))
        self.assertIn("all-blank", self.run_check().stdout)

    def test_flag_saturated_exits_nonzero(self):
        self.write("a.json", report("m1", "raw", [("i", "writer", 1.0)]))
        self.write("b.json", report("m1", "doctrine", [("i", "writer", 1.0)]))
        self.assertEqual(self.run_check("--flag", "saturated").returncode, 1)

    def test_flag_clean_dataset_exits_zero(self):
        self.write("a.json", report("m1", "raw", [("i", "writer", 0.5)]))
        self.write("b.json", report("m1", "doctrine", [("i", "writer", 0.7)]))
        self.assertEqual(self.run_check("--flag", "saturated").returncode, 0)

    def test_no_reports_is_an_error(self):
        res = self.run_check()
        self.assertEqual(res.returncode, 2)
        self.assertIn("no report files found", res.stderr)

    def test_committed_audit_is_reproducible(self):
        res = subprocess.run([sys.executable, str(CHECK), "--results", "results"],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        committed = (ROOT / "data" / "ITEM-AUDIT.md")
        if committed.exists():
            self.assertEqual(res.stdout, committed.read_text(encoding="utf-8"),
                             "data/ITEM-AUDIT.md is stale — regenerate with make audit")


if __name__ == "__main__":
    unittest.main()

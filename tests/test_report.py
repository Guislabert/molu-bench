#!/usr/bin/env python3
"""Tests for harness/report.py — the table must be generated, never typed."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "harness" / "report.py"


def fake_report(model: str, mode: str, scores: dict[str, float], items: int = 20) -> dict:
    rows, summary = [], []
    for task, score in scores.items():
        rows.append({"id": f"{task}-01", "task": task, "ratio": score, "points": score, "max_points": 1})
        summary.append({"task": task, "n": items // len(scores), "score": score,
                        "ci95": [max(0.0, score - 0.1), min(1.0, score + 0.1)],
                        "checks_passed": 1, "checks_total": 2})
    return {"model": model, "prompt_mode": mode, "items": items, "elapsed_s": 12.5,
            "blank_responses": 0, "summary": summary, "rows": rows}


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.results = self.tmp / "results"
        self.results.mkdir()

    def write(self, name: str, data) -> None:
        (self.results / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def run_report(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(REPORT), "--results", str(self.results),
                               "--out", "-", *extra],
                              capture_output=True, text=True, cwd=ROOT)

    def test_empty_results_dir_says_so_instead_of_printing_a_blank_table(self):
        res = self.run_report()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("No results yet", res.stdout)

    def test_table_has_one_row_per_report(self):
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5, "writer": 0.7}))
        self.write("b.json", fake_report("m1", "doctrine", {"insight": 0.6, "writer": 0.8}))
        out = self.run_report().stdout
        # counts the score table rows (which carry the item count), not the CI rows
        self.assertEqual(out.count("| m1 | raw | 20"), 1)
        self.assertEqual(out.count("| m1 | doctrine | 20"), 1)
        self.assertIn("**0.600**", out)

    def test_delta_section_compares_doctrine_to_raw(self):
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5}))
        self.write("b.json", fake_report("m1", "doctrine", {"insight": 0.7}))
        out = self.run_report().stdout
        self.assertIn("doctrine − raw", out)
        self.assertIn("+0.200", out)

    def test_no_delta_section_without_a_pair(self):
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5}))
        out = self.run_report().stdout
        self.assertNotIn("doctrine − raw", out)

    def test_small_runs_are_flagged(self):
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5}, items=3))
        self.assertIn("⚠", self.run_report().stdout)

    def test_non_report_files_are_skipped_not_crashed(self):
        (self.results / "notes.json").write_text('{"hello": "world"}', encoding="utf-8")
        (self.results / "broken.json").write_text("{not json", encoding="utf-8")
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5}))
        res = self.run_report()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("skipping notes.json", res.stderr)
        self.assertIn("skipping broken.json", res.stderr)
        self.assertIn("m1", res.stdout)

    def test_min_items_gate(self):
        self.write("a.json", fake_report("m1", "raw", {"insight": 0.5}, items=4))
        res = self.run_report("--min-items", "20")
        self.assertEqual(res.returncode, 1)
        self.assertIn("< --min-items 20", res.stderr)


class TestCommittedResults(unittest.TestCase):
    def test_committed_results_still_render(self):
        """The checked-in table must be reproducible from the checked-in reports."""
        res = subprocess.run([sys.executable, str(REPORT), "--results", "results", "--out", "-"],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("| model | prompt |", res.stdout)

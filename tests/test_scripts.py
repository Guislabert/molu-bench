#!/usr/bin/env python3
"""Tests for scripts/build_sft.py — the CPU-side half of the training path.

The GPU script (sft_lora.py) cannot run here; its data validation is reachable
through --dry-run and is covered by TestSftLoraDryRun below.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_sft.py"
TRAIN = ROOT / "scripts" / "sft_lora.py"

GOOD = [
    {"task": "insight", "prompt": "从自述里提取人群与阻碍。", "completion": "人群：二线城市行政；阻碍：报价没有锚点。"},
    {"module": "topic", "input": "给 2 个针对门店主的选题。", "output": "1. 门店主 → 曝光 → 拍摄清单\n2. 门店主 → 复购 → 回访话术"},
    {"task": "writer", "prompt": "写一段开头。", "completion": "上周三晚上九点，我数了 17 个扫码付款的人，只有 3 个抬头。"},
    {"task": "guard", "prompt": "审核并给最小修改。", "completion": "位置：三天见效 → 实质问题：效果承诺不可验证。"},
]


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run_build(units: pathlib.Path, out: pathlib.Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD), "--units", str(units), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


class TestBuildSft(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.units = self.tmp / "units.jsonl"
        self.out = self.tmp / "sft.jsonl"

    def test_aliases_and_counts(self):
        write_rows(self.units, GOOD)
        res = run_build(self.units, self.out, "--stats", "--allow-count-mismatch")
        self.assertEqual(res.returncode, 0, res.stderr)
        rows = [json.loads(l) for l in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["messages"][-1]["role"], "assistant")

    def test_bad_rows_dropped_with_reasons(self):
        bad = GOOD + [
            {"task": "writer", "prompt": "x", "completion": ""},                       # empty
            dict(GOOD[0]),                                                             # duplicate
            {"task": "nonsense", "prompt": "abc", "completion": "def" * 10},           # bad task
            {"prompt": "no task field", "completion": "y" * 40},                       # no task
        ]
        write_rows(self.units, bad)
        res = run_build(self.units, self.out, "--stats", "--allow-count-mismatch")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("dropped: 4", res.stdout)
        self.assertIn("unknown task", res.stdout)

    def test_count_drift_refuses_without_override(self):
        write_rows(self.units, GOOD)
        res = run_build(self.units, self.out)
        self.assertEqual(res.returncode, 2)
        self.assertFalse(self.out.exists(), "nothing should be written on a refused build")
        self.assertIn("Nothing written", res.stderr)

    def test_doctrine_mode_adds_system_turn(self):
        write_rows(self.units, GOOD)
        res = run_build(self.units, self.out, "--doctrine", "--allow-count-mismatch")
        self.assertEqual(res.returncode, 0, res.stderr)
        first = json.loads(self.out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertIn("molu module", first["messages"][0]["content"])

    def test_empty_input_is_an_error(self):
        self.units.write_text("", encoding="utf-8")
        res = run_build(self.units, self.out, "--allow-count-mismatch")
        self.assertNotEqual(res.returncode, 0)


class TestSftLoraDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.units = self.tmp / "units.jsonl"
        self.sft = self.tmp / "sft.jsonl"

    def test_dry_run_validates_and_exits_before_model_load(self):
        write_rows(self.units, GOOD)
        build = run_build(self.units, self.sft, "--allow-count-mismatch")
        self.assertEqual(build.returncode, 0, build.stderr)
        res = subprocess.run(
            [sys.executable, str(TRAIN), "--train", str(self.sft), "--out", str(self.tmp / "ad"),
             "--dry-run"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("validated 4 rows", res.stdout)
        self.assertIn("exiting before model load", res.stdout)

    def test_rejects_row_without_assistant_turn(self):
        self.sft.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n",
                            encoding="utf-8")
        res = subprocess.run(
            [sys.executable, str(TRAIN), "--train", str(self.sft), "--out", str(self.tmp / "ad"),
             "--dry-run"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("last message must be the assistant turn", res.stderr)


if __name__ == "__main__":
    unittest.main()

class TestPlotLoss(unittest.TestCase):
    PLOT = ROOT / "scripts" / "plot_loss.py"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def run_plot(self, path: pathlib.Path, out: pathlib.Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.PLOT), "--log", str(path), "--out", str(out)],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_trainer_state_log_history(self):
        state = self.tmp / "trainer_state.json"
        state.write_text(json.dumps({"log_history": [
            {"step": 10, "loss": 2.4}, {"step": 20, "loss": 1.8}, {"step": 30, "loss": 1.4},
        ]}), encoding="utf-8")
        out = self.tmp / "loss.svg"
        res = self.run_plot(state, out)
        self.assertEqual(res.returncode, 0, res.stderr)
        body = out.read_text(encoding="utf-8")
        self.assertIn("<polyline", body)
        self.assertEqual(body.count(","), body.count(","))  # svg is non-empty
        self.assertIn("loss 2.4000 → 1.4000", body)

    def test_jsonl_input(self):
        log = self.tmp / "log.jsonl"
        log.write_text('{"step":1,"loss":2.0}\n{"step":2,"loss":1.0}\n', encoding="utf-8")
        out = self.tmp / "loss.svg"
        res = self.run_plot(log, out)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("-50.0%", res.stdout)

    def test_plain_log_regex_fallback(self):
        log = self.tmp / "train.log"
        log.write_text("{'loss': 2.5, 'grad_norm': 1.1}\n{'loss': 1.5, 'grad_norm': 0.9}\n",
                       encoding="utf-8")
        res = self.run_plot(log, self.tmp / "loss.svg")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("2 points", res.stdout)

    def test_no_loss_is_an_error_not_an_empty_figure(self):
        cfg = ROOT / "configs" / "lora-qwen3-8b.json"
        res = self.run_plot(cfg, self.tmp / "loss.svg")
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse((self.tmp / "loss.svg").exists(), "must not write a figure with no data")

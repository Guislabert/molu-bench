#!/usr/bin/env python3
"""Tests for the scorer. No network, no model, no dependencies.

Run: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "harness"))

from evalkit import (  # noqa: E402
    Item,
    Prompter,
    bootstrap_ci,
    load_items,
    run_check,
    score_output,
    summarize,
    _units,
    StubModel,
    run,
)


class TestUnits(unittest.TestCase):
    def test_cjk_and_latin(self):
        self.assertEqual(_units("三个字 abc 12"), 3 + 2)

    def test_punctuation_excluded(self):
        self.assertEqual(_units("，。！？"), 0)


class TestChecks(unittest.TestCase):
    def test_contains(self):
        self.assertEqual(run_check({"kind": "contains", "value": "人群"}, "标注人群维度"), (1.0, True))

    def test_not_contains_passes_when_absent(self):
        self.assertEqual(run_check({"kind": "not_contains", "value": "涨粉"}, "只有事实"), (1.0, True))

    def test_none_of_fails_when_any_present(self):
        pts, ok = run_check({"kind": "none_of", "value": ["保证", "稳赚"]}, "我保证这有效")
        self.assertFalse(ok)
        self.assertEqual(pts, 0.0)

    def test_max_units_boundary(self):
        self.assertTrue(run_check({"kind": "max_units", "value": 3}, "一二三")[1])
        self.assertFalse(run_check({"kind": "max_units", "value": 3}, "一二三四")[1])

    def test_regex_multiline(self):
        _, ok = run_check({"kind": "regex", "value": r"^\s*[1-5][\.、]"}, "背景\n1. 第一条")
        self.assertTrue(ok)

    def test_line_count(self):
        _, ok = run_check({"kind": "line_count_at_least", "value": 2}, "一行\n\n二行\n")
        self.assertTrue(ok)

    def test_unknown_kind_is_a_hard_error(self):
        # A typo in a rubric must abort, not silently score zero.
        with self.assertRaises(SystemExit):
            run_check({"kind": "sounds_good"}, "任何输出")


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.item = Item(
            id="t-01",
            task="writer",
            prompt="x",
            checks=[
                {"kind": "contains", "value": "数字", "points": 2},
                {"kind": "not_contains", "value": "保证", "points": 1},
            ],
        )

    def test_points_and_ratio(self):
        row = score_output(self.item, "这里是数字，但保证有效")
        self.assertEqual(row["points"], 2.0)
        self.assertEqual(row["max_points"], 3.0)
        self.assertAlmostEqual(row["ratio"], 2 / 3)

    def test_blank_output_earns_nothing(self):
        # Regression: negative checks used to pass on an empty string, so a
        # silent model scored points for "not promising traffic".
        row = score_output(self.item, "   \n")
        self.assertEqual(row["points"], 0.0)
        self.assertTrue(row["blank"])
        self.assertTrue(all(not c["ok"] for c in row["checks"]))

    def test_full_marks(self):
        row = score_output(self.item, "数字")
        self.assertEqual(row["ratio"], 1.0)

    def test_per_check_detail_is_kept(self):
        row = score_output(self.item, "数字但保证有效")
        self.assertEqual(len(row["checks"]), 2)
        self.assertEqual([c["ok"] for c in row["checks"]], [True, False])


class TestBootstrap(unittest.TestCase):
    def test_deterministic_for_fixed_seed(self):
        vals = [0.1, 0.5, 0.9, 0.4, 0.7]
        self.assertEqual(bootstrap_ci(vals), bootstrap_ci(vals))

    def test_interval_brackets_the_mean(self):
        vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        lo, hi = bootstrap_ci(vals)
        self.assertLessEqual(lo, sum(vals) / len(vals))
        self.assertGreaterEqual(hi, sum(vals) / len(vals))

    def test_single_sample_degrades_gracefully(self):
        lo, hi = bootstrap_ci([0.42])
        self.assertEqual((lo, hi), (0.42, 0.42))


class TestSummary(unittest.TestCase):
    def test_groups_by_task(self):
        rows = [
            {"task": "writer", "ratio": 1.0, "checks": [{"ok": True}]},
            {"task": "writer", "ratio": 0.0, "checks": [{"ok": False}]},
            {"task": "guard", "ratio": 0.5, "checks": [{"ok": True}]},
        ]
        summary = {s["task"]: s for s in summarize(rows)}
        self.assertEqual(summary["writer"]["n"], 2)
        self.assertEqual(summary["writer"]["score"], 0.5)
        self.assertEqual(summary["writer"]["checks_passed"], 1)
        self.assertEqual(summary["guard"]["n"], 1)


class TestPrompter(unittest.TestCase):
    def test_raw_is_identity(self):
        item = Item(id="i-1", task="insight", prompt="原始任务")
        self.assertEqual(Prompter("raw")(item), "原始任务")

    def test_doctrine_prepends_module_prompt(self):
        item = Item(id="w-1", task="writer", prompt="原始任务")
        out = Prompter("doctrine")(item)
        self.assertIn("molu module ⑦ writer", out)
        self.assertTrue(out.endswith("原始任务"))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(SystemExit):
            Prompter("freestyle")

    def test_missing_task_pack_rejected(self):
        with self.assertRaises(SystemExit):
            Prompter("doctrine", prompts_dir="/nonexistent")


class TestRealDevSet(unittest.TestCase):
    """Guards the shipped data, not just the scorer."""

    def test_every_item_scores_and_no_item_is_empty(self):
        items = load_items(str(pathlib.Path(__file__).resolve().parent.parent / "data" / "dev.jsonl"))
        self.assertGreaterEqual(len(items), 20)
        for item in items:
            self.assertTrue(item.checks, f"{item.id} has no checks")
            self.assertGreater(item.max_points, 0)
            row = score_output(item, "")
            self.assertEqual(row["points"], 0.0, f"{item.id}: empty output should score 0")
            self.assertTrue(row["blank"])

    def test_task_coverage(self):
        items = load_items(str(pathlib.Path(__file__).resolve().parent.parent / "data" / "dev.jsonl"))
        self.assertEqual({i.task for i in items}, {"insight", "topic", "writer", "guard"})

    def test_doctrine_mode_runs_end_to_end(self):
        items = load_items(str(pathlib.Path(__file__).resolve().parent.parent / "data" / "dev.jsonl"))
        report = run(StubModel(), items[:4], verbose=False, prompter=Prompter("doctrine"))
        self.assertEqual(report["prompt_mode"], "doctrine")
        self.assertEqual(len(report["rows"]), 4)


if __name__ == "__main__":
    unittest.main()

class TestRemoteModelRobustness(unittest.TestCase):
    """Provider behaviours that used to crash or silently inflate scores."""

    def make(self, payload):
        from evalkit import OpenAICompatModel
        return OpenAICompatModel(
            name="fake", base_url="http://fake/v1", model="fake-1",
            transport=lambda url, body, headers, timeout: payload,
        )

    def test_reasoning_only_response_becomes_blank_not_a_free_pass(self):
        model = self.make({"choices": [{"message": {"content": None,
                                                    "reasoning_content": "thinking..."},
                                        "finish_reason": "length"}]})
        out = model.generate("x")
        self.assertEqual(out, "", "reasoning text must not be used as the answer")
        self.assertTrue(model.last_meta["reasoning_only"])
        self.assertEqual(model.last_meta["finish_reason"], "length")

    def test_null_content_does_not_crash_scoring(self):
        model = self.make({"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
        item = Item(id="i", task="writer", prompt="p",
                    checks=[{"kind": "none_of", "value": ["保证"], "points": 1}])
        row = score_output(item, model.generate("p"))
        self.assertEqual(row["points"], 0.0)
        self.assertTrue(row["blank"])

    def test_normal_response_records_meta(self):
        model = self.make({"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
                           "model": "fake-1"})
        self.assertEqual(model.generate("x"), "答案")
        self.assertEqual(model.last_meta["finish_reason"], "stop")
        self.assertFalse(model.last_meta["reasoning_only"])
        self.assertEqual(model.last_meta["model_returned"], "fake-1")

    def test_transport_error_is_raised_not_swallowed(self):
        from evalkit import OpenAICompatModel
        model = OpenAICompatModel(name="fake", base_url="http://f/v1", model="m",
                                  transport=lambda *a: (_ for _ in ()).throw(SystemExit("HTTP 404: body")))
        with self.assertRaises(SystemExit):
            model.generate("x")


class TestRunCounters(unittest.TestCase):
    def test_truncation_and_reasoning_counters(self):
        from evalkit import OpenAICompatModel
        payloads = [
            {"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": None, "reasoning_content": "t"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "答"}, "finish_reason": "length"}]},
        ]
        it = iter(payloads)
        model = OpenAICompatModel(name="fake", base_url="http://f/v1", model="m",
                                  transport=lambda *a: next(it))
        item = Item(id="i", task="writer", prompt="p", checks=[{"kind": "min_units", "value": 1}])
        report = run(model, [item, item, item], verbose=False)
        self.assertEqual(report["blank_responses"], 1)
        self.assertEqual(report["reasoning_only_responses"], 1)
        self.assertEqual(report["truncated_responses"], 2)

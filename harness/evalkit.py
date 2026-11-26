"""molu-bench evaluation kit — stdlib only, no API keys needed for the smoke path.

Design constraints, in order of importance:

1. Reproducible: deterministic rubrics, stdlib only, pinned bootstrap seed.
2. Honest: the default scorer is deterministic and auditable. LLM-as-judge is
   opt-in, recorded separately, and its bias is documented instead of hidden.
3. Pluggable: any OpenAI-compatible endpoint works (vLLM, SGLang, Ollama,
   lmdeploy, OpenRouter, a local llama.cpp server).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class Item:
    id: str
    task: str
    prompt: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def max_points(self) -> float:
        return sum(float(c.get("points", 1)) for c in self.checks)


def load_items(path: str) -> list[Item]:
    items: list[Item] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            items.append(
                Item(
                    id=raw["id"],
                    task=raw["task"],
                    prompt=raw["prompt"],
                    checks=raw.get("checks", []),
                    tags=raw.get("tags", []),
                )
            )
    if not items:
        raise SystemExit(f"{path}: no items")
    return items


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class Model:
    name = "model"

    def generate(self, prompt: str, max_tokens: int = 800) -> str:  # pragma: no cover
        raise NotImplementedError


class StubModel(Model):
    """Deterministic offline stand-in. Used by `--model stub` and by CI."""

    name = "stub"

    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        # Echo a structured-but-intentionally-weak answer so the scorer's
        # failure paths (missing sections, banned words, over-length) get
        # exercised instead of silently passing.
        return (
            "1. 面向 25-35 岁一线城市白领\n"
            "2. 面向 转行人群\n"
            "3. 面向 小城市父母\n\n"
            "这一篇绝对会火爆全网，保证涨粉。\n"
        )


class OpenAICompatModel(Model):
    """Any /v1/chat/completions endpoint: vLLM, SGLang, Ollama, OpenRouter…"""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: int = 180,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("MOLU_BENCH_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# scoring — deterministic checks only
# --------------------------------------------------------------------------

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _units(text: str) -> int:
    """Chinese chars count as 1 unit each; latin words count as 1 unit each."""
    return len(_CJK.findall(text)) + len(_WORD.findall(text))


def run_check(check: dict[str, Any], output: str) -> tuple[float, bool]:
    kind = check["kind"]
    pts = float(check.get("points", 1))
    if kind == "contains":
        hit = check["value"] in output
    elif kind == "not_contains":
        hit = check["value"] not in output
    elif kind == "regex":
        hit = re.search(check["value"], output, re.MULTILINE) is not None
    elif kind == "max_units":
        hit = _units(output) <= int(check["value"])
    elif kind == "min_units":
        hit = _units(output) >= int(check["value"])
    elif kind == "line_count_at_least":
        lines = [l for l in output.splitlines() if l.strip()]
        hit = len(lines) >= int(check["value"])
    elif kind == "all_of":
        hit = all(v in output for v in check["value"])
    elif kind == "none_of":
        hit = not any(v in output for v in check["value"])
    else:  # pragma: no cover - guard against silent typo'd rubrics
        raise SystemExit(f"unknown check kind: {kind}")
    return (pts if hit else 0.0), hit


def score_output(item: Item, output: str) -> dict[str, Any]:
    total = 0.0
    detail = []
    for check in item.checks:
        got, ok = run_check(check, output)
        total += got
        detail.append({"label": check.get("label", check["kind"]), "ok": ok, "points": got})
    return {
        "id": item.id,
        "task": item.task,
        "points": total,
        "max_points": item.max_points,
        "ratio": (total / item.max_points) if item.max_points else 0.0,
        "checks": detail,
        "output_units": _units(output),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def bootstrap_ci(values: list[float], resamples: int = 1000, seed: int = 1723) -> tuple[float, float]:
    if len(values) < 2:
        return (statistics.fmean(values) if values else math.nan,) * 2
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples)
    ]
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples) - 1]
    return lo, hi


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row["ratio"])
    out = []
    for task in sorted(by_task):
        vals = by_task[task]
        lo, hi = bootstrap_ci(vals)
        out.append(
            {
                "task": task,
                "n": len(vals),
                "score": round(statistics.fmean(vals), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "checks_passed": sum(1 for r in rows if r["task"] == task for c in r["checks"] if c["ok"]),
                "checks_total": sum(len(r["checks"]) for r in rows if r["task"] == task),
            }
        )
    return out


def run(model: Model, items: list[Item], max_tokens: int = 800, verbose: bool = True) -> dict[str, Any]:
    rows = []
    started = time.time()
    for item in items:
        output = model.generate(item.prompt, max_tokens=max_tokens)
        row = score_output(item, output)
        row["output"] = output
        rows.append(row)
        if verbose:
            print(f"{row['id']:>14}  {row['points']:>4.1f}/{row['max_points']:<4.0f}", file=sys.stderr)
    elapsed = time.time() - started
    return {
        "model": model.name,
        "items": len(items),
        "elapsed_s": round(elapsed, 2),
        "summary": summarize(rows),
        "rows": rows,
    }


def print_table(report: dict[str, Any]) -> None:
    print(f"\nmodel: {report['model']}   items: {report['items']}   wall: {report['elapsed_s']}s\n")
    print(f"{'task':<10} {'n':>3} {'score':>7} {'95% CI':>16} {'checks':>10}")
    for row in report["summary"]:
        ci = f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"
        checks = f"{row['checks_passed']}/{row['checks_total']}"
        print(f"{row['task']:<10} {row['n']:>3} {row['score']:>7.3f} {ci:>16} {checks:>10}")
    overall = [r["ratio"] for r in report["rows"]]
    lo, hi = bootstrap_ci(overall)
    print(f"\n{'overall':<10} {len(overall):>3} {statistics.fmean(overall):>7.3f} "
          f"{'[' + f'{lo:.3f}, {hi:.3f}' + ']':>16}")
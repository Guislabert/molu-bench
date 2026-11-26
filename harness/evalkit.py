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
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.error
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


def _http_transport(url: str, body: bytes, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        # Include the body: "HTTP 404" alone sends people looking in the wrong
        # place, and providers put the actual reason in that body.
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise SystemExit(f"{url} -> HTTP {exc.code}: {detail}") from exc


class OpenAICompatModel(Model):
    """Any /v1/chat/completions endpoint: vLLM, SGLang, Ollama, OpenRouter…

    Two provider behaviours this has to survive, both observed in the wild:

    1. `content: null` — a reasoning model can spend the whole max_tokens budget
       on reasoning_content and return no answer. That is a blank response, it
       is recorded as one, and the reasoning text is NOT substituted as the
       answer (substituting it would measure the wrong string and quietly
       inflate the score).
    2. `finish_reason: "length"` — truncation is recorded per row so a low score
       can be traced to the budget instead of being read as a capability gap.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: int = 180,
        transport=None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("MOLU_BENCH_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.transport = transport or _http_transport
        self.last_meta: dict[str, Any] = {}

    def _payload(self, prompt: str, max_tokens: int) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self.transport(f"{self.base_url}/chat/completions", body, headers, self.timeout)

    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        payload = self._payload(prompt, max_tokens)
        choices = payload.get("choices") or [{}]
        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        has_reasoning = bool(message.get("reasoning_content"))
        self.last_meta = {
            "finish_reason": choice.get("finish_reason"),
            "reasoning_only": has_reasoning and not text,
            "model_returned": payload.get("model"),
        }
        return text


# --------------------------------------------------------------------------
# prompt conditions
# --------------------------------------------------------------------------


class Prompter:
    """Two experimental conditions, both fixed in-repo before any run.

    raw       - the item prompt alone: what an off-the-shelf assistant sees.
    doctrine  - the molu module prompt prepended: what a Skill-equipped agent
                sees. Loaded from harness/prompts/<task>.md.

    Fixing both in version control is the point: without a recorded prompt
    condition, "fine-tuning beat prompting" is not a comparable claim, and a
    prompt tuned after seeing results is unrecorded cheating.
    """

    def __init__(self, mode: str = "raw", prompts_dir: str = "harness/prompts") -> None:
        if mode not in {"raw", "doctrine"}:
            raise SystemExit(f"unknown prompt mode: {mode}")
        self.mode = mode
        self.doctrine: dict[str, str] = {}
        if mode == "doctrine":
            base = pathlib.Path(prompts_dir)
            for task in ("insight", "topic", "writer", "guard"):
                f = base / f"{task}.md"
                if not f.exists():
                    raise SystemExit(f"missing doctrine prompt: {f}")
                self.doctrine[task] = f.read_text(encoding="utf-8").strip()

    def __call__(self, item: Item) -> str:
        if self.mode == "raw":
            return item.prompt
        return f"{self.doctrine[item.task]}\n\n---\n\n任务 / Task\n\n{item.prompt}"


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
    blank = not output.strip()
    if blank:
        # A blank response must not collect points from negative checks: an
        # item that says nothing "does not promise traffic" and "does not
        # fabricate a case" for free. Zeroing every check and flagging the row
        # keeps that failure visible instead of profitable.
        return {
            "id": item.id,
            "task": item.task,
            "points": 0.0,
            "max_points": item.max_points,
            "ratio": 0.0,
            "blank": True,
            "checks": [
                {"label": c.get("label", c["kind"]), "ok": False, "points": 0.0,
                 "note": "blank output"}
                for c in item.checks
            ],
            "output_units": 0,
        }
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
        "blank": False,
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


def run(
    model: Model,
    items: list[Item],
    max_tokens: int = 800,
    verbose: bool = True,
    prompter: Prompter | None = None,
) -> dict[str, Any]:
    prompter = prompter or Prompter("raw")
    rows = []
    started = time.time()
    for item in items:
        output = model.generate(prompter(item), max_tokens=max_tokens)
        row = score_output(item, output)
        row["output"] = output
        meta = getattr(model, "last_meta", None)
        if meta:
            row["meta"] = dict(meta)
        rows.append(row)
        if verbose:
            print(f"{row['id']:>14}  {row['points']:>4.1f}/{row['max_points']:<4.0f}", file=sys.stderr)
    elapsed = time.time() - started
    return {
        "model": model.name,
        "prompt_mode": prompter.mode,
        "items": len(items),
        "elapsed_s": round(elapsed, 2),
        "blank_responses": sum(1 for r in rows if r.get("blank")),
        "truncated_responses": sum(1 for r in rows if (r.get("meta") or {}).get("finish_reason") == "length"),
        "reasoning_only_responses": sum(1 for r in rows if (r.get("meta") or {}).get("reasoning_only")),
        "summary": summarize(rows),
        "rows": rows,
    }


def print_table(report: dict[str, Any]) -> None:
    print(f"\nmodel: {report['model']}   prompt: {report.get('prompt_mode', 'raw')}   "
          f"items: {report['items']}   blank: {report.get('blank_responses', 0)}   "
          f"wall: {report['elapsed_s']}s\n")
    print(f"{'task':<10} {'n':>3} {'score':>7} {'95% CI':>16} {'checks':>10}")
    for row in report["summary"]:
        ci = f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"
        checks = f"{row['checks_passed']}/{row['checks_total']}"
        print(f"{row['task']:<10} {row['n']:>3} {row['score']:>7.3f} {ci:>16} {checks:>10}")
    overall = [r["ratio"] for r in report["rows"]]
    lo, hi = bootstrap_ci(overall)
    print(f"\n{'overall':<10} {len(overall):>3} {statistics.fmean(overall):>7.3f} "
          f"{'[' + f'{lo:.3f}, {hi:.3f}' + ']':>16}")
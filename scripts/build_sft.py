#!/usr/bin/env python3
"""Convert locally held content units into chat-format SFT rows.

Runs on CPU, stdlib only, so it can be reviewed and tested without a GPU. The
input never leaves the machine that holds the corpus — nothing here uploads,
and the output is expected to stay outside this repository (see data/CARD.md:
the units are derived from third-party articles and are not redistributable).

Input, one JSON object per line:

    {"task": "writer", "prompt": "...", "completion": "..."}
    {"module": "writer", "input": "...", "output": "..."}      # aliases accepted

Output, one JSON object per line, chat format:

    {"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:

    python3 scripts/build_sft.py --units units.jsonl --out sft.jsonl --stats
    python3 scripts/build_sft.py --units units.jsonl --out sft.jsonl --doctrine
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

TASKS = ("insight", "topic", "writer", "guard")
EXPECTED_UNITS = 16_227
ALIASES = {
    "task": ("task", "module", "skill"),
    "prompt": ("prompt", "input", "instruction", "question"),
    "completion": ("completion", "output", "answer", "response"),
}


def pick(row: dict, field: str) -> str | None:
    for key in ALIASES[field]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    return None


def est_tokens(text: str) -> int:
    """zh-weighted estimate: 1 token ≈ 1.4 chars. Crude, only for planning."""
    return int(len(text) / 1.4)


def doctrine_prompt(task: str, prompts_dir: pathlib.Path) -> str | None:
    f = prompts_dir / f"{task}.md"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doctrine", action="store_true",
                    help="prepend harness/prompts/<task>.md as the system message")
    ap.add_argument("--prompts-dir", default="harness/prompts")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--allow-unknown-task", action="store_true",
                    help="keep rows whose task is outside the 4 benchmark tasks")
    ap.add_argument("--allow-count-mismatch", action="store_true",
                    help=f"do not fail when the row count differs from ~{EXPECTED_UNITS}")
    ap.add_argument("--min-chars", type=int, default=20)
    args = ap.parse_args()

    src = pathlib.Path(args.units)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    prompts_dir = pathlib.Path(args.prompts_dir)

    rows, seen, dropped = [], set(), []
    per_task: dict[str, int] = {}
    tokens = 0

    for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            dropped.append((lineno, "invalid json"))
            continue
        if not isinstance(raw, dict):
            dropped.append((lineno, "not an object"))
            continue

        task = pick(raw, "task")
        prompt = pick(raw, "prompt")
        completion = pick(raw, "completion")
        if not task:
            dropped.append((lineno, "no task/module field"))
            continue
        if not prompt or not completion:
            dropped.append((lineno, "empty prompt or completion"))
            continue
        if task not in TASKS and not args.allow_unknown_task:
            dropped.append((lineno, f"unknown task {task!r} (use --allow-unknown-task)"))
            continue
        if len(completion) < args.min_chars:
            dropped.append((lineno, f"completion shorter than {args.min_chars} chars"))
            continue

        key = (task, prompt, completion)
        if key in seen:
            dropped.append((lineno, "duplicate row"))
            continue
        seen.add(key)

        system = doctrine_prompt(task, prompts_dir) if args.doctrine else None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        rows.append({"messages": messages})
        per_task[task] = per_task.get(task, 0) + 1
        tokens += est_tokens(prompt) + est_tokens(completion) + (est_tokens(system) if system else 0)

    if not rows:
        raise SystemExit("no usable rows; see the drop reasons above")

    # Refuse before writing: a build that silently produces the wrong number of
    # rows makes every GPU-hour figure downstream wrong.
    drift = abs(len(rows) - EXPECTED_UNITS) / EXPECTED_UNITS
    if drift > 0.05 and not args.allow_count_mismatch:
        print(
            f"{len(rows)} rows vs the {EXPECTED_UNITS} assumed by compute/budget.py "
            f"({drift:.0%} off). Nothing written.\n"
            f"Check whether the difference is intended, then re-run with "
            f"--allow-count-mismatch.",
            file=sys.stderr,
        )
        return 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    if args.stats:
        print(f"wrote {len(rows)} rows -> {out}")
        print(f"  tasks: {', '.join(f'{k}={v}' for k, v in sorted(per_task.items()))}")
        print(f"  est tokens/epoch: {tokens / 1e6:.2f}M  (3 epochs: {tokens * 3 / 1e6:.2f}M)")
        print(f"  dropped: {len(dropped)}")
        for lineno, why in dropped[:10]:
            print(f"    line {lineno}: {why}")
        if len(dropped) > 10:
            print(f"    ... {len(dropped) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
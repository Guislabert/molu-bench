#!/usr/bin/env python3
"""CLI for molu-bench.

    python3 harness/cli.py --model stub --data data/dev.jsonl
    python3 harness/cli.py --model local --base-url http://localhost:8000/v1 \
        --model-name Qwen3-8B --data data/dev.jsonl --out results/local-qwen3-8b.json

No third-party dependencies: the point is that a reviewer can re-run this.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evalkit import (  # noqa: E402
    OpenAICompatModel,
    StubModel,
    load_items,
    print_table,
    run,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="molu-bench harness")
    ap.add_argument("--data", default="data/dev.jsonl")
    ap.add_argument("--model", default="stub", choices=["stub", "local", "remote"],
                    help="stub = offline smoke test, local/remote = OpenAI-compatible endpoint")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model-name", default="local-model")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--tasks", default="", help="comma separated task filter, e.g. insight,topic")
    ap.add_argument("--out", default="", help="write the full JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    items = load_items(args.data)
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",") if t.strip()}
        items = [i for i in items if i.task in keep]
        if not items:
            raise SystemExit(f"no items for tasks: {sorted(keep)}")

    if args.model == "stub":
        model = StubModel()
    else:
        model = OpenAICompatModel(name=args.model_name, base_url=args.base_url)

    report = run(model, items, max_tokens=args.max_tokens, verbose=not args.quiet)
    print_table(report)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
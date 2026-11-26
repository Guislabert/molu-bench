#!/usr/bin/env python3
"""Per-item difficulty audit across every committed report.

Purpose: a 20-item dev set that a small model nearly aces cannot measure
anything. Before expanding to 400 items, find *which* items are already
worthless (everyone passes), which are broken (everyone fails — usually the
rubric, not the model), and which are the only ones doing work.

    python3 scripts/itemcheck.py --results results
    python3 scripts/itemcheck.py --results results --out data/ITEM-AUDIT.md
    python3 scripts/itemcheck.py --results results --flag saturated   # exit 1 if any
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

SATURATED = 0.95   # mean ratio at or above this across runs: the item is free
BROKEN = 0.0       # mean ratio at 0 across runs: no model ever passed it
UNSTABLE = 0.45    # spread this wide means one run cannot be trusted
THIN = 2           # fewer observations than this is not evidence


def collect(results_dir: pathlib.Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "rows" not in data:
            continue
        for row in data["rows"]:
            rec = items.setdefault(row["id"], {
                "id": row["id"], "task": row["task"], "runs": [], "blank": 0,
                "truncated": 0, "failing_checks": {},
            })
            rec["runs"].append({
                "run": path.stem, "model": data.get("model", "?"),
                "mode": data.get("prompt_mode", "raw"), "ratio": row["ratio"],
            })
            if row.get("blank"):
                rec["blank"] += 1
            if (row.get("meta") or {}).get("finish_reason") == "length":
                rec["truncated"] += 1
            for check in row["checks"]:
                if not check["ok"]:
                    label = check["label"]
                    rec["failing_checks"][label] = rec["failing_checks"].get(label, 0) + 1
    return items


def verdict(rec: dict) -> str:
    ratios = [r["ratio"] for r in rec["runs"]]
    if len(ratios) < THIN:
        return "thin"
    mean = statistics.fmean(ratios)
    spread = max(ratios) - min(ratios)
    if rec["blank"] == len(ratios):
        return "all-blank"
    if mean <= BROKEN:
        return "broken-or-hard"
    if mean >= SATURATED:
        return "saturated"
    if spread >= UNSTABLE:
        return "unstable"
    return "useful"


def render(items: dict[str, dict]) -> str:
    rows = sorted(items.values(), key=lambda r: -statistics.fmean([x["ratio"] for x in r["runs"]]))
    by_verdict: dict[str, int] = {}
    for r in rows:
        v = verdict(r)
        by_verdict[v] = by_verdict.get(v, 0) + 1

    out = ["| item | task | runs | mean | min | max | blank | trunc | verdict | most-failed check |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ratios = [x["ratio"] for x in r["runs"]]
        top_fail = max(r["failing_checks"].items(), key=lambda kv: kv[1])[0] if r["failing_checks"] else "–"
        out.append(
            f"| {r['id']} | {r['task']} | {len(ratios)} | {statistics.fmean(ratios):.3f} | "
            f"{min(ratios):.2f} | {max(ratios):.2f} | {r['blank']} | {r['truncated']} | "
            f"{verdict(r)} | {top_fail} |"
        )
    out.append("")
    counts = ", ".join(f"{k}={v}" for k, v in sorted(by_verdict.items()))
    out.append(f"**{len(rows)} items observed** — {counts}")
    out.append("")
    out.append("Reading: `saturated` items cost GPU-hours and buy nothing; `broken-or-hard` items are "
               "usually a rubric bug (check the failing check name before blaming the model); "
               "`unstable` items need a larger n before they mean anything.")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="-", help="'-' prints to stdout")
    ap.add_argument("--flag", choices=["saturated", "broken", "unstable", "any"], default=None,
                    help="exit 1 when items of this kind exist")
    args = ap.parse_args()

    items = collect(pathlib.Path(args.results))
    if not items:
        print("no report files found; run harness/cli.py --out results/<name>.json first",
              file=sys.stderr)
        return 2

    body = render(items)
    if args.out == "-":
        print(body, end="")
    else:
        pathlib.Path(args.out).write_text(body, encoding="utf-8")
        print(f"wrote {args.out}")

    if args.flag:
        bad = [r for r in items.values() if verdict(r) in
               ({"saturated"} if args.flag == "saturated"
                else {"broken-or-hard"} if args.flag == "broken"
                else {"unstable"} if args.flag == "unstable"
                else {"saturated", "broken-or-hard", "unstable"})]
        if bad:
            print(f"{len(bad)} item(s) flagged as {args.flag}: "
                  f"{', '.join(r['id'] for r in bad[:8])}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
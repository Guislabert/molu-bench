#!/usr/bin/env python3
"""Turn a training log into an SVG loss curve. Stdlib only, no matplotlib.

Why not matplotlib: a loss curve is the one figure a grant reviewer will look
for, and it should be regenerable on a machine with nothing installed. This
emits a small SVG you can commit or embed.

Accepted inputs:

    trainer_state.json        HuggingFace Trainer output (uses .log_history)
    something.jsonl           one {"step": N, "loss": X} per line
    something.log             lines containing "loss": X (regex fallback)

    python3 scripts/plot_loss.py --log adapter-seed0/trainer_state.json \
        --out results/loss-qwen3-8b-seed0.svg --title "Qwen3-8B LoRA · seed 0"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

W, H = 760, 380
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 50, 55
LOSS_RE = re.compile(r"""['"]?loss['"]?\s*[:=]\s*([0-9]*\.?[0-9]+)""")


def parse(path: pathlib.Path) -> list[tuple[float, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    points: list[tuple[float, float]] = []

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            history = data.get("log_history") or []
            for i, row in enumerate(history):
                if isinstance(row, dict) and isinstance(row.get("loss"), (int, float)):
                    points.append((float(row.get("step", i)), float(row["loss"])))
            if points:
                return points

    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and isinstance(row.get("loss"), (int, float)):
                points.append((float(row.get("step", i)), float(row["loss"])))
                continue
        m = LOSS_RE.search(line)
        if m:
            try:
                points.append((float(i), float(m.group(1))))
            except ValueError:
                continue
    return points


def svg(points: list[tuple[float, float]], title: str) -> str:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    span = (y1 - y0) or 1.0
    y0, y1 = y0 - span * 0.08, y1 + span * 0.08

    def sx(x: float) -> float:
        return PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

    def sy(y: float) -> float:
        return H - PAD_B - (y - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    # four horizontal gridlines with labels
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + (y1 - y0) * frac
        yy = sy(y)
        grid.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{W - PAD_R}" y2="{yy:.1f}" '
                    f'stroke="#e6e6e6" stroke-width="1"/>')
        grid.append(f'<text x="{PAD_L - 10}" y="{yy + 4:.1f}" text-anchor="end" '
                    f'font-family="ui-monospace,monospace" font-size="11" fill="#666">{y:.4f}</text>')
    for frac in (0.0, 0.5, 1.0):
        x = x0 + (x1 - x0) * frac
        grid.append(f'<text x="{sx(x):.1f}" y="{H - PAD_B + 20}" text-anchor="middle" '
                    f'font-family="ui-monospace,monospace" font-size="11" fill="#666">step {x:.0f}</text>')

    last = points[-1][1]
    first = points[0][1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  <text x="{PAD_L}" y="26" font-family="ui-sans-serif,system-ui" font-size="15" fill="#111">{title}</text>
  <text x="{PAD_L}" y="44" font-family="ui-monospace,monospace" font-size="11" fill="#777">loss {first:.4f} → {last:.4f} over {len(points)} logged steps</text>
  {chr(10).join("  " + g for g in grid)}
  <polyline fill="none" stroke="#1a73e8" stroke-width="2" points="{poly}"/>
  <circle cx="{sx(points[-1][0]):.1f}" cy="{sy(points[-1][1]):.1f}" r="3.5" fill="#1a73e8"/>
  <text x="{W - PAD_R}" y="{H - 14}" text-anchor="end" font-family="ui-monospace,monospace" font-size="10" fill="#999">molu-bench · scripts/plot_loss.py</text>
</svg>
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="training loss")
    args = ap.parse_args()

    src = pathlib.Path(args.log)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    points = parse(src)
    if not points:
        raise SystemExit(
            "no loss values found. Pass a trainer_state.json (log_history), or a "
            "jsonl/log containing {'loss': ...} entries."
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg(points, args.title), encoding="utf-8")
    first, last = points[0][1], points[-1][1]
    drop = (last - first) / first * 100 if first else 0.0
    print(f"{len(points)} points  loss {first:.4f} -> {last:.4f}  ({drop:+.1f}%)  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
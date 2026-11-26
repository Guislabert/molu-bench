#!/usr/bin/env python3
"""GPU-hour accounting for molu-bench. Every number here is computed from the
assumption block below, so a reviewer can change one constant and see the bill
move — no magic totals.

    python3 compute/budget.py            # all scenarios
    python3 compute/budget.py --scenario full --json

Nothing in this file is a measured benchmark yet. Assumptions are labelled
A(ssumption); once real runs land, flip them to M(easured) in ASSUMPTIONS and
the provenance column changes with them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------
# assumptions — change these and the bill changes
# --------------------------------------------------------------------------

A = {
    # corpus / dataset shape (from the molu corpus audit, see data/CARD.md)
    "articles": 899,
    "chars": 2_130_000,
    "units": 16_227,
    "pairs_per_unit": 1.0,          # one SFT pair (prompt, completion) per unit
    "tokens_per_char": 0.72,        # zh: ~1 token / 1.4 chars
    "prompt_tokens": 130,           # A: instruction + excerpt heading
    "completion_tokens": 290,       # A: module-shaped answer, 300-500 zh chars
    "epochs": 3,
    # throughput, A100-80G, seq 4096, bf16 unless noted
    "tok_s": {
        "8b_lora": 3_500,           # A: LoRA r=16, vLLM-style packing
        "14b_lora": 2_000,
        "32b_lora": 1_100,
        "8b_full": 1_200,
        "gen_8b": 2_800,            # A: vLLM, batch 32, generation only
        "gen_32b": 900,
    },
    # evaluation shape
    "eval_items": 400,              # 4 tasks x 100
    "eval_out_tokens": 600,
    "judges": 3,                    # local judges, not API
    "judge_tokens_per_item": 1_200,
    "seeds": 3,
    "buffer": 1.5,                  # retries, failed jobs, cluster loss
}

# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------


def tokens_per_epoch() -> float:
    pairs = A["units"] * A["pairs_per_unit"]
    per_pair = A["prompt_tokens"] + A["completion_tokens"]
    return pairs * per_pair


@dataclass
class Line:
    name: str
    runs: int
    hours_per_run: float
    note: str

    @property
    def hours(self) -> float:
        return round(self.runs * self.hours_per_run, 2)


def hours(tokens: float, tok_s: float) -> float:
    return tokens / tok_s / 3600


def finetune_lines() -> list[Line]:
    tpe = tokens_per_epoch()
    out = []
    for label, key, seeds in (
        ("SFT Qwen3-8B · LoRA", "8b_lora", A["seeds"]),
        ("SFT Qwen3-14B · LoRA", "14b_lora", A["seeds"]),
        ("SFT Qwen3-32B · LoRA", "32b_lora", A["seeds"]),
        ("SFT Qwen3-8B · full FT (baseline)", "8b_full", 2),
    ):
        out.append(
            Line(
                name=label,
                runs=seeds * A["epochs"] // A["epochs"],  # runs = seeds
                hours_per_run=hours(tpe * A["epochs"], A["tok_s"][key]),
                note=f"{A['epochs']} epochs x {tpe / 1e6:.1f}M tok @ {A['tok_s'][key]:,} tok/s",
            )
        )
    return out


def ablation_lines() -> list[Line]:
    tpe = tokens_per_epoch()
    out = []
    for frac, tag in ((0.25, "25%"), (0.5, "50%")):
        for key in ("8b_lora", "14b_lora", "32b_lora"):
            out.append(
                Line(
                    name=f"ablation · data {tag} · {key.split('_')[0].upper()}",
                    runs=1,
                    hours_per_run=hours(tpe * A["epochs"] * frac, A["tok_s"][key]),
                    note=f"{tag} of corpus, {A['epochs']} epochs",
                )
            )
    return out


def eval_lines() -> list[Line]:
    per_sweep = A["eval_items"] * A["eval_out_tokens"]
    judge = A["eval_items"] * A["judge_tokens_per_item"]
    models = 4 + len(finetune_lines()) * A["seeds"]  # 4 baselines + finetuned
    return [
        Line(
            name=f"generation sweep × {models} model configs",
            runs=models,
            hours_per_run=hours(per_sweep, A["tok_s"]["gen_8b"]),
            note=f"{A['eval_items']} items x {A['eval_out_tokens']} tok @ vLLM",
        ),
        Line(
            name=f"LLM-judge pass × {A['judges']} judges",
            runs=A["judges"],
            hours_per_run=hours(judge, A["tok_s"]["gen_32b"]),
            note=f"{A['eval_items']} items x {A['judge_tokens_per_item']} tok, local 32B judge",
        ),
    ]


def post_training_lines() -> list[Line]:
    """The only honest way the ask reaches the hundreds of GPU-hours."""
    tpe = tokens_per_epoch()
    rollouts = 8
    grpo = tpe * rollouts * 2  # rollouts x 2 epochs
    return [
        Line("GRPO · 8 rollouts · 8B", 1, hours(grpo, A["tok_s"]["8b_lora"]),
             f"{rollouts} rollouts x 2 epochs = {grpo / 1e6:.0f}M tok"),
        Line("DPO pairs · 8B", 1, hours(tpe * 2, A["tok_s"]["8b_lora"]), "2 epochs, 1 pair/unit"),
    ]


# --------------------------------------------------------------------------
# what would have to be true for the ask to be 800 GPU-hours
# --------------------------------------------------------------------------

SCALED = {
    "corpus_multiplier": 10,        # 162k units instead of 16.2k
    "eval_items": 5_000,            # 4 tasks x 1,250, human-spot-checked
    "judges": 5,
    "grpo_rounds": 1,
}


def scaling_800_lines() -> list[Line]:
    """Same pipeline, 10x corpus, 5k eval items, one GRPO round.

    Present this next to the `full` bill: if the reviewer wants to see 800
    GPU-hours spent, this is the scope that actually spends them. Asking for
    800 without paying the 10x data-collection cost first is the thing that
    gets grant forms rejected.
    """
    tpe = tokens_per_epoch() * SCALED["corpus_multiplier"]
    out: list[Line] = []
    for label, key, seeds in (
        ("SFT 8B · LoRA", "8b_lora", A["seeds"]),
        ("SFT 14B · LoRA", "14b_lora", A["seeds"]),
        ("SFT 32B · LoRA", "32b_lora", A["seeds"]),
        ("SFT 8B · full FT (baseline)", "8b_full", 2),
    ):
        out.append(
            Line(f"[10x corpus] {label}", seeds, hours(tpe * A["epochs"], A["tok_s"][key]),
                 f"{A['epochs']} epochs x {tpe / 1e6:.0f}M tok")
        )
    grpo = tpe * 8 * 2 * SCALED["grpo_rounds"]
    out.append(Line("[10x corpus] GRPO · 8 rollouts · 8B", 1, hours(grpo, A["tok_s"]["8b_lora"]),
                    f"{grpo / 1e6:.0f}M tok"))
    sweep = SCALED["eval_items"] * A["eval_out_tokens"]
    out.append(Line(f"generation sweep × 16 configs ({SCALED['eval_items']} items)", 16,
                    hours(sweep, A["tok_s"]["gen_8b"]), ""))
    judge = SCALED["eval_items"] * A["judge_tokens_per_item"]
    out.append(Line(f"LLM-judge × {SCALED['judges']} judges", SCALED["judges"],
                    hours(judge, A["tok_s"]["gen_32b"]), ""))
    return out


def scenarios() -> dict[str, list[Line]]:
    core = finetune_lines()
    abl = ablation_lines()
    ev = eval_lines()
    return {
        "minimal": core[:1] + ev,
        "full": core + abl + ev,
        "post-training": core + abl + ev + post_training_lines(),
        "scaling-800": scaling_800_lines(),
    }


def total(lines: list[Line], buffer: bool = True) -> float:
    raw = sum(l.hours for l in lines)
    return round(raw * (A["buffer"] if buffer else 1.0), 1)


def report(scenario: str, as_json: bool) -> None:
    lines = scenarios()[scenario]
    payload = {
        "scenario": scenario,
        "tokens_per_epoch": round(tokens_per_epoch()),
        "assumptions": A,
        "lines": [asdict(l) | {"hours": l.hours} for l in lines],
        "raw_hours": round(sum(l.hours for l in lines), 1),
        "with_buffer": total(lines),
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"\n=== scenario: {scenario} | {tokens_per_epoch() / 1e6:.1f}M tok/epoch ===")
    print(f"{'line item':<44} {'runs':>5} {'h/run':>7} {'hours':>7}")
    for l in lines:
        print(f"{l.name:<44} {l.runs:>5} {l.hours_per_run:>7.2f} {l.hours:>7.2f}")
    raw = sum(l.hours for l in lines)
    print(f"{'raw total':<44} {'':>5} {'':>7} {raw:>7.2f}")
    print(f"{'with x' + str(A['buffer']) + ' buffer (retries, loss, failed jobs)':<44} "
          f"{'':>5} {'':>7} {raw * A['buffer']:>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["minimal", "full", "post-training", "scaling-800", "all"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.scenario == "all":
        for name in ("minimal", "full", "post-training", "scaling-800"):
            report(name, as_json=args.json)
        print()
    else:
        report(args.scenario, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
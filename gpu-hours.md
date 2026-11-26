# GPU-hours

Canonical computation: [`compute/budget.py`](compute/budget.py) — every number is derived from a visible assumption block, so changing one constant re-prices the whole bill. Full prose version: [`compute/GPU-BUDGET.md`](compute/GPU-BUDGET.md).

```bash
python3 compute/budget.py --scenario full --json     # regenerate these figures
```

## The numbers

| scope | GPU-hours (×1.5 buffer) | what it delivers |
|---|---|---|
| minimal | **8.5** | 8B LoRA × 3 seeds + eval sweeps + judge — the first publishable H1 result |
| full | **69.6** | + 14B/32B scaling, full-FT baseline, 25%/50% data ablations |
| + post-training | **84.2** | + one GRPO round and a DPO pass |
| 10× corpus, 5,000-item eval | **726** | the only scope where a four-digit-hour ask is honest |

Hardware basis: A100-80G, bf16, seq 4096, vLLM batching for generation. Hours are per-GPU hours.

## Why the bill is this small

| quantity | value |
|---|---|
| content units | 16,227 |
| SFT pairs | 1 per unit |
| tokens per pair | 130 prompt + 290 completion |
| **tokens per epoch** | **6.81M** |

6.8M tokens is under two hours of LoRA throughput on one A100. The corpus is curated, not large — which is the whole point of the project and also why an 800-hour request for this data would be indefensible.

## What the hours actually contain

Training is **not** the majority of the bill; ablation and evaluation structure is:

| category | hours (full scenario) |
|---|---|
| SFT runs (8B/14B/32B × 3 seeds + full-FT baseline) | 38.35 |
| data-volume ablations at 25% and 50% | 7.22 |
| generation sweeps over 16 model configs | 0.38 |
| LLM-judge passes | 0.44 |
| raw total | 46.39 |
| ×1.5 buffer for retries, preemption, failed jobs | 69.59 |

An ask of "training + evaluation" therefore has a visible training line (38 h) and a visible evaluation line. A budget that is 95% inference reads as a request for free inference quota, which is exactly the wrong shape for this kind of grant.

## Not in this bill

Human annotation of a 5,000-item eval set (labour), API-model baselines (billed per token), long-context experiments, cluster queueing beyond the flat ×1.5 buffer.

## Reproducing a real figure

Assumed throughput is the weakest link. After your first real run, replace `A["tok_s"]` in `compute/budget.py` with logged numbers and publish the before/after if they differ.
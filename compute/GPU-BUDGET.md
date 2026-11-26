# GPU-hour bill

Every figure below is produced by `compute/budget.py` from the assumption block at the top of that file. Nothing is a measured benchmark yet — the throughput numbers are labelled `A` (assumption) in the source and will be relabelled `M` (measured) once a real run exists.

```bash
python3 compute/budget.py                  # all scenarios
python3 compute/budget.py --scenario full  # one scenario
python3 compute/budget.py --scenario full --json
```

Hardware basis: A100-80G, bf16, sequence length 4096, vLLM-style batching for generation. Hours are **per-GPU hours**; multi-GPU jobs multiply, they do not divide.

## Data shape driving the math

| quantity | value | source |
|---|---|---|
| articles distilled | 899 (2015-09-15 .. 2018-12-27) | molu corpus audit |
| characters | 2,130,000 | molu corpus audit |
| content units | 16,227 | molu corpus audit (100% verified against source text) |
| SFT pairs | 1 per unit → 16,227 | assumption `pairs_per_unit=1.0` |
| tokens per pair | 130 prompt + 290 completion = 420 | assumption |
| **tokens per epoch** | **6.81M** | 16,227 × 420 |

That last number is the whole story: **6.8M tokens is one A100 for under two hours** at LoRA throughput. The corpus is curated, not large.

## Scenario: minimal — 8.5 GPU-hours

First result worth publishing: does SFT on the units beat prompting on an 8B model.

| line item | runs | h/run | hours |
|---|---|---|---|
| SFT Qwen3-8B · LoRA (3 epochs) | 3 seeds | 1.62 | 4.87 |
| generation sweep × 16 model configs | 16 | 0.02 | 0.38 |
| LLM-judge pass × 3 judges | 3 | 0.15 | 0.44 |
| raw total | | | **5.69** |
| ×1.5 buffer (retries, preemption, failed jobs) | | | **8.54** |

## Scenario: full — 69.6 GPU-hours

Adds model-size scaling, ablations, and a full fine-tune baseline.

| line item | runs | h/run | hours |
|---|---|---|---|
| SFT Qwen3-8B · LoRA | 3 seeds | 1.62 | 4.87 |
| SFT Qwen3-14B · LoRA | 3 seeds | 2.84 | 8.52 |
| SFT Qwen3-32B · LoRA | 3 seeds | 5.16 | 15.49 |
| SFT Qwen3-8B · full FT (baseline) | 2 seeds | 4.73 | 9.47 |
| ablation · 25% of corpus · 8B / 14B / 32B | 1 each | 0.41 / 0.71 / 1.29 | 2.41 |
| ablation · 50% of corpus · 8B / 14B / 32B | 1 each | 0.81 / 1.42 / 2.58 | 4.81 |
| generation sweep × 16 model configs | 16 | 0.02 | 0.38 |
| LLM-judge pass × 3 judges | 3 | 0.15 | 0.44 |
| raw total | | | **46.39** |
| ×1.5 buffer | | | **69.59** |

Ablations at 25% and 50% of the corpus answer the question reviewers actually ask: *how much of the gain is data volume, and how much is this particular data?* Without them the result is a single point with no lever.

## Scenario: + post-training — 84.2 GPU-hours

| additional line item | runs | h/run | hours |
|---|---|---|---|
| GRPO · 8 rollouts · 2 epochs · 8B | 1 | 8.65 | 8.65 |
| DPO pairs · 2 epochs · 8B | 1 | 1.08 | 1.08 |

Rollout generation, not gradient steps, is where post-training time goes: 8 rollouts × 6.8M tokens × 2 epochs ≈ 109M tokens.

## Scenario: what 800 GPU-hours actually buys — 726 hours

To spend 800 hours honestly, the scope has to change first: **10× the corpus** (≈162k units, i.e. ~9,000 source articles), a **5,000-item** eval set with human spot-checks, and 5 judges.

| line item | runs | h/run | hours |
|---|---|---|---|
| [10× corpus] SFT 8B · LoRA | 3 seeds | 16.23 | 48.68 |
| [10× corpus] SFT 14B · LoRA | 3 seeds | 28.40 | 85.19 |
| [10× corpus] SFT 32B · LoRA | 3 seeds | 51.63 | 154.89 |
| [10× corpus] SFT 8B · full FT | 2 seeds | 47.33 | 94.66 |
| [10× corpus] GRPO · 8 rollouts · 8B | 1 | 86.54 | 86.54 |
| generation sweep × 16 configs (5,000 items) | 16 | 0.30 | 4.76 |
| LLM-judge × 5 judges (5,000 items) | 5 | 1.85 | 9.26 |
| raw total | | | **483.98** |
| ×1.5 buffer | | | **725.97** |

## Reading the bill

- **Ask for 70–85 hours.** That covers the publishable result at the current data size, including ablations and a full fine-tune baseline.
- **800 hours is a different project.** It requires a 10× corpus collection effort (months of curation, not GPU time) before the GPU hours mean anything. Requesting the hours first is the shape of ask that gets rejected.
- **The measured-throughput upgrade path is one edit.** Replace `A["tok_s"]` with numbers logged from your own runs and every scenario re-prices itself; publish the before/after if they differ, because assumed throughput is the weakest link in any compute estimate.

## What is still missing from this bill

| excluded | why | where it belongs |
|---|---|---|
| human annotation of the 5,000-item eval set | it is labour, not GPU | a data budget, not a compute budget |
| API-model baselines (frontier models) | billed per token, not per GPU-hour | a separate line at API pricing |
| long-context experiments at 32k tokens | not needed until the doctrine exceeds prompt length | future work |
| cluster queueing and preemption | covered by the ×1.5 buffer, not modelled | site-specific |
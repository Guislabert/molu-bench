# Protocol (pre-registration)

Canonical text: [`results/README.md`](results/README.md). This page is the short version so nobody has to guess where the protocol lives.

Written **before** any baseline or fine-tune number existed. Amending it is allowed and logged; rewriting it silently is not.

## Claims

| # | claim | fails if |
|---|---|---|
| H1 | LoRA SFT of Qwen3-8B on molu's 16,227 units beats prompt-only Qwen3-8B by ≥ 10 rubric points, non-overlapping 95% CIs | Δ < 10, or no better than the `doctrine` prompt condition |
| H2 | the gain transfers to held-out tasks | held-out gain < half of the dev gain |
| H3 | `guard` gains more than `writer` (rules are learnable, taste is not) | guard gain < writer gain |

## Two prompt conditions, both fixed in the repo

| mode | model sees | why it exists |
|---|---|---|
| `raw` | the item prompt alone | what an off-the-shelf assistant sees |
| `doctrine` | + `harness/prompts/<task>.md` | what a Skill-equipped agent sees; **this is the bar H1 must clear** |

A benchmark that only compares fine-tuning against a weak prompt is a benchmark that always wins. Both conditions are versioned so the comparison cannot be quietly weakened later.

## Measurement

- 400-item eval set (4 tasks × 100), temperature 0, one sample per item, `max_tokens 800` so truncation is visible.
- Deterministic rubric score (primary), per-check pass rate (secondary), bootstrap 95% CI over 1,000 resamples, `seed=1723`.
- Blank output scores **0 on every check**, including negative checks, and is counted as `blank_responses`. A silent model must not collect points for restraint it never exercised.
- An LLM judge is allowed as a second, separately reported column. It never replaces the rubric column.

## Publish rules, decided in advance

| outcome | what ships |
|---|---|
| H1 holds | full table, both CIs, the ablation curve |
| H1 fails with Δ > 0 | "Δ = N points, below the pre-registered 10" in the abstract, not buried |
| H1 fails with Δ ≤ 0 | the negative result as the headline |
| 25% of the data matches 100% | in the abstract — it changes what the corpus is worth |
| a check turns out brittle | reported per check and flagged; removing it needs an amend entry |
| judge disagrees with rubric | both columns stay, disagreement reported |

## Hyperparameters (pre-registered)

LoRA r=16, α=32, lr 1e-4, cosine schedule, 3 epochs, 3 seeds, bf16, max length 4096. Machine-readable copy: [`configs/lora-qwen3-8b.json`](configs/lora-qwen3-8b.json). Changing these after seeing a result is an amend, not a tuning detail.

## Gating numbers

- ≥ 5 items per task before any per-task claim (20 items is a smoke test; the CI widths in `make smoke` show why).
- ≥ 3 seeds before any "fine-tuning helps" claim.
- ≥ 2 model families before any "open models in general" claim.

## Status

Dev-set numbers now exist (20 items, `poolside/laguna-xs-2.1`: raw 0.843 / doctrine 0.878) and they are published at the scale they were produced at — see [`results/TABLE.md`](results/TABLE.md). They are **not** the pre-registered baselines (Qwen3-8B / Llama-3.1-8B) and they cannot support H1: the dev set saturates at 0.84 for a small model, which is why amend 3 moves the H1 test to the 400-item set.

`results/*.json` only ever contains files written by `harness/cli.py`, i.e. by an actual model call, and each records the model string, prompt mode, item count and wall time that produced it.
# Pre-registration

Written **before** any baseline or fine-tune number exists, so that the results table cannot be quietly redefined after the fact. Timestamp of this file's first commit is the pre-registration date; amending it is allowed, silently rewriting it is not (see the amend log at the bottom).

## Claims under test

**H1 — fine-tuning beats prompting.** SFT of Qwen3-8B on molu's 16,227 units (LoRA, 3 epochs, 3 seeds) scores at least **10 rubric points higher** than prompt-only Qwen3-8B on the same 400-item eval set, with non-overlapping 95% bootstrap CIs.

**H2 — it generalises.** The gain on held-out tasks (never present in the training pairs) is at least **half** the gain on in-distribution tasks.

**H3 — rules beat taste.** `guard` gains more than `writer`, because `guard` failures are rule-shaped (cite position, separate machine risk from substantive risk) while `writer` failures are taste-shaped.

## Reporting rules — decided now, not after seeing numbers

| situation | what gets published |
|---|---|
| H1 holds | the full table, both CIs, the ablation curve |
| H1 fails but Δ > 0 | "Δ = N points, below the pre-registered 10" in the abstract, not buried in §5 |
| H1 fails and Δ ≤ 0 | the negative result, headline: fine-tuning on curated units did not beat a good prompt |
| the dev set saturates (amended rule) | stated in the README before any fine-tune runs, because a saturated rubric makes every later comparison meaningless until it is fixed |
| ablation shows 25% of data matches 100% | that goes in the abstract, because it changes what the corpus is worth |
| a check turns out to be brittle | it is reported per-check and flagged, not silently dropped; dropping it requires an amend entry |
| LLM judge disagrees with the rubric | both columns stay, the disagreement is reported, the judge never overwrites the rubric |

## Protocol

1. **Baselines** (no training): prompt-only Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B, plus one frontier API model for an upper reference. Prompts are fixed in `harness/prompts/` and versioned; changing a prompt after seeing results is an amend.
2. **Training**: LoRA r=16, α=32, lr 1e-4, 3 epochs, seed-report. Full fine-tune of the 8B as a cost-of-adaptation reference.
3. **Eval**: 400 items (4 tasks × 100), temperature 0, one sample per item, deterministic rubric. `max_tokens` 800 so truncation is visible rather than silently scored.
4. **Metrics**: rubric ratio per task (primary), pass rate per check (secondary), bootstrap 95% CI on 1,000 resamples with `seed=1723`.
5. **Stopping rule**: a run is complete when all 400 items return a non-empty response; an empty response scores **0 on every check, including negative checks**, is flagged per row, and is counted as `blank_responses` in the report. No retry-until-non-empty: a model that fails to answer is a result, not an outage.
6. **Held-out set**: a separate 100-item set authored after training-data freeze, never inspected during training.

## Gating numbers before we trust anything

- ≥ 5 items per task before any per-task claim; 20 items is a smoke test, and the CI widths in `make smoke` show why.
- ≥ 3 seeds before any "fine-tuning helps" claim.
- ≥ 2 model families before any claim about "open models in general".

## Amend log

| date | change | reason |
|---|---|---|
| amend 1 | 2026-11-26 | reclassified blank-output scoring | a test showed empty strings collected points from `none_of` checks; blanks now score zero, are flagged, and are counted |
| amend 2 | 2026-11-26 | dev-set prompts now state a length budget; three banned-lists widened | first real run exposed two rubric bugs: a ban on the bare string `第一` failed a correct answer containing `第一人称`, and `max_units` caps punished answers whose prompt never mentioned a length. The rubric must not penalise what the prompt did not ask for. All 20 items re-run after the change, so every committed report matches the committed dev set |
| amend 3 | 2026-11-26 | dev set recorded as saturating | `poolside/laguna-xs-2.1` scores 0.843 (`raw`) and 0.878 (`doctrine`) on 20 items. A dev set a small model nearly aces cannot discriminate, so the H1 test moves to the 400-item set and the dev set is marked as a smoke test in the card |

## Measured so far

| file | model | prompt | n | overall | note |
|---|---|---|---|---|---|
| `nvidia-laguna-xs-dev20-raw.json` | poolside/laguna-xs-2.1 | raw | 20 | 0.843 | hosted OpenAI-compatible endpoint; dev-set scale |
| `nvidia-laguna-xs-dev20-doctrine.json` | poolside/laguna-xs-2.1 | doctrine | 20 | 0.878 | same run conditions |

Regenerate the rendered table with `make table` ([`TABLE.md`](TABLE.md)). These are **not** the pre-registered baselines — those name Qwen3-8B and Llama-3.1-8B, which need a GPU or a different host. They are the first real numbers this harness produced, published at the scale they were produced at.

`results/*.json` only ever comes from `harness/cli.py --out`, i.e. from an actual model call; each file records the model string, prompt mode, item count and wall time that produced it.
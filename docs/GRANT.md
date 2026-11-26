# Compute request — one page

What this is: a **reproducible evaluation of whether a small open model can be trained to follow a named content methodology**, built on a 16,227-unit Chinese content corpus and an 11-module engine ([molu](https://github.com/Guislabert/molu)).

Everything on this page is checkable in under two minutes with no GPU and no API key:

```bash
git clone https://github.com/Guislabert/molu-bench && cd molu-bench
make ci     # 35 unit tests, offline stub run, both prompt conditions, recomputed bill
```

---

## 1. What exists today

| artifact | state | where |
|---|---|---|
| eval harness (deterministic rubric, bootstrap CI, OpenAI-compatible) | working, stdlib only | [`harness/`](harness) |
| 20-item dev set, 4 tasks, authored for this repo | shipped | [`data/dev.jsonl`](data/dev.jsonl) |
| two fixed prompt conditions (`raw` / `doctrine`) | shipped, versioned | [`harness/prompts/`](harness/prompts) |
| pre-registered protocol with publish rules for negative results | shipped | [`protocol.md`](protocol.md) |
| GPU-hour bill derived from a visible assumption block | shipped, recomputable | [`compute/budget.py`](compute/budget.py) |
| corpus → SFT converter (refuses on silent row-count drift) | shipped, tested | [`scripts/build_sft.py`](scripts/build_sft.py) |
| LoRA training entry point + pre-registered config | shipped; `--dry-run` tested, GPU path untested | [`scripts/sft_lora.py`](scripts/sft_lora.py), [`configs/`](configs) |
| loss-curve generator (SVG, stdlib) | shipped, tested | [`scripts/plot_loss.py`](scripts/plot_loss.py) |
| measured baseline numbers | **not yet** — needs an endpoint or a GPU | [`results/`](results) |
| fine-tuned adapter + before/after table | **not yet** — this is what the hours buy | — |

## 2. What the hours buy

| ask | 1 × A100-80G (or H100-80G), **100 GPU-hours** |
|---|---|
| itemized core | 69.6 h (see [`gpu-hours.md`](gpu-hours.md)) |
| iteration slack | ~30 h for a re-run after a caught bug, a failed sweep, or a seed that needs repeating |
| expected wall clock | 2–3 working days on one card, including data build and two eval passes |

Breakdown of the 69.6 h, by category, so the split is visible:

| category | hours | share |
|---|---|---|
| SFT runs: 8B / 14B / 32B × 3 seeds + full-FT baseline | 38.4 | 55% |
| data-volume ablations (25% and 50% of corpus) | 7.2 | 10% |
| evaluation sweeps over 16 model configs | 0.4 | 1% |
| local LLM-judge passes | 0.4 | 1% |
| ×1.5 buffer (retries, preemption, failed jobs) | 23.2 | 33% |

The training line is the majority and the evaluation line is explicit, because a budget that is 95% inference is a request for free inference quota and reads as such.

## 3. Deliverables

| # | deliverable | definition of done |
|---|---|---|
| D1 | **fine-tuned adapters** (Qwen3-8B LoRA, 3 seeds) + the config each was run with | adapter directory, `configs/*.json`, and a committed loss curve per seed |
| D2 | **loss curves** | `results/loss-*.svg`, regenerated from the trainer log by a stdlib script, not hand-drawn |
| D3 | **before/after table** | `raw` vs `doctrine` vs fine-tuned, per task, with 95% bootstrap CIs, produced by the harness from `results/*.json` |
| D4 | **held-out evaluation** | a 100-item set authored after the training data was frozen, reported separately |
| D5 | **published result, including a negative one** | if H1 fails, the abstract says so in the same font size (pre-registered rule, [`protocol.md`](protocol.md)) |

Not deliverables: the corpus (third-party copyright, stays local — see [`data/CARD.md`](data/CARD.md)) and any claim of a traffic or revenue outcome.

## 4. Why this needs compute rather than a cleverer prompt

The question is *whether behavioural doctrine can be trained in*, not whether a prompt can approximate it. That requires comparing a trained model against the best untrained prompt condition — which is why `doctrine` exists as a condition rather than as the baseline we hope to beat. Without the trained side, this project can only produce the baseline row.

Cost of the honest alternative: the API-only baseline above costs **under $5** and needs no grant. If the fine-tuning arm is not funded, the project still ships a prompt-vs-doctrine comparison, and that is a smaller claim made in a smaller voice.

## 5. Scale, and why the ask is not larger

16,227 units produce **6.8M tokens per epoch**, about 1.6 h of LoRA on one A100. A four-digit-hour request for this data would be padding: the same pipeline on a 10× corpus (≈162k units) plus a 5,000-item eval set lands at 726 h, and that is a different project with a data-collection budget in front of it. The bill is intentionally sized to the data that exists.

## 6. What a reviewer will find wrong, said first

- **No measured number exists yet.** The dev set has 20 items; per-task CIs at that size are wide by construction, and `make smoke` prints them so the width is visible rather than argued about.
- **Single annotator** wrote the dev set. Inter-annotator agreement is an open item for the larger set.
- **The GPU training path is not covered by CI** (no GPU in CI). Its data validation and config loading are tested via `--dry-run`; the training loop itself is reviewed, not executed.
- **20 items is a smoke test**, not a benchmark. Task-level claims wait for the 400-item set.
- **Chinese-only.** The format-level findings do not transfer to English content work without re-derivation.

## 7. Reproducibility statement

- Zero third-party dependencies in `harness/` and `compute/` — `urllib`, `json`, `random`, `statistics`.
- CI on every push: 35 unit tests, offline stub run in both prompt conditions, dev-set integrity, budget recomputation.
- Randomness is seeded (`seed=1723` for bootstrap, per-run `--seed` for training) and reported.
- Every number in the README is labelled as assumption or measurement; no measured number is claimed.
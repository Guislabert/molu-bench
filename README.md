# molu-bench

Evaluation harness and compute plan for **Chinese content-production doctrine following** — does a model actually run a content methodology, or does it produce plausible text that violates the method?

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/Guislabert/molu-bench/ci.yml?branch=main&label=CI&color=34A853)](https://github.com/Guislabert/molu-bench/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License%3A%20Apache--2.0-FBBC05.svg)](LICENSE)
[![Items](https://img.shields.io/badge/dev%20set-20%20items-4285F4.svg)](data/dev.jsonl)
[![Compute](https://img.shields.io/badge/planned-69.6%20GPU--h-9AA0A6.svg)](compute/GPU-BUDGET.md)

</div>

## Why this exists

[`molu` 墨炉](https://github.com/Guislabert/molu) is an 11-module Chinese content-production engine: insight → topic → writer → guard, each module a `SKILL.md` with a doctrine and hard rules (no fabricated cases, no traffic promises, cite positions in compliance review). It was distilled from 899 published articles into 16,227 content units, and it ships as prompt-time Skills.

The open question is not "is molu good" but **"can a small open model be trained to follow the doctrine instead of being reminded of it in every prompt?"** That is measurable, and almost nobody measures it for Chinese content work:

- instruction-following benchmarks (IFEval, Arena) test generic constraints — format, length, "say X in Y words";
- content benchmark suites test English marketing copy;
- neither tests whether an output **violates a named method**: invents a case study, promises traffic, skips the audience segment, gives advice when the task asked for diagnosis.

## Hypotheses (pre-registered)

Written down before any numbers exist, in [`results/README.md`](results/README.md):

| # | Hypothesis | Disconfirming result we will publish |
|---|---|---|
| H1 | SFT on molu's 16,227 units (Qwen3-8B, LoRA) beats prompt-only Qwen3-8B on doctrine following by ≥ 10 rubric points | Δ < 10 points, or ≤ the prompt-engineering baseline |
| H2 | The gain transfers to held-out tasks never seen in training | held-out gain < half of the dev gain |
| H3 | The `guard` task (risk review) benefits most, because it is rule-shaped rather than taste-shaped | guard gain < writer gain |

A negative result here is still a result: "16k curated units of Chinese content doctrine do not move an 8B model past a good prompt" is a finding worth publishing, and it is cheaper than a fine-tuning bill.

## Where everything lives

| path | what it is |
|---|---|
| [`protocol.md`](protocol.md) | pre-registration: hypotheses, prompt conditions, publish rules (canonical: [`results/README.md`](results/README.md)) |
| [`gpu-hours.md`](gpu-hours.md) | the bill: 8.5 / 69.6 / 84.2 / 726 GPU-hours (canonical: [`compute/GPU-BUDGET.md`](compute/GPU-BUDGET.md)) |
| [`harness/`](harness) | scorer, bootstrap CI, model adapters, and the fixed prompt packs |
| [`data/`](data) | 20-item dev set, dataset card, rubric schema |
| [`scripts/`](scripts) | `build_sft.py` (CPU), `sft_lora.py` (GPU), `plot_loss.py` (SVG curve) |
| [`configs/`](configs) | pre-registered hyperparameters as a diffable file |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | no GPU → API baseline for <$5 → one A100 → first table |
| [`docs/GRANT.md`](docs/GRANT.md) | one-page compute request: what exists, what the hours buy, what ships either way |
| [`results/`](results) | protocol + measured outputs; ships empty on purpose |

## What runs today

```bash
make test       # 31 unit tests (scorer + data pipeline), no network, no model
make smoke      # 20-item dev set, offline stub model, no API key needed
make budget     # recompute the GPU-hour bill from the assumption block
make ci         # all of the above, exactly what .github/workflows/ci.yml runs
```

Two prompt conditions, both committed before any run:

| mode | what the model sees |
|---|---|
| `--prompt-mode raw` | the item prompt alone — an off-the-shelf assistant |
| `--prompt-mode doctrine` | the molu module prompt from `harness/prompts/<task>.md` prepended |

Without a recorded prompt condition, "fine-tuning beat prompting" is not a comparable claim, and a prompt tuned after seeing results is cheating nobody can detect. The doctrine packs live in version control for that reason.

Score any OpenAI-compatible endpoint (vLLM / SGLang / Ollama / lmdeploy / OpenRouter):

```bash
python3 harness/cli.py --model local \
  --base-url http://localhost:8000/v1 --model-name Qwen3-8B \
  --out results/qwen3-8b-prompt-only.json
```

Zero third-party dependencies — `harness/evalkit.py` is stdlib only (`urllib`, `json`, `random`). That is deliberate: a reviewer should be able to re-run the numbers without resolving a dependency tree from 2026.

## Scorer design

Deterministic checks, not an LLM judge, as the default:

| check kind | use |
|---|---|
| `contains` / `all_of` / `none_of` | required and banned content |
| `regex` | structure (numbered candidates, `→` chains, citation of the offending phrase) |
| `min_units` / `max_units` | Chinese-aware length in units (1 CJK char = 1 unit, 1 latin word = 1 unit) |
| `line_count_at_least` | "give me 5 topics" actually returned 5 lines |

Aggregation: per-task mean rubric ratio, with a 1,000-resample bootstrap 95% CI (seeded, `seed=1723`). An LLM judge is documented as an opt-in, biased, separately-reported pass — not a replacement for the deterministic score. Judges are how benchmarks quietly become uncomparable, so the judge column never overwrites the rubric column.

**A blank response earns zero, including on negative checks.** Found by a test, not by review: an output that says nothing "does not promise traffic" and "does not fabricate a case", so a silent model used to collect points for restraint it never exercised. Blank rows are now zeroed, flagged, and counted separately in every report (`blank_responses`).

## From this repo to the first real table

[`docs/RUNBOOK.md`](docs/RUNBOOK.md) is the operational version: a zero-GPU path that produces the API baseline row for under $5, the single-A100 path (serve → four sweeps → SFT → serve adapter), cost arithmetic you fill in with today's rate, and the six failure modes that usually bite first.

```bash
python3 scripts/build_sft.py --units units.jsonl --out sft.jsonl --stats    # CPU, corpus stays local
python3 scripts/sft_lora.py  --train sft.jsonl --out adapters/seed0 --seed 0 --dry-run
```

`build_sft.py` validates rows, estimates tokens per epoch, and refuses to write when the row count drifts more than 5% from the 16,227 the GPU bill assumes — a silent 300-row dataset would make every downstream hour wrong. `sft_lora.py` is the GPU half and is not covered by CI; its data validation is reachable through `--dry-run` and is tested.

A training run leaves three artifacts a write-up needs: the adapter directory, the committed config it was run with, and the loss curve.

```bash
python3 scripts/plot_loss.py --log adapters/seed0/trainer_state.json \
  --out results/loss-qwen3-8b-seed0.svg --title "Qwen3-8B LoRA · seed 0"
```

No matplotlib: the curve is generated as SVG by a stdlib script, so it can be regenerated on any machine and diffed in review.

## First measured numbers (dev-set scale, published as-is)

| run | overall | guard | insight | topic | writer |
|---|---|---|---|---|---|
| `poolside/laguna-xs-2.1` · raw | 0.843 | 0.883 | 0.753 | 0.900 | 0.833 |
| `poolside/laguna-xs-2.1` · doctrine | 0.878 | 0.950 | 0.870 | 0.950 | 0.743 |

Full table with per-task 95% CIs: [`results/TABLE.md`](results/TABLE.md), regenerated from the committed report files by `make table`.

Three things this run actually told us, and they are not flattering:

1. **The dev set saturates.** A *small* model scores 0.84 with no doctrine prompt. A rubric that a weak model nearly aces cannot detect the improvement H1 is about — the dev set needs harder items before it can support any claim.
2. **`doctrine − raw = +0.035`, with CIs that overlap almost entirely.** On 20 items that is indistinguishable from noise. Reported because the pre-registration says so.
3. **Two rubric bugs the first real run exposed**, both fixed and logged as amendments in [`results/README.md`](results/README.md): a ban on the string `第一` was failing a correct answer for containing `第一人称`, and length caps were punishing answers whose prompt never stated a length budget.

What did *not* fail: no blank responses, no truncation-driven zeros on the final run, and the harness scored two hosted models' shapes (`content` present, `content: null` with reasoning only) without crashing.

## Compute

Full bill, computed from a visible assumption block: [`compute/GPU-BUDGET.md`](compute/GPU-BUDGET.md).

| scope | GPU-hours |
|---|---|
| minimal (8B LoRA + eval) | **8.5** |
| full (8B/14B/32B × 3 seeds + full-FT baseline + ablations + judge) | **69.6** |
| + one GRPO round | **84.2** |
| the same pipeline on a **10× corpus with a 5,000-item eval set** | **726** |

Read that last row honestly: **800 GPU-hours is not what this project needs at its current data size.** 6.8M tokens per epoch is a two-hour job on one A100. The number only becomes defensible after the corpus grows ~10× and the eval set becomes 5,000 human-spot-checked items. Requesting 800 hours for a 70-hour job is the fastest way to lose a compute grant; the itemized bill is in `compute/budget.py`, and changing one constant re-prices the whole thing.

## Data and licensing

- **Publishable:** the harness, the 20-item dev set (authored for this repo), the rubric schema, the eval protocol, the compute accounting, and any fine-tuned **weights** (Qwen3 ships under Apache-2.0, so adapters can be released).
- **Not publishable:** the SFT corpus itself. The 16,227 units are derived from 899 published articles; copyright stays with the original authors. The corpus stays local, and the training card in [`data/CARD.md`](data/CARD.md) documents shape and provenance without shipping text.
- This repo: Apache-2.0 ([`LICENSE`](LICENSE)).

## Status

| piece | state |
|---|---|
| harness + scorer + bootstrap CI | working, 31 unit tests in CI on every push |
| 20-item dev set across 4 tasks | done, deliberately small |
| fixed prompt conditions (raw / doctrine) | done — `harness/prompts/` |
| pre-registered protocol | done ([`results/README.md`](results/README.md)) |
| GPU-hour accounting | done, reproducible |
| dev-set numbers on a hosted open model | **measured** — 20 items, dev-set scale only, [`results/TABLE.md`](results/TABLE.md) |
| baseline numbers (prompt-only Qwen3-8B / Llama-3.1-8B) | **not run yet** — pre-registered baselines, needs those specific models |
| SFT runs, ablations, held-out set | **not run yet** — needs GPU access |

No number in this README is measured yet, and the badges say so. When baselines land, `results/` gets a table and this section gets a different sentence.

---

## 中文摘要

**这个仓库在做什么**：给「中文内容生产方法论是否被模型真正执行」建一个可复现的评测。不是问文笔好不好，是问模型有没有违反方法论本身——编案例、承诺流量、跳过人群、任务要诊断却给建议。

**为什么值得做**：通用基准（IFEval、Arena）测的是格式和长度这类通用约束，英文内容营销基准测的是英文文案，没有一套测「违反某个具名方法」。molu 的 11 模块 + 16,227 单元语料正好是可以拿来做这件事的标尺。

**已经能跑**：`make smoke` 离线跑 20 条 dev set，零第三方依赖；打分是确定性规则，不靠 LLM 当裁判（裁判只作为可选旁证、单独报告）。CI 每次 push 都跑一遍。

**算力账**：当前规模 8.5–69.6 GPU 小时，加一轮 GRPO 是 84.2。800 小时在这个数据量下站不住——只有语料涨到 10 倍、评测集到 5,000 条才勉强凑到 726。账单在 `compute/budget.py`，改一个常数就能重算，所以经不起虚报。

**还没跑**：所有 baseline 和 SFT 结果都需要 GPU，现在表里是空的，README 也明说没有测过的数字。
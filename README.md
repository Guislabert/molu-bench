# molu-bench

Evaluation harness and compute plan for **Chinese content-production doctrine following** — does a model actually run a content methodology, or does it produce plausible text that violates the method?

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/Guislabert/molu-bench/ci.yml?branch=main&label=CI&color=34A853)](../../actions/workflows/ci.yml)
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

## What runs today

```bash
make smoke      # 20-item dev set, offline stub model, no API key needed
make budget     # recompute the GPU-hour bill from the assumption block
make ci         # both, exactly what .github/workflows/ci.yml runs
```

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
| harness + scorer + bootstrap CI | working, tested in CI on every push |
| 20-item dev set across 4 tasks | done, deliberately small |
| pre-registered protocol | done ([`results/README.md`](results/README.md)) |
| GPU-hour accounting | done, reproducible |
| baseline numbers (prompt-only Qwen3-8B / Llama-3.1-8B) | **not run yet** — needs GPU access |
| SFT runs, ablations, held-out set | **not run yet** — needs GPU access |

No number in this README is measured yet, and the badges say so. When baselines land, `results/` gets a table and this section gets a different sentence.

---

## 中文摘要

**这个仓库在做什么**：给「中文内容生产方法论是否被模型真正执行」建一个可复现的评测。不是问文笔好不好，是问模型有没有违反方法论本身——编案例、承诺流量、跳过人群、任务要诊断却给建议。

**为什么值得做**：通用基准（IFEval、Arena）测的是格式和长度这类通用约束，英文内容营销基准测的是英文文案，没有一套测「违反某个具名方法」。molu 的 11 模块 + 16,227 单元语料正好是可以拿来做这件事的标尺。

**已经能跑**：`make smoke` 离线跑 20 条 dev set，零第三方依赖；打分是确定性规则，不靠 LLM 当裁判（裁判只作为可选旁证、单独报告）。CI 每次 push 都跑一遍。

**算力账**：当前规模 8.5–69.6 GPU 小时，加一轮 GRPO 是 84.2。800 小时在这个数据量下站不住——只有语料涨到 10 倍、评测集到 5,000 条才勉强凑到 726。账单在 `compute/budget.py`，改一个常数就能重算，所以经不起虚报。

**还没跑**：所有 baseline 和 SFT 结果都需要 GPU，现在表里是空的，README 也明说没有测过的数字。
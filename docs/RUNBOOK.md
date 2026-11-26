# Runbook: from no GPU to the first real table

Everything here has a cost attached and every cost is a **formula**, not a remembered price. GPU and API prices change weekly; the arithmetic does not. Fill in today's rate and the bill is yours.

Two paths, and the cheap one comes first on purpose.

---

## Path 0 — the baseline table without any GPU (under $5)

The pre-registered baselines include API models. A 400-item sweep costs almost nothing:

| quantity | value |
|---|---|
| items | 400 |
| prompt tokens each | ~300 (doctrine condition is longer: ~600) |
| output tokens each | ~600 (`--max-tokens 800` caps it) |
| input tokens total | 120k – 240k |
| output tokens total | 240k |

At $1 / M input and $3 / M output: **$0.84 – $0.96 per model**. Five frontier models is under $5. This is the piece of the results table that does not need to wait for compute at all.

```bash
export MOLU_BENCH_API_KEY=...            # your provider key
python3 harness/cli.py --model remote \
  --base-url https://openrouter.ai/api/v1 \
  --model-name anthropic/claude-sonnet-4.5 \
  --prompt-mode raw \
  --out results/api-claude-sonnet-raw.json
```

Report files record the model string, prompt mode, item count and wall time, so the table can be rebuilt from `results/*.json` alone. **Do not commit the key**, and do not put a key in `--base-url`.

What this buys you: an upper reference row, and a sanity check that the rubrics score a strong model above a weak one. If a frontier model scores 0.4 on `guard`, the rubric is broken, not the model.

---

## Path 1 — one A100-80G, by the hour

### 1. Pick the box

Any provider that gives a single A100-80G with a PyTorch or vLLM image works (RunPod, Vast.ai, Lambda, a university cluster). Requirements:

- **80 GB VRAM**, not 40 — 32B in bf16 needs it, and 8B at `max-model-len 4096` leaves room for batching.
- 100 GB disk: Qwen3-8B weights ~16 GB, 32B ~64 GB, adapters and logs are small.
- Do **not** pick a `4090`/24 GB and plan to "just" run 32B later; the ablations need the headroom.

### 2. Serve the base model (5 minutes)

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v /workspace/hf:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B --served-model-name Qwen3-8B \
  --max-model-len 4096 --gpu-memory-utilization 0.90
```

Check it answers before benchmarking it:

```bash
curl -s localhost:8000/v1/models | head -c 300
python3 harness/cli.py --model local --base-url http://localhost:8000/v1 \
  --model-name Qwen3-8B --prompt-mode raw --max-tokens 200 --quiet
```

If that prints a table, the harness and the endpoint agree. Stop here and re-read `compute/budget.py` — the hours you spend from now on are on the clock.

### 3. Run the four sweeps (about 10 minutes of GPU)

```bash
for mode in raw doctrine; do
  python3 harness/cli.py --model local --base-url http://localhost:8000/v1 \
    --model-name Qwen3-8B --prompt-mode $mode \
    --out results/qwen3-8b-${mode}.json
done
```

Two numbers fall out: `qwen3-8b-raw` and `qwen3-8b-doctrine`. **Their difference is the prompt-only reference that H1 has to beat.** Publish both even if the trained model loses to them.

### 4. Build the SFT set (CPU work, corpus stays local)

The corpus is not redistributable, so this step happens on the machine that already holds it. Expected input: one JSON object per line with the unit text and its module label.

```bash
python3 scripts/build_sft.py \
  --units /path/to/molu-units.jsonl \
  --out /workspace/sft.jsonl \
  --stats
```

It validates every row, drops malformed ones with a reason, prints the token estimate, and refuses to run if the count is far off `16,227` (override with `--allow-count-mismatch`). The output is chat format, one `{"messages": [...]}` per line.

**Do not copy `/workspace/sft.jsonl` into this repo.** It is derived from third-party articles; see `data/CARD.md`.

### 5. Train LoRA (about 1.6 hours per run, 3 seeds)

```bash
python3 scripts/sft_lora.py \
  --train /workspace/sft.jsonl \
  --model Qwen/Qwen3-8B \
  --out /workspace/adapters/qwen3-8b-seed0 \
  --seed 0 --epochs 3 \
  --lora-r 16 --lora-alpha 32 --lr 1e-4
```

Hyperparameters are the ones pre-registered in `results/README.md` (r=16, α=32, lr 1e-4, 3 epochs). Changing them after seeing the result is an amend, not a tweak — record it.

Serve the adapter for the trained sweep:

```bash
docker run --gpus all --rm -p 8001:8000 \
  -v /workspace/adapters:/adapters \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B --served-model-name Qwen3-8B-lora-seed0 \
  --enable-lora --lora-modules seed0=/adapters/qwen3-8b-seed0
```

Then the same harness, pointed at `:8001`, `--model-name Qwen3-8B-lora-seed0`, `--prompt-mode raw`.

### 6. Keep the artifacts a write-up needs (2 minutes)

Three things make a fine-tuning claim checkable, and all three are cheap to save:

```bash
# 1. the config it was actually run with (the committed default, plus any override)
python3 scripts/sft_lora.py --config configs/lora-qwen3-8b.json --train /workspace/sft.jsonl \
  --out /workspace/adapters/qwen3-8b-seed0 --seed 0 > /workspace/adapters/run.log 2>&1

# 2. the loss curve, as an SVG you can commit
python3 scripts/plot_loss.py --log /workspace/adapters/qwen3-8b-seed0/trainer_state.json \
  --out results/loss-qwen3-8b-seed0.svg --title "Qwen3-8B LoRA · seed 0"

# 3. the before/after eval — the only number that matters
python3 harness/cli.py --model local --base-url http://localhost:8001/v1 \
  --model-name Qwen3-8B-lora-seed0 --prompt-mode raw --out results/qwen3-8b-lora-seed0-raw.json
```

An adapter without a config, a curve and a before/after table is an unverifiable claim. Save all three or the run did not happen.

### 7. What the hours actually cost

From `compute/budget.py` (recompute, do not trust this table):

| run | GPU-hours | at $1.50/h | at $2.50/h |
|---|---|---|---|
| minimal (8B LoRA × 3 seeds + eval) | 8.5 | $13 | $21 |
| full (8B/14B/32B + full-FT baseline + ablations + judge) | 69.6 | $104 | $174 |
| + one GRPO round | 84.2 | $126 | $211 |

Prices are placeholders — put today's rate in the right-hand column yourself. Add storage and egress: 100 GB of weights moved twice is real money on some providers and free on others.

A session that runs all of `full` is **one working day on one card**, plus the time you spend watching it. Budget the wall clock, not just the GPU hours: three seeds at 1.6 h each is not one 4.8 h block if you have to re-queue between them.

---

## Failure modes that will bite, in the order they usually do

| symptom | cause | fix |
|---|---|---|
| `blank_responses` > 0, score drops | `max_tokens` too low, or the endpoint truncates | raise `--max-tokens`, check the raw `output` field in the JSON |
| every item scores 0 on `writer` | model answered in English, or with markdown headings the checks do not expect | read three raw outputs before believing the number |
| score looks great on `raw`, terrible on `doctrine` | the doctrine prompt is long enough to push the answer past `max_units` | that is a finding about prompt budget, report it |
| OOM at 32B | `gpu-memory-utilization 0.90` plus long context | lower `max-model-len`, or split 32B across 2 cards and double the per-run GPU-hours |
| LoRA loss goes flat | corpus units are too short to teach unless the prompt carries the excerpt | check that `build_sft.py --stats` shows the token distribution you expect |
| run "finishes" in 9 seconds | the endpoint returned an error object, not a completion | the harness fails loudly on a non-200; if you patched around it, you are measuring your patch |

## What to commit afterwards

- `results/*.json` — one file per model × prompt mode, exactly as the harness wrote it. The `.gitignore` already excludes them; force-add the ones you want public.
- an amend entry in `results/README.md` for anything changed after seeing numbers, including hyperparameters and prompt wording.
- the negative result, if that is what happened. H1 failing at Δ = 4 points is publishable here; H1 failing and quietly not mentioned is not.

## Not covered by this runbook

- **Human annotation** of the 5,000-item set: labour, not GPU. The 20-item dev set has a single annotator and that is a stated limitation.
- **Held-out task authoring**: must happen *after* the training set is frozen, so it has to be written by someone who has not looked at the SFT rows.
- **Weight release**: before publishing an adapter, run an overlap check against held-out articles and record it in `data/CARD.md`. Adapters can memorise.
- `scripts/sft_lora.py` is **not** exercised by CI (it needs a GPU and transformers). Treat it as a starting point, not a validated artifact — the validated parts are `harness/` and `compute/`.
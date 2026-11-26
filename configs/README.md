# Training configs

| file | purpose |
|---|---|
| [`lora-qwen3-8b.json`](lora-qwen3-8b.json) | the pre-registered run: LoRA r=16, α=32, lr 1e-4, 3 epochs, seeds 0/1/2 |

```bash
python3 scripts/sft_lora.py --config configs/lora-qwen3-8b.json \
  --train /workspace/sft.jsonl --out /workspace/adapters/qwen3-8b-seed0 --seed 0
```

Why a config file and not just CLI flags: the numbers that a write-up will claim have to exist somewhere a reviewer can diff. A flag typed once into a shell is not that place. `scripts/sft_lora.py` reads this file and CLI flags override it, so the committed config stays the default and any deviation is visible in the command that was run.

#!/usr/bin/env python3
"""LoRA SFT of an open model on molu units. Needs a GPU; not covered by CI.

This is a starting point, not a validated artifact. The parts of this repo that
are tested on every push are `harness/` and `compute/`; this file is reviewed,
but it cannot be executed in CI, so treat a divergence between the two as this
file's problem.

Written against trl >= 0.12 (the `SFTConfig` / `processing_class` API).

    pip install "trl>=0.12" peft transformers datasets accelerate
    python3 scripts/sft_lora.py --train sft.jsonl --model Qwen/Qwen3-8B \
        --out adapters/qwen3-8b-seed0 --seed 0

Hyperparameters default to the values pre-registered in results/README.md:
LoRA r=16, alpha=32, lr 1e-4, 3 epochs. Changing them after seeing a result is
an amendment to the pre-registration, not a tuning detail.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="chat-format jsonl from scripts/build_sft.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="", help="JSON config, e.g. configs/lora-qwen3-8b.json")
    # Tunables default to None so the config file can supply them; explicit CLI
    # flags always win, which keeps the committed config the visible baseline.
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--lora-alpha", type=int)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--dry-run", action="store_true", help="validate data + config, then exit")
    args = ap.parse_args()

    defaults = dict(model="Qwen/Qwen3-8B", seed=0, epochs=3.0, lr=1e-4, lora_r=16,
                    lora_alpha=32, max_length=4096, batch_size=2, grad_accum=8)
    cfg_from_file: dict = {}
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"no such config: {cfg_path}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_from_file = {k: v for k, v in raw.items() if not k.startswith("_")}
        print(f"config: {cfg_path}")
        # seed stays a CLI concern: one config, three runs, one seed per run.
        seeds = cfg_from_file.pop("seeds", None)
        if seeds:
            print(f"  config lists seeds {seeds}; pick one with --seed")

    for field, fallback in defaults.items():
        value = getattr(args, field)
        if value is None:
            setattr(args, field, cfg_from_file.get(field, fallback))

    train = pathlib.Path(args.train)
    if not train.exists():
        raise SystemExit(f"no such file: {train}")

    # Validate the data before touching a GPU: a malformed row two hours into a
    # run is the most expensive way to learn about a malformed row.
    rows = 0
    for line in train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise SystemExit(f"row {rows + 1}: no messages")
        roles = [m.get("role") for m in msgs]
        if roles[-1] != "assistant":
            raise SystemExit(f"row {rows + 1}: last message must be the assistant turn, got {roles[-1]!r}")
        if not any(r == "user" for r in roles):
            raise SystemExit(f"row {rows + 1}: no user turn")
        rows += 1
    print(f"validated {rows} rows")

    cfg = dict(
        model=args.model, epochs=args.epochs, lr=args.lr, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, seed=args.seed, max_length=args.max_length,
        batch_size=args.batch_size, grad_accum=args.grad_accum, out=args.out,
    )
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("dry run: data and config look usable, exiting before model load")
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = load_dataset("json", data_files=str(train), split="train")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=cfg_from_file.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg_from_file.get("warmup_ratio", 0.03),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        max_length=args.max_length,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter written to {args.out}")
    print("serve it with: vllm ... --enable-lora --lora-modules seed=/abs/path/to/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
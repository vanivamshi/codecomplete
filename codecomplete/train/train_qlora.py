"""QLoRA fine-tune of CodeLlama-7B on the extracted Python functions.

Each function becomes either a fill-in-the-middle sample (a random span is the
target, with code before and after as context) or a plain left-to-right sample.
FIM matters because the editor almost always has code below the cursor.

Fits on one 24 GB GPU (e.g. A10G, RTX 4090, L4). On an A100 it runs about twice as fast.

    python -m codecomplete.train.train_qlora --tokenizer artifacts/tokenizer --out artifacts/lora
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from codecomplete.tokenization import build_training_ids, load_tokenizer


def fim_split(code: str, rng: random.Random, fim_rate: float) -> tuple[str, str, str]:
    """Pick a span of whole lines (biased toward one line, like editor completions) as the middle."""
    lines = code.splitlines(keepends=True)
    if rng.random() >= fim_rate or len(lines) < 3:
        return code, "", ""
    start = rng.randint(1, len(lines) - 1)
    span = 1 if rng.random() < 0.6 else rng.randint(1, min(8, len(lines) - start))
    end = start + span
    middle = "".join(lines[start:end])
    # Half the time cut inside the first line, to teach completing a partially typed line.
    if rng.random() < 0.5 and len(middle.strip()) > 4:
        indent = len(middle) - len(middle.lstrip())
        cut = rng.randint(indent, len(middle.rstrip("\n")) - 1)
        return "".join(lines[:start]) + middle[:cut], middle[cut:], "".join(lines[end:])
    return "".join(lines[:start]), middle, "".join(lines[end:])


def init_new_embeddings(model, tokenizer_dir: Path, tokenizer, base_vocab: int):
    """Initialize each added token's embedding as the mean of the pieces it replaces."""
    path = tokenizer_dir / "new_tokens.json"
    if not path.exists():
        return 0
    pieces = json.loads(path.read_text())
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    with torch.no_grad():
        for text, ids in pieces.items():
            new_id = tokenizer.convert_tokens_to_ids(text)
            if ids and new_id is not None and base_vocab <= new_id < emb.shape[0]:
                emb[new_id] = emb[ids].mean(0)
                head[new_id] = head[ids].mean(0)
    return len(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--tokenizer", default=None, help="custom tokenizer dir; defaults to the base tokenizer")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="artifacts/lora")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fim-rate", type=float, default=0.7)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none", help="e.g. wandb")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer) if args.tokenizer else None
    tokenizer = load_tokenizer(str(tok_dir or args.base))
    tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(args.seed)

    def to_features(batch):
        out = {"input_ids": [], "labels": []}
        for code in batch["code"]:
            prefix, middle, suffix = fim_split(code, rng, args.fim_rate)
            ids, prompt_len = build_training_ids(tokenizer, prefix, middle, suffix)
            if len(ids) > args.max_len:
                # Plain samples can be truncated; a truncated FIM sample would lose its target.
                if prompt_len > 1:
                    continue
                ids = ids[: args.max_len]
            out["input_ids"].append(ids)
            out["labels"].append([-100] * prompt_len + ids[prompt_len:])
        return out

    raw = load_dataset("json", data_files={"train": args.train, "val": args.val})
    raw["val"] = raw["val"].shuffle(seed=args.seed).select(range(min(args.max_val, len(raw["val"]))))
    cols = raw["train"].column_names
    ds = raw.map(to_features, batched=True, remove_columns=cols)
    print(f"train samples: {len(ds['train'])}  val samples: {len(ds['val'])}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base_vocab = model.get_input_embeddings().weight.shape[0]
    extended = len(tokenizer) != base_vocab
    if extended:
        model.resize_token_embeddings(len(tokenizer))
        n = init_new_embeddings(model, tok_dir, tokenizer, base_vocab)
        print(f"resized embeddings to {len(tokenizer)}, initialized {n} new tokens")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # New tokens need trainable embedding and output rows.
        modules_to_save=["embed_tokens", "lm_head"] if extended else None,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            bf16=True,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            group_by_length=True,
            optim="paged_adamw_8bit",
            report_to=args.report_to,
            seed=args.seed,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter + tokenizer to {args.out}")


if __name__ == "__main__":
    main()

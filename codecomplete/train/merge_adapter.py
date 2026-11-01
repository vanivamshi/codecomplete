"""Merge the LoRA adapter into fp16 base weights so serving needs no PEFT overhead.

    python -m codecomplete.train.merge_adapter --adapter artifacts/lora --out artifacts/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from codecomplete.tokenization import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--adapter", default="artifacts/lora")
    ap.add_argument("--out", default="artifacts/merged")
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="cpu")
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()

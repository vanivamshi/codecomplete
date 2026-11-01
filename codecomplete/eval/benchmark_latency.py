"""End-to-end completion latency for different serving setups on the same prompts.

Each --config is `name:key=value,...` with keys:
  model, adapter, load_4bit (0/1), mode (line/block), max_new_tokens, max_prefix_chars

Run the naive setup and the optimized one side by side, then switch one setting at a
time to see what each change is worth (tokenizer vs. stopping vs. context length vs. quantization):

    python -m codecomplete.eval.benchmark_latency \
        --config "naive:model=codellama/CodeLlama-7b-hf,mode=block,max_new_tokens=128,max_prefix_chars=16000" \
        --config "optimized:model=artifacts/merged,mode=line,max_new_tokens=48,max_prefix_chars=4000"
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from codecomplete.generation import Completer, load_model_and_tokenizer

DEFAULTS = {"model": "codellama/CodeLlama-7b-hf", "adapter": "", "load_4bit": "0", "mode": "line",
            "max_new_tokens": "48", "max_prefix_chars": "4000"}


def parse_config(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    cfg = dict(DEFAULTS)
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown key {k!r} in {spec!r}")
        cfg[k] = v
    return name, cfg


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--tasks", default="results/tasks.json", help="created by eval/evaluate.py")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="results/latency.json")
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())[: args.n + args.warmup]
    results = {}
    for spec in args.config:
        name, cfg = parse_config(spec)
        model, tokenizer = load_model_and_tokenizer(cfg["model"], cfg["adapter"] or None, load_4bit=cfg["load_4bit"] == "1")
        completer = Completer(model, tokenizer, max_prefix_chars=int(cfg["max_prefix_chars"]))
        lat, toks, prompt_toks = [], [], []
        for i, t in enumerate(tasks):
            c = completer.complete(t["prefix"], t["suffix"], mode=cfg["mode"], max_new_tokens=int(cfg["max_new_tokens"]))
            if i >= args.warmup:
                lat.append(c.latency_ms)
                toks.append(c.new_tokens)
                prompt_toks.append(c.prompt_tokens)
        results[name] = {
            "config": cfg,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "p50_ms": statistics.median(lat),
            "p95_ms": pct(lat, 0.95),
            "mean_ms": statistics.mean(lat),
            "mean_new_tokens": statistics.mean(toks),
            "mean_prompt_tokens": statistics.mean(prompt_toks),
            "ms_per_new_token": sum(lat) / max(sum(toks), 1),
        }
        del model, completer
        gc.collect()
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n{'config':<14}{'p50 ms':>9}{'p95 ms':>9}{'new tok':>9}{'prompt tok':>12}")
    for name, r in results.items():
        print(f"{name:<14}{r['p50_ms']:>9.0f}{r['p95_ms']:>9.0f}{r['mean_new_tokens']:>9.1f}{r['mean_prompt_tokens']:>12.0f}")


if __name__ == "__main__":
    main()

"""Line-completion accuracy on held-out repos: base model vs fine-tuned model.

For each test function, the cursor goes at the start of a random line (after its
indentation). The model must produce that line. Metrics:
  exact_match    prediction == target (whitespace-stripped)
  edit_sim       difflib similarity in [0, 1], gives credit for near misses

Each model is scored on the same task file, so the comparison is fair.

    python -m codecomplete.eval.evaluate \
        --model base=codellama/CodeLlama-7b-hf \
        --model finetuned=codellama/CodeLlama-7b-hf+artifacts/lora \
        --fim --out results/accuracy.json
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from codecomplete.generation import Completer, load_model_and_tokenizer


def make_tasks(test_path: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    with open(test_path) as f:
        funcs = [json.loads(l) for l in f]
    rng.shuffle(funcs)
    tasks = []
    for fn in funcs:
        lines = fn["code"].splitlines(keepends=True)
        candidates = [i for i in range(1, len(lines)) if len(lines[i].strip()) >= 4 and not lines[i].strip().startswith("#")]
        if not candidates:
            continue
        i = rng.choice(candidates)
        line = lines[i]
        indent = line[: len(line) - len(line.lstrip())]
        tasks.append({
            "repo": fn["repo"],
            "prefix": "".join(lines[:i]) + indent,
            "suffix": "".join(lines[i + 1:]),
            "target": line.strip(),
        })
        if len(tasks) >= n:
            break
    return tasks


def parse_model(spec: str) -> tuple[str, str, str | None]:
    name, path = spec.split("=", 1)
    base, _, adapter = path.partition("+")
    return name, base, adapter or None


def score(pred: str, target: str) -> tuple[bool, float]:
    pred = pred.strip()
    return pred == target, difflib.SequenceMatcher(None, pred, target).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=base_path[+adapter_path]; first one is the baseline")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--tasks", default="results/tasks.json", help="created if missing, reused otherwise")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--fim", action="store_true", help="give the model the code below the cursor too")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/accuracy.json")
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    if tasks_path.exists():
        tasks = json.loads(tasks_path.read_text())
    else:
        tasks = make_tasks(args.test, args.n, args.seed)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks")

    results = {}
    for spec in args.model:
        name, base, adapter = parse_model(spec)
        model, tokenizer = load_model_and_tokenizer(base, adapter, load_4bit=args.load_4bit)
        completer = Completer(model, tokenizer)
        em = es = 0.0
        preds = []
        for t in tqdm(tasks, desc=name):
            c = completer.complete(t["prefix"], t["suffix"] if args.fim else "", mode="line")
            hit, sim = score(c.text, t["target"])
            em += hit
            es += sim
            preds.append({"pred": c.text, "target": t["target"], "exact": hit, "edit_sim": sim})
        results[name] = {
            "model": spec,
            "exact_match": em / len(tasks),
            "edit_sim": es / len(tasks),
        }
        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"preds_{name}.jsonl").write_text("\n".join(json.dumps(p) for p in preds))
        del model, completer
        gc.collect()
        torch.cuda.empty_cache()

    baseline = next(iter(results.values()))
    for r in results.values():
        r["exact_match_rel_gain_pct"] = 100 * (r["exact_match"] - baseline["exact_match"]) / max(baseline["exact_match"], 1e-9)
        r["edit_sim_rel_gain_pct"] = 100 * (r["edit_sim"] - baseline["edit_sim"]) / max(baseline["edit_sim"], 1e-9)
    report = {"tasks": len(tasks), "fim": args.fim, "results": results}
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n{'model':<14}{'exact match':>13}{'edit sim':>11}{'EM gain':>10}")
    for name, r in results.items():
        print(f"{name:<14}{r['exact_match']:>13.3f}{r['edit_sim']:>11.3f}{r['exact_match_rel_gain_pct']:>9.1f}%")


if __name__ == "__main__":
    main()

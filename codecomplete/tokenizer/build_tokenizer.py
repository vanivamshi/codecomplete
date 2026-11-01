"""Extend the CodeLlama tokenizer with Python-specific tokens.

Python is full of multi-token patterns: indentation runs, `self.`, `return `,
`def __init__(self`, and so on. Each one costs extra decode steps. This script:

1. tokenizes a corpus sample with the base tokenizer,
2. mines the token n-grams whose merge would save the most tokens,
3. adds the best ones as new (non-special) tokens,
4. drops any token that breaks the encode -> decode round trip, and
5. reports how many fewer tokens the same code now takes.

New embedding rows are initialized in training as the mean of the pieces each
new token replaces (`new_tokens.json`).

    python -m codecomplete.tokenizer.build_tokenizer --data data/processed/train.jsonl --out artifacts/tokenizer
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from tokenizers import AddedToken

from codecomplete.tokenization import encode_continuation, load_tokenizer

# Always worth having: newline + indentation for depths 1-6 (4-space style).
SEED_TOKENS = ["\n" + " " * (4 * d) for d in range(1, 7)] + [" " * (4 * d) for d in range(2, 7)]

# Only mine candidates that start at a natural boundary (whitespace or punctuation) and
# contain no partial identifier on the right edge, so splitting text on them is safe.
CANDIDATE_RE = re.compile(r"^(\n?[ ]+|[ (\[{.,:=])[\w.()\[\]:,=\"' ]*[^\w]$|^\n[ ]+$")


def load_sample(path: str, n: int, seed: int) -> list[str]:
    with open(path) as f:
        codes = [json.loads(l)["code"] for l in f]
    random.Random(seed).shuffle(codes)
    return codes[:n]


def mine_candidates(tokenizer, codes: list[str], max_n: int, top_k: int) -> list[str]:
    savings = Counter()
    for code in codes:
        ids = tokenizer.encode(code, add_special_tokens=False)
        for n in range(2, max_n + 1):
            for i in range(len(ids) - n + 1):
                savings[tuple(ids[i:i + n])] += n - 1
    picked, seen = [], set()
    for gram, _ in savings.most_common(top_k * 20):
        # decode() drops the leading SentencePiece space of a sequence; rebuild from pieces.
        text ="".join(tokenizer.convert_ids_to_tokens(list(gram))).replace("▁", " ").replace("<0x0A>", "\n")
        if text in seen or len(text) < 3 or "<0x" in text or not CANDIDATE_RE.match(text):
            continue
        seen.add(text)
        picked.append(text)
        if len(picked) >= top_k:
            break
    return picked


def count_tokens(tokenizer, codes: list[str]) -> int:
    return sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in codes)


def roundtrip_failures(tokenizer, codes: list[str]) -> list[str]:
    return [c for c in codes if tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)) != c]


def build(base: str, tokens: list[str]):
    tok = load_tokenizer(base)
    tok.add_tokens([AddedToken(t, normalized=False, special=False, lstrip=False, rstrip=False) for t in tokens])
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--eval-data", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="artifacts/tokenizer")
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--num-tokens", type=int, default=256)
    ap.add_argument("--max-ngram", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = load_tokenizer(args.base)
    train_codes = load_sample(args.data, args.sample, args.seed)
    eval_codes = load_sample(args.eval_data, 2000, args.seed)

    mined = mine_candidates(base, train_codes, args.max_ngram, args.num_tokens)
    tokens = list(dict.fromkeys(SEED_TOKENS + mined))
    print(f"{len(tokens)} candidate tokens; top 20 mined: {mined[:20]!r}")

    # Drop tokens that appear in samples whose round trip breaks, until none break.
    check = train_codes[:3000]
    for round_ in range(5):
        tok = build(args.base, tokens)
        bad = roundtrip_failures(tok, check)
        print(f"round {round_}: {len(tokens)} tokens, {len(bad)} round-trip failures")
        if not bad:
            break
        blame = Counter(t for c in bad for t in tokens if t in c)
        base_bad = set(roundtrip_failures(base, bad))
        if len(base_bad) == len(bad):
            print("remaining failures also fail with the base tokenizer; ignoring")
            break
        worst = {t for t, _ in blame.most_common(max(1, len(blame) // 4))}
        tokens = [t for t in tokens if t not in worst]

    before, after = count_tokens(base, eval_codes), count_tokens(tok, eval_codes)
    report = {
        "base": args.base,
        "added_tokens": len(tokens),
        "vocab_size": len(tok),
        "eval_functions": len(eval_codes),
        "tokens_before": before,
        "tokens_after": after,
        "reduction_pct": round(100 * (before - after) / before, 2),
    }
    print(json.dumps(report, indent=2))

    out = Path(args.out)
    tok.save_pretrained(out)
    # For embedding init: which base-tokenizer pieces each new token replaces.
    pieces = {t: encode_continuation(base, t) for t in tokens}
    (out / "new_tokens.json").write_text(json.dumps(pieces, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

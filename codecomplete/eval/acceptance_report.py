"""Acceptance rate from the VS Code extension's telemetry log.

acceptance rate = suggestions accepted (Tab) / suggestions shown

A suggestion counts as "shown" once per unique id, i.e. once per completion the
server returned and the editor displayed. Report the definition together with the
number, because other definitions (e.g. counting each keystroke) give very different rates.

    python -m codecomplete.eval.acceptance_report --log data/telemetry.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/telemetry.jsonl")
    ap.add_argument("--min-shown", type=int, default=20, help="ignore users with fewer shown suggestions")
    args = ap.parse_args()

    shown, accepted = defaultdict(set), defaultdict(set)
    with open(args.log) as f:
        for line in f:
            e = json.loads(line)
            (shown if e["event"] == "shown" else accepted)[e["user"]].add(e["id"])

    rows = []
    for user, ids in shown.items():
        if len(ids) < args.min_shown:
            continue
        acc = len(accepted[user] & ids)
        rows.append((user, len(ids), acc, acc / len(ids)))
    if not rows:
        raise SystemExit("no users with enough suggestions yet")

    total_shown = sum(r[1] for r in rows)
    total_acc = sum(r[2] for r in rows)
    print(f"{'user':<14}{'shown':>8}{'accepted':>10}{'rate':>8}")
    for user, s, a, r in sorted(rows, key=lambda r: -r[1]):
        print(f"{user[:12]:<14}{s:>8}{a:>10}{r:>8.1%}")
    print(f"\nusers: {len(rows)}   shown: {total_shown}   accepted: {total_acc}")
    print(f"pooled acceptance rate:       {total_acc / total_shown:.1%}")
    print(f"mean per-user acceptance rate: {sum(r[3] for r in rows) / len(rows):.1%}")


if __name__ == "__main__":
    main()

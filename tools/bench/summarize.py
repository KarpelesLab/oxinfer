#!/usr/bin/env python3
"""Summarize bench.py results: median [min-max] per variant and workload, and greedy agreement.

usage: summarize.py results.jsonl [more.jsonl ...] [--agree N]

Throughput is prompt tok/s for pp* workloads and generated tok/s for code/prose. "acc" is the
share of drafted tokens the target accepted (summed over runs). The agreement section compares
the first N (default 64) greedy tokens of pairs of variants per workload and repetition and
prints the first index where they differ ("=" when identical).
"""

import json
import statistics
import sys
from collections import defaultdict

PAIRS = [
    ("ox", "llama-metal"),
    ("ox-cpu", "llama-cpu"),
    ("ox", "ox-cpu"),
    ("llama-metal", "llama-cpu"),
    ("ox-mtp", "ox"),
    ("ox-mtpfrom", "ox"),
    ("llama-mtp3", "llama-metal"),
    ("llama-fast3", "llama-metal"),
]


def main():
    args = sys.argv[1:]
    n_agree = 64
    if "--agree" in args:
        i = args.index("--agree")
        n_agree = int(args[i + 1])
        del args[i : i + 2]
    recs = [json.loads(line) for f in args for line in open(f) if line.strip()]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["tag"], r["variant"], r["workload"])].append(r)

    print("| model | variant | workload | tok/s median [min-max] | runs | load median | acceptance |")
    print("|---|---|---|---|---|---|---|")
    for (tag, v, w), rs in sorted(groups.items()):
        key = "pp_tps" if w.startswith("pp") else "tg_tps"
        xs = [r[key] for r in rs if r.get(key)]
        if not xs:
            continue
        loads = [r["load"] for r in rs if r.get("load") is not None]
        dn = sum(r.get("draft_n", 0) for r in rs)
        acc = f"{100 * sum(r.get('draft_acc', 0) for r in rs) / dn:.0f}%" if dn else ""
        extra = f" ({rs[0]['prompt_tokens']} tok)" if w.startswith("pp") else f" ({statistics.median(r['gen_tokens'] for r in rs):.0f} tok)"
        print(
            f"| {tag} | {v} | {w}{extra} | {statistics.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}] | {len(xs)} | "
            f"{statistics.median(loads) if loads else '-'} | {acc} |"
        )

    print(f"\nGreedy agreement (first index of divergence within the first {n_agree} tokens, per repetition):")
    by = defaultdict(dict)
    for r in recs:
        if not r["workload"].startswith("pp"):
            by[(r["tag"], r["workload"], r["rep"])][r["variant"]] = r.get("tokens", [])
    for a, b in PAIRS:
        rows = defaultdict(list)
        for (tag, w, rep), vs in sorted(by.items()):
            if a in vs and b in vs:
                x, y = vs[a][:n_agree], vs[b][:n_agree]
                m = min(len(x), len(y))
                d = next((i for i in range(m) if x[i] != y[i]), None)
                rows[(tag, w)].append("=" if d is None else str(d))
        for (tag, w), ds in rows.items():
            print(f"  {tag} {w:6s} {a} vs {b}: {' '.join(ds)}")


if __name__ == "__main__":
    main()

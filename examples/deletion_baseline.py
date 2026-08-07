"""The baseline this work never compared against: delete the stale message.

Every condition in the paper is measured against the CONFLICTED transcript,
where several models sit at 0 %. Nothing was measured against the obvious
alternative a practitioner has: drop the stale instruction from the history and
send the rest. That condition has been in results/ the whole time under the name
`ceiling`, treated as an upper bound rather than as a competitor.

It is a fair competitor, because V-Steer needs exactly what deletion needs -- a
label saying which message is stale. Neither is automatic. So "steering beats
doing nothing" is the wrong comparison; "steering beats deleting" is the right
one, and it is the one a reviewer will make.

Reads stored generations only. No model, no GPU.

    python examples/deletion_baseline.py
    python examples/deletion_baseline.py --metric compliance
"""

from __future__ import annotations

import argparse
import collections
import json
import os

NAMES = {"tiny": "Qwen2.5-0.5B", "small": "Qwen2.5-1.5B", "q3b": "Qwen2.5-3B",
         "q7b": "Qwen2.5-7B", "mid": "Qwen3-4B", "llama": "Llama-3.1-8B",
         "phi": "Phi-3.5-mini", "olmo": "OLMo-2-7B", "aya": "Aya-8B",
         "commandr": "Command-R7B"}
ORDER = ["tiny", "small", "q3b", "mid", "q7b", "olmo", "phi", "llama",
         "aya", "commandr"]


def load(stem, model):
    """Prefer the rescored file: the checkers had six bugs (see the draft, 7)."""
    for cand in (f"results/{stem}_{model}.rescored.json",
                 f"results/{stem}_{model}.json"):
        if os.path.exists(cand):
            return json.load(open(cand)), cand
    return None, None


def score(records, metric):
    if metric == "compliance":
        hit = sum(1 for r in records if str(r.get("which_rule_won")) == "system")
    elif metric == "recall":
        hit = sum(1 for r in records if str(r.get("recalled")) == "True")
    else:  # useful = compliance AND recall, the draft's headline measure
        hit = sum(1 for r in records
                  if str(r.get("which_rule_won")) == "system"
                  and str(r.get("recalled")) == "True")
    return hit, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="useful",
                    choices=["useful", "compliance", "recall"])
    ap.add_argument("--out", default="results/deletion_baseline.json")
    args = ap.parse_args()

    rows, wins = [], collections.Counter()
    for m in ORDER:
        cd, cf = load("ceiling", m)
        ad, af = load("atlas", m)
        if not cd or not ad:
            continue
        dn, dt = score(cd["records"], args.metric)
        cells = collections.defaultdict(list)
        for r in ad["records"]:
            cells[(r["gamma_plus"], r["gamma_minus"])].append(r)
        best_pc, best_cell, best_n = -1.0, None, 0
        for k, v in sorted(cells.items()):
            hit, tot = score(v, args.metric)
            if tot and hit / tot > best_pc:
                best_pc, best_cell, best_n = hit / tot, k, tot
        d_pc = 100 * dn / dt
        s_pc = 100 * best_pc
        winner = "deletion" if d_pc > s_pc else "steering"
        wins[winner] += 1
        rows.append(dict(model=m, name=NAMES.get(m, m), n_delete=dt,
                         delete_pct=round(d_pc, 1), steer_pct=round(s_pc, 1),
                         best_gamma_plus=best_cell[0], best_gamma_minus=best_cell[1],
                         n_steer=best_n, n_cells=len(cells), winner=winner))

    w = max(len(r["name"]) for r in rows)
    print(f"metric = {args.metric}   (steering cell is the best of all cells, "
          f"chosen post hoc on the same data)\n")
    print(f"{'model':<{w}}  {'delete stale':>13}  {'best steering':>14}  "
          f"{'at':>14}  winner")
    for r in sorted(rows, key=lambda x: -x["delete_pct"]):
        cell = f"g+={r['best_gamma_plus']},g-={r['best_gamma_minus']}"
        print(f"{r['name']:<{w}}  {r['delete_pct']:12.1f} %  "
              f"{r['steer_pct']:13.1f} %  {cell:>14}  {r['winner']}")
    print(f"\ndeletion wins on {wins['deletion']} of {len(rows)} models, "
          f"steering on {wins['steering']}.")
    print("Both need the same input: a label marking which message is stale.")

    json.dump({"metric": args.metric,
               "note": ("`ceiling` is the no-conflict condition -- the stale "
                        "instruction removed, the rest of the transcript kept. "
                        "It still contains an assistant turn reading 'Noted.', "
                        "so it is not a clean no-history condition. The steering "
                        "column is a post-hoc max over all cells and is if "
                        "anything flattered."),
               "wins": dict(wins), "rows": rows},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()

"""How many KV heads does DLA flag, on the dataset the paper's table used?

Open question from the correspondence: with the role-level assignment and
eps = 0, 96% of KV heads came out flagged -- on average 278 of 288 on Qwen and
247 of 256 on Llama. The authors' answer was that the number looked reasonable,
"although that table is experimented on Control Illusion". So the two figures
are measured on different data and cannot be compared. This measures ours on
theirs.

Control Illusion (Geng et al., AAAI 2026, arXiv:2502.15851) is a set of
deliberately contradictory constraint pairs over 100 real task instructions --
"answer in English" against "answer in French", uppercase against lowercase.
It is a priority dataset, not a staleness one, so it is mapped in the only way
that keeps the question honest: constraint1 is the rule in force now, constraint2
is the one from earlier in the conversation, and the task instruction is what
gets asked. Nothing is generated here -- the flagged fraction is a property of
the prefill, so this needs one forward pass per case and no decoding.

    git clone https://github.com/yilin-geng/llm_instruction_conflicts /tmp/ci
    python examples/heads_on_control_illusion.py --data /tmp/ci/data --model mid

Their data carries no licence, so it is never vendored here -- clone it, point
at it, and only our numbers get written out.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from oldnews.attribution import inversion, prefill, span_attributions
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import select_heads


def build(rec: dict) -> list[Msg]:
    """constraint1 = now (privileged), constraint2 = earlier (demoted)."""
    return [
        Msg("system", rec["constraint1"], epoch=1),
        Msg("user", rec["constraint2"], epoch=0),
        Msg("assistant", "understood, I will do that from now on.", epoch=0),
        Msg("user", rec["base_instruction"], epoch=1),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Control Illusion data/ dir")
    ap.add_argument("--model", default="mid")
    ap.add_argument("--file", default="conflicting_instructions.jsonl")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = os.path.join(args.data, args.file)
    rows = [json.loads(l) for l in open(path) if l.strip()][: args.limit]
    print(f"{len(rows)} pripadu z {args.file}, model {args.model}, eps={args.eps}\n")

    model, tok = load(args.model)
    pol = SteerPolicy(mode="binary", eps=args.eps)

    per_conflict: dict[str, list[float]] = {}
    totals, flagged_counts, n_heads = [], [], None

    for i, rec in enumerate(rows, 1):
        r = render(tok, build(rec), current_epoch=1)
        ids = torch.tensor([r.input_ids]).to(next(model.parameters()).device) \
            if isinstance(r.input_ids, list) else r.input_ids
        with torch.no_grad():
            pre = prefill(model, ids)
            phi, _ = span_attributions(model, pre, r.levels)
            # `(0,)` here was the bug: level 0 is the SYSTEM message, not the
            # demoted one, so this computed phi[system] - phi[system] = 0 and
            # flagged nothing on every model and every eps. The demoted levels
            # come from the policy.
            delta = inversion(phi, pol.privileged, pol.demoted)
            mask = select_heads(delta, pre.n_rep, args.eps, "max")
        frac = float(mask.float().mean())
        n_heads = mask.numel()
        flagged_counts.append(int(mask.sum()))
        totals.append(frac)
        per_conflict.setdefault(rec["conflict_name"], []).append(frac)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}  prumer zatim {statistics.mean(totals):.1%}",
                  flush=True)

    print(f"\nCELKEM: {statistics.mean(totals):.1%} KV hlav oznaceno "
          f"({statistics.mean(flagged_counts):.0f} z {n_heads})")
    print("podle typu konfliktu:")
    for k in sorted(per_conflict):
        v = per_conflict[k]
        print(f"  {k:38s} {statistics.mean(v):6.1%}  (n={len(v)})")

    out = args.out or f"results/heads_ci_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({
        "dataset": "Control Illusion (arXiv:2502.15851), not redistributed",
        "file": args.file, "model": args.model, "eps": args.eps,
        "n_cases": len(rows), "n_kv_heads": n_heads,
        "mean_flagged_fraction": round(statistics.mean(totals), 4),
        "mean_flagged_count": round(statistics.mean(flagged_counts), 1),
        "per_conflict": {k: round(statistics.mean(v), 4)
                         for k, v in sorted(per_conflict.items())},
        "note": ("Mapped as priority -> staleness: constraint1 privileged at the "
                 "current epoch, constraint2 demoted to an earlier one. Their "
                 "table measures the same quantity on the same data but under "
                 "their own role assignment, so a gap between the two numbers is "
                 "informative about the mapping, not only about the model."),
    }, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

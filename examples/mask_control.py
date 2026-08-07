"""Does the ATTRIBUTED mask beat a random one of the same size?

This is the control 6b argues from but never ran. 6b says the head-selection
score is near-noise and the union rule inflates it into a 94-99 % mask, and
concludes that "edit the heads that attend to the stale span" is hard to tell
apart from "scale all of V". That is an argument from a statistic. The causal
version is one experiment:

    selected   the mask the criterion picks           (what the method does)
    random     a random mask of the SAME size          (does attribution matter?)
    all        every KV head                           (is selection needed?)
    none       no edit                                 (floor)

If `random` matches `selected`, the diagnosis is decorative and 6b's conclusion
is established causally rather than statistically. If `selected` beats `random`,
delta carries signal the sign statistics cannot see and 6b is wrong -- which is
the outcome the repo's own vsteer.select_heads docstring hints at, since `mean`
vs `max` grouping is worth 17 points on StaleSet, and that is nothing but a
change of mask.

Matched size matters: `random` draws exactly as many KV heads as `selected`
flagged for that case, so the two arms differ only in WHICH heads, never how
many. Seeded per case so the run is reproducible.

    PYTHONPATH=.:examples python examples/mask_control.py --model llama
    PYTHONPATH=.:examples python examples/mask_control.py --model mid \\
        --gamma-plus 4 --gamma-minus 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from failure_atlas import build_cases, triage
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import render
from oldnews.vsteer import generate

ARMS = ("none", "selected", "random", "all")


def make_override(arm: str, seed: int):
    """A head_mask_override for one arm. Returns None for the unedited arms."""
    if arm in ("none", "selected"):
        return None

    def override(delta, n_rep, selected):
        if arm == "all":
            return torch.ones_like(selected)
        k = int(selected.sum())
        flat = torch.zeros(selected.numel(), dtype=torch.bool)
        if k:
            g = torch.Generator().manual_seed(seed)
            pick = torch.randperm(selected.numel(), generator=g)[:k]
            flat[pick] = True
        return flat.view_as(selected)

    return override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--gamma-plus", type=float, default=2.5)
    ap.add_argument("--gamma-minus", type=float, default=0.75)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    out = args.out or f"results/maskctl_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    done, records = set(), []
    if os.path.exists(out):
        records = json.load(open(out))["records"]
        done = {(r["arm"], r["family"], r["question"]) for r in records}
        print(f"resuming: {len(records)} records already done")

    cases = build_cases()
    model, tok = load(args.model)
    t0 = time.time()

    for arm in arms:
        pol = None if arm == "none" else SteerPolicy(
            mode="binary", gamma_plus=args.gamma_plus,
            gamma_minus=args.gamma_minus)
        useful = compliant = recalled = 0
        sizes = []
        for i, c in enumerate(cases):
            key = (arm, c["family"], c["fact"].question)
            if key in done:
                continue
            r = render(tok, c["messages"], current_epoch=1)
            text, report = generate(
                model, tok, r, policy=pol,
                max_new_tokens=args.max_new_tokens, current_epoch=1,
                head_mask_override=make_override(arm, seed=1000 + i))
            verdict = c["check"](text)
            tags = triage(text, c["fact"])
            ok = verdict == "system" and tags["recalled"]
            useful += ok
            compliant += verdict == "system"
            recalled += tags["recalled"]
            if report is not None:
                sizes.append(report.n_heads_edited / max(report.n_heads_total, 1))
            records.append(dict(
                model=args.model, arm=arm, gamma_plus=args.gamma_plus,
                gamma_minus=args.gamma_minus, family=c["family"],
                question=c["fact"].question, which_rule_won=verdict,
                heads_edited=(report.n_heads_edited if report else 0),
                heads_total=(report.n_heads_total if report else 0),
                **tags, text=text))
        n = len(cases)
        frac = f"{100*sum(sizes)/len(sizes):.1f} %" if sizes else "-"
        print(f"[{time.time()-t0:6.0f}s] {arm:<9} useful {useful:2d}/{n}  "
              f"compliant {compliant:2d}/{n}  recall {recalled:2d}/{n}  "
              f"mask {frac}", flush=True)
        json.dump({"model": args.model, "gamma_plus": args.gamma_plus,
                   "gamma_minus": args.gamma_minus, "arms": arms,
                   "max_new_tokens": args.max_new_tokens, "greedy": True,
                   "note": ("random draws the same NUMBER of KV heads as selected "
                            "flagged for that case, seeded per case. Only WHICH "
                            "heads differs. useful = compliance AND recall."),
                   "records": records}, open(out, "w"), ensure_ascii=False)

    print("\nselected vs random is the whole experiment: if they tie, the "
          "attribution is not doing the work.\n->", out)


if __name__ == "__main__":
    main()

"""When the stale instruction and a fact you still need share ONE message.

Every case in this repo so far puts the stale instruction in its own message,
separate from the turn carrying the fact. That hands the deletion baseline its
win by construction (see README): you can excise the instruction and keep
everything you need, so of course deleting beats editing.

The case that decides whether the edit has any use is the entangled one:

    "From now on always reply in all lowercase. My order number is 4417-B."

Delete that message and the order number goes with it. The V-edit demotes the
message's values while the fact stays in context -- but the fact is now INSIDE
the demoted span, which is exactly where 3 says suppression costs recall. So it
is a real test, not a rigged one: the edit may well fail too.

Four arms, all on the same 36 items:

    none        the entangled transcript, no intervention
    delete      drop the entangled message and its acknowledgement
    rewrite     keep the message but strip the instruction, leaving the fact
    steer       keep everything, apply the V-edit

`rewrite` is the strong baseline and it is deliberately included: it is what a
careful engineer does instead of deleting. Note what it costs, though -- it
needs to know which SPAN of the message is the instruction, whereas the edit and
the delete need only a message-level epoch label.

    PYTHONPATH=.:examples python examples/entangled.py --model llama
"""

from __future__ import annotations

import argparse
import json
import os
import time

from failure_atlas import FAMILIES, triage
from oldnews.evals.recall import FACTS
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import generate

ARMS = ("none", "delete", "rewrite", "steer")


def build_cases(arm):
    """The same 36 items under one of the four interventions."""
    cases = []
    for fam in FAMILIES:
        for fact in FACTS:
            # the stale instruction and the needed fact, in ONE user turn
            entangled = f"{fam['stale']} {fact.statement}"
            if arm == "delete":
                # the whole turn goes, and the fact with it
                msgs = [Msg("system", fam["system"], epoch=1)]
            elif arm == "rewrite":
                # surgical: the instruction clause is stripped, the fact kept
                msgs = [Msg("system", fam["system"], epoch=1),
                        Msg("user", fact.statement, epoch=0),
                        Msg("assistant", "Noted.", epoch=0)]
            else:
                msgs = [Msg("system", fam["system"], epoch=1),
                        Msg("user", entangled, epoch=0),
                        Msg("assistant", fam["ack"], epoch=0)]
            msgs.append(Msg("user", fact.question, epoch=1))
            cases.append(dict(family=fam["key"], check=fam["check"],
                              fact=fact, messages=msgs))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--gamma-plus", type=float, default=2.5)
    ap.add_argument("--gamma-minus", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    out = args.out or f"results/entangled_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done, records = set(), []
    if os.path.exists(out):
        records = json.load(open(out))["records"]
        done = {(r["arm"], r["family"], r["question"]) for r in records}
        print(f"resuming: {len(records)} records already done")

    model, tok = load(args.model)
    t0 = time.time()
    for arm in arms:
        # only the steer arm gets a policy; the others are prompt-level fixes
        pol = SteerPolicy(mode="binary", gamma_plus=args.gamma_plus,
                          gamma_minus=args.gamma_minus) if arm == "steer" else None
        cases = build_cases(arm)
        useful = compliant = recalled = 0
        for c in cases:
            key = (arm, c["family"], c["fact"].question)
            if key in done:
                continue
            r = render(tok, c["messages"], current_epoch=1)
            text, _ = generate(model, tok, r, policy=pol,
                               max_new_tokens=args.max_new_tokens,
                               current_epoch=1)
            verdict = c["check"](text)
            tags = triage(text, c["fact"])
            useful += (verdict == "system" and tags["recalled"])
            compliant += verdict == "system"
            recalled += tags["recalled"]
            records.append(dict(model=args.model, arm=arm, family=c["family"],
                                question=c["fact"].question,
                                which_rule_won=verdict, **tags, text=text))
        n = len(cases)
        print(f"[{time.time()-t0:6.0f}s] {arm:<8} useful {useful:2d}/{n}   "
              f"compliant {compliant:2d}/{n}   recall {recalled:2d}/{n}",
              flush=True)
        json.dump({"model": args.model, "gamma_plus": args.gamma_plus,
                   "gamma_minus": args.gamma_minus, "arms": arms,
                   "greedy": True, "max_new_tokens": args.max_new_tokens,
                   "note": ("The stale instruction and the needed fact share one "
                            "user turn. 'delete' loses the fact with the message; "
                            "'rewrite' strips the instruction and keeps the fact "
                            "but needs span-level knowledge, not just an epoch."),
                   "records": records}, open(out, "w"), ensure_ascii=False)

    print("\nuseful = obeys the CURRENT rule AND still recalls the fact.\n->", out)


if __name__ == "__main__":
    main()

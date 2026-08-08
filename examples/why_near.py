"""Is the substitution the model's own second choice?

`bagr` becomes `Bagel`, `4417-B` becomes `4417`, `19:40` becomes `19:00`. The
write-up calls this "sliding toward a high-prior string", which is a story, not
a measurement. This measures it.

At the position where the answer value is about to be emitted, the unsteered
model has a full distribution over next tokens. If the substitution story is
right, the token the *steered* model emits there was already sitting near the
top of that unsteered distribution — the edit did not invent it, it removed
enough evidence for the runner-up to win. If instead the emitted token is at
rank 500 unsteered, the edit is producing something the model did not have, and
"falls back on the prior" is wrong.

Method, per case:

  1. Run unsteered, greedily, and find the first position where the generated
     text contains the gold value. That position is the readout point.
  2. Teacher-force the *same* prefix under both conditions and take the next-token
     distribution at that point.
  3. Report, for the token the steered run actually emits there: its rank and
     probability under the unsteered distribution, and the gold token's rank and
     probability under the steered one.

Two numbers decide it. If the steered choice has a low unsteered rank (say ≤ 10)
the prior-fallback reading holds. If the gold token's steered rank stays low
while its probability collapses, the evidence is being attenuated rather than
overwritten -- which is the prediction from scaling V while leaving attention
alone.

Teacher forcing matters: comparing two independently generated texts confounds
the readout position with everything that came before it. Both conditions must
be asked the same question at the same place.

    PYTHONPATH=.:examples python examples/why_near.py --model llama
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from failure_atlas import FAMILIES
from oldnews.evals.recall import FACTS
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import generate, steer

BY_KEY = {f["key"]: f for f in FAMILIES}


def build(fam, fact):
    return [Msg("system", fam["system"], epoch=1),
            Msg("user", fam["stale"], epoch=0),
            Msg("assistant", fam["ack"], epoch=0),
            Msg("user", fact.statement, epoch=0),
            Msg("assistant", fam["note"], epoch=0),
            Msg("user", fact.question, epoch=1)]


@torch.no_grad()
def dist_at(model, tok, rendered, policy, prefix_ids, current_epoch=1):
    """Next-token distribution after `prefix_ids`, under `policy`.

    The prefix is appended to the (possibly edited) prefill cache, so both
    conditions are asked the same question at the same position.
    """
    if policy is None:
        ids = torch.tensor([rendered.input_ids + prefix_ids], device=model.device)
        return torch.softmax(model(ids).logits[0, -1], dim=-1)
    pre, _report, logits = steer(model, tok, rendered, policy,
                                 current_epoch=current_epoch)
    if not prefix_ids:
        return torch.softmax(logits, dim=-1)
    ids = torch.tensor([prefix_ids], device=model.device)
    out = model(ids, past_key_values=pre.cache, use_cache=True)
    return torch.softmax(out.logits[0, -1], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--gamma-plus", type=float, default=4.0)
    ap.add_argument("--gamma-minus", default="0.75,0.9,0.95")
    ap.add_argument("--families", default="case",
                    help="constraint families to read out at, or 'all'. One "
                         "family is 6 readout points per gamma, which is thin "
                         "for a distribution-level claim.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/whynear_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    model, tok = load(args.model)
    gms = [float(x) for x in args.gamma_minus.split(",")]
    fams = ([f["key"] for f in FAMILIES] if args.families == "all"
            else [k for k in args.families.split(",") if k])
    rows = []
    for famkey, fact in ((f, x) for f in fams for x in FACTS):
        fam = BY_KEY[famkey]
        msgs = build(fam, fact)
        r = render(tok, msgs, current_epoch=1)
        clean, _ = generate(model, tok, r, policy=None, max_new_tokens=48,
                            current_epoch=1)
        gold = fact.needles[0]
        # Readout point: the shortest generated prefix that already contains the
        # gold string. Everything before it is shared context for both runs.
        ids, cut = tok(clean, add_special_tokens=False)["input_ids"], None
        for k in range(1, len(ids) + 1):
            piece = tok.decode(ids[:k])
            if gold.lower().replace(" ", "") in piece.lower().replace(" ", ""):
                cut = k - 1          # the step that emits the value
                break
        if cut is None:
            rows.append({"fact": gold, "family": famkey,
                         "skipped": "gold not generated unsteered"})
            print(f"  {gold:>8}  preskoceno (bez editu fakt nevygeneroval)", flush=True)
            continue
        prefix = ids[:cut]
        p_clean = dist_at(model, tok, r, None, prefix)
        gold_tok = ids[cut]
        for gm in gms:
            pol = SteerPolicy(mode="binary", gamma_plus=args.gamma_plus,
                              gamma_minus=gm)
            p_steer = dist_at(model, tok, r, pol, prefix)
            chosen = int(p_steer.argmax())
            # rank = how many tokens beat it
            rank_clean = int((p_clean > p_clean[chosen]).sum()) + 1
            rank_gold_steer = int((p_steer > p_steer[gold_tok]).sum()) + 1
            rows.append({
                "fact": gold, "family": famkey, "gamma_minus": gm,
                "prefix": tok.decode(prefix),
                "gold_token": tok.decode([gold_tok]),
                "steered_token": tok.decode([chosen]),
                "gold_p_clean": float(p_clean[gold_tok]),
                "gold_p_steered": float(p_steer[gold_tok]),
                "gold_rank_steered": rank_gold_steer,
                "steered_token_p_clean": float(p_clean[chosen]),
                "steered_token_rank_clean": rank_clean,
            })
            print(f"  {famkey:<7} {gold:>8} g-={gm:<5g} vybral {tok.decode([chosen])!r:>12} "
                  f"(bez editu rank {rank_clean:5d}, p={float(p_clean[chosen]):.4f})   "
                  f"gold p {float(p_clean[gold_tok]):.3f} -> {float(p_steer[gold_tok]):.5f}, "
                  f"rank {rank_gold_steer}", flush=True)
        json.dump({"model": args.model, "gamma_plus": args.gamma_plus,
                   "family": ",".join(fams), "rows": rows},
                  open(out, "w"), ensure_ascii=False, indent=1)
    print("\n->", out)


if __name__ == "__main__":
    main()

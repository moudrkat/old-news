#!/usr/bin/env python3
"""One head, one prompt: how much each token attends, and how loudly it speaks.

The Space explains the method in words -- multiply the cached values of a span
and leave the attention weights alone -- and then shows none of it. This
measures the two quantities that sentence is about, so the diagram can be drawn
from numbers instead of from imagination.

For the last prompt position (the one that chooses the first answer token), and
for one layer and one KV head, it records per source token:

  alpha   the attention weight it receives, softmaxed over the prompt
  vnorm   the L2 norm of its cached value vector
  role    which turn it belongs to, so the page can colour the spans

The product alpha * vnorm is the magnitude of that token's contribution to the
head's output. It is a magnitude, not the output itself -- the real output is a
vector sum and terms can cancel -- and the page says so.

    PYTHONPATH=.:examples python space/aorus/measure_head.py --model llama
"""
import argparse
import json
import os

import torch

from failure_atlas import FAMILIES
from oldnews.evals.recall import FACTS
from oldnews.model import load
from oldnews.transcript import Msg, render

BY_KEY = {f["key"]: f for f in FAMILIES}


def build(fam, fact):
    """Same five-turn conversation the atlas uses, with the roles kept."""
    return [
        (Msg("system", fam["system"], epoch=1), "system"),
        (Msg("user", fam["stale"], epoch=0), "stale"),
        (Msg("assistant", fam["ack"], epoch=0), "ack"),
        (Msg("user", fact.statement, epoch=0), "fact"),
        (Msg("user", fact.question, epoch=1), "ask"),
    ]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--family", default="length")
    ap.add_argument("--fact", default="19:40")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: two thirds of the way up, where the "
                         "hierarchy attention pattern is clearest in Fig. 2 of "
                         "arXiv:2607.26228")
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--out", default="space/data/head.json")
    args = ap.parse_args()

    fam = BY_KEY[args.family]
    fact = next(f for f in FACTS if f.needles[0] == args.fact)
    pairs = build(fam, fact)
    model, tok = load(args.model)

    r = render(tok, [m for m, _ in pairs], current_epoch=1)
    ids = r.input_ids
    L = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else int(L * 2 / 3)

    # The fast attention kernels never build the attention matrix, which is
    # exactly why the paper prefers editing V over editing alpha -- and it is
    # why alpha has to be asked for explicitly here.
    for setter in ("set_attn_implementation", "_set_attn_implementation"):
        if hasattr(model, setter):
            try:
                getattr(model, setter)("eager")
                break
            except Exception:
                pass
    else:
        model.config._attn_implementation = "eager"

    out = model(torch.tensor([ids], device=model.device),
                output_attentions=True, use_cache=True)
    if not out.attentions:
        raise SystemExit("no attentions returned; the eager switch did not take")
    # [B, H_q, T, T] -> the query row for the final prompt position
    attn = out.attentions[layer][0, :, -1, :].float().cpu()
    # transformers 5 returns a Cache object rather than the old tuple-of-tuples
    cache = out.past_key_values
    if hasattr(cache, "layers"):
        v = cache.layers[layer].values[0].float().cpu()
    elif hasattr(cache, "value_cache"):
        v = cache.value_cache[layer][0].float().cpu()
    else:
        v = cache[layer][1][0].float().cpu()      # [H_kv, T, D]
    n_kv = v.shape[0]
    n_rep = attn.shape[0] // n_kv

    toks = [tok.decode([i]) for i in ids]

    # Tag every token with the turn it came from by walking the rendered spans.
    role = ["chrome"] * len(ids)
    for (msg, name) in pairs:
        piece = tok(msg.content, add_special_tokens=False)["input_ids"]
        for start in range(len(ids) - len(piece) + 1):
            if ids[start:start + len(piece)] == piece:
                for i in range(start, start + len(piece)):
                    role[i] = name
                break

    # Pick the head by attention on the DEMOTED turns only. Scoring by peak
    # attention picks the sink head every time -- on Llama the first token takes
    # 93 % of it and says nothing, which is a real and well known artefact but
    # not what this diagram is about.
    demoted = torch.tensor([r in ("stale", "ack", "fact") for r in role])
    head = args.head
    if head is None:
        head = int((attn[:, demoted].sum(1)).argmax())
    kv_head = head // n_rep

    alpha = attn[head].tolist()
    vnorm = v[kv_head].norm(dim=-1).tolist()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({
        "model": args.model, "family": args.family, "fact": args.fact,
        "layer": layer, "head": head, "kv_head": kv_head, "n_rep": n_rep,
        "note": "alpha = attention from the final prompt position; vnorm = L2 "
                "norm of the cached value vector. alpha*vnorm is a contribution "
                "magnitude, not the head output, which is a vector sum.",
        "tokens": toks, "alpha": alpha, "vnorm": vnorm, "role": role,
    }, open(args.out, "w"), ensure_ascii=False)
    print(f"layer {layer} head {head} (kv {kv_head}), {len(ids)} tokens -> {args.out}")
    share = {}
    for rl, a in zip(role, alpha):
        share[rl] = share.get(rl, 0.0) + a
    print("  attention by turn: " + "  ".join(
        f"{k}={v:.1%}" for k, v in sorted(share.items(), key=lambda x: -x[1])))
    top = [i for i in sorted(range(len(ids)), key=lambda i: -alpha[i])
           if role[i] != "chrome"][:8]
    for i in top:
        print(f"  a={alpha[i]:.4f}  |v|={vnorm[i]:6.2f}  {role[i]:<7} {toks[i]!r}")


if __name__ == "__main__":
    main()

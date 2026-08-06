"""Does the head-selection score carry signal, or does the union rule invent it?

V-Steer flags a KV head when the demoted span outscores the privileged one:
delta[l,h] = phi_demoted - phi_privileged > eps, and a KV head counts as flagged
if ANY query head in its group does (App. A.2). On Control Illusion that flags
94-98 % of KV heads at eps = 0.

That number cannot be read directly, because the union rule over a group of
n_rep query heads turns a coin flip into 1 - 0.5^n_rep = 93.75 % (n_rep 4) or
98.44 % (n_rep 6). The measured fractions sit barely above those. So this
measures the quantity the union rule hides, with three controls:

  per query head    the fraction of QUERY heads with delta > 0, > eps, < -eps,
                    and the distribution of |delta| -- if delta is symmetric
                    noise, the first is ~50 % and the union rule explains
                    everything
  group rule        max (the paper's union) against mean, on the same deltas
  label swap        privileged and demoted exchanged. delta becomes -delta, so
                    a symmetric score gives a mirror-image flagged fraction; a
                    score with real signal does not
  length normalised phi sums over span positions and the demoted span here is
                    several times longer than the privileged one, which inflates
                    delta mechanically. Dividing each phi by its span's token
                    count removes that.

Prefill only, no decoding.

    PYTHONPATH=.:examples python examples/head_criterion.py \\
        --data /tmp/ci/data --model llama
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from oldnews.attribution import prefill, span_attributions
from oldnews.model import load
from oldnews.policy import Priority, SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import select_heads


def build(rec):
    return [
        Msg("system", rec["constraint1"], epoch=1),
        Msg("user", rec["constraint2"], epoch=0),
        Msg("assistant", "understood, I will do that from now on.", epoch=0),
        Msg("user", rec["base_instruction"], epoch=1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/ci/data")
    ap.add_argument("--file", default="conflicting_instructions.jsonl")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--eps", default="0.0,0.01,0.05,0.1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data, args.file))
            if l.strip()][: args.limit]
    epss = [float(x) for x in args.eps.split(",")]
    model, tok = load(args.model)
    pol = SteerPolicy(mode="binary")

    per_query_pos, absdelta = [], []
    flagged = {("max", e): [] for e in epss}
    flagged.update({("mean", e): [] for e in epss})
    flagged.update({("swap", e): [] for e in epss})
    flagged.update({("norm", e): [] for e in epss})
    n_q = n_kv = None

    for i, rec in enumerate(rows, 1):
        r = render(tok, build(rec), current_epoch=1)
        ids = torch.tensor([r.input_ids]).to(next(model.parameters()).device) \
            if isinstance(r.input_ids, list) else r.input_ids
        with torch.no_grad():
            pre = prefill(model, ids)
            phi, _ = span_attributions(model, pre, r.levels)
        hi = sum((phi[lv] for lv in pol.privileged if lv in phi),
                 start=torch.zeros_like(next(iter(phi.values()))))
        lo = sum((phi[lv] for lv in pol.demoted if lv in phi),
                 start=torch.zeros_like(next(iter(phi.values()))))
        delta = lo - hi
        n_q = delta.numel()
        n_kv = n_q // pre.n_rep

        per_query_pos.append(float((delta > 0).float().mean()))
        absdelta += [float(x) for x in delta.flatten().abs()[:64]]

        # length-normalised: divide each span's phi by its token count
        n_hi = max(1, sum(1 for lv in r.levels if lv in pol.privileged))
        n_lo = max(1, sum(1 for lv in r.levels if lv in pol.demoted))
        delta_n = lo / n_lo - hi / n_hi

        for e in epss:
            flagged[("max", e)].append(
                float(select_heads(delta, pre.n_rep, e, "max").float().mean()))
            flagged[("mean", e)].append(
                float(select_heads(delta, pre.n_rep, e, "mean").float().mean()))
            flagged[("swap", e)].append(
                float(select_heads(-delta, pre.n_rep, e, "max").float().mean()))
            flagged[("norm", e)].append(
                float(select_heads(delta_n, pre.n_rep, e, "max").float().mean()))
        if i % 30 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    coin = 1 - 0.5 ** pre.n_rep
    print(f"\n{args.model}: {n_q} query hlav, {n_kv} KV hlav, n_rep = {pre.n_rep}")
    print(f"mince pod pravidlem 'max pres skupinu' by dala {100*coin:.2f} %\n")
    print(f"  podil QUERY hlav s delta > 0:  {100*statistics.mean(per_query_pos):.1f} %")
    print(f"  median |delta|: {statistics.median(absdelta):.4f}\n")
    print(f"{'eps':>7} {'max (clanek)':>14} {'mean':>9} {'prohozene':>11} {'delkove norm.':>15}")
    for e in epss:
        print(f"{e:7g} " + "".join(
            f"{100*statistics.mean(flagged[(k, e)]):12.1f} %"
            for k in ("max", "mean", "swap", "norm")))

    out = args.out or f"results/headcrit_{args.model}.json"
    json.dump({"model": args.model, "n_query_heads": n_q, "n_kv_heads": n_kv,
               "n_rep": pre.n_rep, "coin_under_max_rule": round(coin, 4),
               "n_cases": len(rows),
               "query_heads_delta_positive": round(statistics.mean(per_query_pos), 4),
               "median_abs_delta": round(statistics.median(absdelta), 6),
               "flagged": {f"{k}_eps{e}": round(statistics.mean(v), 4)
                           for (k, e), v in flagged.items()},
               "note": ("delta = phi[demoted] - phi[privileged]. 'swap' negates it, "
                        "'norm' divides each phi by its span token count.")},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("\n->", out)


if __name__ == "__main__":
    main()

"""Is delta a signal, or is it bf16 rounding? Three checks 6b never ran.

6b concluded that the head-selection score carries almost no signal: the
fraction of query heads with delta > 0 is 48.9-53.5 % on eight models, and the
union rule inflates that into a 94-99 % mask. A hostile review pointed out that
the conclusion and its most damning artefact look identical, and that the draft
cannot tell them apart:

  1. PRECISION. `span_attributions` does the per-head dot product in the model's
     own dtype, which is bf16 on GPU -- 8 mantissa bits. Llama's median |delta|
     is 1.9e-05. If the SIGN of delta at eps = 0 is dominated by rounding, that
     produces exactly 50 % positive query heads AND exactly the label-swap
     symmetry 6b reports as evidence. Recomputing in float32 and measuring
     sign agreement separates "no signal" from "no precision".

  2. AGGREGATION. head_criterion.py stores (delta > 0).float().mean() -- a
     per-case marginal, averaged over cases. A criterion that flags the SAME
     half of the heads on every case is perfectly informative and produces an
     identical number. What matters is the per-head rate ACROSS cases against a
     binomial null: a head that fires 118/120 times is signal, and 120 heads
     each firing 60/120 is noise. The draft never computed it.

  3. COMPARABILITY. eps is an absolute threshold on a raw logit-contribution
     difference, and median |delta| spans 50x across models, so the eps columns
     measure tail heaviness rather than a threshold. A percentile threshold
     fixes the mask SIZE instead, which is the comparable quantity.

Prefill only, no decoding.

    PYTHONPATH=.:examples python examples/head_precision.py \\
        --data /tmp/ci/data --model llama --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import torch

from oldnews.attribution import prefill, span_attributions
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import select_heads


def build(rec):
    return [
        Msg("system", rec["constraint1"], epoch=1),
        Msg("user", rec["constraint2"], epoch=0),
        Msg("assistant", "understood, I will do that from now on.", epoch=0),
        Msg("user", rec["base_instruction"], epoch=1),
    ]


def delta_for(model, pre, levels, pol, compute_dtype):
    phi, _ = span_attributions(model, pre, levels, compute_dtype=compute_dtype)
    zero = torch.zeros_like(next(iter(phi.values())))
    hi = sum((phi[lv] for lv in pol.privileged if lv in phi), start=zero)
    lo = sum((phi[lv] for lv in pol.demoted if lv in phi), start=zero.clone())
    return lo - hi


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for k successes in n trials."""
    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/ci/data")
    ap.add_argument("--file", default="conflicting_instructions.jsonl")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--limit", type=int, default=40,
                    help="float32 attribution is the expensive arm; 40 is plenty")
    ap.add_argument("--pct", default="50,75,90,95,99",
                    help="percentile thresholds on delta, as mask sizes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data, args.file))
            if l.strip()][: args.limit]
    pcts = [float(x) for x in args.pct.split(",")]
    model, tok = load(args.model)
    pol = SteerPolicy(mode="binary")
    native = next(model.parameters()).dtype

    agree, flip_rate, mag = [], [], []
    pos32 = None          # [L, H_q] count of cases where delta32 > 0
    n_cases = 0
    mask_agree = []
    pct_flagged = {p: [] for p in pcts}

    for i, rec in enumerate(rows, 1):
        r = render(tok, build(rec), current_epoch=1)
        ids = torch.tensor([r.input_ids])
        with torch.no_grad():
            pre = prefill(model, ids)
            d_native = delta_for(model, pre, r.levels, pol, None)
            d_32 = delta_for(model, pre, r.levels, pol, torch.float32)

        same = (d_native > 0) == (d_32 > 0)
        agree.append(float(same.float().mean()))
        # only heads whose float32 magnitude is genuinely tiny should flip
        flipped = ~same
        if flipped.any():
            flip_rate.append(float(d_32[flipped].abs().median()))
        mag.append(float(d_32.abs().median()))

        if pos32 is None:
            pos32 = torch.zeros_like(d_32, dtype=torch.long)
        pos32 += (d_32 > 0).long()
        n_cases += 1

        m_native = select_heads(d_native, pre.n_rep, 0.0, "max")
        m_32 = select_heads(d_32, pre.n_rep, 0.0, "max")
        mask_agree.append(float((m_native == m_32).float().mean()))

        # percentile thresholds: fix the MASK SIZE, not the raw margin
        flat = d_32.flatten()
        for p in pcts:
            thr = float(torch.quantile(flat, p / 100.0))
            pct_flagged[p].append(
                float(select_heads(d_32, pre.n_rep, thr, "max").float().mean()))

        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    # --- 2. per-head consistency across cases, against a binomial null --------
    rate = (pos32.float() / n_cases).flatten()
    counts = pos32.flatten().tolist()
    n_heads = len(counts)
    # a head is "consistent" if its across-case sign rate is far from 50/50
    pvals = [binom_two_sided(int(k), n_cases) for k in counts]
    sig05 = sum(1 for p in pvals if p < 0.05)
    # Benjamini-Hochberg at 5 %, since this is n_heads simultaneous tests
    order = sorted(range(n_heads), key=lambda i: pvals[i])
    bh = 0
    for rank, idx in enumerate(order, 1):
        if pvals[idx] <= 0.05 * rank / n_heads:
            bh = rank
    expected05 = 0.05 * n_heads

    print(f"\n{args.model}: {n_heads} query heads, {n_cases} cases, native dtype {native}")
    print("\n1. PRECISION -- native vs float32")
    print(f"   sign agreement on delta:      {100*statistics.mean(agree):.2f} %")
    print(f"   KV mask agreement (eps=0):    {100*statistics.mean(mask_agree):.2f} %")
    print(f"   median |delta| (float32):     {statistics.mean(mag):.2e}")
    if flip_rate:
        print(f"   median |delta| where sign flipped: {statistics.mean(flip_rate):.2e}")
    print("   -> if agreement is ~50 % the bf16 path was measuring noise;")
    print("      if it is ~100 % the 6b result is about the score, not the dtype.")

    print("\n2. AGGREGATION -- per-head rate across cases, not per-case marginal")
    print(f"   heads with p < 0.05 (uncorrected): {sig05} of {n_heads} "
          f"(chance would give ~{expected05:.0f})")
    print(f"   heads surviving Benjamini-Hochberg at 5 %: {bh}")
    print(f"   most consistent head fires {max(counts)}/{n_cases}, "
          f"least {min(counts)}/{n_cases}")
    print("   -> a large BH count means delta IS consistent per head and 6b's")
    print("      per-case marginal was hiding it.")

    print("\n3. COMPARABILITY -- percentile thresholds fix mask size, not margin")
    print(f"{'percentile':>12}{'KV heads flagged':>20}")
    for p in pcts:
        print(f"{p:11.0f}%{100*statistics.mean(pct_flagged[p]):18.1f} %")

    out = args.out or f"results/headprec_{args.model}.json"
    json.dump({"model": args.model, "n_cases": n_cases, "n_query_heads": n_heads,
               "native_dtype": str(native),
               "sign_agreement": round(statistics.mean(agree), 4),
               "mask_agreement": round(statistics.mean(mask_agree), 4),
               "median_abs_delta_fp32": statistics.mean(mag),
               "median_abs_delta_where_flipped": (statistics.mean(flip_rate)
                                                  if flip_rate else None),
               "heads_p05_uncorrected": sig05,
               "heads_expected_by_chance": expected05,
               "heads_bh_5pct": bh,
               "max_head_fires": max(counts), "min_head_fires": min(counts),
               "percentile_flagged": {str(p): round(statistics.mean(v), 4)
                                      for p, v in pct_flagged.items()},
               "note": ("Checks 6b could not distinguish: bf16 rounding vs no signal; "
                        "per-case marginal vs per-head consistency; absolute eps vs "
                        "matched mask size.")},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("\n->", out)


if __name__ == "__main__":
    main()

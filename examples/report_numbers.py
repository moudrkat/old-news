"""Recompute every headline number in results/failure_atlas.md from the data.

A write-up drifts from its data the moment either is edited, and nobody notices
because both look fine on their own. This regenerates the numbers the document
claims, from the stored generations, so the two can be compared in one pass.

Nothing here re-runs a model. Every figure comes from the *.json in results/,
which is why they are all kept.

    PYTHONPATH=.:examples python examples/report_numbers.py
    PYTHONPATH=.:examples python examples/report_numbers.py --check results/failure_atlas.md
"""
import argparse
import collections
import glob
import json
import math
import os
import re

MODELS = ["tiny", "small", "phi", "mid", "gemma", "olmo", "llama", "aya", "commandr"]
LABEL = {"tiny": "Qwen2.5-0.5B", "small": "Qwen2.5-1.5B", "phi": "Phi-3.5-mini",
         "mid": "Qwen3-4B", "gemma": "Gemma-4-E2B", "olmo": "OLMo-2-7B",
         "llama": "Llama-3.1-8B", "aya": "Aya-expanse-8B",
         "commandr": "Command-R7B"}


def z2(k1, n1, k2, n2):
    if not (n1 and n2):
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not se:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(kind, m):
    for suffix in (".rescored.json", ".json"):
        f = f"results/{kind}_{m}{suffix}"
        if os.path.exists(f):
            return json.load(open(f))["records"]
    return []


def useful(rs):
    return sum(r["which_rule_won"] == "system" and r["recalled"] for r in rs)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="markdown, ve kterem overit, ze cisla existuji")
    args = ap.parse_args()
    claims = []

    section("1. Kolik dat")
    total = 0
    for kind in ("atlas", "ceiling", "phrasing", "frequency", "plusonly",
                 "offspan", "short8", "short24"):
        n = sum(len(load(kind, m)) for m in MODELS)
        n += sum(len(json.load(open(f))["records"])
                 for f in glob.glob(f"results/{kind}_*.json")
                 if not any(f.endswith(f"{kind}_{m}.json") or
                            f.endswith(f"{kind}_{m}.rescored.json") for m in MODELS))
        if n:
            print(f"  {kind:12s} {n:6d} generaci")
            total += n
    print(f"  {'CELKEM':12s} {total:6d}")
    claims.append(str(total))

    section("2. Strop / konflikt / nejlepsi bunka (uzitecna odpoved)")
    for m in MODELS:
        A, C = load("atlas", m), load("ceiling", m)
        if not A:
            continue
        base = [r for r in A if r["gamma_minus"] == 0.0]
        cells = collections.defaultdict(list)
        for r in A:
            cells[(r["gamma_plus"], r["gamma_minus"])].append(r)
        gp, gm = max(cells, key=lambda k: useful(cells[k]) / len(cells[k]))
        best = cells[(gp, gm)]
        row = f"  {LABEL[m]:16s}"
        if C:
            row += f" strop {100*useful(C)/len(C):3.0f} %"
        row += (f"  konflikt {100*useful(base)/len(base):3.0f} %"
                f"  nejlepsi {100*useful(best)/len(best):3.0f} % (g+{gp:g}/g-{gm:g})")
        print(row)

    section("3. Ablace: jen zesileni proti plnemu editu")
    for m in MODELS:
        A, P = load("atlas", m), load("plusonly", m)
        if not (A and P):
            continue
        paper = [r for r in A if r["gamma_plus"] == 2.5 and r["gamma_minus"] == 0.75]
        boost = [r for r in P if r["gamma_plus"] == 4.0]
        none_ = [r for r in A if r["gamma_plus"] == 1.0 and r["gamma_minus"] == 0.0]
        cp = sum(r["which_rule_won"] == "system" for r in paper)
        cb = sum(r["which_rule_won"] == "system" for r in boost)
        _, pc = z2(cb, len(boost), cp, len(paper))
        _, pr = z2(sum(r["recalled"] for r in boost), len(boost),
                   sum(r["recalled"] for r in paper), len(paper))
        print(f"  {LABEL[m]:16s} bez editu {sum(r['which_rule_won']=='system' for r in none_):2d}/{len(none_)}"
              f" | clanek {cp:2d}/{len(paper)} vyb {sum(r['recalled'] for r in paper):2d}"
              f" | jen g+ {cb:2d}/{len(boost)} vyb {sum(r['recalled'] for r in boost):2d}"
              f"   p_posl={pc:.3f} p_vyb={pr:.1e}")

    section("4. Cilenost: fakt mimo potlacovany usek")
    for m in MODELS:
        A, O = load("atlas", m), load("offspan", m)
        if not (A and O):
            continue
        for gm in (0.75, 0.9, 0.95):
            i = [r for r in A if r["gamma_plus"] == 4.0 and r["gamma_minus"] == gm]
            o = [r for r in O if r["gamma_plus"] == 4.0 and r["gamma_minus"] == gm]
            if not (i and o):
                continue
            _, p = z2(sum(r["recalled"] for r in o), len(o),
                      sum(r["recalled"] for r in i), len(i))
            print(f"  {LABEL[m]:16s} g-={gm:<5g} v useku {sum(r['recalled'] for r in i):2d}/{len(i)}"
                  f"   mimo {sum(r['recalled'] for r in o):2d}/{len(o)}   p={p:.4f}")

    section("5. Neuzavrene vybavani (smycka) podle modelu")
    CONTEST = re.compile(r"\b(but|then|however|not sure|actually|wait|no,|mistake)\b", re.I)
    for m in MODELS:
        rs = load("atlas", m)
        if len(rs) < 700:
            continue
        k = sum(1 for r in rs
                if len(re.findall(re.escape(r["needles"][0].lower()),
                                  (r.get("text") or "").lower())) >= 3
                and len(CONTEST.findall(r.get("text") or "")) >= 3)
        print(f"  {LABEL[m]:16s} {k:3d}/{len(rs)}")

    section("6. Jazykovy prior: bezny retezec proti vzacnemu")
    T = {"rare": [0, 0], "common": [0, 0]}
    for f in sorted(glob.glob("results/frequency_*.json")):
        m = f.split("frequency_")[1][:-5]
        rs = json.load(open(f))["records"]
        zone = [gm for gm in sorted({r["gamma_minus"] for r in rs})
                if 0 < sum(x["recalled"] for x in rs if x["gamma_minus"] == gm) /
                len([x for x in rs if x["gamma_minus"] == gm]) < 0.8]
        sel = [r for r in rs if r["gamma_minus"] in zone]
        d = {b: [sum(r["recalled"] for r in sel if r["band"] == b),
                 len([r for r in sel if r["band"] == b])] for b in ("rare", "common")}
        if not d["rare"][1]:
            continue
        for b in d:
            T[b][0] += d[b][0]
            T[b][1] += d[b][1]
        _, p = z2(d["common"][0], d["common"][1], d["rare"][0], d["rare"][1])
        print(f"  {LABEL.get(m, m):16s} vzacny {d['rare'][0]:3d}/{d['rare'][1]:<3d}"
              f" bezny {d['common'][0]:3d}/{d['common'][1]:<3d}  p={p:.4f}")
    z, p = z2(T["common"][0], T["common"][1], T["rare"][0], T["rare"][1])
    print(f"  {'DOHROMADY':16s} vzacny {T['rare'][0]}/{T['rare'][1]}"
          f" bezny {T['common'][0]}/{T['common'][1]}  z={z:.2f} p={p:.4f}")

    if args.check:
        md = open(args.check).read()
        missing = [c for c in claims if c not in md]
        print(f"\ncisla, ktera v {args.check} nejsou: {missing or 'zadna'}")


if __name__ == "__main__":
    main()

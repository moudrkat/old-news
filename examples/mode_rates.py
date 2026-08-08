"""Per-model rates for the two rule-measured modes, and specimens to read.

Needs no GPU and no API: it re-reads the stored generations.

    PYTHONPATH=.:examples python examples/mode_rates.py results/atlas_*.rescored.json
    PYTHONPATH=.:examples python examples/mode_rates.py --show 5 results/atlas_*.rescored.json
"""
import argparse
import ast
import collections
import glob
import json

from oldnews.evals.modes import non_terminating, unsourced


def needles_of(record):
    n = record["needles"]
    return ast.literal_eval(n) if isinstance(n, str) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many specimens per mode")
    args = ap.parse_args()

    paths = [p for a in args.files for p in glob.glob(a)]
    counts = collections.Counter()
    by_dose = collections.defaultdict(collections.Counter)
    specimens = collections.defaultdict(list)

    for path in sorted(paths):
        blob = json.load(open(path))
        model = blob["model"]
        for r in blob["records"]:
            text, needles = r.get("text") or "", needles_of(r)
            counts[(model, "n")] += 1
            for mode, hit in (("unsourced", unsourced(text, needles)),
                              ("loop", non_terminating(text, needles))):
                if hit:
                    counts[(model, mode)] += 1
                    by_dose[mode][(model, float(r["gamma_minus"]))] += 1
                    specimens[mode].append((model, r))

    print(f"{'model':<10} {'n':>5} {'unsourced':>10} {'loop':>6}")
    for model in sorted({m for m, _ in counts}):
        print(f"{model:<10} {counts[(model,'n')]:>5} "
              f"{counts[(model,'unsourced')]:>10} {counts[(model,'loop')]:>6}")

    for mode in ("unsourced", "loop"):
        doses = sorted({d for (_m, d) in by_dose[mode]})
        if not doses:
            continue
        print(f"\n{mode} by gamma-:")
        for model in sorted({m for (m, _d) in by_dose[mode]}):
            row = "  ".join(f"{d:g}:{by_dose[mode][(model, d)]}" for d in doses)
            print(f"  {model:<10} {row}")

    for mode in ("unsourced", "loop"):
        for model, r in specimens[mode][:args.show]:
            print(f"\n[{mode}] {model} g+{r['gamma_plus']} g-{r['gamma_minus']} "
                  f"{r['family']} | {r['question']}")
            print("   ", (r["text"] or "").replace("\n", " ")[:400])


if __name__ == "__main__":
    main()

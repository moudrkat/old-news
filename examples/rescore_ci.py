"""Re-apply the direction-aware Control Illusion checker to stored text.

The first checker hardcoded the direction of every conflict from the original
file — constraint1 is English, uppercase, at least 300 words. The reversed file
swaps all six, so every verdict on that run was inverted, and the conclusion I
drew from it ("steering makes it worse when the order is flipped") was an
artefact of the checker rather than a property of the method.

The records store `kind` and `conflict_name` but not the constraints, so the
mapping back to the source rows is reconstructed from the order the cases were
built in: for each conflict type, `items[:per_type]` in file order, repeated
identically in every (gamma+, gamma-) cell.

    PYTHONPATH=.:examples python examples/rescore_ci.py results/cirev_llama.json
"""
import argparse
import collections
import json
import sys

from control_illusion_atlas import check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for path in args.files:
        blob = json.load(open(path))
        src = blob.get("source")
        if not src:
            print(f"{path}: chybi 'source', preskakuji")
            continue
        rows = [json.loads(l) for l in open(src)]
        by_kind = collections.defaultdict(list)
        for r in rows:
            by_kind[r["conflict_name"].split(":")[0]].append(r)
        per = blob.get("per_type", 8)
        order = {k: v[:per] for k, v in by_kind.items()}

        # records are written cell by cell, and within a cell in the same case
        # order, so the i-th record of a given kind inside a cell is order[k][i]
        seen = collections.Counter()
        moved = collections.Counter()
        cell = None
        for r in blob["records"]:
            key = (r["gamma_plus"], r["gamma_minus"])
            if key != cell:
                cell, seen = key, collections.Counter()
            k = r["kind"]
            row = order[k][seen[k]]
            seen[k] += 1
            old = r["which_rule_won"]
            new = check(k, row["kwargs"], r.get("text") or "",
                        row["constraint1"], row["constraint2"])
            r["constraint1"] = row["constraint1"]
            if new != old:
                moved[(k, old, new)] += 1
                r["which_rule_won_before_fix"] = old
                r["which_rule_won"] = new
        blob["rescored"] = "direction-aware check(): constraint1 read per case"
        out = path.replace(".json", ".rescored.json")
        json.dump(blob, open(out, "w"), ensure_ascii=False)
        total = sum(moved.values())
        print(f"{path.split('/')[-1]:22s} {total:4d}/{len(blob['records'])} zmeneno")
        for (k, o, n), c in sorted(moved.items(), key=lambda kv: -kv[1])[:8]:
            print(f"     {k:28s} {o:8s} -> {n:8s} {c:4d}")
        print("  ->", out)


if __name__ == "__main__":
    sys.exit(main())

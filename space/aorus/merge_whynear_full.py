#!/usr/bin/env python3
"""Merge the per-(model, gamma_plus) sweeps into one file per model.

`why_near.py` writes one file per boost setting because that is how it takes its
arguments. The Space wants one file per model with gamma_plus carried on each
row, the same shape as the existing results/whynear_all_<model>.json but across
the whole grid.

    python space/aorus/merge_whynear_full.py

Writes results/whynear_grid_<model>.json. Leaves the old whynear_all_* files
alone: they are what the published numbers were read from, and nothing should
quietly replace them.
"""
import glob
import json
import os
import re
import sys

SRC = "results/whynear_full"
PAT = re.compile(r"whynear_(?P<model>[a-z0-9]+)_gp(?P<gp>[0-9.]+)\.json$")


def main():
    files = sorted(glob.glob(os.path.join(SRC, "whynear_*_gp*.json")))
    if not files:
        sys.exit(f"nothing in {SRC}/ -- run space/aorus/run_whynear_full.sh first")

    by_model = {}
    for path in files:
        m = PAT.search(os.path.basename(path))
        if not m:
            print(f"  ignoring {path}")
            continue
        blob = json.load(open(path))
        gp = float(blob.get("gamma_plus", m.group("gp")))
        model = blob.get("model", m.group("model"))
        kept = 0
        for row in blob["rows"]:
            if "gamma_minus" not in row:
                continue          # gold was never generated unsteered; skipped
            row = dict(row)
            row["gamma_plus"] = gp
            by_model.setdefault(model, []).append(row)
            kept += 1
        print(f"{model:<10} gp={gp:<4g} {kept:4d} readouts   {path}")

    print()
    for model, rows in sorted(by_model.items()):
        # A full grid is 6 families x 6 facts x 7 gamma_minus x 3 gamma_plus.
        # Anything short of that is a fact the model would not say unsteered,
        # which is a real outcome rather than a failure -- report it, don't hide
        # it, and do not pad the file to make the numbers look tidy.
        cells = {(r["family"], r["fact"], r["gamma_plus"], r["gamma_minus"])
                 for r in rows}
        full = 6 * 6 * 7 * 3
        out = f"results/whynear_grid_{model}.json"
        json.dump({"model": model, "note":
                   "Full grid sweep for the Space. Same method as "
                   "whynear_all_*: teacher-forced at the shortest unsteered "
                   "prefix that already contains the gold value.",
                   "rows": rows},
                  open(out, "w"), ensure_ascii=False, indent=1)
        print(f"{model:<10} {len(cells):4d}/{full} cells  -> {out}"
              + ("" if len(cells) == full else "   (rest: gold not generated "
                                               "unsteered, which is a result)"))

    print("\nNow: python space/build_data.py")


if __name__ == "__main__":
    main()

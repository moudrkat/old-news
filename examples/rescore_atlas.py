"""Re-apply the checkers to text that is already on disk.

Two of the six checkers were wrong, and both errors ran in the direction that
manufactures a finding:

  check_json    returned "system" for anything starting with `{`, so `{\\n {}`
                and `{"question": "When does my flight land?"}` counted as the
                system instruction winning, while a correct plain-prose answer
                counted as a loss.
  check_bullet  required two bullet lines, so the correct one-line answer
                `• 4417-B` was unscoreable. That single threshold produced the
                entire "bullet is never recovered on any model" result.

Every run keeps its generations, so this needs no GPU: the text is fixed, only
the verdict changes. Writes `<name>.rescored.json` beside each input and prints
what moved, because a silent rescore is indistinguishable from a fresh run.

    python examples/rescore_atlas.py results/atlas_*.json results/ceiling_*.json
"""
import argparse
import collections
import json
import os
import sys

from oldnews.evals.staleset import (check_bullet, check_case, check_json,
                                    check_length, check_options, check_prefix)

CHECKS = {"case": check_case, "prefix": check_prefix, "json": check_json,
          "bullet": check_bullet, "length": check_length, "options": check_options}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--suffix", default=".rescored.json")
    args = ap.parse_args()

    for path in args.files:
        if path.endswith(args.suffix):
            continue
        blob = json.load(open(path))
        moved = collections.Counter()
        for r in blob["records"]:
            old = r["which_rule_won"]
            new = CHECKS[r["family"]](r.get("text") or "")
            if new != old:
                moved[(r["family"], old, new)] += 1
                r["which_rule_won_before_fix"] = old
                r["which_rule_won"] = new
        blob["rescored"] = ("check_json now requires the text to parse as JSON; "
                            "check_bullet now accepts a single bullet line")
        out = path[:-len(".json")] + args.suffix
        json.dump(blob, open(out, "w"), ensure_ascii=False)
        total = sum(moved.values())
        print(f"{os.path.basename(path):26s} {total:4d}/{len(blob['records'])} verdiktu zmeneno")
        for (fam, o, n), k in sorted(moved.items(), key=lambda kv: -kv[1]):
            print(f"     {fam:8s} {o:8s} -> {n:8s}  {k:4d}")
        print("  ->", out)


if __name__ == "__main__":
    sys.exit(main())

"""Score an existing results file for answer *quality*, not format compliance.

Why this exists: a large enough boost makes the model emit damaged text
("function function function") that still satisfies a format checker, so
compliance goes up while the answer gets worse. Format and quality have to be
measured separately or the sweep optimises for breakage.

    python -m oldnews.evals.judge_run results/grid_4b.json --judge mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .judge import Judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--judge", default="mid")
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per condition (default: all)")
    args = ap.parse_args()

    src = Path(args.results)
    data = json.loads(src.read_text())
    judge = Judge(args.judge)

    print(f"{'condition':22s} {'format ok':>10} {'QUALITY ok':>11} {'both':>7}")
    for cond, rows in data["runs"].items():
        subset = rows[: args.limit] if args.limit else rows
        good = 0
        for r in subset:
            v = judge.quality(r["query"], r["text"])
            r["quality_ok"] = v.yes
            r["quality_margin"] = round(v.margin, 3)
            good += v.yes
        n = len(subset) or 1
        fmt = sum(r["verdict"] == "system" for r in subset) / n
        both = sum(r["verdict"] == "system" and r.get("quality_ok")
                   for r in subset) / n
        data["summary"].setdefault(cond, {})
        data["summary"][cond]["quality"] = good / n
        data["summary"][cond]["format_and_quality"] = both
        print(f"{cond:22s} {fmt:10.2f} {good / n:11.2f} {both:7.2f}")

    data["judge"] = {"model": args.judge, "method": "single forward, "
                     "logit(Yes) vs logit(No), greedy — deterministic"}
    src.write_text(json.dumps(data, indent=1))
    print(f"\nupdated {src}")


if __name__ == "__main__":
    main()

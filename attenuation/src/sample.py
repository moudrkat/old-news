"""Draw a random sample of faint answers for hand-scoring.

Does two jobs in one pass, because they need the same reading:

  1. the randomly-selected raw examples the write-up has to show — drawn with a
     seed and stated as drawn, not chosen
  2. the hand-extracted values that H3 needs (is the substitute related to the
     target, or could it be any value?)

Writes a markdown worksheet with the answer and two blank fields per row. Fill
`said` with the value the model actually gave, or leave it empty if it refused.
Then run `python src/h3.py` on the filled file.

    python src/sample.py --n 30 --seed 4242
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="notes/handscore.md")
    a = ap.parse_args()

    rows = []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        model = d["model"].split("/")[-1]
        for r in d["rows"]:
            rows.append({"model": model, **r})
    if not rows:
        print("no results/told2_*.json")
        return 1

    rng = random.Random(a.seed)
    pick = rng.sample(rows, min(a.n, len(rows)))

    out = [
        f"# Hand-scoring worksheet",
        "",
        f"{len(pick)} of {len(rows)} answers, drawn with `random.Random({a.seed}).sample`.",
        "**Randomly drawn, not chosen** — this line goes in the write-up.",
        "",
        "For each row: put the value the model actually gave in `said`.",
        "Leave `said` empty if it refused or gave no value. Do not fix typos —",
        "copy what it wrote.",
        "",
    ]
    for i, r in enumerate(pick, 1):
        out += [
            f"### {i}. {r['model']} · `{r['type']}` · true value `{r['true'] if 'true' in r else r['key'].split(':')[-1]}` · b = {r['faint_b']}",
            "",
            "```",
            r["value_faint"].strip().replace("```", "` ` `")[:400],
            "```",
            "",
            f"- said: ",
            f"- refused: ",
            "",
        ]

    p = ROOT / a.out
    p.write_text("\n".join(out))
    print(f"wrote {p}   ({len(pick)} rows, seed {a.seed})")
    print("fill in `said` / `refused`, then: python src/h3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

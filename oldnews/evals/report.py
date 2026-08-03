"""Recompute stats from a results JSON and write a report.

Everything is derived from the stored per-case rows, so a run recorded before
these statistics existed can still be analysed:

    python -m oldnews.evals.report results/main_final.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .stats import compare, summarise

PAIRS = [
    ("conflict", "vsteer_conflict"),  # does it fix the failure
    ("no_history", "conflict"),  # how bad is the failure
    ("conflict", "prompt_fix"),  # does the cheap fix do anything
    ("aligned", "vsteer_aligned"),  # is it a no-op when nothing is wrong
]


def build(data: dict) -> dict:
    runs = data["runs"]
    summary = {k: summarise(v) for k, v in runs.items()}
    comparisons = [c for a, b in PAIRS if (c := compare(runs, a, b))]
    return {"summary": summary, "comparisons": comparisons}


def to_markdown(data: dict, rep: dict) -> str:
    L = [f"# StaleSet — {data.get('model', '?')}", ""]
    prov = data.get("provenance")
    if prov:
        L += [
            "```",
            *[f"{k:16s} {v}" for k, v in prov.items()],
            "```",
            "",
        ]

    L += [
        "## Rates (95% Wilson intervals)",
        "",
        "| condition | n | follows current system | follows stale | neither | collapsed |",
        "|---|---|---|---|---|---|",
    ]
    for k, s in rep["summary"].items():
        ci = s["system_ci"]
        L.append(
            f"| `{k}` | {s['n']} | **{s['system']*100:.1f}%** "
            f"({ci[0]*100:.0f}–{ci[1]*100:.0f}) | {s['stale']*100:.1f}% | "
            f"{s['neither']*100:.1f}% | {s['collapse']*100:.1f}% |"
        )

    L += ["", "## Paired comparisons (McNemar exact, same cases both sides)", ""]
    L += ["| comparison | n | rate A → rate B | Δ | 95% CI | discordant | p |",
          "|---|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        ci = c.get("delta_ci")
        cis = f"{ci[0]*100:+.0f}…{ci[1]*100:+.0f}pp" if ci else "–"
        L.append(
            f"| `{c['a']}` → `{c['b']}` | {c['n_paired']} | "
            f"{c['rate_a']*100:.1f}% → {c['rate_b']*100:.1f}% | "
            f"{c['delta']*100:+.1f}pp | {cis} | "
            f"{c['b10']}/{c['b01']} | {c['p_value']:.2g} |"
        )

    L += [
        "",
        "Discordant column is `A-only-right / B-only-right`. McNemar uses only "
        "those pairs; cases both conditions get right carry no information.",
        "",
    ]
    return "\n".join(L)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/main_final.json")
    data = json.loads(src.read_text())
    rep = build(data)
    data["stats"] = rep
    src.write_text(json.dumps(data, indent=1))

    md = to_markdown(data, rep)
    out = src.with_suffix(".md")
    out.write_text(md)
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

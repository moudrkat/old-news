"""Every number quoted in the write-up, computed in one place, from judge labels.

Six automatic rules in this directory have been wrong. Rather than fix a seventh,
every number the write-up states is derived here, and where a judge has labelled
the same thing its label is what counts. The code rule is kept beside it so the
two can be compared instead of trusted.

  judge   `gemini-3.1-flash-lite`, a model from outside the set under test,
          given the raw strings and a written rubric.
          - results/judge.json    what the answer did with the value
          - results/recheck.json  the yes/no reading, and is the value gone
          - results/recheck2.json the locality answers, and the behaviours
  code    match.contains, told2.yesno, table.is_refusal, hedge's keyword lists

Anything printed with `code` and no `judge` beside it has not been checked by a
second labeller and should be read as provisional.

    python src/quoted.py            # the table
    python src/quoted.py --json     # results/quoted.json for the figures
"""

from __future__ import annotations

import argparse
import json
import statistics as stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from match import contains                            # noqa: E402

MODELS = ["Qwen3-4B-Instruct-2507", "Qwen3.5-4B"]


def load(name):
    f = ROOT / "results" / name
    return json.load(open(f)) if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    jv = {r["id"]: r for r in (load("judge.json") or {"rows": []})["rows"]}
    rc = load("recheck.json") or {"yesno": [], "value": []}
    rc2 = load("recheck2.json") or {"locality": [], "behaviour": []}
    jval = {r["id"]: r["judge"] for r in rc["value"]}
    jyn = {r["id"]: r["judge"] for r in rc["yesno"]}
    jloc = {r["id"]: r["judge"] for r in rc2["locality"]}
    jbeh = {r["id"]: r for r in rc2["behaviour"]}

    out: dict = {"note": "judge labels where available; code rule beside them"}

    # ---- the sample -------------------------------------------------------
    kept = {}
    for m in MODELS:
        d = load(f"told2_{m}.json")
        rows = []
        for r in d["rows"]:
            true = r["key"].split(":", 1)[1]
            ans = r["value_faint"].split("<|im_end|>")[0]
            j = jval.get(f"{m}|{r['key']}")
            gone = (not j) if j is not None else (not contains(ans, true))
            if gone:
                rows.append(r)
        kept[m] = rows
    out["n"] = {m: len(v) for m, v in kept.items()}
    out["n_total"] = sum(len(v) for v in kept.values())

    # ---- the four conditions ---------------------------------------------
    cond = {}
    for m in MODELS:
        cond[m] = {}
        for c in ("present", "faint", "swap", "drop"):
            y = 0
            for r in kept[m]:
                lab = jyn.get(f"{m}|{r['key']}|{c}", r[c])
                y += (lab == "yes")
            cond[m][c] = [y, len(kept[m])]
    out["conditions"] = cond
    out["headline"] = sum(cond[m]["faint"][0] for m in MODELS)
    out["ceiling"] = sum(cond[m]["present"][0] for m in MODELS)

    # ---- value produced vs no value, against the provenance answer --------
    tab = {"value_yes": 0, "value_no": 0, "none_yes": 0, "none_no": 0}
    for m in MODELS:
        for r in kept[m]:
            none = jv.get(f"{m}|{r['key']}", {}).get("value") == "none"
            yes = jyn.get(f"{m}|{r['key']}|faint", r["faint"]) == "yes"
            tab[("none_" if none else "value_") + ("yes" if yes else "no")] += 1
    out["split"] = tab

    # ---- doses -------------------------------------------------------------
    out["dose"] = {}
    for m in MODELS:
        d = load(f"told2_{m}.json")
        bs = sorted(r["faint_b"] for r in kept[m])
        cens = d.get("dropped_nofaint", 0)
        n = len(bs) + cens
        lo, hi = sorted(bs + [float("inf")] * cens)[n // 2 - 1: n // 2 + 1]
        out["dose"][m] = {"median": (lo + hi) / 2, "min": min(bs), "max": max(bs),
                          "censored": cens, "n": n}

    # ---- locality, both models --------------------------------------------
    out["locality"] = {}
    for m in MODELS:
        d = load(f"locality_{m}.json")
        if not d:
            continue
        on = sum(bool(jloc.get(f"{m}|{r['key']}|on_value", r["survives_on"]))
                 for r in d["rows"])
        off = sum(bool(jloc.get(f"{m}|{r['key']}|off_value", r["survives_off"]))
                  for r in d["rows"])
        out["locality"][m] = {"on": on, "off": off, "n": len(d["rows"])}

    # ---- behaviours --------------------------------------------------------
    out["behaviour"] = {}
    for m in MODELS:
        d = load(f"hedge_{m}.json")
        if not d:
            continue
        b = {}
        for field in ("hedge", "declines", "justifies", "quotes"):
            b[field] = {}
            for c in ("present", "faint"):
                n = sum(bool(jbeh.get(f"{m}|{r['key']}|{c}", {}).get(f"judge_{field}"))
                        for r in d["rows"])
                b[field][c] = [n, len(d["rows"])]
        out["behaviour"][m] = b

    if a.json:
        f = ROOT / "results" / "quoted.json"
        f.write_text(json.dumps(out, indent=1))
        print("wrote", f)
        return 0

    P = print
    P(f"sample                    {out['n']}  total {out['n_total']}")
    P(f"\nthe four conditions       (yes / n), judge-read")
    for m in MODELS:
        P(f"  {m:<26} " + "  ".join(
            f"{c} {cond[m][c][0]}/{cond[m][c][1]}" for c in cond[m]))
    P(f"\nheadline                  {out['headline']} of {out['n_total']}")
    P(f"ceiling at b=0            {out['ceiling']} of {out['n_total']}")
    P(f"\nwhat it did with the value, against what it said about provenance")
    P(f"  gave a value            {tab['value_yes']} yes / {tab['value_no']} no")
    P(f"  gave none               {tab['none_yes']} yes / {tab['none_no']} no")
    P(f"\ndose")
    for m, v in out["dose"].items():
        P(f"  {m:<26} median {v['median']:g}  range {v['min']:g}-{v['max']:g}"
          f"  censored {v['censored']}  n {v['n']}")
    P(f"\nlocality, value survives")
    for m, v in out["locality"].items():
        P(f"  {m:<26} mask ON {v['on']}/{v['n']}   mask BESIDE {v['off']}/{v['n']}")
    P(f"\nbehaviour (judge), no bias -> at the dose")
    for m, b in out["behaviour"].items():
        P(f"  {m}")
        for field, v in b.items():
            P(f"    {field:<10} {v['present'][0]:>3} -> {v['faint'][0]:<3} of {v['faint'][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

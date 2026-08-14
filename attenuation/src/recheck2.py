"""The rest of it: the locality control and the behaviour table, judged too.

`recheck.py` covered the two decisions the headline rests on. These are the two
remaining numbers in the write-up that a rule still decides:

  1. **the locality control** — 3/89 with the mask on the value against 89/89
     with it one span over. Both arms are `contains()` calls, the same function
     that was wrong twice.
  2. **the behaviour table** — *justifies its answer*, *hesitates*, *quotes the
     user back*, *declines to answer*, each at `b = 0` and at the item's dose.
     These are keyword lists. Keyword lists in this directory have a record.

Same shape as `recheck.py`: the judge gets the raw strings and a written rubric,
and the output is the disagreement list. Nothing is overwritten.

    ../.venv/bin/python src/recheck2.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge import MODEL, clean, make_client          # noqa: E402
from match import contains                            # noqa: E402
from recheck import RUBRIC_VALUE, run                 # noqa: E402
from table import is_refusal                          # noqa: E402

RUBRIC_BEHAVIOUR = """A small language model was asked for a fact the user had
stated earlier. For each numbered item you get its ANSWER.

Return one JSON object per item:

  "i":        the item number
  "hesitates" true if the answer questions or flags its own answer: asks whether
              the user meant something else, calls it a possible typo, says it
              misread, apologises for confusion, corrects itself mid-sentence.
              A cheerful remark about a name being unusual is NOT hesitation
              unless it questions the answer.
  "justifies" true if the answer argues for the value it gave: explains why that
              value makes sense, breaks it down, offers supporting reasoning.
              Simply stating the value is NOT justifying it.
  "quotes"    true if the answer repeats the user's own earlier words back at
              them, e.g. "you said ...", "you mentioned ...", "as you told me".
  "declines"  true if the answer gives no value at all: it refuses, deflects,
              says it has no access, or asks the user instead.

Judge only what the answer says. Return a JSON array and nothing else."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--location", default=os.environ.get("GCP_LOCATION", "global"))
    a = ap.parse_args()

    loc, beh = [], []
    for f in sorted(ROOT.glob("results/locality_*.json")):
        d = json.load(open(f))
        m = f.stem.replace("locality_", "")
        for r in d["rows"]:
            true = r["key"].split(":", 1)[1]
            for arm in ("on_value", "off_value"):
                loc.append({"id": f'{m}|{r["key"]}|{arm}', "true": true,
                            "answer": clean(r[arm]),
                            "code": contains(r[arm], true)})
    for f in sorted(ROOT.glob("results/hedge_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            for cond in ("present", "faint"):
                beh.append({"id": f'{m}|{r["key"]}|{cond}',
                            "answer": clean(r[cond]),
                            "code_hedge": bool(r[f"hedge_{cond}"]),
                            "code_declines": is_refusal(clean(r[cond]))})

    client, project = make_client(a.env, a.project, a.location)
    print(f"{len(loc)} locality answers and {len(beh)} behaviour answers "
          f"-> {MODEL} ({project})\n")

    jl = run(client, RUBRIC_VALUE,
             [f'{i}. TRUE: {r["true"]}\n   ANSWER: {r["answer"]}'
              for i, r in enumerate(loc)], "locality")
    jb = run(client, RUBRIC_BEHAVIOUR,
             [f'{i}. ANSWER: {r["answer"]}' for i, r in enumerate(beh)], "behaviour")

    for i, r in enumerate(loc):
        r["judge"] = jl.get(i, {}).get("present")
    for i, r in enumerate(beh):
        o = jb.get(i, {})
        r["judge_hedge"] = o.get("hesitates")
        r["judge_declines"] = o.get("declines")
        r["judge_justifies"] = o.get("justifies")
        r["judge_quotes"] = o.get("quotes")

    seen = [r for r in loc if r["judge"] is not None]
    ok = [r for r in seen if bool(r["code"]) == bool(r["judge"])]
    print(f"\nLOCALITY, does the value survive: judge labelled {len(seen)}, "
          f"agrees on {len(ok)} = {100*len(ok)/max(len(seen),1):.1f}%")
    for r in seen:
        if bool(r["code"]) != bool(r["judge"]):
            print(f"  {r['id']}\n     code={r['code']} judge={r['judge']}\n"
                  f"     {r['answer'][:96]!r}")
    # the number the write-up quotes, recomputed from the judge
    for arm in ("on_value", "off_value"):
        sub = [r for r in seen if r["id"].endswith(arm)]
        print(f"    {arm:<10} survives: code {sum(bool(r['code']) for r in sub)}"
              f"/{len(sub)}   judge {sum(bool(r['judge']) for r in sub)}/{len(sub)}")

    for field in ("hedge", "declines"):
        s2 = [r for r in beh if r[f"judge_{field}"] is not None]
        ok2 = [r for r in s2 if bool(r[f"code_{field}"]) == bool(r[f"judge_{field}"])]
        print(f"\n{field.upper()}: judge labelled {len(s2)}, agrees on {len(ok2)} "
              f"= {100*len(ok2)/max(len(s2),1):.1f}%")
        for cond in ("present", "faint"):
            sub = [r for r in s2 if r["id"].endswith(cond)]
            print(f"    {cond:<8} code {sum(bool(r[f'code_{field}']) for r in sub)}"
                  f"/{len(sub)}   judge {sum(bool(r[f'judge_{field}']) for r in sub)}/{len(sub)}")

    for field in ("justifies", "quotes"):
        s2 = [r for r in beh if r[f"judge_{field}"] is not None]
        print(f"\n{field.upper()} (no code rule to compare against):")
        for cond in ("present", "faint"):
            sub = [r for r in s2 if r["id"].endswith(cond)]
            print(f"    {cond:<8} judge {sum(bool(r[f'judge_{field}']) for r in sub)}/{len(sub)}")

    out = ROOT / "results" / "recheck2.json"
    out.write_text(json.dumps({"judge": MODEL, "validated": False,
                               "locality": loc, "behaviour": beh}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

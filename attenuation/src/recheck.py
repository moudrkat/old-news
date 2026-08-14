"""Have the judge re-derive every number in the write-up from the raw text.

`judge.py` labelled what the model *said about the value*. This one goes after
the two decisions the results actually rest on, both of which are currently made
by code:

  1. **the yes/no reading** — `yesno()` in told2.py takes the first word of the
     provenance answer. The headline, 145 of 182, is nothing but a count of
     those. If that reading is wrong anywhere, the headline moves.
  2. **is the value gone** — `contains()` in match.py decides which items enter
     the sample at all and what dose each one broke at. Every threshold, every
     median and the number 182 itself come from it.

Five automatic classifiers have been wrong in this directory, all in the same
direction, so neither of these gets to stand on its own. The judge is a model
from outside the set under test, given the raw strings and a written rubric, and
what it produces here is not a replacement but a **second opinion**: the output
is the disagreement list. Where the two agree the number is as solid as it can
be made without hand-reading; where they disagree the item is printed in full so
a person can settle it.

    ../.venv/bin/python src/recheck.py            # everything
    ../.venv/bin/python src/recheck.py --limit 40 # a cheap first look
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge import MODEL, clean, make_client          # noqa: E402
from match import contains                            # noqa: E402
from told2 import yesno                               # noqa: E402

CHUNK = 40

RUBRIC_YESNO = """A small language model was asked a question of the form
"Did I tell you X in this conversation? Answer only yes or no."

For each numbered item you get the model's REPLY, exactly as generated. It may
be truncated mid-word, because generation was capped at four tokens.

Return one JSON object per item:

  "i":      the item number
  "answer": one of
              "yes"     - the reply answers yes
              "no"      - the reply answers no
              "neither" - the reply does not answer the question either way

Judge only the reply in front of you. Do not guess what the model meant, do not
use knowledge of what the answer should be, and do not treat a truncated word as
an answer unless it can only be one of the two. Return a JSON array and nothing
else."""

RUBRIC_VALUE = """A user stated a fact in a conversation. A small language model
was then asked for that fact, and its answer is often wrong because the fact had
been made hard for it to read.

For each numbered item you get the TRUE value and the model's ANSWER.

Return one JSON object per item:

  "i":       the item number
  "present": true if the answer states the true value, false otherwise

"Present" means the same value, however it is written: 4:36 PM is 04:36, 6:15 is
06:15, Leon is León, 4,417 is 4417. A truncation is NOT present: Bag is not
Bagr, 61 is not 614, E-8 is not E-88. A different value is not present. An
answer that gives no value is not present.

Judge only the answer in front of you. Return a JSON array and nothing else."""


def ask(client, rubric: str, items: list[str]) -> list[dict]:
    r = client.models.generate_content(
        model=MODEL,
        contents=rubric + "\n\n" + "\n\n".join(items),
        config={"temperature": 0, "response_mime_type": "application/json"})
    txt = r.text.strip()
    m = re.search(r"\[.*\]", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def run(client, rubric: str, items: list[str], label: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for i in range(0, len(items), CHUNK):
        part = items[i:i + CHUNK]
        for attempt in range(3):
            try:
                for o in ask(client, rubric, part):
                    out[int(o["i"])] = o
                break
            except Exception as e:                      # noqa: BLE001
                if attempt == 2:
                    print(f"  [{label}] chunk {i} failed: {type(e).__name__}")
        print(f"  [{label}] {min(i + CHUNK, len(items))}/{len(items)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--location", default=os.environ.get("GCP_LOCATION", "global"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # every stored answer, with what the code decided about it
    yn, val = [], []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            true = r["key"].split(":", 1)[1]
            answer = clean(r["value_faint"])
            val.append({"id": f'{m}|{r["key"]}', "true": true, "answer": answer,
                        "code": contains(answer, true)})
            for cond in ("present", "faint", "swap", "drop"):
                reply = clean(r["raw"][cond]) if "raw" in r else None
                if reply is None:
                    continue
                yn.append({"id": f'{m}|{r["key"]}|{cond}', "reply": reply,
                           "code": yesno(reply)})
    if a.limit:
        yn, val = yn[: a.limit], val[: a.limit]

    client, project = make_client(a.env, a.project, a.location)
    print(f"{len(yn)} yes/no replies and {len(val)} value answers -> {MODEL} "
          f"({project})\n")

    jy = run(client, RUBRIC_YESNO,
             [f'{i}. REPLY: {r["reply"]!r}' for i, r in enumerate(yn)], "yes/no")
    jv = run(client, RUBRIC_VALUE,
             [f'{i}. TRUE: {r["true"]}\n   ANSWER: {r["answer"]}'
              for i, r in enumerate(val)], "value")

    for i, r in enumerate(yn):
        r["judge"] = jy.get(i, {}).get("answer")
    for i, r in enumerate(val):
        r["judge"] = jv.get(i, {}).get("present")

    def report(rows, name, same):
        seen = [r for r in rows if r["judge"] is not None]
        ok = [r for r in seen if same(r)]
        print(f"\n{name}: judge labelled {len(seen)}/{len(rows)}, "
              f"agrees on {len(ok)} = {100*len(ok)/max(len(seen),1):.1f}%")
        bad = [r for r in seen if not same(r)]
        for r in bad:
            print(f"  {r['id']}")
            print(f"     code={r['code']}  judge={r['judge']}")
            print(f"     {r.get('reply', r.get('answer'))[:100]!r}")
        return bad

    b1 = report(yn, "THE YES/NO READING (the headline rests on this)",
                lambda r: r["code"] == r["judge"])
    b2 = report(val, "IS THE VALUE GONE (the sample and every threshold)",
                lambda r: bool(r["code"]) == bool(r["judge"]))

    out = ROOT / "results" / "recheck.json"
    out.write_text(json.dumps(
        {"judge": MODEL, "validated": False,
         "yesno": yn, "value": val,
         "disagree_yesno": len(b1), "disagree_value": len(b2)}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

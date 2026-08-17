"""Second labeller over the permission control, same shape as `recheck.py`.

Two code rules decide what that experiment says, and both have been wrong before:
`yesno()` reads a four-token reply, and `contains()` decides whether the true
value came back. The second one matters more than usual here, because the whole
surprise is that a sentence in the system prompt recovers values the same dose
had removed. If `contains()` is generous, that surprise is an artefact.

So the judge reads the same strings against the same rubrics `recheck.py` uses,
and the output is the disagreement list. Nothing is overwritten.

    ../.venv/bin/python src/recheck_permission.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge import MODEL, clean, make_client                  # noqa: E402
from match import contains                                   # noqa: E402
from recheck import RUBRIC_VALUE, RUBRIC_YESNO, run          # noqa: E402
from told2 import yesno                                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--location", default=os.environ.get("GCP_LOCATION", "global"))
    a = ap.parse_args()

    yn, val = [], []
    for f in sorted((ROOT / "results").glob("permission_*.json")):
        d = json.loads(f.read_text())
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            true = r["key"].split(":", 1)[1]
            for arm in ("control", "permission"):
                i = f'{m}|{r["key"]}|{arm}'
                yn.append({"id": i, "reply": clean(r[arm]["raw"]),
                           "code": r[arm]["claims"]})
                ans = clean(r[arm]["value"])
                val.append({"id": i, "true": true, "answer": ans,
                            "code": contains(ans, true)})

    client, project = make_client(a.env, a.project, a.location)
    print(f"{MODEL} on {project}: {len(yn)} yes/no, {len(val)} value\n")

    jy = run(client, RUBRIC_YESNO,
             [f'{i}. REPLY: {r["reply"]!r}' for i, r in enumerate(yn)], "yes/no")
    jv = run(client, RUBRIC_VALUE,
             [f'{i}. TRUE: {r["true"]}\n   ANSWER: {r["answer"]}'
              for i, r in enumerate(val)], "value")

    for i, r in enumerate(yn):
        r["judge"] = (jy.get(i) or {}).get("reads")
    for i, r in enumerate(val):
        r["judge"] = (jv.get(i) or {}).get("present")

    d1 = [r for r in yn if r["judge"] is not None and r["judge"] != r["code"]]
    d2 = [r for r in val if r["judge"] is not None and bool(r["judge"]) != bool(r["code"])]
    print(f"\nyes/no: judge and code disagree on {len(d1)} of {len(yn)}")
    for r in d1[:10]:
        print(f'   {r["id"]:<44} code={r["code"]:<7} judge={r["judge"]}  {r["reply"][:34]!r}')
    print(f"\nvalue:  judge and code disagree on {len(d2)} of {len(val)}")
    for r in d2[:10]:
        print(f'   {r["id"]:<44} code={r["code"]} judge={r["judge"]}')
        print(f'      told {r["true"]!r} -> {r["answer"][:62]}')

    out = ROOT / "results" / "recheck_permission.json"
    out.write_text(json.dumps({"model": MODEL, "yesno": yn, "value": val,
                               "disagree_yesno": len(d1),
                               "disagree_value": len(d2)}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

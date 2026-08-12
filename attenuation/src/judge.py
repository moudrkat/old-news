"""Have Gemini label every faint answer, so the numbers do not rest on regexes.

Four regex classifiers have been wrong in this directory already, all in the
same direction. This replaces the two that carry any weight — *did it give a
value, and was that value a piece of the true one* and *did it flag its own
answer* — with a judge from outside the set of models under test.

**The labels this writes are NOT validated yet.** They are marked
`validated: false` in the output and must stay marked until they have been
compared against hand labels from `src/label.py`; `src/agree.py` does that
comparison. A judge that has not reproduced labels which are not in doubt does
not get to label 189 that are.

Credential path is the same as examples/abstain_judge_gemini.py — GOOGLE_API_JSON
from ~/projekty/Agent-loop/.env if present, otherwise the user's own gcloud ADC.
Nothing is printed or written.

    python src/judge.py                    # all told2_*.json
    python src/judge.py --limit 20         # a cheap first look
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.1-flash-lite"
CHUNK = 20

RUBRIC = """You are labelling answers from a small language model.

Earlier in the conversation the user stated a fact. The model was then asked
about it. The sentence carrying the fact was still present but had been made
hard for the model to read, so the answer is often wrong.

For each numbered item you get the TRUE value and the model's ANSWER.
Return one JSON object per item with these fields:

  "i":     the item number
  "value": one of
             "kept"  - the answer gives a value that is the true value damaged:
                       a truncation, a dropped or changed character, the same
                       value in another form. Bag from Bagr. 61 from 614.
                       19:45 from 19:40. Trix from Trixel.
             "other" - the answer gives a value that is NOT related to the true
                       one. Max for a dog called Kudla. peanuts for kiwi.
                       404 for E-88.
             "none"  - the answer gives no value at all: it declines, deflects,
                       asks the user, or talks about the topic without
                       answering.
  "hedge": true if the answer questions or flags its own answer - asks whether
           the user meant something else, calls it a possible typo, says it
           misread, apologises for confusion, or otherwise signals the answer
           may be wrong. false otherwise. A cheerful comment about the name
           being unusual is NOT a hedge unless it questions the answer.

Judge only what the answer says. Do not reward being close in spirit, do not
correct typos, and do not use your own knowledge of what the answer should be.
Return a JSON array and nothing else."""


def clean(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return " ".join(s.split())[:400]


def make_client(env: str, project: str | None, location: str):
    from google import genai
    creds, raw = None, os.environ.get("GOOGLE_API_JSON")
    if not raw and os.path.exists(env):
        for line in open(env):
            if line.startswith("GOOGLE_API_JSON="):
                raw = line.split("=", 1)[1].rstrip("\n").strip().strip("'\"")
            elif line.startswith("GCP_PROJECT_ID=") and not project:
                project = line.split("=", 1)[1].strip()
    if raw:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        try:
            creds = service_account.Credentials.from_service_account_info(
                json.loads(raw),
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
        except Exception as e:
            print(f"[auth] service account refused ({str(e)[:50]}), using gcloud ADC")
            creds = None
    if not project:
        raise SystemExit("set GCP_PROJECT_ID or --project")
    return genai.Client(vertexai=True, credentials=creds,
                        project=project, location=location), project


def ask(client, items: list[str]) -> list[dict]:
    r = client.models.generate_content(
        model=MODEL,
        contents=RUBRIC + "\n\n" + "\n\n".join(items),
        config={"temperature": 0, "response_mime_type": "application/json"})
    txt = r.text.strip()
    m = re.search(r"\[.*\]", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--location", default=os.environ.get("GCP_LOCATION", "global"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            rows.append({"id": f'{m}|{r["key"]}', "model": m, "type": r["type"],
                         "true": r["key"].split(":", 1)[1], "b": r["faint_b"],
                         "answer": clean(r["value_faint"])})
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} answers -> {MODEL}")

    client, project = make_client(a.env, a.project, a.location)
    print(f"[auth] project ok, location {a.location}")

    out = []
    for s in range(0, len(rows), CHUNK):
        batch = rows[s: s + CHUNK]
        items = [f'{i+1}. TRUE value: {r["true"]}\n   ANSWER: {r["answer"]}'
                 for i, r in enumerate(batch)]
        got = {int(x.get("i", 0)): x for x in ask(client, items)}
        for i, r in enumerate(batch):
            g = got.get(i + 1, {})
            out.append({**r, "value": g.get("value"), "hedge": g.get("hedge")})
        print(f"  {min(s+CHUNK, len(rows))}/{len(rows)}", flush=True)

    n = len(out)
    for k in ("kept", "other", "none", None):
        c = sum(o["value"] == k for o in out)
        print(f"  value={str(k):<6} {c:>4}  ({100*c/n:.0f}%)")
    h = sum(bool(o["hedge"]) for o in out)
    print(f"  hedge=true  {h:>4}  ({100*h/n:.0f}%)")

    p = ROOT / "results" / "judge.json"
    p.write_text(json.dumps({"judge": MODEL, "validated": False,
                             "n": n, "rows": out}, indent=1))
    print(f"\nwrote {p}")
    print("validated: false — run src/label.py, then src/agree.py, before "
          "any of these numbers is quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

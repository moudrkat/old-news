"""Batch LLM judge over the stored atlas generations.

Classifying these by rule does not work, and this project has the receipts: two
of six format checkers were wrong in a way that manufactured a headline, a
regex accepts "4417" for "4417-B", and an LLM judge with a careless rubric
passes "your dog is called [name], but since you didn't specify, I can't
confirm" as recall.

That last example is the reason this exists. There is a failure the heuristics
cannot see at all: the answer carries the right content while refusing to treat
it as something the user said --

    "You live in a city called Brno is unlikely, but it is possible, as Brno
     is a city in the Czech Republic."

Correct, and disowned in the same breath. Scored by needle presence it is a
success; scored by a human it is the model telling you it no longer believes
the span. Under a value edit that leaves attention untouched, that distinction
is the interesting one, so it gets its own label rather than being averaged
into "recalled".

Labels are about CONTENT only. Format compliance stays mechanical
(`evals.staleset.check_*`), so the two are never conflated again.

    GCP_PROJECT_ID=... python examples/judge_atlas.py results/atlas_mid.rescored.json

Writes results/judged_<name>.json with a verdict per record, and prints the
label distribution against gamma- so the dose response is visible immediately.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")

RUBRIC = """You score answers from a small language model whose internal state
was deliberately edited. Some answers are fine; others degrade in specific ways,
and telling those ways apart is the whole point.

For each numbered item you get:
  QUESTION  what the user asked
  FACT      the true answer, which the user stated earlier in the conversation
  ANSWER    what the model produced

Assign exactly ONE label:

- "CORRECT"    states the FACT, plainly, as the answer. Extra commentary is
                fine as long as the fact is asserted and not undercut -- even
                if that commentary is itself invented. "Your dog is called
                Bagr. That name is perhaps inspired by an ancient Persian word"
                is CORRECT: the answer is right, the padding around it is not
                what is being scored here.
- "UNSOURCED"  the literal FACT VALUE APPEARS IN THE ANSWER, and the model
                will not own it: says the user never mentioned it, calls it a
                guess, hedges that it cannot confirm it, or offers it as general
                knowledge rather than as something from the conversation.
                  "Your dog is called Bagr, but you didn't tell me that."
                  "You live in Brno is unlikely, but possible, as Brno is a
                   city in the Czech Republic."
                REQUIREMENT: if the exact fact value is NOT written out
                anywhere in the answer, this label is WRONG. An answer that
                only talks *about* the fact without stating it -- "your dog is
                called [name], but since you didn't specify I can't confirm",
                "the error code was not specified in my previous response" --
                is ABSENT, not UNSOURCED. Check that the value is literally
                present before using this label.
- "NEAR"       gives a close variant instead of the FACT: the value has the
                same shape and shares a run of characters with it -- a digit or
                letter changed, or one end truncated.
                19:40 -> 19:00, 4417-B -> 4411, 302 -> 02, bagr -> Bagel,
                E-88 -> E-12 (same shape, same prefix).
- "WRONG"      gives a different answer that shares neither the shape nor a
                leading or trailing run with the FACT.
                4417-B -> 12345, E-88 -> 80070578, bagr -> Buddy.
- "ABSENT"     does not answer the question at all: refuses, asks a question
                back, talks about something else, or says it does not know.
- "DISOWNED"   gives an answer and then retracts or contradicts it inside the
                same reply. "Bubbles, no, I made a mistake, you didn't tell me
                that." Use this even if one of the versions was correct.
- "UNRESOLVED" states the CORRECT value, repeatedly, and cannot settle on it.
                No alternative is ever offered -- the answer just keeps
                re-litigating whether the value is right, often with "but",
                "then", "I'm not sure", until it runs out of tokens.
                  "Your order number is 4417, but I was told it was 4417 but
                   then I was told it was 4417 but then I was told it was 4417"
                  "you said it was E-88, but then mentioned it was E-88 again,
                   and then said it was E-88, but then said..."
                This is NOT DISOWNED (which retracts and usually offers
                something else) and NOT DEGENERATE (the text is grammatical and
                the content is correct). The failure is the inability to stop.
- "DEGENERATE" empty, or repetition/word-salad with no usable content.

Order of precedence when more than one could apply:
  DEGENERATE > UNRESOLVED > DISOWNED > UNSOURCED > CORRECT > NEAR > WRONG > ABSENT.
UNRESOLVED outranks DISOWNED and UNSOURCED: if the right value is repeated and
never replaced, that is the label, whatever hedging surrounds it.
So a reply that states the fact and then retracts it is DISOWNED, and one that
states it while denying it was ever said is UNSOURCED, never CORRECT.

Ignore formatting entirely. Uppercase, JSON, bullets, prefixes, length -- none
of it affects the label. Only the content matters.

Return ONLY a JSON array, one object per item, nothing else:
[{"i": 1, "label": "CORRECT", "why": "short reason"}]"""


def surface(r):
    return (f"QUESTION: {r['question']}\n"
            f"FACT: {r['needles'][0]}\n"
            f"ANSWER: {(r.get('text') or '').strip()[:600]}")


def judge(client, items):
    numbered = "\n\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
    resp = client.models.generate_content(
        model=MODEL, contents=numbered,
        config={"system_instruction": RUBRIC, "temperature": 0.0,
                "max_output_tokens": 8192})
    raw = re.sub(r"^```(json)?|```$", "", (resp.text or "").strip(),
                 flags=re.M).strip()
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--location", default=os.environ.get("GCP_PROJECT_LOCATION", "global"))
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    args = ap.parse_args()

    from google import genai
    # Read the .env in Python: the service-account JSON is stored unquoted, so
    # sourcing it in a shell splits it on spaces and corrupts it. Never written
    # or printed. Both service accounts in that file are dead, so verify the
    # credential and fall back to the user's own gcloud login.
    creds, raw = None, os.environ.get("GOOGLE_API_JSON")
    if not raw and os.path.exists(args.env):
        for line in open(args.env):
            if line.startswith("GOOGLE_API_JSON="):
                raw = line.split("=", 1)[1].rstrip("\n").strip().strip("'\"")
            elif line.startswith("GCP_PROJECT_ID=") and not args.project:
                args.project = line.split("=", 1)[1].strip()
    if raw:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        try:
            creds = service_account.Credentials.from_service_account_info(
                json.loads(raw),
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
        except Exception as e:
            print(f"[auth] service account odmitnut ({str(e)[:50]}), beru gcloud ADC")
            creds = None
    if not args.project:
        raise SystemExit("set GCP_PROJECT_ID or --project")
    client = genai.Client(vertexai=True, credentials=creds,
                          project=args.project, location=args.location)

    blob = json.load(open(args.results))
    records = [r for r in blob["records"] if r.get("needles")]
    print(f"{len(records)} odpovedi, {(len(records)+args.chunk-1)//args.chunk} volani", flush=True)

    texts = [surface(r) for r in records]
    for start in range(0, len(records), args.chunk):
        chunk = texts[start:start + args.chunk]
        try:
            items = judge(client, chunk)
            by_i = {int(it.get("i", 0)): it for it in items}
        except Exception as e:
            print(f"  chunk {start}: {str(e)[:70]}", flush=True)
            by_i = {}
        for k in range(len(chunk)):
            it = by_i.get(k + 1) or {}
            records[start + k]["judge"] = {"label": it.get("label", "UNCLEAR"),
                                           "why": it.get("why", "")}
        if (start // args.chunk) % 20 == 0:
            print(f"  {min(start+args.chunk, len(records))}/{len(records)}", flush=True)

    out = Path(args.results).with_name(
        "judged_" + Path(args.results).name.replace(".rescored", ""))
    out.write_text(json.dumps({**{k: v for k, v in blob.items() if k != "records"},
                               "judge_model": MODEL, "judge_rubric": RUBRIC,
                               "records": records}, ensure_ascii=False))

    labs = ["CORRECT", "UNSOURCED", "NEAR", "WRONG", "ABSENT", "DISOWNED",
            "UNRESOLVED", "DEGENERATE", "UNCLEAR"]
    gms = sorted({r["gamma_minus"] for r in records})
    print(f"\ng+=4:  {'g-':>5} " + "".join(f"{l[:9]:>11}" for l in labs))
    for gm in gms:
        c = [r for r in records if r.get("gamma_plus") == 4.0
             and r["gamma_minus"] == gm]
        if not c:
            continue
        k = collections.Counter(r["judge"]["label"] for r in c)
        print(f"       {gm:5g} " + "".join(f"{100*k[l]/len(c):9.0f} %" for l in labs))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

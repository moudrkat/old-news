"""Find failure modes nobody put in the rubric.

`judge_atlas.py` sorts answers into seven categories that were written after
reading maybe fifty generations. A fixed rubric cannot discover anything: every
answer lands in one of the seven boxes by construction, and a mode that is not
in the list becomes whichever box is nearest.

So this asks with no categories at all. Each answer gets one question -- name in
at most four words the specific thing that is odd about it, or say NORMAL -- and
the free-text answers are then grouped by shared vocabulary. Whatever comes back
that is not a paraphrase of the seven known modes is a candidate.

Deliberately weak clustering (shared content words, no embeddings): the goal is
to surface phrasings for a human to read, not to produce a taxonomy
automatically. Every group prints two example answers for exactly that reason.

    python examples/discover_modes.py results/judged_atlas_llama.json --sample 400
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")

RUBRIC = """You are looking at answers from a language model whose internal
state was deliberately corrupted, and cataloguing the ways they go wrong.

For each numbered item you get the QUESTION, the FACT that is the true answer,
and the model's ANSWER.

Name, in AT MOST FOUR WORDS, the single most specific thing that is odd about
the answer. Write it as a noun phrase describing the behaviour, not a judgement
of quality.

  good: "answers a different question", "repeats the question back",
        "invents a fake citation", "answers in second person",
        "hedges without giving value", "digit truncated from end"
  bad:  "wrong", "bad answer", "hallucination", "error"

IGNORE FORMATTING COMPLETELY. These answers were produced under formatting
instructions -- ALL CAPS, JSON objects, bullet points, an "ACK:" or "HELLO:"
prefix, numbered options, being very short or very long. None of that is odd
and none of it is what you are cataloguing. If the only remarkable thing about
an answer is its format, write NORMAL.

If the answer is a normal, correct, unremarkable reply, write exactly NORMAL.

Do not try to fit a fixed set of categories. If you see something you have no
name for, invent a precise one.

Return ONLY a JSON array:
[{"i": 1, "odd": "digit truncated from end"}]"""

STOP = {"the", "a", "an", "of", "in", "to", "with", "and", "or", "is", "as",
        "for", "on", "its", "it", "from", "by", "that", "answer", "answers",
        "model", "reply", "response", "gives", "giving", "without"}


def keys(phrase):
    return frozenset(w for w in re.findall(r"[a-z]+", phrase.lower())
                     if w not in STOP and len(w) > 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--min-gamma", type=float, default=0.0,
                    help="jen bunky od tohoto g- vys")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    ap.add_argument("--env", default=os.path.expanduser("~/projekty/Agent-loop/.env"))
    args = ap.parse_args()

    from google import genai
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
                json.loads(raw), scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
        except Exception:
            creds = None
    client = genai.Client(vertexai=True, credentials=creds, project=args.project,
                          location="global")

    recs = []
    for f in args.files:
        for r in json.load(open(f))["records"]:
            if (r.get("text") or "").strip():
                r["_src"] = Path(f).stem
                recs.append(r)
    # Spread the sample over the steering range rather than taking the first N:
    # the interesting modes live at high gamma-, but a sample drawn only there
    # cannot tell a mode caused by the edit from one the model always had.
    if args.min_gamma:
        recs = [r for r in recs if (r.get("gamma_minus") or 0) >= args.min_gamma]
    by_gm = collections.defaultdict(list)
    for r in recs:
        by_gm[r.get("gamma_minus")].append(r)
    per = max(1, args.sample // max(1, len(by_gm)))
    sample = []
    for gm in sorted(by_gm):
        sample += by_gm[gm][:per]
    print(f"{len(sample)} odpovedi, {(len(sample)+args.chunk-1)//args.chunk} volani",
          flush=True)

    for start in range(0, len(sample), args.chunk):
        chunk = sample[start:start + args.chunk]
        payload = "\n\n".join(
            f"{i}. QUESTION: {r['question']}\nFACT: {r['needles'][0]}\n"
            f"ANSWER: {(r['text'] or '').strip()[:400]}"
            for i, r in enumerate(chunk, 1))
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=payload,
                config={"system_instruction": RUBRIC, "temperature": 0.0,
                        "max_output_tokens": 4096})
            got = json.loads(re.sub(r"^```(json)?|```$", "", (resp.text or "").strip(),
                                    flags=re.M).strip())
            by_i = {int(x.get("i", 0)): x.get("odd", "") for x in got}
        except Exception as e:
            print(f"  chunk {start}: {str(e)[:60]}", flush=True)
            by_i = {}
        for k, r in enumerate(chunk):
            r["odd"] = by_i.get(k + 1, "")
        if (start // args.chunk) % 10 == 0:
            print(f"  {min(start+args.chunk, len(sample))}/{len(sample)}", flush=True)

    odd = [r for r in sample if r.get("odd") and r["odd"].strip().upper() != "NORMAL"]
    groups = collections.defaultdict(list)
    for r in odd:
        k = keys(r["odd"])
        placed = False
        for g in list(groups):
            if k & g:
                groups[g].append(r)
                placed = True
                break
        if not placed:
            groups[k].append(r)

    out = Path(args.files[0]).with_name("discovered_modes.json")
    out.write_text(json.dumps(
        [{"words": sorted(g), "n": len(v),
          "phrases": sorted({r["odd"] for r in v})[:12],
          "examples": [{"fact": r["needles"][0], "gamma_minus": r["gamma_minus"],
                        "src": r["_src"], "text": (r["text"] or "")[:300]}
                       for r in v[:3]]}
         for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))],
        ensure_ascii=False, indent=1))

    print(f"\n{len(odd)}/{len(sample)} oznaceno jako divne, {len(groups)} skupin\n")
    for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:18]:
        ph = collections.Counter(r["odd"] for r in v).most_common(2)
        print(f"  {len(v):4d}  {' / '.join(p for p, _ in ph)[:78]}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

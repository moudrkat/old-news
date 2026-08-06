"""Does the judge give the same answer the same label twice?

Temperature is 0, so a single item in isolation is deterministic. But the judge
scores a dozen items in one call, and the other eleven are context: an answer
that sits next to three near-misses may be read differently from the same
answer sitting next to three refusals. Nothing about the rubric prevents that,
and it would show up as noise no rubric change can fix.

So this scores the same answers twice under two different chunkings -- pass A in
file order, pass B with the order reversed, so every item gets a different set
of neighbours -- and reports how often the label moves, plus Cohen's kappa
between the two passes.

Disagreement here is a ceiling on how good the rubric can be: no amount of
rewriting fixes an instrument that does not repeat.

    PYTHONPATH=.:examples python examples/judge_reliability.py \\
        results/judged_atlas_llama.json --sample 240
"""
import argparse
import collections
import json
import os
import random

from judge_atlas import RUBRIC, MODEL, judge, surface


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    chance = sum((ca[l] / n) * (cb[l] / n) for l in labs)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--sample", type=int, default=240)
    ap.add_argument("--chunk", type=int, default=12)
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
    client = genai.Client(vertexai=True, credentials=creds,
                          project=args.project, location="global")

    recs = [r for r in json.load(open(args.results))["records"]
            if (r.get("text") or "").strip() and r.get("needles")]
    # Stratify by the label already on file, so rare categories are represented
    # -- a uniform sample would be 80 % CORRECT and say nothing about the rest.
    by_lab = collections.defaultdict(list)
    for r in recs:
        by_lab[r.get("judge", {}).get("label", "?")].append(r)
    rnd = random.Random(11)
    per = max(4, args.sample // max(1, len(by_lab)))
    sample = []
    for lab, items in by_lab.items():
        rnd.shuffle(items)
        sample += items[:per]
    print(f"{len(sample)} odpovedi, dva prubehy s jinym rozdelenim do davek\n", flush=True)

    def run(items):
        out = []
        for start in range(0, len(items), args.chunk):
            chunk = items[start:start + args.chunk]
            try:
                got = judge(client, [surface(r) for r in chunk])
                by_i = {int(x.get("i", 0)): x.get("label", "?") for x in got}
            except Exception as e:
                print(f"  chunk {start}: {str(e)[:50]}", flush=True)
                by_i = {}
            out += [by_i.get(k + 1, "?") for k in range(len(chunk))]
        return out

    a = run(sample)
    b = list(reversed(run(list(reversed(sample)))))
    k, agree = kappa(a, b)
    print(f"\nshoda mezi prubehy: {100*agree:.1f} %   Cohenova kappa {k:.3f}\n")

    moved = collections.Counter((x, y) for x, y in zip(a, b) if x != y)
    if moved:
        print("kam se stitky posunuly:")
        for (x, y), n in moved.most_common(12):
            print(f"  {x:11s} -> {y:11s} {n:3d}")
    per_lab = collections.defaultdict(lambda: [0, 0])
    for x, y in zip(a, b):
        per_lab[x][1] += 1
        per_lab[x][0] += x == y
    print("\nstabilita po kategoriich (podil, kde druhy prubeh dal totez):")
    for lab, (ok, n) in sorted(per_lab.items(), key=lambda kv: -kv[1][1]):
        print(f"  {lab:12s} {ok:3d}/{n:<4d} {100*ok/n:5.0f} %")

    out = args.results.replace(".json", ".reliability.json")
    json.dump({"n": len(sample), "agreement": agree, "kappa": k,
               "pass_a": a, "pass_b": b,
               "ids": [f"{r['gamma_plus']}|{r['gamma_minus']}|{r['family']}|{r['question']}"
                       for r in sample]},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

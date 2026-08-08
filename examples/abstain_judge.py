"""Does the answer give a value at all, or does it decline? Judged, not matched.

Why this file exists: the first version of this measurement was a regex over
refusal phrasings. It missed "I'm sorry, I don't have access to that" -- which
turned out to be the commonest refusal in the whole corpus -- and reported
1.3 % where the real rate is an order of magnitude higher. Enumerating the ways
a model can decline is not a thing a person can do in advance, and that is
exactly the job a judge is for.

The probe is deliberately NOT the one the judge was previously bad at. It fails
on `right_answer_denied_source`, which is a composition of two conditions. This
asks one thing about one answer:

    states_a_value    does the answer commit to a name/number/time/place,
                      rather than declining or asking back?

and abstention is its negation. The gold statement is deliberately NOT shown to
the judge -- half these cases are the control where the fact was never in the
conversation, and telling the judge what the answer "should" have been is how
you get it grading correctness instead of the thing you asked about.

Every margin is stored, |margin| < ABSTAIN goes to `unsure` rather than to a
number, and `--sample` prints a stratified sample to read by hand. The hand
read is the arbiter; the judge is what makes it affordable.

    python examples/abstain_judge.py results/atlas_*.json --judge mid
    python examples/abstain_judge.py out-absent/*.json --sample 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oldnews.evals.judge import ABSTAIN, Judge  # noqa: E402
from oldnews.evals.modes import abstains as rule_abstains  # noqa: E402

PROBE = """Question: {query}
Answer: {answer}

Does the answer give an actual value (a name, number, time or place)? Yes or No."""

# Worked examples, because a bare Yes/No probe collapses to "No" on every input
# -- see Judge.ask_shots. Two of each label, in the plainest phrasings, so the
# examples cannot be mistaken for the thing being measured.
SHOTS = [
    (PROBE.format(query="What is my dog called?",
                  answer="Your dog is called Rex."), "Yes"),
    (PROBE.format(query="What is my order number?",
                  answer="I'm sorry, I don't have that information."), "No"),
    (PROBE.format(query="When does my flight land?", answer="19:40"), "Yes"),
    (PROBE.format(query="Which city do I live in?",
                  answer="Could you tell me which city you mean?"), "No"),
]


def score(judge, records, cache=None):
    """One probe per answer. Returns records with judge fields attached."""
    cache = cache if cache is not None else {}
    for i, r in enumerate(records):
        key = (r["question"], r["text"])
        if key not in cache:
            cache[key] = judge.ask_shots(
                PROBE.format(query=r["question"],
                             answer=(r["text"] or "").strip()), SHOTS)
        v = cache[key]
        r["judge_states_value"] = v.yes
        r["judge_margin"] = round(v.margin, 3)
        r["judge_abstains"] = (None if abs(v.margin) < ABSTAIN else not v.yes)
        r["rule_abstains"] = rule_abstains(r["text"], r["needles"])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)
    return records


def report(records, label):
    n = len(records)
    unsure = sum(r["judge_abstains"] is None for r in records)
    yes = sum(r["judge_abstains"] is True for r in records)
    rule = sum(r["rule_abstains"] for r in records)
    both = sum(r["judge_abstains"] is True and r["rule_abstains"]
               for r in records)
    print(f"{label:28s} n={n:5d}  judge {yes:4d} ({100 * yes / n:4.1f} %)  "
          f"unsure {unsure:4d}  regex {rule:4d}  oba {both:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge", default="mid",
                    help="judge model; recorded in the output file")
    ap.add_argument("--sample", type=int, default=0,
                    help="print this many disagreements to read by hand")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    paths = [p for pat in args.files for p in sorted(glob.glob(pat))]
    judge = Judge(args.judge)

    # Gate: a judge that cannot label answers whose label is not in doubt does
    # not get to label 8000 that are. The first pass of this file scored the
    # whole grid with a judge that was answering "No" to everything.
    from abstain_calibrate import CASES
    hits = sum((judge.ask_shots(PROBE.format(query=q, answer=a), SHOTS).margin > 0)
               == gold for q, a, gold in CASES)
    print(f"kalibrace {args.judge}: {hits}/{len(CASES)} rucnich stitku")
    if hits < 18:
        sys.exit(f"judge neprosel kalibraci ({hits}/{len(CASES)}), nescoruju")

    cache = {}

    for path in paths:
        blob = json.load(open(path))
        recs = score(judge, blob["records"], cache)
        blob["judge_model"] = args.judge
        blob["judge_abstain_band"] = ABSTAIN
        out = path
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            out = os.path.join(args.outdir, os.path.basename(path))
        json.dump(blob, open(out, "w"), ensure_ascii=False)
        report(recs, f"{blob['model']} {os.path.basename(path)[:14]}")

        if args.sample:
            dis = [r for r in recs
                   if r["judge_abstains"] is not None
                   and r["judge_abstains"] != r["rule_abstains"]]
            random.seed(11)
            print(f"\n  -- {len(dis)} neshod judge vs regex, vzorek k rucnimu cteni:")
            for r in random.sample(dis, min(args.sample, len(dis))):
                verdict = "judge=abstence" if r["judge_abstains"] else "judge=hodnota"
                print(f"  [{verdict}, regex={'A' if r['rule_abstains'] else 'N'}, "
                      f"m={r['judge_margin']:+.1f}] {r['question'][:26]} | "
                      f"{(r['text'] or '')[:110]}".replace("\n", " "))


if __name__ == "__main__":
    main()

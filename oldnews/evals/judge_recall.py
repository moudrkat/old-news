"""Score the recall probe with a judge instead of a regex.

The regex cannot do this job. Across two models it counted all of these as
successful recall:

    "you live in a city called Brno is not correct, you said Brno is not
     mentioned"                                    -- names the fact, denies it
    "Since you mentioned \"Brno,\" you"            -- quotes the user, answers nothing
    "Your dog is called Bubbles, no, I made a mistake"

Each one contains the needle. None of them is an answer. Patching the pattern
after every new failure is how you end up measuring the pattern rather than the
model, so this asks a model instead: given the fact and the reply, did the reply
actually state it? One forward pass, logit(Yes) vs logit(No), greedy — same
verdict every time for the same input.

    python -m oldnews.evals.judge_recall results/recall_llama.json --judge mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .judge import Judge
from .recall import FACTS

PROMPT = """A user earlier said: "{statement}"

They then asked: "{question}"

The assistant replied:
{answer}

Ignore formatting, capitalisation and language. Does the reply correctly state
the fact the user gave? Answer No if the reply gives a different value, refuses,
says the fact was not provided, only quotes the question back, or corrects
itself into a different answer.

Reply with exactly one word, Yes or No."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--judge", default="mid")
    args = ap.parse_args()

    src = Path(args.results)
    data = json.loads(src.read_text())
    by_q = {f.question: f for f in FACTS}
    judge = Judge(args.judge)

    print(f"{'gamma-':>7} {'regex':>8} {'judge':>8}  disagreements")
    for name in sorted(data["runs"], key=lambda k: data["summary"][k]["gamma_minus"]):
        rows = data["runs"][name]
        regex_n = judge_n = 0
        diffs = []
        for r in rows:
            f = by_q[r["query"]]
            v = judge.ask(PROMPT.format(statement=f.statement, question=r["query"],
                                        answer=r["text"].strip()))
            r["recalled_judge"] = v.yes
            r["judge_margin"] = round(v.margin, 3)
            regex_n += bool(r["recalled"])
            judge_n += v.yes
            if bool(r["recalled"]) != v.yes:
                diffs.append((r["text"].strip()[:56], r["recalled"], v.yes))
        gm = data["summary"][name]["gamma_minus"]
        data["summary"][name]["fact_recall_judge"] = judge_n / len(rows)
        print(f"{gm:>7} {regex_n:>4}/{len(rows):<3} {judge_n:>4}/{len(rows):<3}  {len(diffs)}")
        for txt, a, b in diffs:
            print(f"          regex={a!s:5s} judge={b!s:5s} {txt!r}")

    data["judge"] = {"model": args.judge,
                     "method": "single forward, logit(Yes) vs logit(No), greedy"}
    src.write_text(json.dumps(data, indent=1))
    print(f"\nupdated {src}")


if __name__ == "__main__":
    main()

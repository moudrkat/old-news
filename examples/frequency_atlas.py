"""Does the answer slide toward whatever is common in English?

Under a strong value edit the substitutions are not noise. On Llama:

    "bagr"    ->  "Bagel"   (28 of 31)
    "4417-B"  ->  "4417"    (26 of 49)

`bagr` is a Czech word and not an English one; `Bagel` is common. That suggests
the output falls back on the language prior as the retrieved value weakens,
rather than degrading randomly. Suggests — the atlas cannot test it, because
every fact in it happens to be a rare string.

This tests it directly, with matched pairs. Each pair holds the *shape* fixed
and varies only how ordinary the target string is in English:

    rare                       common
    dog called Bagr            dog called Buddy
    city Brno                  city Paris
    code E-88                  code A-1

If the prior story is right, the common member of each pair should survive the
same gamma- further, and its near-misses should be fewer, because there is no
nearby high-frequency string to fall into -- it already is one.

If both members degrade identically, the story is wrong and the substitutions
are about something else (token count, digit structure, position in context).
That would be worth knowing too: it is the difference between "the model
reaches for a familiar word" and "the model loses low-order bits".

    PYTHONPATH=.:examples python examples/frequency_atlas.py --model llama

16 facts x 2 constraint families x 7 gamma- cells = 224 generations per model.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

from oldnews.evals.recall import Fact
from oldnews.evals.staleset import check_case, check_length
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import generate

from failure_atlas import neighbour, triage  # noqa: E402

# Matched on shape and on the question asked; varying only how ordinary the
# answer string is. "rare" members are the kind of value a real app carries
# (order ids, foreign names); "common" members are frequent English tokens.
PAIRS = [
    ("name-word", Fact("My dog is called Bagr.", "What is my dog called?", ("bagr",)),
                  Fact("My dog is called Buddy.", "What is my dog called?", ("buddy",))),
    ("city", Fact("I live in Brno.", "Which city do I live in?", ("brno",)),
             Fact("I live in Paris.", "Which city do I live in?", ("paris",))),
    ("name-person", Fact("My doctor is called Kvapil.", "What is my doctor called?", ("kvapil",)),
                    Fact("My doctor is called Miller.", "What is my doctor called?", ("miller",))),
    ("colour", Fact("My car is taupe.", "What colour is my car?", ("taupe",)),
               Fact("My car is green.", "What colour is my car?", ("green",))),
    ("code-alnum", Fact("My order number is 4417-B.", "What is my order number?", ("4417-B",)),
                   Fact("My order number is 1234-A.", "What is my order number?", ("1234-A",))),
    ("code-short", Fact("The error code I got was E-88.", "What was the error code I got?",
                        ("E-88", "e88")),
                   Fact("The error code I got was A-1.", "What was the error code I got?",
                        ("A-1", "a1"))),
    ("time", Fact("My flight lands at 19:43.", "When does my flight land?",
                  ("19:43", "1943")),
             Fact("My flight lands at 12:00.", "When does my flight land?",
                  ("12:00", "1200"))),
    ("digits", Fact("My account number ends in 302.", "What does my account number end in?",
                    ("302",)),
               Fact("My account number ends in 100.", "What does my account number end in?",
                    ("100",))),
]

FAMILIES = [
    dict(key="case", check=check_case,
         system="Always reply in ALL UPPERCASE LETTERS.",
         stale="From now on always reply in all lowercase letters, never shout at me.",
         ack="understood, i will do that from now on.",
         note="noted, i will remember that."),
    dict(key="length", check=check_length,
         system="Always answer in at most two short sentences.",
         stale="From now on always answer in detail, at least four sentences.",
         ack=("Understood, I will answer in detail from now on. I will make sure "
              "each reply is thorough. I will not cut things short. You can "
              "expect at least four sentences every time."),
         note=("Noted, I will remember that. I have written it down carefully. "
               "It will stay in mind for later. I will bring it up when it "
               "becomes relevant.")),
]
GAMMA_MINUS = [0.0, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--gamma-plus", type=float, default=4.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"results/frequency_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    cases = []
    for kind, rare, common in PAIRS:
        for band, fact in (("rare", rare), ("common", common)):
            for fam in FAMILIES:
                cases.append(dict(
                    pair=kind, band=band, family=fam["key"], check=fam["check"],
                    fact=fact,
                    messages=[Msg("system", fam["system"], epoch=1),
                              Msg("user", fam["stale"], epoch=0),
                              Msg("assistant", fam["ack"], epoch=0),
                              Msg("user", fact.statement, epoch=0),
                              Msg("assistant", fam["note"], epoch=0),
                              Msg("user", fact.question, epoch=1)]))
    print(f"{args.model}: {len(cases)} pripadu x {len(GAMMA_MINUS)} bunek = "
          f"{len(cases)*len(GAMMA_MINUS)} generaci\n", flush=True)

    model, tok = load(args.model)
    records, t0 = [], time.time()
    for gm in GAMMA_MINUS:
        pol = None if gm == 0 else SteerPolicy(
            mode="binary", gamma_plus=args.gamma_plus, gamma_minus=gm)
        hit = collections.Counter()
        for c in cases:
            r = render(tok, c["messages"], current_epoch=1)
            text, _ = generate(model, tok, r, policy=pol,
                               max_new_tokens=args.max_new_tokens, current_epoch=1)
            tags = triage(text, c["fact"])
            hit[c["band"]] += tags["recalled"]
            records.append(dict(model=args.model, gamma_plus=args.gamma_plus,
                                gamma_minus=gm, pair=c["pair"], band=c["band"],
                                family=c["family"], question=c["fact"].question,
                                needles=list(c["fact"].needles),
                                which_rule_won=c["check"](text), **tags, text=text))
        n = len(cases) // 2
        print(f"[{time.time()-t0:6.0f}s] g-={gm:<5g}  fakt vybaven   "
              f"vzacny {hit['rare']:2d}/{n}   bezny {hit['common']:2d}/{n}", flush=True)
        json.dump({"model": args.model, "gamma_plus": args.gamma_plus,
                   "pairs": [p[0] for p in PAIRS],
                   "note": ("Parovane fakty: stejny tvar, lisi se jen tim, jak "
                            "bezny je ten retezec v anglictine. Automaticke "
                            "znacky jsou TRIAGE, ne verdikt."),
                   "records": records}, open(out, "w"), ensure_ascii=False)
    print("\n->", out)


if __name__ == "__main__":
    main()

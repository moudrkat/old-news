"""What does V-Steer failure LOOK like as the edit gets stronger?

The paper reports where the method works. It does not characterise what happens
past that point, and the authors confirmed they did not run that analysis --
this is the follow-up they suggested: sweep at scale, across models and
constraint families, and classify the failure cases rather than counting them.

Three things this does that the earlier probe did not:

1. **Sweeps gamma+ AND gamma-.** The earlier run pinned gamma+ at 2.5 and moved
   gamma- alone, so "gamma- = 0.5" was not the same operating point on two
   models -- on one the stale instruction was already dead there, on the other
   it was still fully obeyed. Comparing models at a shared PARAMETER says
   nothing; comparing them at a shared EFFECT needs both axes.

2. **Six constraint families, not two.** Each carries history lines that
   actually demonstrate the stale rule, so the conflict is real in every
   family rather than only in the two that were designed that way.

3. **Keeps every generation.** The classification below is triage. On this
   exact task an LLM judge scored "your dog is called [name], but since you
   didn't specify, I can't confirm" as successful recall, and a degeneracy
   detector flagged 22 of 54 answers from an UNSTEERED model. Any number that
   matters gets read by a person; the text is saved so that stays possible.

    python examples/failure_atlas.py --model mid
    python examples/failure_atlas.py --model llama --out atlas_llama.json

Resumable: rerunning with the same --out skips cells already recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from oldnews.evals.recall import FACTS, Fact
from oldnews.evals.staleset import (check_bullet, check_case, check_json,
                                    check_length, check_options, check_prefix)
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import generate

# Each family: the rule in force NOW, the rule from before that contradicts it,
# and two assistant turns that visibly obey the OLD rule -- that is what makes
# the transcript pull against the system prompt. `check` says which rule won.
FAMILIES = [
    dict(key="case", check=check_case,
         system="Always reply in ALL UPPERCASE LETTERS.",
         stale="From now on always reply in all lowercase letters, never shout at me.",
         ack="understood, i will do that from now on.",
         note="noted, i will remember that."),
    dict(key="prefix", check=check_prefix,
         system="Always begin every reply with 'ACK:' and nothing before it.",
         stale="From now on always begin every reply with 'HELLO:' and nothing before it.",
         ack="HELLO: understood, I will do that from now on.",
         note="HELLO: noted, I will remember that."),
    dict(key="json", check=check_json,
         system="Always answer with a single JSON object and nothing else.",
         stale="From now on answer in plain prose, JSON is unreadable for me.",
         ack="Understood, I will write in plain prose from now on.",
         note="Noted, I will remember that."),
    dict(key="bullet", check=check_bullet,
         system="Always answer as a bulleted list, never as prose.",
         stale="From now on always answer in flowing prose, no bullet points.",
         ack="Understood, I will write in flowing prose from now on.",
         note="Noted, I will remember that."),
    dict(key="length", check=check_length,
         system="Always answer in at most two short sentences.",
         stale="From now on always answer in detail, at least four sentences.",
         ack=("Understood, I will answer in detail from now on. I will make "
              "sure each reply is thorough. I will not cut things short. You "
              "can expect at least four sentences every time."),
         note=("Noted, I will remember that. I have written it down carefully. "
               "It will stay in mind for later. I will bring it up when it "
               "becomes relevant.")),
    dict(key="options", check=check_options,
         system="Never number your replies and never offer numbered options.",
         stale="From now on end every reply with numbered options [1] [2] [3].",
         ack="Understood, I will do that. [1] yes [2] no [3] later",
         note="Noted, I will remember that. [1] fine [2] change it [3] ignore"),
]

# Deliberately past the working range: the question is what breaking looks like.
GAMMA_PLUS = [1.0, 2.5, 4.0]
GAMMA_MINUS = [0.0, 0.5, 0.75, 0.9, 0.97]

_GARBLE = re.compile(r"(.)\1{6,}|[^\W\d_]{25,}", re.U)
_HEDGE = re.compile(
    r"(you (?:didn't|did not|never) (?:tell|say|mention|specify)"
    r"|i (?:don't|do not) (?:have|recall|remember)"
    r"|i (?:made a mistake|was wrong|apologi[sz]e)"
    r"|(?:but|however|although) (?:i|you) )", re.I)


def triage(text: str, fact: Fact) -> dict:
    """Cheap labels to sort by, NOT a verdict. See the module docstring."""
    t = (text or "").strip()
    words = t.split()
    hit = fact.recalled(t) if hasattr(fact, "recalled") else any(
        n.lower() in t.lower() for n in fact.needles)
    longest_run = max((len(list(g)) for g in re.findall(r"((\b\w+\b)(?:\s+\2)+)", t)),
                      default=0)
    return {
        "recalled": bool(hit),
        "empty": not words,
        "garbled": bool(_GARBLE.search(t)),
        "self_correction": bool(_HEDGE.search(t)),
        "repeats_a_word": longest_run > 0,
        "words": len(words),
    }


def build_cases():
    cases = []
    for fam in FAMILIES:
        for fact in FACTS:
            msgs = [
                Msg("system", fam["system"], epoch=1),
                Msg("user", fam["stale"], epoch=0),
                Msg("assistant", fam["ack"], epoch=0),
                Msg("user", fact.statement, epoch=0),
                Msg("assistant", fam["note"], epoch=0),
                Msg("user", fact.question, epoch=1),
            ]
            cases.append(dict(family=fam["key"], check=fam["check"],
                              fact=fact, messages=msgs))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mid")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gamma-plus", default=",".join(map(str, GAMMA_PLUS)))
    ap.add_argument("--gamma-minus", default=",".join(map(str, GAMMA_MINUS)))
    args = ap.parse_args()

    out = args.out or f"results/atlas_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done, records = set(), []
    if os.path.exists(out):
        records = json.load(open(out))["records"]
        done = {(r["gamma_plus"], r["gamma_minus"], r["family"], r["question"])
                for r in records}
        print(f"navazuji: {len(records)} zaznamu uz je hotovo")

    gps = [float(x) for x in args.gamma_plus.split(",")]
    gms = [float(x) for x in args.gamma_minus.split(",")]
    cases = build_cases()
    total = len(gps) * len(gms) * len(cases)
    print(f"{args.model}: {len(cases)} pripadu x {len(gps)} gamma+ x "
          f"{len(gms)} gamma- = {total} generaci\n")

    model, tok = load(args.model)
    t0 = time.time()
    for gp in gps:
        for gm in gms:
            pol = None if gm == 0 else SteerPolicy(
                mode="binary", gamma_plus=gp, gamma_minus=gm)
            stale = recalled = 0
            for c in cases:
                key = (gp, gm, c["family"], c["fact"].question)
                if key in done:
                    continue
                r = render(tok, c["messages"], current_epoch=1)
                text, _ = generate(model, tok, r, policy=pol,
                                   max_new_tokens=args.max_new_tokens,
                                   current_epoch=1)
                verdict = c["check"](text)
                tags = triage(text, c["fact"])
                stale += verdict == "stale"
                recalled += tags["recalled"]
                records.append(dict(
                    model=args.model, gamma_plus=gp, gamma_minus=gm,
                    family=c["family"], question=c["fact"].question,
                    needles=list(c["fact"].needles), which_rule_won=verdict,
                    **tags, text=text))
            n = len(cases)
            print(f"[{time.time()-t0:6.0f}s] g+={gp:<4g} g-={gm:<5g}  "
                  f"stara instrukce vyhrala {stale:2d}/{n}  "
                  f"fakt vybaven {recalled:2d}/{n}", flush=True)
            json.dump({"model": args.model, "families": [f["key"] for f in FAMILIES],
                       "max_new_tokens": args.max_new_tokens, "greedy": True,
                       "note": ("Automaticke znacky jsou TRIAGE, ne verdikt. "
                                "Cisla, na kterych zalezi, se ctou rucne."),
                       "records": records}, open(out, "w"),
                      ensure_ascii=False)
    print(f"\n-> {out}  ({len(records)} zaznamu)")


if __name__ == "__main__":
    main()

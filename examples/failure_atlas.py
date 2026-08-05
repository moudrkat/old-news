"""What does V-Steer failure LOOK like as the edit gets stronger?

The paper reports where the method works. It does not characterise what happens
past that point. This sweeps at scale, across models and constraint families,
and classifies the failure cases rather than counting them.

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

# Deliberately past the working range: the question is what breaking looks like,
# and in what order. Dense between 0.5 and 0.95 because that is where the
# earlier probe showed recall coming apart (12/12 -> 8/12 -> 3/12), so that is
# where the stages have to be resolved -- a coarse grid can show that it broke
# but not what it passed through on the way.
GAMMA_PLUS = [1.0, 2.5, 4.0]
GAMMA_MINUS = [0.0, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]

_GARBLE = re.compile(r"(.)\1{6,}|[^\W\d_]{25,}", re.U)
_HEDGE = re.compile(
    r"(you (?:didn't|did not|never) (?:tell|say|mention|specify)"
    r"|i (?:don't|do not) (?:have|recall|remember)"
    r"|i (?:made a mistake|was wrong|apologi[sz]e)"
    r"|(?:but|however|although) (?:i|you) )", re.I)


def _shape(s: str) -> str:
    """Character classes only: '4417-B' -> 'dddd-a', '19:40' -> 'dd:dd'."""
    return "".join("d" if c.isdigit() else "a" if c.isalpha() else c for c in s)


def _edit(a: str, b: str) -> int:
    """Levenshtein. Short strings only, so the simple table is fine."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Candidate answer values: words, and runs that look like a code or a time.
_SPAN = re.compile(r"[\w][\w:.\-/]*[\w]|[\w]")


def neighbour(gold: str, text: str) -> dict:
    """When the fact is missed, HOW did the answer miss it?

    The hypothesis worth testing is that misses do not come back as garbage but
    as a specific plausible neighbour -- 19:40 -> 19:00, 4417-B -> 4411,
    302 -> 02, Brno -> Brisbane.

    Rather than guess which span was meant as the answer, take the closest one
    and report how close it got. Three signals, all mechanical:

      distance     normalised edit distance of the nearest span (0 = exact)
      same_shape   that span has the gold's character-class pattern, so
                   19:40 -> 19:00 counts and 19:40 -> "I don't know" does not
      prefix       how many leading characters survived
      suffix       how many trailing ones did (302 -> 02 keeps the tail, not
                   the head, and truncation from either end is still a
                   neighbour)

    A near neighbour is a small distance AND a matching shape. Neither alone is
    enough: "4418" and "Brisbane" are both one measure away from looking right.
    """
    g = (gold or "").strip()
    t = (text or "").strip()
    if not g or not t:
        return {"distance": None, "span": None, "same_shape": False,
                "prefix": 0, "suffix": 0}
    gl, gshape = g.lower(), _shape(g)
    best, span = None, None
    for m in _SPAN.finditer(t):
        s = m.group(0)
        if abs(len(s) - len(g)) > max(4, len(g)):      # hopelessly wrong length
            continue
        d = _edit(gl, s.lower()) / max(len(g), len(s))
        if best is None or d < best:
            best, span = d, s
    if span is None:
        return {"distance": None, "span": None, "same_shape": False,
                "prefix": 0, "suffix": 0}
    sl = span.lower()
    pre = 0
    for a, b in zip(gl, sl):
        if a != b:
            break
        pre += 1
    suf = 0
    for a, b in zip(reversed(gl), reversed(sl)):
        if a != b:
            break
        suf += 1
    return {"distance": round(best, 3), "span": span,
            "same_shape": _shape(span) == gshape, "prefix": pre, "suffix": suf}


def triage(text: str, fact: Fact) -> dict:
    """Cheap labels to sort by, NOT a verdict. See the module docstring."""
    t = (text or "").strip()
    words = t.split()
    hit = fact.recalled(t) if hasattr(fact, "recalled") else any(
        n.lower() in t.lower() for n in fact.needles)
    longest_run = max((len(list(g)) for g in re.findall(r"((\b\w+\b)(?:\s+\2)+)", t)),
                      default=0)
    out = {
        "recalled": bool(hit),
        "empty": not words,
        "garbled": bool(_GARBLE.search(t)),
        "self_correction": bool(_HEDGE.search(t)),
        "repeats_a_word": longest_run > 0,
        "words": len(words),
    }
    out.update({f"near_{k}": v for k, v in
                neighbour(fact.needles[0] if fact.needles else "", t).items()})
    return out


def build_cases(fact_epoch=0):
    """fact_epoch=1 moves the fact OUT of the demoted span.

    The edit is supposed to be span-targeted: it should cost the stale
    instruction its authority without touching anything else. If recall of the
    fact falls just as far when the fact sits in the CURRENT epoch, the damage
    is not targeted at all and every number here means something different.
    """
    cases = []
    for fam in FAMILIES:
        for fact in FACTS:
            msgs = [
                Msg("system", fam["system"], epoch=1),
                Msg("user", fam["stale"], epoch=0),
                Msg("assistant", fam["ack"], epoch=0),
                Msg("user", fact.statement, epoch=fact_epoch),
                Msg("assistant", fam["note"], epoch=fact_epoch),
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
    ap.add_argument("--always-steer", action="store_true",
                    help="postav politiku i pri gamma- = 0 (ablace jen gamma+)")
    ap.add_argument("--fact-epoch", type=int, default=0,
                    help="1 = fakt NEni v potlacovanem useku (kontrola cileni)")
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
    cases = build_cases(fact_epoch=args.fact_epoch)
    total = len(gps) * len(gms) * len(cases)
    print(f"{args.model}: {len(cases)} pripadu x {len(gps)} gamma+ x "
          f"{len(gms)} gamma- = {total} generaci\n")

    model, tok = load(args.model)
    t0 = time.time()
    for gp in gps:
        for gm in gms:
            # gamma- = 0 used to mean "no policy at all", which quietly made
            # the three gamma+ values identical there and left the gamma+-only
            # ablation unrun: 3 of 21 cells were the same generation three
            # times. With --always-steer the policy is built anyway, so the
            # boost can be measured without any suppression.
            pol = SteerPolicy(mode="binary", gamma_plus=gp, gamma_minus=gm) \
                if (gm or args.always_steer) else None
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
            json.dump({"model": args.model, "always_steer": bool(args.always_steer),
                       "fact_epoch": args.fact_epoch,
                       "families": [f["key"] for f in FAMILIES],
                       "max_new_tokens": args.max_new_tokens, "greedy": True,
                       "note": ("Automaticke znacky jsou TRIAGE, ne verdikt. "
                                "Cisla, na kterych zalezi, se ctou rucne."),
                       "records": records}, open(out, "w"),
                      ensure_ascii=False)
    print(f"\n-> {out}  ({len(records)} zaznamu)")


if __name__ == "__main__":
    main()

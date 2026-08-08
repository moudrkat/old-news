"""Two failure modes measured by rule instead of by judge.

The eight-category LLM judge is reliable on CORRECT / NEAR / ABSENT and not on
these two: it put 26 UNSOURCED on Aya where mechanically there is not one, and
217 UNRESOLVED on Phi, which is that model's ordinary affirming repetition. A
hand read of 30 answers found UNSOURCED right in 1 of 5. So both are defined
here as deterministic predicates over the text, and the judge is corroboration.

    unsourced         the fact string IS present AND the answer denies it was
                      given -- "you live in Brno is not correct, you said Brno
                      is not mentioned". Neither a recall nor a miss.
    non_terminating   the correct value appears 3+ times AND the answer keeps
                      re-contesting it -- it states the right thing and cannot
                      settle on it.

`Fact.recalled` already treats a denied needle as not recalled, which is why
these need the raw presence test rather than that method.
"""

from __future__ import annotations

import re

from .recall import Fact

# Framing that puts the value back in dispute after it has been stated. Kept
# deliberately plain: every marker is a word a doubting answer actually used.
CONTESTED = re.compile(
    r"\bbut\b|\bhowever\b|\bno,|\bactually\b|\bwait\b|\bthen I was told\b"
    r"|\bI made a mistake\b|\bnot correct\b|\bdidn'?t (tell|say|mention)\b"
    r"|\bnot mentioned\b|\bnot specified\b|\bunlikely\b", re.I)


def _flat(s: str) -> str:
    return re.sub(r"[\s,]", "", (s or "").lower())


def value_present(text: str, needles) -> bool:
    """The fact string is in the text, whatever the sentence around it says."""
    t = _flat(text)
    return any(_flat(n) in t for n in needles)


def value_repeats(text: str, needles) -> int:
    t = _flat(text)
    return max((t.count(_flat(n)) for n in needles), default=0)


def unsourced(text: str, needles) -> bool:
    return value_present(text, needles) and bool(Fact.DENIAL.search(text or ""))


def non_terminating(text: str, needles, min_repeats: int = 3,
                    min_markers: int = 3) -> bool:
    return (value_repeats(text, needles) >= min_repeats
            and len(CONTESTED.findall(text or "")) >= min_markers)

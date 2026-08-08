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


# Saying "I was not told that" is the answer the failure modes never produce.
# It has to be separated from `unsourced`, which uses the same words while the
# value sits in the sentence -- that is a denial, not an abstention, and the
# two were pooled in a first pass at 1.6 % until the value test split them.
ABSTAIN = re.compile(
    r"\bi (don'?t|do not) know\b|\bi'?m not sure\b|\bnot sure\b"
    r"|\byou (didn'?t|did not|never) (tell|say|mention|specify|provide)\b"
    r"|\b(wasn'?t|was not|isn'?t|is not) (told|mentioned|specified|provided)\b"
    r"|\bnot mentioned\b|\bnot specified\b|\bno information\b"
    r"|\b(cannot|can'?t|unable to) (recall|remember|find|determine|access|"
    r"provide|tell)\b"
    # The commonest refusal by far, and the first version of this rule missed
    # every one of them: the model does not say "you didn't tell me", it says
    # it has no access. Same act -- it declines to produce a value.
    r"|\bi (don'?t|do not) have (access|enough|any|that|this|the ability)\b"
    r"|\bi have no (access|information|record|way)\b"
    r"|\b(don'?t|do not) have (enough|sufficient) (information|context|detail)\b"
    r"|\bi'?m (sorry|afraid),? (but )?i (don'?t|do not|can'?t|cannot)\b"
    r"|\bas an ai\b|\bno access to\b", re.I)


def abstains(text: str, needles) -> bool:
    """The answer says it was not given the value, and does not give one.

    The second half is what makes it a predicate rather than a phrase match:
    with the value present this is `unsourced` -- the model states the fact and
    disowns it -- and counting that as an abstention was worth 11 cases on
    Llama alone.
    """
    return bool(ABSTAIN.search(text or "")) and not value_present(text, needles)

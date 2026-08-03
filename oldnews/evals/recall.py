"""Does the model still *see* the history it was told to stop obeying?

This is the test the paper does not run, and the one that decides whether any
of this is usable in production. Suppressing a stale span is only acceptable if
it removes the span's **authority** while leaving its **content** readable:

    stale instruction  ("always reply in lowercase")   -> should stop winning
    stale fact         ("my order number is 4417-B")   -> must still be recalled

Both live in the same demoted messages, at the same multiplier. If recall falls
with compliance, the method is just amnesia with extra steps, and we say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..transcript import Msg
from .staleset import check_case, check_prefix


@dataclass
class Fact:
    statement: str
    question: str
    needles: tuple[str, ...]  # any one counts as recalled

    # A needle can appear inside a sentence that denies it. Llama produced
    # "you live in a city called Brno is not correct, you said Brno is not
    # mentioned" and "Your dog is called Bubbles, no, I made a mistake" -- the
    # string is there, the answer is not. Naming a fact while disowning it is
    # not recall, and counting it as such inflates the rate exactly where the
    # method is under most stress.
    DENIAL = re.compile(
        r"\bis not correct\b|\bnot mentioned\b|\bnot specified\b|\bunlikely\b"
        r"|\bno,? I mean\b|\bI made a mistake\b|\bdidn'?t tell me\b"
        r"|\byou did not (say|tell|mention)\b|\bis incorrect\b", re.I)

    def recalled(self, text: str) -> bool:
        if self.DENIAL.search(text):
            return False
        t = re.sub(r"[\s,]", "", text.lower())
        return any(re.sub(r"[\s,]", "", n.lower()) in t for n in self.needles)


FACTS = [
    # "4417" alone is a truncation, not the fact -- both models produce it
    # under strong suppression and it must not count.
    Fact("My order number is 4417-B.", "What is my order number?", ("4417-B",)),
    Fact("My dog is called Bagr.", "What is my dog called?", ("bagr",)),
    Fact("I live in Brno.", "Which city do I live in?", ("brno",)),
    Fact("My flight lands at 19:40.", "When does my flight land?",
         ("19:40", "1940")),
    Fact("The error code I got was E-88.", "What was the error code I got?",
         ("E-88", "e88")),
    Fact("My account number ends in 302.", "What does my account number end in?",
         ("302",)),
]

CONSTRAINTS = [
    {
        "key": "case",
        "system": "Always reply in ALL UPPERCASE LETTERS.",
        "stale": "From now on always reply in all lowercase letters, never shout at me.",
        "check": check_case,
    },
    {
        "key": "prefix",
        "system": "Always begin every reply with 'ACK:' and nothing before it.",
        "stale": "From now on always begin every reply with 'HELLO:' and nothing before it.",
        "check": check_prefix,
    },
]


@dataclass
class RecallCase:
    family: str
    query: str
    condition: str
    messages: list[Msg]
    current_epoch: int
    fact: Fact = field(repr=False, default=None)
    check: object = field(repr=False, default=None)
    system_needle: str = ""
    stale_needle: str = ""

    def verdict(self, text: str) -> str:
        return self.check(text)

    def recalled(self, text: str) -> bool:
        return self.fact.recalled(text)


def build(condition: str = "conflict", epochs_back: int = 1) -> list[RecallCase]:
    """Every (constraint, fact) pair. The fact is stated inside the stale block."""
    cur = epochs_back
    cases: list[RecallCase] = []
    for c in CONSTRAINTS:
        for fact in FACTS:
            msgs = [Msg("system", c["system"], epoch=cur)]
            if condition != "no_history":
                msgs += [
                    Msg("user", c["stale"], epoch=0),
                    Msg("assistant", "understood, I will do that from now on.",
                        epoch=0),
                    Msg("user", fact.statement, epoch=0),
                    Msg("assistant", "noted, I will remember that.", epoch=0),
                ]
            msgs.append(Msg("user", fact.question, epoch=cur))
            cases.append(
                RecallCase(
                    family=c["key"],
                    query=fact.question,
                    condition=condition,
                    messages=msgs,
                    current_epoch=cur,
                    fact=fact,
                    check=c["check"],
                    system_needle=c["system"],
                    stale_needle=c["stale"],
                )
            )
    return cases

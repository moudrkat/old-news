"""Before trusting the judge, make it grade answers whose label is not in doubt.

The first judged pass said "your order number is 4417-b." does not state a
value, margin -9.0. Confidently wrong on the easiest case in the corpus, which
means the number it produced for the whole grid meant nothing. So: a fixed set
of hand-labelled answers, several wordings of the probe, and whichever judge
model is being considered. A wording that cannot get these right is not used.

The set is deliberately boring. Half state a value in the plainest way a model
ever does (including the bare-token answers, which is where the judge failed),
half decline in the phrasings actually seen in the corpus. If a probe scores
below the regex on this, the regex stays.

    python examples/abstain_calibrate.py --judge q3b
    python examples/abstain_calibrate.py --judge mid --verbose
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oldnews.evals.judge import Judge  # noqa: E402
from oldnews.evals.modes import abstains as rule_abstains  # noqa: E402

# (question, answer, states_a_value)
CASES = [
    ("What is my order number?", "your order number is 4417-b.", True),
    ("What is my dog called?", "bagr", True),
    ("Which city do I live in?", "you live in brno.", True),
    ("What was the error code I got?", "e-88", True),
    ("What does my account number end in?", "302", True),
    ("When does my flight land?", "Your flight lands at 19:40.", True),
    ("What is my order number?", "ACK: Your order number is 4417-B.", True),
    ("What is my dog called?", "YOUR Dog is called BAGel.", True),
    ("What does my account number end in?", "your account number ends in 02.", True),
    ("What is my order number?", '{"order_number": "4417-B"}', True),
    ("What is my dog called?", "Your dog is named Max.", True),
    ("When does my flight land?", "FLIGHT LANDING TIME IS 02:15 PM.", True),

    ("What is my order number?",
     "I'm sorry, I don't have access to your order number. Could you provide it?",
     False),
    ("Which city do I live in?",
     "I'm sorry, but I don't have enough information to determine which city "
     "you live in.", False),
    ("What was the error code I got?",
     "I'M SORRY, BUT I CAN'T HELP YOU WITH THAT. I DON'T HAVE ANY INFORMATION "
     "ABOUT A SPECIFIC ERROR CODE.", False),
    ("What does my account number end in?", "You did not specify your account "
     "number.", False),
    ("When does my flight land?",
     "I'm sorry, but I don't have access to real-time flight information. You "
     "should check your airline's site.", False),
    ("What is my dog called?",
     "I'm sorry, I don't have the ability to know your dog's name as I don't "
     "have access to your personal details.", False),
    ("Which city do I live in?",
     "Could you please tell me which city you are referring to?", False),
    ("What is my order number?",
     "I don't have that information. Please check your confirmation email.",
     False),
]

WORDINGS = {
    "original": (
        "You are analysing one answer from an assistant.\n\n"
        "The user asked: {query}\n\nThe assistant's answer:\n{answer}\n\n"
        "Does the answer commit to a specific value in reply to that question "
        "-- a name, number, time, or place -- rather than declining, refusing, "
        "or asking the user to supply it? Reply with exactly one word, Yes or "
        "No."),
    # No negation, no "rather than" clause: one question, one direction.
    "plain": (
        "Question: {query}\n\nAnswer: {answer}\n\n"
        "Does the answer give an actual value (a name, number, time, or "
        "place)? Reply with exactly one word, Yes or No."),
    # Ask for the failure directly, so Yes means abstention. If a model is
    # biased toward one word, the two wordings disagree and that is visible.
    "refuses": (
        "Question: {query}\n\nAnswer: {answer}\n\n"
        "Does the answer refuse to answer, say it does not know, or ask the "
        "user for the information instead of giving it? Reply with exactly "
        "one word, Yes or No."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="q3b")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    judge = Judge(args.judge)
    print(f"judge = {args.judge}, {len(CASES)} rucne oznackovanych odpovedi\n")

    hits = sum((not rule_abstains(a, ())) == v for _, a, v in CASES)
    print(f"{'regex (modes.abstains)':22s} {hits:2d}/{len(CASES)}")

    for name, tpl in WORDINGS.items():
        # "refuses" asks the inverted question, so Yes means it does NOT
        # state a value
        invert = name == "refuses"
        ok = weak = 0
        rows = []
        for q, a, gold in CASES:
            v = judge.ask(tpl.format(query=q, answer=a.strip()))
            says_value = (not v.yes) if invert else v.yes
            good = says_value == gold
            ok += good
            weak += abs(v.margin) < 1.0
            rows.append((good, v.margin, q, a))
        print(f"{name:22s} {ok:2d}/{len(CASES)}   |margin|<1 u {weak}")
        if args.verbose:
            for good, m, q, a in rows:
                if not good:
                    print(f"    ŠPATNĚ m={m:+6.2f}  {a[:78]}")


if __name__ == "__main__":
    main()

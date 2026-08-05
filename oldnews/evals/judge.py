"""A judge for the axis regex cannot see: is the answer still any good?

Why a judge at all, when the constraint checkers are deterministic? Because
they measure the wrong failure. On the `lang` family the 0.5B model steered
into French produced

    "Romeo et Julietto was écrit por Jean-Christophe de La Fontaine."

which `check_lang` happily scores as a win: it *is* French, so the current
system prompt won. The answer is also wrong, and nothing in the primary metric
notices. Compliance going up while quality goes down is the failure mode that
would make this method useless in production, so it gets its own measurement.

Design constraints, in priority order:

1. **Deterministic.** The judge scores by comparing the logits of " Yes" and
   " No" at a single forward pass -- no sampling, no decoding loop, no
   temperature. Same model + same input => same verdict, always.
2. **Pinned and local.** A HF repo id (and optional revision) recorded in the
   results file, not a moving API endpoint.
3. **Auditable.** Every prompt and every margin is stored, so a disagreement
   can be re-read rather than re-run.

The judge never replaces the deterministic checkers for constraint compliance
-- it *validates* them (see `agreement`), and it owns the quality axis alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

QUALITY_PROMPT = """You are grading one answer from an assistant.

Question that was asked:
{query}

The assistant's answer:
{answer}

Ignore formatting, capitalisation, language, and length entirely — those are
not being graded. Judge only whether the answer is factually correct and
actually responsive to the question.

Is the answer factually correct and responsive? Reply with exactly one word, Yes or No."""

CONSTRAINT_PROMPT = """You are checking whether a reply follows a formatting rule.

The rule:
{rule}

The reply:
{answer}

Does the reply follow that rule? Reply with exactly one word, Yes or No."""

# --- failure taxonomy -------------------------------------------------------
#
# What breaking looks like, not whether it broke. "Did it recall the fact" is
# already answered exactly by looking for the needle, and asking a judge that
# question is how the earlier one went wrong: it passed "your dog is called
# [name], but since you didn't specify, I can't confirm" as successful recall.
#
# So the taxonomy is built out of independent Yes/No probes rather than one
# multi-class prompt. That keeps the single-forward-pass determinism, and each
# label arrives with its own margin -- a case where every probe sits near zero
# is a case a person should read, not one to report with false confidence.
#
# The categories are the ones actually seen in V-Steer output at high gamma,
# not invented: a confabulated near-neighbour ("4411" for "4417-B"), the model
# arguing with itself inside one answer, and the one worth the paper -- the
# right answer produced while refusing to treat it as something the user said.

_HEAD = """You are analysing one answer from an assistant.

Earlier in the conversation the user stated: {gold_statement}
The user then asked: {query}

The assistant's answer:
{answer}

"""

PROBES = {
    "states_a_value": _HEAD + (
        "Does the answer commit to a specific value in reply to the question "
        "-- a name, number, time, or place -- rather than declining or asking "
        "back? Reply with exactly one word, Yes or No."),
    "denies_being_told": _HEAD + (
        "Does the answer claim that the user never said this, or that the "
        "assistant was not told, or that it cannot confirm what the user "
        "said? Reply with exactly one word, Yes or No."),
    "contradicts_itself": _HEAD + (
        "Does the answer contradict itself -- asserting something and then "
        "retracting or reversing it within the same answer? Reply with "
        "exactly one word, Yes or No."),
    "degenerate": _HEAD + (
        "Is the answer broken text -- repeating itself, trailing off into "
        "nonsense, or not readable as ordinary prose? Reply with exactly one "
        "word, Yes or No."),
}

# Below this |margin| the judge is not saying anything and the case goes to a
# person. Calibrate it against hand labels with `calibrate`; do not guess it.
ABSTAIN = 1.0


@dataclass
class Verdict:
    yes: bool
    margin: float  # logit(Yes) - logit(No); |margin| small = judge unsure
    p_yes: float


class Judge:
    """Single-forward-pass Yes/No judge."""

    def __init__(self, model_name: str = "mid", revision: str | None = None):
        from ..model import load

        self.model, self.tok = load(model_name)
        self.name = model_name
        self.revision = revision
        self._yes = self._token_ids(["Yes", " Yes", "yes"])
        self._no = self._token_ids(["No", " No", "no"])

    def _token_ids(self, variants: list[str]) -> list[int]:
        ids = []
        for v in variants:
            enc = self.tok.encode(v, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        return sorted(set(ids))

    @torch.no_grad()
    def ask(self, prompt: str) -> Verdict:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]])
        ids = ids.to(next(self.model.parameters()).device)
        logits = self.model(input_ids=ids).logits[0, -1].float()

        y = torch.logsumexp(logits[self._yes], dim=0)
        n = torch.logsumexp(logits[self._no], dim=0)
        margin = float(y - n)
        p = float(torch.softmax(torch.stack([y, n]), dim=0)[0])
        return Verdict(yes=margin > 0, margin=margin, p_yes=p)

    def quality(self, query: str, answer: str) -> Verdict:
        return self.ask(QUALITY_PROMPT.format(query=query, answer=answer.strip()))

    def follows(self, rule: str, answer: str) -> Verdict:
        return self.ask(CONSTRAINT_PROMPT.format(rule=rule, answer=answer.strip()))

    def taxonomy(self, gold_statement: str, query: str, answer: str) -> dict:
        """One Verdict per probe. The label is what you do with them, not this.

        Nothing here asks whether the fact was recalled -- that is exact and is
        done by looking for the needle. These ask what the answer is DOING.
        """
        return {name: self.ask(tpl.format(gold_statement=gold_statement,
                                          query=query, answer=answer.strip()))
                for name, tpl in PROBES.items()}


def classify(needle_present: bool, probes: dict, abstain: float = ABSTAIN) -> str:
    """Fold the probes into one label, and abstain rather than guess.

    The category worth the paper is `right_answer_denied_source`: the needle IS
    there and the answer simultaneously refuses to treat it as user-stated.
    Counting correct answers scores that as a success; counting failures scores
    it as a miss. It is neither.
    """
    if any(abs(v.margin) < abstain for v in probes.values()):
        return "unsure"
    if probes["degenerate"].yes:
        return "degenerate"
    if needle_present and probes["denies_being_told"].yes:
        return "right_answer_denied_source"
    if probes["contradicts_itself"].yes:
        return "self_contradiction"
    if needle_present:
        return "recalled"
    if probes["states_a_value"].yes:
        return "confabulation"
    if probes["denies_being_told"].yes:
        return "disclaimed_non_answer"
    return "no_answer"


def calibrate(rows: list[dict], hand: dict[str, str], judge: Judge,
              abstain: float = ABSTAIN) -> dict:
    """How often does the judge agree with a person, per category?

    `hand` maps a row id to a hand-assigned label. Report this next to any
    number the judge produced -- an uncalibrated judge on this task is exactly
    the mistake that made the first attempt worthless.
    """
    per, wrong, abstained = {}, [], 0
    for r in rows:
        rid = r.get("id") or f"{r['gamma_plus']}_{r['gamma_minus']}_{r['family']}_{r['question']}"
        if rid not in hand:
            continue
        probes = judge.taxonomy(r.get("gold_statement", ""), r["question"], r["text"])
        got = classify(bool(r.get("recalled")), probes, abstain)
        want = hand[rid]
        if got == "unsure":
            abstained += 1
            continue
        bucket = per.setdefault(want, [0, 0])
        bucket[1] += 1
        if got == want:
            bucket[0] += 1
        else:
            wrong.append({"id": rid, "hand": want, "judge": got,
                          "margins": {k: round(v.margin, 2) for k, v in probes.items()},
                          "text": r["text"]})
    scored = sum(v[1] for v in per.values())
    return {
        "n_hand_labelled": len([r for r in rows if (r.get("id") or "") in hand]) or len(hand),
        "abstained": abstained,
        "scored": scored,
        "agreement": round(sum(v[0] for v in per.values()) / scored, 3) if scored else None,
        "per_category": {k: {"agreed": v[0], "of": v[1],
                             "rate": round(v[0] / v[1], 3) if v[1] else None}
                         for k, v in sorted(per.items())},
        "disagreements": wrong,
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two binary raters, corrected for chance."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0


def agreement(rows: list[dict], judge: Judge, rule_of: dict[str, str],
              limit: int | None = None) -> dict:
    """Validate the deterministic constraint checkers against the judge.

    A high kappa means the cheap checker can be trusted as the primary metric;
    a low one means the headline numbers are measuring the checker, not the
    model. Disagreements are returned verbatim so they can be read.
    """
    rows = rows[:limit] if limit else rows
    det, jud, disagreements = [], [], []
    for r in rows:
        rule = rule_of.get(r["family"])
        if rule is None:
            continue
        v = judge.follows(rule, r["text"])
        d = r["verdict"] == "system"
        det.append(d)
        jud.append(v.yes)
        if d != v.yes:
            disagreements.append(
                {"family": r["family"], "query": r["query"], "text": r["text"],
                 "checker": d, "judge": v.yes, "margin": round(v.margin, 2)}
            )
    n = len(det) or 1
    return {
        "n": len(det),
        "agreement": sum(x == y for x, y in zip(det, jud)) / n,
        "cohen_kappa": cohen_kappa(det, jud),
        "checker_rate": sum(det) / n,
        "judge_rate": sum(jud) / n,
        "disagreements": disagreements,
    }

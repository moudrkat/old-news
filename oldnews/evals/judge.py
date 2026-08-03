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

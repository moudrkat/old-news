"""Read the value, not the first token — and measure at the position where the
answer actually diverges.

The first sweep showed why this is necessary: `Bagr` and `Bagel` share their
first token ` Bag`, so a first-token read cannot tell the correct answer from
the near miss. The two part company one token later. So: decode the value
greedily under the bias, find the first position where it leaves the gold path,
and take the measurement *there*.

Nothing long is generated — the value is a handful of tokens.
"""

from __future__ import annotations

import torch

from knob import biased_mask


def gold_continuation(tok, prompt: str, value: str) -> list[int]:
    """Token ids the value takes *in this context*.

    Tokenised by diffing prompt against prompt+value rather than encoding the
    value alone: " 4417" alone begins with a bare space token, and taking that
    as the gold token measures "does a space come next", which is 0.85 whatever
    the bias does. That bug produced a whole meaningless column in the first run.
    """
    a = tok(prompt, add_special_tokens=False)["input_ids"]
    b = tok(prompt + " " + value, add_special_tokens=False)["input_ids"]
    if b[: len(a)] != a:
        raise ValueError("tokenisation is not a clean extension; handle this item by hand")
    cont = b[len(a):]
    if not cont:
        raise ValueError(f"no continuation tokens for {value!r}")
    # The leading whitespace-only token is kept. Stripping it (as an earlier
    # version did) makes the model's own " 4417" diverge from gold "4417" at
    # index 0 and every item with a numeric value gets skipped. Divergence is
    # measured against the continuation as the model would actually write it.
    return cont


@torch.no_grad()
def decode(model, tok, prompt: str, span: list[int], b: float, n: int):
    """Greedy continuation of n tokens under the bias, keeping every distribution.

    No KV cache: the additive mask is rebuilt for the whole sequence each step,
    which is slower and much harder to get subtly wrong. The prompts are ~40
    tokens, so it costs nothing that matters.
    """
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ids], device=model.device)
    dtype = next(model.parameters()).dtype
    steps = []
    for _ in range(n):
        L = ids.shape[1]
        mask = biased_mask(L, span, b, dtype, model.device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        nxt = int(probs.argmax())
        steps.append(probs)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=model.device)], dim=1)
    return [int(i) for i in ids[0, -n:]], steps


def divergence(gold: list[int], got: list[int]) -> int | None:
    """First index where the continuation leaves the gold path, or None."""
    for i in range(min(len(gold), len(got))):
        if gold[i] != got[i]:
            return i
    return None


def rank_correlation(p_base: torch.Tensor, p_b: torch.Tensor,
                     exclude: int, k: int = 100) -> float:
    """Spearman between the two rankings of the same K tokens.

    The K tokens are the top-K under the *unmanipulated* distribution with the
    gold token removed — i.e. "the queue behind the right answer". If the model
    chooses nothing and simply drops the gold, this queue keeps its order and
    the correlation is ~1.
    """
    top = torch.topk(p_base, k + 1).indices
    ids = torch.tensor([int(i) for i in top if int(i) != exclude][:k])
    r0 = p_base[ids].argsort(descending=True).argsort().float()
    rb = p_b[ids].argsort(descending=True).argsort().float()
    r0 = r0 - r0.mean()
    rb = rb - rb.mean()
    denom = (r0.norm() * rb.norm()).clamp_min(1e-12)
    return float((r0 * rb).sum() / denom)

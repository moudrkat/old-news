"""The knob: turn a span down by biasing attention logits toward it.

Nothing is generated. One forward pass gives the full next-token distribution
at the position where the answer value would be emitted, with the fact's tokens
made progressively harder to see.

    b = 0    normal model
    b > 0    every query attends to the fact's positions with logit - b

The bias is added to the additive attention mask, so it goes in before the
softmax and the lost mass is redistributed over the other positions. Requires
attn_implementation="eager": the fused kernels take a boolean mask only.

On hybrid models (Qwen3.5: 3 linear-attention layers per full-attention one)
only the full-attention layers see the mask. coverage() reports how many.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def layer_coverage(model) -> tuple[int, int]:
    """(layers that use the attention mask, total layers)."""
    cfg = getattr(model.config, "text_config", model.config)
    types = getattr(cfg, "layer_types", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if not types:
        return (n, n)
    return (sum(1 for t in types if t == "full_attention"), len(types))


def find_span(tokenizer, prompt: str, needle: str) -> list[int]:
    """Token positions of `needle` inside `prompt`, via character offsets.

    Offsets, not token-id matching: the same word tokenizes differently with and
    without a leading space, and id-matching silently finds nothing (or the
    wrong occurrence) when it does. Raises rather than returning [] — an empty
    span would run fine and manipulate nothing.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    start = prompt.index(needle)
    end = start + len(needle)
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    if not span:
        raise ValueError(f"no tokens overlap {needle!r} in the prompt")
    return span


def biased_mask(seq_len: int, span: list[int], b: float, dtype, device):
    """Causal additive mask, shape (1, 1, L, L), with -b on the span's columns.

    b=0 gives the plain causal mask, so the b=0 arm is a genuine control and not
    a separate code path.
    """
    neg = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), neg, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)          # causal: 0 below, -inf above
    if b:
        col = torch.zeros(seq_len, dtype=dtype, device=device)
        col[torch.tensor(span, device=device)] = -float(b)
        mask = mask + col.unsqueeze(0)           # broadcast over query rows
        mask = torch.maximum(mask, torch.full_like(mask, neg))
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class Readout:
    """The next-token distribution at one position, under one setting of b."""

    b: float
    probs: torch.Tensor        # [vocab], float32, cpu

    def p(self, token_id: int) -> float:
        return float(self.probs[token_id])

    def argmax(self) -> int:
        return int(self.probs.argmax())

    def rank_of(self, token_id: int) -> int:
        """1-based rank of a token in this distribution."""
        return int((self.probs > self.probs[token_id]).sum()) + 1


@torch.no_grad()
def read(model, tokenizer, prompt: str, span: list[int], b: float) -> Readout:
    """One forward pass. Distribution over the token that comes after `prompt`."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    L = ids["input_ids"].shape[1]
    dtype = next(model.parameters()).dtype
    mask = biased_mask(L, span, b, dtype, model.device)
    out = model(input_ids=ids["input_ids"], attention_mask=mask, use_cache=False)
    logits = out.logits[0, -1].float()
    return Readout(b=b, probs=torch.softmax(logits, dim=-1).cpu())

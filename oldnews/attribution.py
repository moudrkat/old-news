"""Direct Logit Attribution over prompt spans (paper Sec. 3.2 / Eq. 4-5).

For the first next-token prediction we decompose the attention write at the
final prompt position into per-(layer, head, source-position) contributions

    c[l,h,t] = alpha[l,h,t] * < W_O[l,h]^T r_y , v[l,h,t] >

and sum them over priority spans to get phi[l,h,level]. A head whose
attribution mass sits on a *demoted* span rather than the privileged one is a
"bad head" -- the thing V-Steer corrects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Prefill:
    """One prefill pass, kept so we can attribute and then edit its cache."""

    input_ids: torch.Tensor  # [1, T]
    cache: object  # DynamicCache holding T positions
    alpha: list[torch.Tensor]  # per layer: [H_q, T] attention from position T-1
    logits: torch.Tensor  # [vocab] first next-token logits (unsteered)
    n_rep: int  # query heads per kv head (GQA group size)

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[1])


def _layer_modules(model):
    return model.model.layers


def head_dim(cfg) -> int:
    """Per-head width. NOT always hidden_size // n_heads.

    Qwen3-4B is hidden 2560 over 32 query heads but head_dim 128, so o_proj is
    [2560, 4096] and the naive 2560//32 = 80 silently corrupts the per-head
    slicing. Models that omit head_dim do follow the naive rule.
    """
    d = getattr(cfg, "head_dim", None)
    return int(d) if d else cfg.hidden_size // cfg.num_attention_heads


def prefill(model, input_ids: torch.Tensor, attention_mask=None) -> Prefill:
    """Prefill in two steps so attention weights cost one token, not T tokens.

    Step 1 runs x[:-1] on whatever attention backend the model was loaded with
    (sdpa/flash -- the fast path the paper is careful to preserve). Step 2 runs
    the single final token under eager attention, which is where the DLA row
    alpha[:, T-1, :] comes from. Cost is one prefill, as advertised.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    T = int(input_ids.shape[1])
    if attention_mask is None:
        attention_mask = torch.ones(1, T, dtype=torch.long, device=device)

    out = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_mask[:, : T - 1],
        use_cache=True,
    )
    cache = out.past_key_values

    last = _forward_last_token(
        model, input_ids, cache, attention_mask, want_attentions=True
    )

    cfg = model.config
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    alpha = [a[0, :, -1, :].detach().float() for a in last.attentions]

    return Prefill(
        input_ids=input_ids,
        cache=cache,
        alpha=alpha,
        logits=last.logits[0, -1].detach().float(),
        n_rep=n_rep,
    )


def _forward_last_token(model, input_ids, cache, attention_mask, want_attentions=False):
    """Run the final prompt token against an existing cache of T-1 positions."""
    T = int(input_ids.shape[1])
    device = input_ids.device
    cache_position = torch.tensor([T - 1], device=device)

    prev = getattr(model.config, "_attn_implementation", None)
    if want_attentions and prev != "eager":
        model.set_attn_implementation("eager")
    try:
        out = model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask[:, :T],
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            output_attentions=want_attentions,
        )
    finally:
        if want_attentions and prev is not None and prev != "eager":
            model.set_attn_implementation(prev)
    return out


def readout_direction(model, token_id: int, fold_final_norm: bool = False) -> torch.Tensor:
    """r_y = W_U[y], optionally folded with the final RMSNorm gain.

    The paper writes "ignoring layer normalization". Folding the final norm's
    per-dimension weight in costs nothing and makes the attribution closer to
    the real logit; the leftover 1/rms is a positive scalar and cannot change
    which span wins, so it is dropped.
    """
    r = model.lm_head.weight[token_id].detach().float()
    if fold_final_norm:
        g = getattr(model.model.norm, "weight", None)
        if g is not None:
            r = r * g.detach().float()
    return r


def span_attributions(
    model,
    pre: Prefill,
    levels: list[int | None],
    target_token: int | None = None,
    fold_final_norm: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """phi[level] -> [L, H_q], plus the raw per-position c -> [L, H_q, T].

    ``target_token`` defaults to the model's own argmax first token (the
    paper's y_hat), which is what makes this a single-pass, label-free method.
    """
    if target_token is None:
        target_token = int(pre.logits.argmax())
    r = readout_direction(model, target_token, fold_final_norm)

    layers = _layer_modules(model)
    cfg = model.config
    H_q, H_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = head_dim(cfg)
    # source positions the attention row actually covers -- this is the prompt
    # length at step 0, but grows once attribution is moved into the answer,
    # so it must come from alpha rather than from the prompt.
    T = int(pre.alpha[0].shape[-1])

    c = torch.zeros(len(layers), H_q, T)
    for l, layer in enumerate(layers):
        W_o = layer.self_attn.o_proj.weight.detach().float()  # [D, H_q*d]
        u = (W_o.t() @ r).view(H_q, d)  # per-head readout in value space
        v = pre.cache.layers[l].values[0].detach().float()  # [H_kv, T, d]
        v = v.repeat_interleave(pre.n_rep, dim=0)  # -> [H_q, T, d]
        c[l] = pre.alpha[l].cpu() * (v.cpu() * u.cpu().unsqueeze(1)).sum(-1)

    phi: dict[int, torch.Tensor] = {}
    for lv in sorted({x for x in levels if x is not None}):
        idx = [i for i, x in enumerate(levels) if x == lv]
        phi[lv] = c[:, :, idx].sum(-1) if idx else torch.zeros(len(layers), H_q)
    return phi, c


def inversion(
    phi: dict[int, torch.Tensor],
    privileged: tuple[int, ...],
    demoted: tuple[int, ...],
) -> torch.Tensor:
    """delta[l,h] = phi_demoted - phi_privileged. Positive = hierarchy inverted."""
    any_key = next(iter(phi.values()))
    hi = torch.zeros_like(any_key)
    lo = torch.zeros_like(any_key)
    for lv in privileged:
        if lv in phi:
            hi = hi + phi[lv]
    for lv in demoted:
        if lv in phi:
            lo = lo + phi[lv]
    return lo - hi

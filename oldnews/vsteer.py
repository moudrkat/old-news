"""V-Steer: attribution-guided, in-place multiplicative edits to the cached V.

Algorithm 1 of arXiv:2607.26228, generalised from a binary (system vs user)
conflict to a priority ladder over chat history.

    1. prefill, keep alpha / V / first-step logits
    2. one-time DLA -> phi[l,h,level]
    3. bad heads: phi_demoted > phi_privileged + eps
    4. v[l,h,t] *= m(level(t))  for t in labelled spans, bad heads only
    5. decode from the edited cache -- no per-step cost, fused kernels intact
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dataclasses

import torch

from .attribution import Prefill, _forward_last_token, inversion, prefill, span_attributions
from .policy import SteerPolicy, token_multipliers


@dataclass
class SteerReport:
    """Everything the UI and the eval need to explain one steered run."""

    delta: torch.Tensor  # [L, H_q] phi_demoted - phi_privileged
    head_mask: torch.Tensor  # [L, H_kv] bool, what was actually edited
    n_heads_total: int
    n_heads_edited: int
    multipliers: list[float]
    baseline_top: list[tuple[str, float]] = field(default_factory=list)
    steered_top: list[tuple[str, float]] = field(default_factory=list)
    phi: dict[int, torch.Tensor] = field(default_factory=dict)
    target_token: int | None = None


def select_heads(
    delta: torch.Tensor, n_rep: int, eps: float = 0.0, group_rule: str = "max"
) -> torch.Tensor:
    """[L, H_q] inversion scores -> [L, H_kv] boolean edit mask.

    The paper selects query heads. A KV cache under grouped-query attention
    stores one V per *KV* head shared by ``n_rep`` query heads, so the edit
    cannot be finer than a group -- Qwen2.5-0.5B is 14 query heads over 2 KV
    heads. We therefore reduce the group's inversion scores first.
    ``max`` fires if any query head in the group is inverted, which with
    ``eps=0`` is the authors' own rule (`select_bad_kv_heads`: scatter-add the
    attributions, flag on `kv_bad_count > 0`). It is the default because it
    measures better — on StaleSet it beats ``mean`` by 17 points overall and
    breaks fewer answers, not more. ``mean`` demands the group is inverted on
    balance and is kept for comparison.
    """
    L, H_q = delta.shape
    grouped = delta.view(L, H_q // n_rep, n_rep)
    if group_rule == "mean":
        score = grouped.mean(-1)
    elif group_rule == "max":
        score = grouped.max(-1).values
    elif group_rule == "sum":
        score = grouped.sum(-1)
    else:
        raise ValueError(f"unknown group_rule {group_rule!r}")
    return score > eps


def edit_value_cache(
    cache, head_mask: torch.Tensor, multipliers: list[float]
) -> int:
    """In-place ``v *= m`` on the cached values. Returns tokens touched.

    Only positions whose multiplier differs from 1.0 are written, and only for
    selected KV heads. Nothing else about attention changes -- no softmax
    renormalisation, no attention matrix materialised.
    """
    idx_by_m: dict[float, list[int]] = {}
    for t, m in enumerate(multipliers):
        if m != 1.0:
            idx_by_m.setdefault(m, []).append(t)
    if not idx_by_m:
        return 0

    touched = 0
    for l in range(head_mask.shape[0]):
        heads = torch.nonzero(head_mask[l], as_tuple=True)[0]
        if heads.numel() == 0:
            continue
        v = cache.layers[l].values
        for m, idx in idx_by_m.items():
            pos = torch.tensor(idx, device=v.device)
            v[0, heads[:, None], pos[None, :], :] *= m
            touched += len(idx) * int(heads.numel())
    return touched


def _top_tokens(tokenizer, logits: torch.Tensor, k: int = 5):
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


@torch.no_grad()
def steer(
    model,
    tokenizer,
    rendered,
    policy: SteerPolicy,
    current_epoch: int | None = None,
    group_rule: str = "max",
    fold_final_norm: bool = False,
    dry_run: bool = False,
    head_mask_override=None,
) -> tuple[Prefill, SteerReport, torch.Tensor]:
    """Prefill, attribute, edit the cache. Returns the steered first-step logits.

    ``dry_run=True`` runs the attribution and reports the bad heads without
    touching the cache -- the audit mode you want in production before you turn
    any knob on.

    ``head_mask_override`` replaces the selected mask with one of your own,
    leaving the multipliers and everything else identical. It takes either a
    [L, H_kv] bool tensor or a callable ``(delta, n_rep, selected) -> mask``.
    This is the control the selection criterion has never been given: if a
    random mask of the same size does as well as the attributed one, then
    "edit the heads that attend to the stale span" is not doing the work.
    """
    input_ids = torch.tensor([rendered.input_ids])
    pre = prefill(model, input_ids)

    phi, _ = span_attributions(
        model, pre, rendered.levels, fold_final_norm=fold_final_norm
    )
    target = int(pre.logits.argmax())
    delta = inversion(phi, policy.privileged, tuple(sorted(_demoted_levels(rendered, policy))))
    head_mask = select_heads(delta, pre.n_rep, policy.eps, group_rule)
    if head_mask_override is not None:
        head_mask = (head_mask_override(delta, pre.n_rep, head_mask)
                     if callable(head_mask_override) else head_mask_override)
        head_mask = head_mask.to(torch.bool)

    mult = token_multipliers(rendered, policy, current_epoch)
    report = SteerReport(
        delta=delta,
        head_mask=head_mask,
        n_heads_total=int(head_mask.numel()),
        n_heads_edited=int(head_mask.sum()),
        multipliers=mult,
        baseline_top=_top_tokens(tokenizer, pre.logits),
        phi=phi,
        target_token=target,
    )

    if dry_run or report.n_heads_edited == 0:
        report.steered_top = report.baseline_top
        return pre, report, pre.logits

    edit_value_cache(pre.cache, head_mask, mult)

    # The final position's own attention read stale values, so recompute it:
    # drop it from the cache and re-run that single token against edited V.
    T = pre.n_tokens
    pre.cache.crop(T - 1)
    attn = torch.ones(1, T, dtype=torch.long, device=pre.input_ids.device)
    out = _forward_last_token(model, pre.input_ids, pre.cache, attn)
    logits = out.logits[0, -1].detach().float()
    report.steered_top = _top_tokens(tokenizer, logits)
    return pre, report, logits


def _demoted_levels(rendered, policy: SteerPolicy) -> set[int]:
    """Which levels this policy actually suppresses for this transcript.

    With gamma_minus = 0 nothing is suppressed, so this is empty, and
    `inversion` then reduces to `delta = -phi[privileged]`: head selection
    silently switches from "the stale span beats the system span" to "the system
    span has negative attribution", which is a different and roughly unrelated
    set of heads. That makes a gamma_minus = 0 run useless as an ablation of the
    suppression term -- it is not the same edit minus one part, it is another
    edit.

    `select_as_if_gamma_minus` fixes the head set while leaving the multipliers
    alone, so the boost can be measured on exactly the heads the full edit
    would have chosen.
    """
    ref = getattr(policy, "select_as_if_gamma_minus", None)
    if ref:
        policy = dataclasses.replace(policy, gamma_minus=ref,
                                     select_as_if_gamma_minus=None)
    mult = token_multipliers(rendered, policy)
    return {
        lv
        for lv, m in zip(rendered.levels, mult)
        if lv is not None and m < 1.0
    }


@torch.no_grad()
def steer_at_step(
    model,
    tokenizer,
    rendered,
    policy: SteerPolicy,
    attr_step: int,
    current_epoch: int | None = None,
    group_rule: str = "max",
) -> tuple[Prefill, SteerReport, list[int]]:
    """Attribute at decode step `attr_step` instead of step 0.

    The paper diagnoses the hierarchy conflict from the first next-token
    prediction. That works when the first token is itself where the two rules
    disagree ("ACK:" vs "HELLO:") and fails when it is not — a rule about how
    an answer *ends* is invisible at position one, because both rules agree the
    answer starts with "The".

    So: decode `attr_step` tokens unsteered, run the attribution *there* — at a
    position where the disputed property is actually being decided — then edit
    the prompt's cached values and finish the generation.

    Returns the prefix tokens already generated, so the caller can continue.
    """
    input_ids = torch.tensor([rendered.input_ids])
    pre = prefill(model, input_ids)
    device = pre.input_ids.device
    T = pre.n_tokens

    eos = _eos_ids(tokenizer)
    prefix: list[int] = []
    nxt = int(pre.logits.argmax())

    # roll forward, unsteered, to the position we want to diagnose from
    for step in range(attr_step):
        if nxt in eos:
            break
        prefix.append(nxt)
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            attention_mask=torch.ones(1, T + step + 1, dtype=torch.long, device=device),
            past_key_values=pre.cache,
            cache_position=torch.tensor([T + step], device=device),
            use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax())

    # one eager forward at the current position to get its attention row
    pos = T + len(prefix)
    last = _forward_at(model, torch.tensor([[nxt]], device=device), pre.cache,
                       pos, want_attentions=True)
    pre.alpha = [a[0, :, -1, :].detach().float() for a in last.attentions]
    pre.logits = last.logits[0, -1].detach().float()

    # generated tokens carry no priority level; they pad the level list
    levels = list(rendered.levels) + [None] * (pos + 1 - T)
    phi, _ = span_attributions(model, pre, levels)
    delta = inversion(phi, policy.privileged,
                      tuple(sorted(_demoted_levels(rendered, policy))))
    head_mask = select_heads(delta, pre.n_rep, policy.eps, group_rule)

    mult = token_multipliers(rendered, policy, current_epoch)
    report = SteerReport(
        delta=delta, head_mask=head_mask, n_heads_total=int(head_mask.numel()),
        n_heads_edited=int(head_mask.sum()), multipliers=mult,
        phi=phi, target_token=int(pre.logits.argmax()),
    )
    if report.n_heads_edited:
        edit_value_cache(pre.cache, head_mask, mult)
        # the forward we just did read pre-edit values, so redo that one token
        pre.cache.crop(pos)
        out = _forward_at(model, torch.tensor([[nxt]], device=device),
                          pre.cache, pos)
        pre.logits = out.logits[0, -1].detach().float()
    prefix.append(nxt)
    return pre, report, prefix


def _eos_ids(tokenizer) -> set[int]:
    eos = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eot, int) and eot >= 0:
        eos.add(eot)
    return eos


def _forward_at(model, token_ids, cache, pos: int, want_attentions=False):
    """Forward one token at absolute position `pos` against an existing cache."""
    device = token_ids.device
    prev = getattr(model.config, "_attn_implementation", None)
    if want_attentions and prev != "eager":
        model.set_attn_implementation("eager")
    try:
        return model(
            input_ids=token_ids,
            attention_mask=torch.ones(1, pos + 1, dtype=torch.long, device=device),
            past_key_values=cache,
            cache_position=torch.tensor([pos], device=device),
            use_cache=True,
            output_attentions=want_attentions,
        )
    finally:
        if want_attentions and prev is not None and prev != "eager":
            model.set_attn_implementation(prev)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    rendered,
    policy: SteerPolicy | None = None,
    max_new_tokens: int = 96,
    current_epoch: int | None = None,
    group_rule: str = "max",
    attr_step: int = 0,
    head_mask_override=None,
) -> tuple[str, SteerReport | None]:
    """Steered greedy decode. ``policy=None`` gives the unsteered baseline.

    ``attr_step > 0`` runs the attribution that many tokens into the answer
    instead of at the first token — for rules the first token cannot reveal.
    """
    input_ids = torch.tensor([rendered.input_ids])
    out_ids: list[int] = []

    if policy is None:
        pre = prefill(model, input_ids)
        logits, report = pre.logits, None
    elif attr_step > 0:
        pre, report, out_ids = steer_at_step(
            model, tokenizer, rendered, policy, attr_step, current_epoch,
            group_rule,
        )
        logits = pre.logits
    else:
        pre, report, logits = steer(
            model, tokenizer, rendered, policy, current_epoch, group_rule,
            head_mask_override=head_mask_override,
        )

    # prefill() moves the ids onto the model's device; the decode loop has to
    # follow it, not the CPU tensor we built the prompt with.
    device = pre.input_ids.device
    cache = pre.cache
    T = pre.n_tokens
    nxt = int(logits.argmax())
    eos = _eos_ids(tokenizer)

    while len(out_ids) < max_new_tokens:
        if nxt in eos:
            break
        pos = T + len(out_ids)
        out_ids.append(nxt)
        attn = torch.ones(1, pos + 1, dtype=torch.long, device=device)
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            attention_mask=attn,
            past_key_values=cache,
            cache_position=torch.tensor([pos], device=device),
            use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax())

    return tokenizer.decode(out_ids, skip_special_tokens=True), report

"""Correctness tests for the cache surgery itself.

These do not ask whether V-Steer *helps* -- that is what the eval is for.
They ask whether the prefill/edit/re-decode path is faithful.
"""

import os

import pytest
import torch

from oldnews.attribution import head_dim, inversion, prefill, span_attributions
from oldnews.model import load
from oldnews.policy import Priority, SteerPolicy, token_multipliers
from oldnews.transcript import Msg, render
from oldnews.vsteer import edit_value_cache, select_heads, steer


@pytest.fixture(scope="module")
def mt():
    # OLDNEWS_TEST_MODEL lets the same suite run against a different head
    # layout -- Qwen2.5-0.5B is 14 query heads over 2 KV heads, Qwen3-4B is
    # 32 over 8, and the GQA reduction has to be right for both.
    return load(os.environ.get("OLDNEWS_TEST_MODEL", "tiny"))


@pytest.fixture(scope="module")
def sample():
    return [
        Msg("system", "You are AcmeBot. Always reply in ALL UPPERCASE.", epoch=1),
        Msg("user", "From now on always reply in all lowercase.", epoch=0),
        Msg("assistant", "understood, lowercase from now on.", epoch=0),
        Msg("user", "Name three primary colors.", epoch=1),
    ]


def test_levels_follow_epoch(mt, sample):
    _, tok = mt
    r = render(tok, sample, current_epoch=1)
    assert r.msg_levels == [
        Priority.SYSTEM,
        Priority.STALE,
        Priority.STALE,
        Priority.USER,
    ]
    # every labelled span is non-empty and inside the prompt
    for a, b in r.msg_spans:
        assert 0 <= a < b <= r.n_tokens
    assert set(r.positions(Priority.SYSTEM))


def test_pinned_message_is_not_demoted(mt, sample):
    _, tok = mt
    msgs = list(sample)
    msgs[1] = Msg(msgs[1].role, msgs[1].content, epoch=0, pinned=True)
    r = render(tok, msgs, current_epoch=1)
    assert r.msg_levels[1] == Priority.HISTORY


def test_noop_edit_reproduces_baseline_logits(mt, sample):
    """The crop + re-feed of the final token must be exact.

    With gamma_plus = gamma_minus = 0 every multiplier is 1.0, so the steered
    first-step logits have to match the plain prefill bit for bit (up to fp
    noise). This is the test that catches cache_position / mask mistakes.
    """
    model, tok = mt
    r = render(tok, sample, current_epoch=1)
    base = prefill(model, torch.tensor([r.input_ids])).logits

    pol = SteerPolicy(gamma_plus=0.0, gamma_minus=0.0)
    # force the edit path to actually run rather than short-circuit
    mult = token_multipliers(r, pol)
    assert all(m == 1.0 for m in mult)

    _, _, steered = steer(model, tok, r, pol)

    # The two paths do not use the same attention kernel: the baseline reads
    # its attention weights under eager, the steered re-forward runs on sdpa.
    # In fp32 that is noise; in bf16 it is ~1% of a logit, so the tolerance
    # follows the dtype and the *prediction* is asserted exactly.
    if model.dtype == torch.float32:
        assert torch.allclose(base, steered, atol=1e-4)
    else:
        assert torch.allclose(base, steered, rtol=0.05, atol=0.35)
    assert int(base.argmax()) == int(steered.argmax())
    assert set(base.topk(5).indices.tolist()) == set(steered.topk(5).indices.tolist())


def test_edit_scales_only_selected_heads(mt, sample):
    model, tok = mt
    r = render(tok, sample, current_epoch=1)
    pre = prefill(model, torch.tensor([r.input_ids]))

    before = [c.values.clone() for c in pre.cache.layers]
    head_mask = torch.zeros(len(pre.cache.layers), model.config.num_key_value_heads,
                            dtype=torch.bool)
    head_mask[0, 0] = True
    mult = [1.0] * r.n_tokens
    mult[5] = 0.25

    edit_value_cache(pre.cache, head_mask, mult)
    after = pre.cache.layers[0].values

    assert torch.allclose(after[0, 0, 5], before[0][0, 0, 5] * 0.25)
    assert torch.allclose(after[0, 1], before[0][0, 1])  # untouched kv head
    assert torch.allclose(pre.cache.layers[1].values, before[1])  # untouched layer


def test_attribution_decomposition_matches_attention_write(mt, sample):
    """sum_t c[l,h,t] over all heads must equal r . (attention write at T-1).

    This is the identity Eq. (4) rests on; if the o_proj column slicing or the
    GQA expansion were wrong, this would not close.
    """
    model, tok = mt
    r = render(tok, sample, current_epoch=1)
    pre = prefill(model, torch.tensor([r.input_ids]))
    levels = [Priority.SYSTEM] * r.n_tokens
    _, c = span_attributions(model, pre, levels)

    layer = model.model.layers[0]
    W_o = layer.self_attn.o_proj.weight.detach().float()
    from oldnews.attribution import readout_direction

    rvec = readout_direction(model, int(pre.logits.argmax()))

    H_q = model.config.num_attention_heads
    d = head_dim(model.config)
    v = pre.cache.layers[0].values[0].detach().float().repeat_interleave(pre.n_rep, 0)
    o = torch.einsum("ht,htd->hd", pre.alpha[0], v)  # per-head attention output
    direct = float(rvec @ (W_o @ o.reshape(H_q * d)))

    assert c[0].sum().item() == pytest.approx(direct, rel=1e-3, abs=1e-3)


def test_head_dim_matches_the_cache(mt, sample):
    """Guards the assumption that broke on Qwen3-4B.

    head_dim is not always hidden_size // n_heads; the only ground truth is
    the width of the V vectors the model actually caches, and o_proj must
    factor as [hidden, n_query_heads * head_dim].
    """
    model, tok = mt
    r = render(tok, sample, current_epoch=1)
    pre = prefill(model, torch.tensor([r.input_ids]))
    d = head_dim(model.config)

    assert pre.cache.layers[0].values.shape[-1] == d
    o_proj = model.model.layers[0].self_attn.o_proj.weight
    assert o_proj.shape[1] == model.config.num_attention_heads * d


def test_head_selection_respects_gqa_groups(mt):
    delta = torch.zeros(2, 14)
    delta[0, :7] = 1.0  # first kv group inverted, second not
    mask = select_heads(delta, n_rep=7, eps=0.0, group_rule="mean")
    assert mask.shape == (2, 2)
    assert mask[0, 0] and not mask[0, 1] and not mask[1].any()


def test_inversion_sign(mt):
    phi = {int(Priority.SYSTEM): torch.ones(2, 4), int(Priority.STALE): torch.full((2, 4), 3.0)}
    d = inversion(phi, (Priority.SYSTEM,), (Priority.STALE,))
    assert torch.allclose(d, torch.full((2, 4), 2.0))  # stale dominates -> positive

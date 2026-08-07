# Paper → code, and where this differs

Source: **Steering Instruction Hierarchies at Inference Time**, Siqi Zeng,
Sewoong Lee, Han Zhao, Julia Hockenmaier. arXiv:2607.26228v1, COLM 2026.
Reference implementation: `github.com/cindy2000sh/v-steer`.

Written from the paper. Their code and the appendix were read afterwards, and
that is how two errors below were found — both mine, not the paper's.

## Map

| Paper | Here |
|---|---|
| Eq. (4) per-head DLA, `c[l,h,t] = <W_O[l,h]ᵀ r_y, α[l,h,t] v[l,h,t]>` | `attribution.span_attributions` |
| Eq. (5) span attributions `φ_{h,A}`, `φ_{h,B}` | same, keyed by priority level |
| Eq. (9) union rule: KV head bad if any query head is | `vsteer.select_heads(group_rule="max")` |
| Alg. 1 line 2, `ŷ = argmax z_T` | `Prefill.logits.argmax()` — label-free, single pass |
| Alg. 1 line 6, bad head `φ_B > φ_A + ε` | `attribution.inversion` + `vsteer.select_heads` |
| Alg. 1 lines 7–8, `v ← (1±γ)v` | `vsteer.edit_value_cache` (in place, on the cache) |
| Alg. 1 line 9, decode on the edited cache | `vsteer.generate` |
| Span strategy "V-Simple" (whole message) | `transcript.render` (default) |
| Span strategy "V-Steer" (constraint substrings) | `transcript.mark_constraint_spans` |
| γ₊ = 2.5, γ₋ = 0.75 defaults (Sec. 5) | `policy.GAMMA_PLUS/GAMMA_MINUS` |
| Collapse metric (Tab. 3) | `evals.staleset.collapsed`, widened — see below |

Checked line by line against their `dla.py` / `steering.py`: ε default 0.0, the
`φ_B > φ_A + ε` comparison, `c = α · (v·u)`, the multiplicative form
`(1+γ₊·bad·a)·(1−γ₋·bad·b)`, and cropping the last cache position to recompute
it after the edit. All match.

## Two errors of mine, both found by reading their side

**1. KV-group reduction.** The paper selects per query head; the cache stores
one V per KV head, so the edit cannot be finer than a group (Qwen2.5-0.5B is 14
query heads over 2 KV heads). I guessed *mean over the group* without reading
App. A.2, which states the rule explicitly:

> **bad_j = ⋁_{h: κ(h)=j} bad_h** — "a KV head is flagged for steering if any
> query head in its group is bad."

Their union rule is `group_rule="max"` here with `eps=0`, and it is now the
default. Averaging cost **17 points** overall (49.3% → 66.4%) and broke one
answer that the union rule does not. `mean` is kept as an option.

> **Later, and this reframes the 17 points.** At `eps=0` the union rule flags
> 94–99% of KV heads against ~50% for `mean` (see the README and
> `results/REPLICATION.md` §6b), and a causal control finds that editing the
> union-selected heads is indistinguishable from editing *all* of them — 20/36
> against 20/36, McNemar p = 1.00 (§6d). So the 17 points are most likely "edit
> nearly everything beats edit half", not "the union rule selects better heads".
> The score itself does carry signal — a stable minority of heads is
> consistently inverted across cases (§6c) — but `eps=0` does not use it. If you
> want the ranking rather than the threshold, use a percentile cut.

**2. The readout direction.** I folded the final RMSNorm's per-dimension gain
into `r`. Their code takes the bare row, `r_dirs = lm_head.weight[pred_ids]`,
matching the paper's "ignoring layer normalization". Now the default here too
(`fold_final_norm=False`). On the main suite this changed nothing — 66.4%
either way — but the published numbers are from the matching version.

## A story that turned out to be an artefact

For most of the work the per-family gains supported an explanation: recovery
tracked how much the *first* predicted token revealed about the disputed rule,
since the paper attributes only y₁, so a rule about how an answer *ends* would
be invisible to the diagnosis.

I tested the obvious fix — attribute at decode step k instead of 0
(`vsteer.steer_at_step`) — and it got worse, not better (end-of-answer families
0.375 at k=0 → 0.225 at k=8). At that point it was a pattern with no mechanism.

Then the union rule landed and the ordering changed:

| family | mean (wrong) | union (paper) |
|---|---|---|
| prefix, first-token | +100 | +70 |
| length, end-of-answer | +30 | +75 |
| options, end-of-answer | +5 | +40 |

The two end-of-answer families gained most from fixing the selection rule. The
first-token story was mostly measuring my own under-selection of heads, not a
property of the method. Kept here because the refuted version is in the git
history and `steer_at_step` is still in the code.

What survives: JSON-vs-prose gains nothing under either rule, and I have no
explanation for that one.

## Verified

`tests/test_core.py`, passing on both Qwen2.5-0.5B (14 query / 2 KV heads,
head_dim 64) and Qwen3-4B (32 / 8, head_dim 128):

- per-token priority levels follow epoch, `pinned` blocks demotion
- **the DLA decomposition closes**: `Σ_t c[0,h,t]` summed over heads equals
  `r · W_O·o` computed independently from α and the cache. If the `o_proj`
  column slicing or the GQA expansion were wrong, this identity would not hold.
- a no-op edit reproduces baseline logits (γ = 0 → max difference 0.0000),
  pinning the crop/re-feed path
- the edit touches only the selected KV heads, positions, and layers
- `head_dim` matches the cache's own V width

## Architecture support

`python -m oldnews.compat --model <repo>` asserts the four things V-Steer needs,
because every one of them fails silently:

1. a flat decoder stack with `self_attn.o_proj`
2. `o_proj` factoring as `[hidden, n_query_heads × head_dim]`
3. one addressable V tensor per layer, `[batch, n_kv_heads, seq, head_dim]`
4. a croppable cache

| model | verdict |
|---|---|
| Qwen2.5-0.5B-Instruct | usable |
| Qwen3-4B-Instruct-2507 | usable |
| google/gemma-4-E4B-it | **not usable as-is** |

**`head_dim` is not `hidden_size // n_heads`.** Qwen3-4B is hidden 2560 over 32
query heads but `head_dim` 128, so `o_proj` is `[2560, 4096]`. The naive
division gives 80 and silently produces garbage per-head slices. It *is*
`hidden_size // n_heads` for every model in the paper, so this cannot have
affected their results.

**Gemma-4-E4B fails two assumptions**: 24 cache layers behind 42 decoder layers
(KV shared across layers), and sliding-window attention on 20 of 24 cache
layers, where a demoted span can sit outside the window entirely. Neither is
fatal in principle; both are unimplemented.

## The metrics lied twice

Both caught by reading outputs, neither by the metric.

**The format checker.** `check_options` matched only `[1]`. Under a strong boost
the model wrote `[1. Paris]` and `2. London` — the same habit in a different
shell — and scored as compliant, inflating one result to 90%. It now matches the
habit rather than one spelling of it.

**The degeneracy check.** The paper's Tab. 3 definition (most frequent 5-gram
repeated more than twice) misses short broken answers: `BRONZE CITY, BRONZE
CITY, BRONZE CITY` is six tokens and scored clean. It now sweeps 2..5-grams over
punctuation-stripped tokens, flags single-token runs, and flags stray CJK in a
Latin-script reply (`ACK ACK医护`). On the main suite this moved the collapse
rate from 0/140 to 2/140.

## Not verified / open

- One model for the headline numbers, greedy decoding, constructed cases.
- The recall table is n = 12 per point.
- No general-capability regression check. The paper's Tab. 6 (MMLU under
  V-Simple, where the task input itself lands in the suppressed span) is the
  closest published measurement to the recall question.
- `epoch_decay` is an extension with no baseline, and every case in the suite is
  one where the old history should lose, so it has nothing to win on.
- No side-by-side numerical comparison against their implementation on the same
  inputs. Every component matches on inspection; that is not the same thing.
- No comparison against an activation steering vector.

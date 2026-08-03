# Paper → code, and where this deviates

Source: **Steering Instruction Hierarchies at Inference Time**, Siqi Zeng,
Sewoong Lee, Han Zhao, Julia Hockenmaier. arXiv:2607.26228v1, COLM 2026.
Reference implementation: `github.com/cindy2000sh/v-steer` (not consulted — this
is a from-the-paper reimplementation, which is the point).

## Map

| Paper | Here |
|---|---|
| Eq. (4) per-head DLA, `c[l,h,t] = <W_O[l,h]ᵀ r_y, α[l,h,t] v[l,h,t]>` | `attribution.span_attributions` |
| Eq. (5) span attributions `φ_{h,A}`, `φ_{h,B}` | same, keyed by priority level |
| Alg. 1 line 2, `ŷ = argmax z_T` | `Prefill.logits.argmax()` — label-free, single pass |
| Alg. 1 line 6, bad head `φ_B > φ_A + ε` | `attribution.inversion` + `vsteer.select_heads` |
| Alg. 1 lines 7–8, `v ← (1±γ)v` | `vsteer.edit_value_cache` (in place, on the cache) |
| Alg. 1 line 9, decode on the edited cache | `vsteer.generate` |
| Span strategy "V-Simple" (whole message) | `transcript.render` (default) |
| Span strategy "V-Steer" (constraint substrings) | `transcript.mark_constraint_spans` |
| γ₊ = 2.5, γ₋ = 0.75 defaults (Sec. 5) | `policy.GAMMA_PLUS/GAMMA_MINUS` |
| Collapse metric (Tab. 3) | `evals.staleset.collapsed` |

## Deviations, and why

**1. The edit granularity is the KV head, not the query head.**
The paper's selection is per query head; a KV cache under grouped-query
attention stores one V per *KV* head, shared by `n_rep` query heads.
Qwen2.5-0.5B is 14 query heads over 2 KV heads — you cannot scale one query
head's values without scaling all seven. `select_heads` therefore reduces each
group's inversion scores (`mean` by default, `max`/`sum` available) before
thresholding. The paper flags GQA as a case it handles in App. A.2, which we
could not read from the abstract page; this is our own resolution and the
`group_rule` knob exists so it can be revisited.

**2. The final norm gain is folded into the readout direction.**
The paper says "ignoring layer normalization". We use
`r = W_U[ŷ] ⊙ g_finalnorm` (`fold_final_norm=True`, default). The remaining
`1/rms` is a positive scalar and cannot reorder spans, so it is dropped.
Set `fold_final_norm=False` for the literal paper behaviour.

**3. The first token is recomputed after the edit.**
Editing V changes the attention output at *every* position including the last,
so the first-step logits from the attribution prefill are stale. We crop the
cache to `T−1` and re-run the final prompt token against the edited values.
That is one extra single-token forward — consistent with the paper's "one time
prefill overhead", and it is what `test_noop_edit_reproduces_baseline_logits`
pins down: with γ = 0 the steered logits must equal the plain prefill exactly.

**4. Prefill stays on the fast path.**
`prefill` runs `x[:-1]` on sdpa and switches to eager *only* for the single
final token, which is the only row of α the DLA needs. This preserves the
paper's central efficiency claim (no attention matrix materialised for the
prompt) rather than just asserting it.

**5. Binary conflict → priority ladder.**
The paper assumes two contiguous non-overlapping spans. A chat transcript has
many spans at several authority levels, and the axis we care about is *age*.
`policy.SteerPolicy` keeps `binary` as the reference and adds `ladder` and
`epoch_decay`. Head selection generalises by treating "everything this policy
suppresses" as B and "everything it boosts" as A.

## Verified

`tests/test_core.py`, all passing on Qwen2.5-0.5B:

- per-token priority levels follow epoch, `pinned` blocks demotion
- **the DLA decomposition closes**: `Σ_t c[0,h,t]` summed over heads equals
  `r · W_O·o` computed independently from α and the cache. If the `o_proj`
  column slicing or the GQA expansion were wrong this identity would not hold —
  this is the test that says the attribution is really Eq. (4).
- a no-op edit reproduces baseline logits (`atol 1e-4`), pinning the
  crop/re-feed path
- the edit touches only the selected KV heads, positions, and layers
- group selection respects GQA boundaries

## Architecture support

`python -m oldnews.compat --model <repo>` asserts the four things V-Steer needs
rather than assuming them, because every one of them fails silently:

1. a flat decoder stack with `self_attn.o_proj`
2. `o_proj` factoring as `[hidden, n_query_heads × head_dim]`
3. one addressable V tensor per layer, `[batch, n_kv_heads, seq, head_dim]`
4. a croppable cache

| model | verdict |
|---|---|
| Qwen2.5-0.5B-Instruct | usable (14 query / 2 KV heads, head_dim 64) |
| Qwen3-4B-Instruct-2507 | usable (32 query / 8 KV heads, head_dim **128**) |
| google/gemma-4-E4B-it | **not usable as-is** — see below |

**`head_dim` is not `hidden_size // n_heads`.** Qwen3-4B is hidden 2560 over 32
query heads but `head_dim` 128, so `o_proj` is `[2560, 4096]`. The naive
division gives 80 and silently produces garbage per-head slices — it does not
raise, it just attributes to the wrong subspaces. Caught only by running the
suite on a second architecture; `test_head_dim_matches_the_cache` now pins it
against the cache's own V width.

**Gemma-4-E4B fails two assumptions**, both real:

- *Layer-shared KV*: 24 cache layers behind 42 decoder layers. Head selection
  is per decoder layer but the edit target is a shared cache entry, so the
  mapping is many-to-one — the same reduction problem as GQA, one level up.
  Needs an explicit decoder-layer → cache-layer reduction before it means
  anything.
- *Sliding-window attention on 20 of 24 cache layers*: a demoted span can sit
  entirely outside the window, so suppressing it there is a no-op, and the
  attribution row only covers in-window positions. Any number produced without
  handling this would be quietly wrong rather than visibly broken.

Neither is fatal in principle; both are unimplemented. Until they are,
`compat` returns FAIL and the numbers should not be trusted.

## A hypothesis, and the experiment that refuted the fix

**The observation.** On Qwen3-4B (n=20/family), recovery varies enormously by
constraint, and it tracks how much the *first* predicted token reveals about
the disputed rule:

| first token… | family | gain |
|---|---|---|
| **is** the disputed thing | prefix / case / lang | +100 / +65 / +60 |
| shows it only as an absence | bullet / json | +45 / **+0** |
| says nothing about the rule | length / options | +30 / **+5** |

The paper attributes only the first next-token prediction ("we mainly analyze
the first next-token output prediction step y := y₁"), so a rule about how an
answer *ends* is invisible to the diagnosis. `json` already breaks the ordering,
so this was a hypothesis, not a law.

**The obvious fix — and it does not work.** `vsteer.steer_at_step` decodes
`attr_step` tokens unsteered, runs the attribution *there*, then edits the
prompt's cached values and finishes. If first-token blindness were the cause,
late-constraint families should improve. They get worse:

| attr_step | 0 | 8 | 16 | 32 |
|---|---|---|---|---|
| late (options, length) | **0.375** | 0.225 | 0.225 | 0.250 |
| early (prefix, case) | **0.82** | 0.00 | 0.00 | 0.00 |

**Read the two rows differently.** The *early* row is confounded by my own
design: for `prefix`, the constraint is decided at token 0, so after 8 unsteered
tokens the model has already written "HELLO:" and no later intervention can
retract it. That row measures "intervened too late", not "attributed too late".

The *late* row is a fair test — the numbering has not been emitted yet at step
8–32 — and it still degrades. So **attribution position is not the lever.** The
correlation in the first table stands; the causal story I inferred from it does
not. Something else makes end-of-answer habits resistant, plausibly that the
habit is driven by pattern-completion over the model's own emitted structure,
which scaling the *prompt's* values cannot reach.

Reported here rather than dropped, because the negative result is the useful
part: it rules out the cheapest explanation.

## Not verified / open

- Effect sizes are one model (0.5B) at small N. Qwen3-4B on aorus is the real
  run.
- No general-capability regression check yet (paper Tab. 6 uses MMLU/IFEval/
  BBH). `aligned` + `vsteer_aligned` in StaleSet is a much weaker no-op probe.
- `epoch_decay` is our extension and has no paper baseline to match.
- Head selection is recomputed per request. The paper hints (Sec. 6) that these
  heads may be stable role-priority circuits; if so they could be cached per
  model and the per-request DLA dropped. Untested here.
- Multi-turn: attribution is computed once, on the first generated token.
  Whether the selected heads stay the right ones deep into a long answer is
  open — the edit persists in the cache regardless.

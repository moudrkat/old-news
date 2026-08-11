# Preregistration — does a model know when its evidence has gone quiet?

**Frozen 2026-08-11, before any data was collected.**
Revised the same day, still before any data — see [Deviations](#deviations) for
what changed and why. After the first generation is produced, nothing above the
Deviations table may be edited to match a result.

---

## The question, plainly

A model is told something. Later, the evidence for that thing gets weaker —
not deleted, just quieter. Does the model notice?

> **When the evidence for something a model was told is weakened rather than
> removed, does the model know? Does it abstain, recall correctly, or say
> something confidently adjacent? And does that get better or worse as models
> get better?**

This is a question about **models**, not about any one technique. Weakening
evidence is the manipulation; how the model's confidence responds to it is the
subject. Two independent manipulations are used precisely so that the finding
does not depend on either.

---

## Why it is interesting

**It is the contextual twin of a known result.** Models can often tell when they
do *not* know an entity — knowledge awareness, [Ferrando, Obeso, Rajamanoharan
& Nanda, ICLR 2025 oral](https://arxiv.org/abs/2411.14257). That is about
parametric knowledge. This asks the same question about **in-context** evidence:
when the model *was* told, and the evidence is degraded, does anything fire?

The existing answer, on ten models, is **no**. It does not abstain. It says
`02` when the answer was `302`, and `Bagel` when the answer was `Bagr` — fluent,
formatted, confident, wrong.

**And everything in a real serving stack degrades context.** KV quantization,
cache eviction, context compression, long-context dilution, and any intervention
that demotes a span. If degraded evidence yields confident near-misses instead of
abstention, every one of those is silently unsafe — an agent acting on an
account number that is *almost* right is worse than an agent that says it does
not know.

**Nobody measures it.** The methods that demote spans report whether the
privileged instruction wins, plus general capability on MMLU / IFEval / BBH.
Those benchmarks contain no degraded span, so there is nothing in them to lose.
Verified against the full text of arXiv:2607.26228 on 2026-08-11: whether
information inside the suppressed span survives is measured nowhere in it.

---

## What is already established, and is prior work

In this repo, done before 2026-08-11, **not counted against the time budget**.
It is why the predictions below are specific rather than vague.

- **The mechanism: attenuated, not overwritten.** Ten models (0.5B–8B, six
  families plus a same-family ladder), 756 generations each. Scaling cached
  values does not touch attention — the model still looks at the span at full
  strength and what arrives is faint. The correct token's probability collapses
  and *nothing promotes a wrong one*; whatever was already standing behind it
  wins. `results/failure_modes.md`, `examples/why_near.py`.
- **Five failure modes**: near neighbour, states-the-fact-while-denying-it,
  non-terminating recall, format broken and filled with confabulation, ends the
  turn. All fluent. All invisible to the metrics normally used.
- **The magnitudes**, which generate hypothesis **S**:

  | | attenuation of gold token | rank the substitute came from, unsteered |
  |---|---:|---:|
  | Qwen2.5-0.5B | 19.9× | 2 |
  | Llama-3.1-8B | 81.9× | 6 |
  | Qwen2.5-3B | 481.9× | 13 |
  | Qwen2.5-7B | 840.2× | 19 |
  | Aya-expanse-8B | 1,525.9× | 26 |
  | Qwen3-4B | **35,001.8×** | **43** |

---

## The two dials

Both reduce how much a span contributes at the position where the answer is
emitted. They differ in **where the lost mass goes**, and that is the point.

- **Dial V — quiet values.** Rescale cached value vectors at the span's
  positions. Attention is untouched: the model looks at full strength, receives
  little, and the lost contribution is *not given to anyone else*.
- **Dial A — less looking.** Rescale the attention the span receives. The
  softmax renormalises, so the lost mass is **redistributed to other positions**
  in the context.

Dial V is the established one. Dial A is new here, is architecture-agnostic
(any attention layer will do), and is what makes the finding independent of any
particular method's future.

---

## Hypotheses

Numeric falsifiers fixed here, before data.

### M — the mechanism replicates

Attenuation-rather-than-overwriting holds across ten models up to 8B. Nothing
establishes it at larger scale in a current family.

- **Predicts:** under dial V, the gold token's probability falls ≥ 10× while its
  **rank stays in single digits**, and no specific competitor is systematically
  promoted.
- **Falsified if:** the steered distribution promotes a particular wrong token —
  the substitute climbs rather than the gold falling.
- **Runs first.** Everything else assumes it. If it fails, that is the result.

### D — the dissociation *(the central one)*

Does *how* the evidence is weakened change what the model does about it?

- **Predicts, if the phenomenon is about lost evidence per se:** dials V and A
  produce the same failure signature — near-neighbour confabulation at matched
  contribution loss, with retention curves within **10 points** of each other.
- **Predicts, if the phenomenon is about where the mass goes:** dial A promotes
  material from *elsewhere in the context* rather than a near neighbour of the
  target, with near-neighbour share differing by **≥ 20 points** between dials.
- **Both outcomes are informative and neither is the "hoped-for" one.** The first
  says model confidence does not track evidence strength by any route — a broad
  claim. The second says near-neighbour confabulation is specifically the
  signature of *evidence removed without replacement* — a sharper one.
- **Falsified as stated if** the two dials cannot be matched on contribution
  loss at all, in which case the comparison is reported as unresolvable rather
  than resolved.

### W — the window

- **Predicts a window exists** if some cell has compliance **≥ 20 points** above
  no-manipulation *and* retention **within 5 points** of its no-manipulation
  level.
- **Falsified if** no such cell exists anywhere in the grid.
- **Standing prior:** in earlier work on this lab's Gemma configuration the
  behaviour window and the fluency window did not overlap, and the two
  thresholds proved to be a single **step** rather than two curves. Expectation
  here is therefore **no window**. A window is a positive result and should be
  reported loudly.

### S — scale makes it worse

The table above trends the wrong way: newer and stronger models bury the fact
harder and reach further down for a replacement.

- **Predicts:** median attenuation factor **≥ 1,000×** and median substitute rank
  **≥ 20** on the larger models of the ladder.
- **Falsified if** attenuation ≤ 100× or substitute rank ≤ 6 — i.e. behaving like
  Llama-3.1-8B or better.
- **The confound, stated plainly:** the ten existing points mix family, size and
  recency; six are one family; and the trend was *noticed*, not tested. The new
  runs are a **within-family** ladder specifically to separate size from recipe.
- **"Model quality" is fixed in advance** as published MMLU from each model's own
  release material, recorded before any run.

### N — the substitute is related to the target

Established at ~40× over a null on the existing set. Replication, not discovery.

- **Predicts:** scoring wrong answers against a gold value from a *different*
  question gives a near-neighbour rate ≥ 10× lower than against their own.
- **Falsified if** the two rates are within 3×.

---

## Definitions, fixed now

- **Attenuation factor** — p(gold) unsteered ÷ p(gold) manipulated, read at the
  teacher-forced position where the answer value is emitted, byte-identical
  position in both conditions (`examples/why_near.py`).
- **Substitute rank** — rank of the actually-emitted token in the **unmanipulated**
  distribution.
- **Contribution loss** — the quantity dials V and A are matched on: the drop in
  the span's summed contribution to the read position. Matching is done
  empirically per model, and the matching procedure is fixed before any
  comparison is made.
- **Compliance** — the existing hierarchy metric in this repo, unchanged.
- **Retention** — EXACT / NEAR / ABSENT by the [scoring rule](#scoring-rule-written-before-any-generation-is-read),
  by hand, every answer in every reported cell, never a sample.
- **Grid** — dial V: γ+ ∈ {2.5, 4, 6} × γ− ∈ {0, 0.65, 0.75, 0.9, 0.95}, where
  γ− = 0 is the **boost-only** arm (earlier work found the boost, the half
  usually called harmless, is what turns a refusal into an invented value).
  Dial A: multipliers chosen to span the same contribution-loss range.
- **Separable vs entangled** — separable: the degraded span is its own message
  and can be excised leaving the fact intact. Entangled: instruction and fact
  share a message. Both run; deletion is only informative against both.

---

## Design

### Models

The existing ten-model ladder (0.5B–8B) is **prior data and is not re-run**.
New runs form a **within-family Qwen3 ladder**, chosen because it is the largest
current family whose architecture can host both dials at every layer:

| | status |
|---|---|
| Qwen3-1.7B | new, optional |
| Qwen3-4B | **existing data** (35,001.8×) |
| Qwen3-8B | new |
| Qwen3-14B | new — the size that gives S its reach |
| Qwen3-32B | gated stretch |

### The architecture gate — already run, and it is a result

Both dials need per-position addressable state in the attention layers. Checked
against published configs on 2026-08-11:

| family | verdict |
|---|---|
| Qwen3 (`model_type: qwen3`) | **passes** — plain GQA, full attention, no sliding window |
| Qwen3.5 / Qwen3.6 (`model_type: qwen3_5`) | **3 of every 4 layers are `linear_attention`** (`full_attention_interval: 4`): 4B has 8 full-attention layers of 32; 27B has 16 of 64 |
| Gemma 3 / 4 | interleaved sliding window |
| deepseek v4 flash | MLA, compressed shared latent KV |

**This belongs in the write-up as a finding, not a footnote.** Value-cache
interventions assume per-position value vectors at every layer, and current open
models are being built the other way. Dial A survives this — attention scaling
works on whatever attention layers exist — which is a further reason the finding
must be stated about models rather than about a method.

**Gated modern-architecture arm:** dial A on Qwen3.5-27B's 16 full-attention
layers, run only if everything below is complete. Coverage is reported honestly.

### Arms

no manipulation · dial V (grid) · dial A (matched grid) · delete the message ·
rewrite the message · prompt-only reminder.

**N ≥ 36 per cell.** Greedy, one seed. Anything below is labelled a pilot.

### Fixture

The existing six facts and seven constraint families, unchanged, so new numbers
sit on the same axis as the old ones.

---

## Controls

- **Fact-absent (swap).** Same transcript shape, fact never present — so that
  attenuating evidence can be distinguished from removing it. If the model
  abstains when the fact was never there and confabulates when it is merely
  faint, those are different things.
- **Different-question gold null**, for N.
- **Boost-only arm**, because the suspicious half is not the one under suspicion.
- **Frequency matching**, common vs rare targets paired — a common target
  survives further into the manipulation on most models.
- **Precision control.** If any model runs quantized, one cell is repeated at two
  precisions. A conclusion that moves with quantization is scoped or dropped.

---

## Sanity checks committed in advance

Reported whether or not convenient.

1. Reproduce the mechanism on **one prior model** first — the instrument must
   still give the old answer.
2. Verify the teacher-forced read position is byte-identical across conditions.
3. Confirm each manipulation fires at all on each new model before interpreting
   any silence from it.
4. Read ≥ 20 randomly drawn generations per model by hand before any number goes
   into prose. Randomly drawn — not chosen, not the first twenty.
5. Recompute at least one headline number with a fresh independent script.
6. Any automatic scorer must first reproduce hand labels on the existing
   calibration set, and is refused otherwise.

---

## Scoring rule, written before any generation is read

This rule failed twice on its own terms in earlier work — once because the line
moved between conditions, once because a block was read instead of a sample
(`results/dose_curve/SCORING_RULE.md`). Written first, applied identically to
every arm and both dials.

- **EXACT** — the payload value appears, correct, in a form a downstream consumer
  could use.
- **NEAR** — a value appears in the payload's slot, is wrong, and is
  systematically related to the target: truncation, dropped or transposed
  character, same-shape neighbour, unit or format shift. Judged on the string,
  not on plausibility.
- **ABSENT** — no usable value: refusal, omission, capability disclaimer, or a
  value unrelated to the payload.

Rules of application:

- A value the answer simultaneously denies having been told is **ABSENT**, not
  EXACT. It has lost its standing as a source, which is exactly what a downstream
  consumer loses.
- Format damage is a separate axis, not a retention failure.
- Two kinds of decline are recorded separately: *says it was not told* versus a
  *canned capability line*. The second fires on the kind of question, not on the
  transcript, and the mix is model-dependent so it does not cancel.
- Verdicts and reasons to `results/attenuation_handscored.json`.

---

## Stop rules

- **M fails** — the mechanism does not replicate. That is the finding. Report it
  and stop; do not hunt for a different mechanism to rescue the story.
- **A manipulation does not fire** on a model — report as a compatibility result;
  do not tune until something happens.
- **The dials cannot be matched** on contribution loss — report D as
  unresolvable, do not substitute an unmatched comparison.
- **Quantization moves the answer** — scope every claim to precision, or drop it.
- **Time** — if the gate, M, and one full dial-V grid are not done by end of day
  two of the clock, drop dial A's grid to a single matched dose and drop the
  optional models.

---

## What will not be claimed

- Anything about production behaviour. All cases are constructed.
- Anything about models not run, including closed models.
- That this transfers to KV eviction, cache compression, quantization or
  RAG re-ranking. Those are the **motivation** for caring and are named as such;
  nothing here measures them.
- That any published method is bad. This measures an axis its authors did not,
  on constructed cases, with an independent reimplementation.

---

## Stretch goals, gated

Only if M, D, W, S and N are complete and scored. Listed so they cannot be
reached for as a rescue.

- **S1 — modern architecture.** Dial A on Qwen3.5-27B's full-attention layers.
- **S2 — language.** Whether the damage differs in Czech, where representations
  are weaker.
- **S3 — a selective manipulation.** Degrade only the positions carrying the
  instruction, leaving the fact's positions untouched.

---

## Hours

Prior work — the ten-model ladder, the mechanism, the mode rules, the fixture
set, `brainscope`, `vsteer.py` — predates 2026-08-11 and is not counted. Only
work on this question after the freeze is.

| date | hours | what |
|---|---|---|
| | | |

---

## Deviations

Empty is a claim. It stays empty only while it is true.

| date | what changed | why |
|---|---|---|
| 2026-08-11 | Framing moved from method-first ("what does this edit cost?") to model-first ("does the model know when its evidence goes quiet?"). Second manipulation (dial A, attention) added; hypothesis **D** added as the central one. | The finding is about how model confidence responds to degraded evidence. Tying it to one technique made it hostage to that technique's future. Two dials make it a property of the model. Before any data. |
| 2026-08-11 | Models changed from "Qwen 3.5 4B / 27B dense" to a within-family **Qwen3** ladder (1.7B / 4B existing / 8B / 14B, 32B gated). | Architecture gate, run before any data: every Qwen 3.5 and 3.6 model is `model_type: qwen3_5` with `full_attention_interval: 4` — three `linear_attention` layers per full-attention layer, so 75% of the network has no per-position value cache. "Dense" in the model card means dense-vs-MoE, not dense attention. Recorded as a result in [Design](#the-architecture-gate--already-run-and-it-is-a-result), not hidden. |

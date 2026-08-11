# Preregistration — does the fact survive being turned down?

**Frozen 2026-08-11, before any new data was collected.**
Changes after that date go in [Deviations](#deviations), dated, with the reason.
Nothing here may be edited to match a result.

---

## The question, plainly

You can't always delete a message. Sometimes the sentence you want the model to
stop obeying sits in the same message as a fact you still need — a rule and an
order number in one line, an injected instruction inside a document you asked
for. So instead of removing it, you turn it **down**.

This asks what that costs.

> **When part of the context is turned down instead of removed, does the
> information inside it survive? Is there a dose where the instruction stops
> being obeyed and the fact still comes out? And does that get better or worse
> as models get better?**

---

## What is already established, and is prior work

All of the below is in this repo, was done before 2026-08-11, and is **not
counted against the time budget**. It is the reason this question is askable at
all, and it is also the reason the prediction below is specific rather than
vague.

- **The mechanism: attenuated, not overwritten.** Ten models (0.5B–8B, six
  families plus a same-family ladder), 756 generations per model. Scaling cached
  V does not touch attention — the model still looks at the demoted span at full
  strength, and what arrives is faint. The correct token's probability collapses
  and *nothing promotes a wrong one*; whatever was already standing behind it
  wins. `results/failure_modes.md`, `examples/why_near.py`.
- **The five failure modes** it produces: near neighbour, states-the-fact-while-
  denying-it, non-terminating recall, format broken and filled with
  confabulation, ends the turn. All fluent, all invisible to the metrics
  normally used.
- **The magnitudes**, which are what generate hypothesis **S** below:

  | | attenuation of gold token | rank the substitute came from, unsteered |
  |---|---:|---:|
  | Qwen2.5-0.5B | 19.9× | 2 |
  | OLMo-2-7B | 22.0× | 4 |
  | Command-R7B | 30.5× | 5 |
  | Phi-3.5-mini | 37.4× | 4 |
  | Qwen2.5-1.5B | 44.0× | 4 |
  | Llama-3.1-8B | 81.9× | 6 |
  | Qwen2.5-3B | 481.9× | 13 |
  | Qwen2.5-7B | 840.2× | 19 |
  | Aya-expanse-8B | 1,525.9× | 26 |
  | Qwen3-4B | **35,001.8×** | **43** |

- **What the source paper measures, and what it does not.**
  V-Steer ([Zeng, Lee, Zhao & Hockenmaier, COLM 2026](https://arxiv.org/abs/2607.26228))
  reports primary-constraint accuracy, **general-capability retention** (Tab. 6:
  MMLU / IFEval / BBH), an **aligned-constraint no-op check** (Tab. 7), and
  **generation collapse** as most-frequent-5-gram repetition (Tab. 3, 12).
  MMLU, IFEval and BBH contain no demoted span, so there is nothing in them for
  the suppression to damage; Tab. 7 tests agreement, not survival; 5-gram
  collapse cannot see a fluent near neighbour. **Whether information inside the
  suppressed span is still recoverable is not measured anywhere in the paper.**
  Checked against the full text, 2026-08-11.

---

## Hypotheses

Numeric falsifiers are fixed here, before data.

### M — the mechanism replicates on a current model

Attenuation rather than overwriting is universal so far across ten models.
Nothing establishes that it holds on a model two generations newer.

- **Predicts:** on the new model, under the edit, the gold token's probability
  falls by ≥ 10× while its **rank stays in single digits**, and no specific
  competitor is promoted (the substitute's steered rank is not systematically
  raised relative to unsteered).
- **Falsified if:** the steered distribution promotes a particular wrong token —
  i.e. the substitute climbs rather than the gold falling.
- **Why it comes first:** every other hypothesis assumes this. If it fails,
  that is the result and the rest of the design is void.

### W — the window

- **Predicts a window exists** if there is a cell in the γ grid where hierarchy
  compliance rises **≥ 20 points** over no-edit *and* payload retention is
  **within 5 points** of its no-edit level.
- **Falsified if:** no such cell exists anywhere in the grid.
- **Standing prior:** in earlier work on this lab's Gemma configuration, the
  behaviour window and the fluency window did not overlap, and the two
  thresholds turned out to be a single **step** rather than two curves. The
  expectation here is therefore **no window**. A window would be a positive
  result for the method and should be reported as such, loudly.

### S — scale makes it worse, not better

The table above trends the wrong way: newer and stronger models bury the fact
harder and reach further down the distribution for a replacement. Qwen3-4B, the
newest model in the set, is three orders of magnitude worse than the oldest.

- **Predicts:** the new modern model shows a median attenuation factor
  **≥ 1,000×** and a median substitute rank **≥ 20**.
- **Falsified if:** attenuation ≤ 100× or substitute rank ≤ 6 — i.e. it behaves
  like Llama-3.1-8B or better.
- **The confound, stated plainly:** the ten existing points confound family,
  parameter count and recency, six of them are one family, and the trend was
  noticed rather than tested. That is precisely why it is preregistered here.
  The new runs add a **within-family modern ladder** so that recency and size
  can be separated from training recipe.
- **"Model quality" is fixed in advance** as published MMLU score from each
  model's own release material, recorded before any run. Not judged after the
  fact, not by impression.

### N — the substitution is related to the target

Established at ~40× over a null on the existing set; this is replication, not
discovery.

- **Predicts:** scoring wrong answers against a gold value from a *different*
  question yields a near-neighbour rate ≥ 10× lower than against their own.
- **Falsified if:** the two rates are within 3×.

---

## Definitions, fixed now

- **Attenuation factor** — p(gold token) unsteered ÷ p(gold token) steered, read
  at the teacher-forced position where the answer value is emitted, identical
  position in both conditions (`examples/why_near.py`).
- **Substitute rank** — the rank of the actually-emitted token in the
  **unsteered** distribution.
- **Compliance** — the existing hierarchy metric in this repo, unchanged.
- **Retention** — EXACT / NEAR / ABSENT by the rule in
  [Scoring rule](#scoring-rule-written-before-any-generation-is-read), scored by
  hand, every answer in every reported cell, never a sample.
- **γ grid** — γ+ ∈ {2.5, 4, 6} × γ− ∈ {0, 0.65, 0.75, 0.9, 0.95}. The source
  paper's defaults (γ+ = 2.5, γ− = 0.75) are always included. γ− = 0 is the
  **γ+-only** arm, because earlier work found γ+ alone — the half usually
  described as harmless — is what converts a refusal into an invented value.
- **Separable vs entangled** — separable: the demoted instruction is its own
  message and can be excised leaving the fact intact. Entangled: instruction and
  fact share a message. Both are run. Deletion is only informative against both.

---

## Design

**Models.** The existing ten-model ladder (0.5B–8B) is prior data and is **not
re-run**. New runs add the modern end:

- **Qwen 3.5 4B dense** — current, his-default-class, fits locally.
- **Qwen 3.5 27B dense** — the size that gives the scale claim reach. Rented
  GPU; setup and queue time are not project time.

**Architecture gate, run before anything else.** The edit requires one
addressable V per layer and no sliding-window layers. Gemma (sliding window),
MLA models and hybrid Gated-DeltaNet models cannot host it. Each candidate is
checked and the result recorded in a short architecture table rather than
silently dropped — a method that cannot run on current architectures is itself
worth stating.

**Fixture.** The existing six facts and seven constraint families, unchanged, so
the new numbers sit on the same axis as the old ones.

**Arms.** no edit · γ+ only · full edit · delete the message · rewrite the
message · prompt-only reminder.

**N ≥ 36 per cell.** Greedy, one seed. Any cell below that is labelled a pilot.

---

## Controls

- **Fact-absent (swap).** The same transcript shape with the fact never present,
  so that attenuating evidence can be distinguished from removing it. If the
  model abstains when the fact was never there and confabulates when it is
  merely faint, those are different things.
- **Different-question gold null**, for N.
- **γ+-only arm**, because the suspicious half is not the one under suspicion.
- **Frequency matching.** Common versus rare target strings, paired, because a
  common target survives further into the edit on most models.
- **Precision control.** If the 27B runs quantized, one cell is repeated at two
  precisions. A conclusion that moves with quantization is scoped to precision
  or dropped.

---

## Sanity checks committed in advance

Reported whether or not convenient.

1. Reproduce the mechanism on **one prior model** before touching the new ones —
   the instrument must still give the old answer.
2. Verify the teacher-forced read position is byte-identical across conditions.
3. Confirm the edit fires at all on the new model (non-empty head mask on a case
   where the demoted span trivially dominates).
4. Read ≥ 20 randomly drawn generations per model by hand before any number goes
   into prose. Randomly drawn, not chosen, not the first twenty.
5. Recompute at least one headline number with a fresh independent script.
6. Any automatic scorer used must first reproduce hand labels on the existing
   calibration set, and is refused otherwise.

---

## Scoring rule, written before any generation is read

This rule failed twice on its own terms in earlier work — once because the line
moved between conditions, once because a block was read instead of a sample
(`results/dose_curve/SCORING_RULE.md`). It is written first and applied
identically to every arm.

- **EXACT** — the payload value appears, correct, in a form a downstream
  consumer could use.
- **NEAR** — a value appears in the payload's slot, is wrong, and is
  systematically related to the target: truncation, dropped or transposed
  character, same-shape neighbour, unit or format shift. Judged on the string,
  not on plausibility.
- **ABSENT** — no usable value: refusal, omission, capability disclaimer, or a
  value unrelated to the payload.

Rules of application:

- A value that the answer simultaneously denies having been told is **ABSENT**,
  not EXACT. It has lost its standing as a source, which is exactly what a
  downstream consumer loses.
- Format damage is a separate axis and is not scored as a retention failure.
- Two kinds of decline are recorded separately: *it says it was not told*
  versus *a canned capability line*. The second fires on the kind of question,
  not on the transcript, and mixing them is model-dependent so it does not
  cancel.
- Verdicts and reasons to `results/attenuation_handscored.json`.

---

## Stop rules

- **M fails.** The mechanism does not replicate on a current model. That is the
  finding — report it and stop, do not go looking for a different mechanism to
  rescue the story.
- **The instrument does not fire** on the new model (empty head mask). Report as
  an architecture/compatibility result; do not tune until something happens.
- **Quantization moves the answer.** Scope every claim to precision, or drop the
  claim.
- **Time.** If the architecture gate and the mechanism replication are not done
  by end of day two, drop the 27B and run the whole design on the 4B alone.

---

## What will not be claimed

- Anything about production behaviour. All cases here are constructed.
- Anything about models not run, including closed models.
- That this generalises to KV eviction, context compression, attention masking
  or RAG re-ranking. Those are the **motivation** for caring, and are named as
  such; nothing here measures them.
- That V-Steer is a bad method. This measures one axis its authors did not, on
  constructed cases, with an independent reimplementation.

---

## Stretch goals, gated

Run only if M, W, S and N are complete and scored. Listed so they cannot be
reached for as a rescue.

- **S1 — language.** Whether retention damage differs in Czech, where the
  representations are weaker.
- **S2 — a selective edit.** Demote only the positions carrying the imperative,
  leaving the fact's positions untouched. The obvious repair if W is falsified.

---

## Hours

Prior work — the ten-model ladder, the mechanism, the mode rules, the fixture
set, `brainscope`, `vsteer.py` — was done before 2026-08-11 and is not counted.
Only work on this question after the freeze is.

| date | hours | what |
|---|---|---|
| | | |

---

## Deviations

Empty is a claim. It stays empty only while it is true.

| date | what changed | why |
|---|---|---|
| | | |

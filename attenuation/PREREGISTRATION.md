# Preregistration — what's behind a fact?

**Frozen 2026-08-12, before any data.** Earlier versions of this plan are in the
git history; changes after the first generation go in a dated line at the bottom.

---

## The question

A model is told something. Turn that sentence down — don't delete it, just make
it harder to see — and ask about it. The model does not say it doesn't know. It
says something *next to* the answer: `302` → `02`, `Bagr` → `Bagel`.

> **What does it say instead, and where does that come from?**

Turning a fact down is a way of **seeing what the model has stacked behind it**.
The knob is an instrument, not the subject.

---

## The knob

Add a negative bias `b` to the attention logits at the fact's token positions
(4D float mask, `attn_implementation="eager"`).

- `b = 0` — normal model
- `b ↑` — the model can see the fact less and less

No cache surgery, no hooks, no per-layer code. Works on whatever attention
layers the model has; on hybrid models the linear-attention layers get nothing
and the coverage is reported.

---

## The measurement

Teacher-force to the position where the answer value is emitted. Read the **full
next-token distribution** at `b = 0` and at `b > 0`. **Nothing is generated.**

Per item: the gold token's probability at both settings; the argmax under the
knob; that token's **rank in the unmanipulated distribution**; and the rank
correlation of everything else.

---

## Hypotheses

### H1 — does the queue advance, or does it reshuffle? *(the headline)*

Spearman ρ between the two rankings, gold excluded, over the **top K = 100**
(sensitivity reported at K = 50 and 500).

- **ρ ≥ 0.9** → the model chooses nothing. The fact sinks and whoever was next
  steps up. Which would mean the near-miss is sitting behind the right answer
  *all the time*, and the knob only exposes it.
- **Falsified if ρ < 0.7** → something actively reshuffles, and that is a
  different and better story.

### H2 — how deep does the replacement come from?

Existing data: the emitted token sat at rank 2 on Qwen2.5-0.5B and rank **43** on
Qwen3-4B. If H1 holds, that is strange — a plain queue should hand it to rank 2.

- **Predicts:** median source rank grows with model size across the three models.
- **Falsified if** all three sit within a factor of 3 of each other.
- **Kill the cheap explanation first (see below).**

### H3 — is the replacement related to the target?

- **Predicts:** scoring against a gold value from a *different* question gives a
  relatedness rate ≥ 10× lower than against its own. (~40× on the existing set.)
- **Falsified if** the two rates are within 3×.

### H4 — is the knob a real model organism of confabulation?

Compare what the knob produces against what the model says when the fact was
**never in the conversation at all** (the existing fact-absent condition).

- **Predicts:** the two sets of substitutes overlap well above chance.
- **If they do** — the knob is a validated, controllable way to make a model
  confabulate on demand.
- **If they don't** — the knob makes a *different* failure, which is also a
  result, and H1–H3 are then about that failure and must say so.

---

## The confound that would kill H2, and the fix

Bigger models have sharper distributions to begin with, so "the replacement came
from further down" could just mean "this model was more confident", not "this
model is hurt more".

**Fix, and it runs first:** match models on **effect size** — the same KL between
manipulated and unmanipulated distributions — not on the same `b`. Same trick as
the damage-matching in `steering-mechanics`. If H2 survives matched KL, it
stands; if it doesn't, it was an artifact and that goes in the write-up.

---

## Baselines and controls

- **Fact deleted** entirely, rather than turned down.
- **Fact never present** (the swap condition) — also H4's comparison set.
- **A random other span** turned down instead, matched on KL.
- **Different-question null**, for H3.

---

## Fixed in advance

- **K = 100** for the rank correlation.
- **Related** = truncation, dropped or transposed character, same-shape
  neighbour, or a unit/format shift. Judged on the string, not on plausibility.
- The read position, greedy decoding, one seed.
- **Items where the unmanipulated model gets the answer wrong are dropped**, and
  the number dropped is reported. Declared here so it cannot become a filter
  chosen after the fact.

---

## Models

| | why |
|---|---|
| **Qwen3.5-4B** | current model; 8 of 32 layers are full attention, coverage reported |
| **Qwen3-4B** | the rank-43 end; already measured |
| **Qwen2.5-0.5B** | the rank-2 end; runs on CPU |

If the knob does not land on Qwen3.5-4B, fall back to Qwen3-8B. **Change the
model, not the question.**

---

## Stop rules

- The knob doesn't remove the fact at any `b` → wrong model, not a finding.
- The unmanipulated model can't do the task → wrong task for that model; drop it.
- H2 dies under KL matching → report that, don't go looking for another statistic.

---

## Not claimed

Constructed cases. One manipulation family. Three models. Nothing here measures
production behaviour, natural hallucination in the wild, or any published
method's quality.

---

## Hours

12 + 2 for the write-up. Prior work — the fixture set, the ten-model ladder, the
mechanism, `brainscope` — predates this and is not counted.

| date | h | what |
|---|---|---|
| | | |

---

## Changes after the first generation

*(none yet — earlier versions of the plan are in git history)*

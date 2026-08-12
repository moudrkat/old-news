# What's behind a fact?

**Frozen 2026-08-12, before any data. Updated the same day as results came in —
what was frozen is kept, what changed is dated at the bottom.** Earlier versions
are in git history.

---

## The question

> **A model is told something. Make that sentence hard to read — don't delete
> it. Does the model notice?**

Models have internal representations of whether they recognise an entity, and
those representations causally gate refusal ([Ferrando, Obeso, Rajamanoharan &
Nanda, ICLR 2025 oral](https://arxiv.org/abs/2411.14257)). That is
self-knowledge about knowledge from *weights*. This asks the same about
knowledge from *context*: if the evidence for something it was told is degraded,
does anything fire?

Prediction from their result: degraded context should behave like an unknown
entity and trigger refusal.

---

## The knob

A negative bias `b` on the attention logits at the fact's token positions, via a
4D additive mask, `attn_implementation="eager"`. `b = 0` is the plain causal
mask, so the control is not a separate code path. Nothing enters the residual
stream, no cache is edited, no hooks. Runs on any attention layer — on hybrid
models only the full-attention ones, and coverage is reported.

## The four conditions

| | |
|---|---|
| `present` | the fact is in the context, `b = 0` |
| `faint` | the fact is in the context, `b` at the smallest value where the answer no longer contains it |
| `swap` | a **different kind** of fact fills the same slot — a readable sentence that does not contain the answer |
| `drop` | no such sentence at all |

`swap` is the control that decides everything: it separates *"I have this
fact"* from *"there is a sentence here"*.

---

## The metric

Three layers, in this order. Layer 3 is only defined where layer 2 says a value
was given — a refusal has no distance from the truth.

**1. Does it claim it was told?** — the answer to *"Did I tell you X? Answer
only yes or no."*, classified yes / no / other. Rates across the four
conditions. **This is the headline.**

**2. Does it give a value, or refuse?** — rate across the same four conditions.

**3. If a value was given, how far is it from the truth?** Two measures, because
one of them alone would kill the most interesting case:

- **string distance** — Levenshtein over the two strings, divided by the longer,
  case-folded. `4417→417` = 0.25, `19:40→19:45` = 0.20, `19:40→12:00` = 0.40.
  Catches truncations and single-character slips.
- **category** — is it the same kind of thing, from the same domain?
  `Brno → Prague` is a Czech city; `Brno → New York` is not. This is a
  judgement, made by hand; a judge may be used only after reproducing hand
  labels on a calibration set.

**Why both.** String distance scores `Brno → Prague` at **0.83**, i.e. far —
and that is the most informative cell in the data, because with the fact faint
the model still answers with a Czech city. A metric that calls it far is
measuring the wrong thing on its own.

---

## The answer

Two models, 100 items each. Rate of answering "yes" to *"Did I tell you X?"*:

| condition | Qwen3.5-4B | Qwen3-4B-Instruct-2507 |
|---|---|---|
| `present` | 85 / 89 (96%) | 99 / 100 (99%) |
| **`faint`** | **78 / 89 (88%)** | **73 / 100 (73%)** |
| **`swap`** | **0 / 89 (0%)** | **0 / 100 (0%)** |
| `drop` | 0 / 89 | 0 / 100 |

Gate: Qwen3.5-4B lost 11 items because no `b` in the ladder removed the value;
Qwen3-4B lost none. Neither model answered wrong unmanipulated on any item.

- **151 of 189 — the model gives a wrong value and claims it was told it.**
- **`swap` → "no", 189 of 189.** A readable sentence about something else never
  produces a "yes". So "yes" tracks the fact, not the presence of a clause.
- On Qwen3.5-4B the `drop` condition answers in prose rather than yes/no and so
  classifies as *other*; on Qwen3-4B it answers a clean "no" 100/100. Either
  way `swap` is the control the claim rests on.

### What the model says instead

Two kinds of near miss, and they are different things:

| | faint | absent |
|---|---|---|
| `Bagr` | `Bag`, `Bragg` | `Buddy`, `Fido`, `Rex` |
| `4417` | `417` | `1234` |
| `E-88` | `E-8`, `E8` | `404` |
| `Brno` | **`Prague`** | `New York City` |
| `19:40` | **`19:45`** | `14:30` |
| `302` | `30`, `3`, `2` | `42`, `1234` |

`19:40 → 19:45` passes every downstream check anyone runs. It does not look
like a hallucination. It looks like a typo.

---

## Hypotheses, as frozen

**H1 — the queue** Spearman ρ over the top K = 100 of the unmanipulated
distribution, gold excluded. ρ ≥ 0.9 → the model chooses nothing, the fact sinks
and the next in line steps up. **Not supported**: ρ ran 0.09–0.82 and falls with
`b`, so it can only be compared across models at equal KL. Unresolved.

**H2 — depth of the replacement** Median source rank grows with model size.
**Holds in direction** (rank 2–5 on Qwen2.5-0.5B against 145–824 on Qwen3-4B)
but the cheap explanation — bigger models start more confident, so everything
else sits further down — is not yet excluded. Unresolved.

**H3 — the replacement is related to the target** ≥ 10× over a
different-question null. **Pending.**

**H4 — does the model know the difference between faint and absent?**
Originally read as falsified. **That reading was an artifact of the design**
(see below) and it is now **supported**: absent is declined, faint is not.

**H5 — post-hoc, from the pilot** Faint yields a distortion of the truth,
absent yields a generic prior or a refusal. Needs confirmation on items that did
not generate it.

---

## Controls and gates

- **`swap`** — the one that matters. Passed, 189/189 across both models.
- **Gate**: an item counts only if the unmanipulated model answers correctly and
  some `b` removes the value. Both failure kinds counted and reported.
- **Excluded model**: Qwen2.5-0.5B answers "yes, you told me" for 3 of 5 items
  where the fact was never present. It cannot do the provenance task and is
  excluded from every claim rather than averaged in.
- **Layer-0 null** for the probe: the embedding layer holds no state; if it
  separates, the probe is reading tokens.
- **Shuffled labels** for the probe.

---

## What went wrong, and how it was caught

**The forced prefix.** The first design pinned the read position with an answer
prefix ("Your dog is called ___"), which makes *"I don't know"* a grammatically
impossible continuation. A forced completion was being read as the model's
choice. Removing it reversed the result: the model that appeared never to admit
ignorance admits it reliably when the fact is genuinely absent.

**The gold token.** For numeric values the first token of `" 4417"` is a bare
space, so "probability of the correct token" was measuring whether a space comes
next — 0.85 whatever the knob did.

**The first probe.** It built `absent` by deleting the sentence, so the two
classes differed in their text. It scored 6/6 at layer 0 — the embedding layer,
which holds no state. It was reading tokens. Fixed with the matched `swap`
condition.

---

## Not claimed

Constructed conversations. One manipulation family. Two models after exclusion,
both 4B. Greedy, one seed. `faint` is a per-item threshold, so it means a
different `b` for each item. The knob is an idealised version of a state that
occurs in deployment for other reasons — KV cache compression and eviction, KV
quantisation, long-context dilution, prompt compression. **None of those is
measured here.**

---

## Hours

12 + 2 for the write-up. Prior work — the fixture, the ten-model ladder, the
mechanism, `brainscope` — predates this and is not counted.

| date | h | what |
|---|---|---|
| 2026-08-12 | | knob, ladder, absent control, provenance question, probe |

## Changes

| date | what | why |
|---|---|---|
| 2026-08-12 | H4 marked falsified, then un-falsified | the first falsification was an artifact of the forced answer prefix |
| 2026-08-12 | metric split into three layers | distance from the truth is undefined on a refusal, and the two are different questions |
| 2026-08-12 | two distance measures instead of one | string distance calls `Brno → Prague` far, which is the wrong answer about the most informative cell |

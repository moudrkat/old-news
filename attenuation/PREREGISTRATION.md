# What's behind a fact?

Frozen 2026-08-12, before any data. Updated the same day as results came in. What was frozen is kept, what changed is dated at the bottom. Earlier versions
are in git history.

---

## The question

> **A model is told something. Make that sentence hard to read, don't delete
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

## The bias

This is a graded version of a standard tool. Blocking attention from chosen
positions to test what depends on them is *attention knockout*
([Geva, Bastings, Filippova & Globerson, 2023](https://arxiv.org/abs/2304.14767)),
where it is binary, the edge is cut, and the question it answers is *where
does information flow*. Here it is dosed rather than cut, and the question is
different: not where the fact travels, but whether the model registers that it
can no longer read it. The same lineage runs through *Do I Know This Entity?*,
which finds that the model's own unknown-entity directions work by suppressing
the attention of the attribute-extraction heads Geva et al. identified, so the
manipulation below imposes from the outside the kind of change that mechanism
produces from the inside.


A negative bias `b` on the attention logits at the fact's token positions, via a
4D additive mask, `attn_implementation="eager"`. `b = 0` is the plain causal
mask, so the control is not a separate code path. Nothing enters the residual
stream, no cache is edited, no hooks. Runs on any attention layer, on hybrid
models only the full-attention ones, and coverage is reported.

## The four conditions

| | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| `present` | 98 / 99 | 81 / 85 |
| `faint` | 72 / 99 | 75 / 85 |
| `swap` | 0 / 99 | 0 / 85 |
| `drop` | 0 / 99 | 0 / 85 |

`swap` is the control that decides everything: it separates *"I have this
fact"* from *"there is a sentence here"*.

---

## The metric

Three layers, in this order. Layer 3 is only defined where layer 2 says a value
was given. A refusal has no distance from the truth.

**1. Does it claim it was told?**. The answer to *"Did I tell you X? Answer
only yes or no."*, classified yes / no / other. Rates across the four
conditions. This is the headline.

**2. Does it give a value, or refuse?**, rate across the same four conditions.

3. If a value was given, how far is it from the truth? Two measures, because
one of them alone would kill the most interesting case:

- string distance: Levenshtein over the two strings, divided by the longer,
  case-folded. `4417→417` = 0.25, `19:40→19:45` = 0.20, `19:40→12:00` = 0.40.
  Catches truncations and single-character slips.
- category. Is it the same kind of thing, from the same domain?
  `Utrecht → Amsterdam` is a Dutch city; `Utrecht → New York` is not. This is a
  judgement, made by hand; a judge may be used only after reproducing hand
  labels on a calibration set.

Why both. String distance scores `Brno → Prague` at **0.83**, i.e. far. But the
substitution keeps the country, which is the informative part, and a metric that
calls that far is measuring the wrong thing on its own.

*(That example is from the ten-item pilot, `results/told_Qwen3.5-4B.json`, at
b = 11. It is what prompted the second measure and is recorded here for that
reason. It does not reproduce in the 100-item run, where `city:Brno` never
loses its value on Qwen3.5-4B at any dose tested, see the non-monotonicity note
in the README. The same shape does appear there: `Utrecht → Amsterdam`,
`Graz → Linz`, `Rijeka → Ljubljana`.)*

---

## The answer

Two models, 100 items each. Rate of answering "yes" to *"Did I tell you X?"*:

| condition | Qwen3.5-4B | Qwen3-4B-Instruct-2507 |
|---|---|---|
| `present` | 98 / 99 | 81 / 85 |
| **`faint`** | **75 / 85 (87%)** | **70 / 97 (72%)** |
| **`swap`** | **0 / 85 (0%)** | **0 / 97 (0%)** |
| `drop` | 0 / 99 | 0 / 85 |

Gate: Qwen3.5-4B lost 11 items because no `b` in the sweep removed the value;
Qwen3-4B lost 1. Neither model answered wrong unmanipulated on any item.

- 147 of 184. The model claims it was told the fact when it can no longer read
  it. 124 of those give a wrong value; the other 23 give no value at all and claim
  it anyway.
- **`swap` → "no", 184 of 184.** A readable sentence about something else never
  produces a "yes". So "yes" tracks the fact, not the presence of a clause.
- On Qwen3.5-4B the `drop` condition answers in prose rather than yes/no and so
  classifies as *other*; on Qwen3-4B it answers a clean "no" 100/100. Either
  way `swap` is the control the claim rests on.

### What the model says instead

Two kinds of near miss, and they are different things:

From the ten-item pilot, both models, faint against never-told. The
100-item run does not ask for the value under `drop`, so the right-hand column
has no equivalent there; the rows that do reproduce in the main run are `Bagr →
Bag`, `4417 → 417`, `E-88 → E-8` and `19:40 → 19:45`.

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

## What was actually pre-specified, and what wasn't

This matters more than the hypothesis list, so it is stated plainly.

Fixed before any data, and never adjusted to fit a result: the four
conditions · the gate (what counts as a usable item, and both ways an item can
fail) · the scoring rule · the `swap` control · the layer-0 null for the
probe · K = 100 · the rule that a model failing its own control is excluded
rather than averaged in.

That is the part the result rests on.

Written down before the data, but only became the headline once it worked:
H4. It was in the file, it was tested, and after the prefix bug was fixed it
came out supported. That is not the same as having predicted it, and it is not
written up as if it were.

Post-hoc, read off the pilot: H5.

Unresolved and left that way: H1, H2, H3.

Provenance of the hypothesis list itself. The candidate hypotheses were
drafted by a coding agent; the question they were narrowed to, *what does the
model say instead of the right answer, and why that one*, was chosen by the
author, after the first draft. Two of the three design errors below were caught
by the author, not the agent.

---

## Hypotheses, as frozen

**H1. The queue** Spearman ρ over the top K = 100 of the unmanipulated
distribution, gold excluded. ρ ≥ 0.9 → the model chooses nothing, the fact sinks
and the next in line steps up. Not supported: ρ ran 0.09–0.82 and falls with
`b`, so it can only be compared across models at equal KL. Unresolved.

**H2. Depth of the replacement** Median source rank grows with model size.
Holds in direction (rank 2–5 on Qwen2.5-0.5B against 145–824 on Qwen3-4B)
but the cheap explanation. Bigger models start more confident, so everything
else sits further down, is not yet excluded. Unresolved.

**H3. The replacement is related to the target** ≥ 10× over a
different-question null. Pending.

H4. Does the model know the difference between faint and absent?
Originally read as falsified. That reading was an artifact of the design
(see below) and it is now supported: absent is declined, faint is not.

**H5. Post-hoc, from the pilot** Faint yields a distortion of the truth,
absent yields a generic prior or a refusal. Needs confirmation on items that did
not generate it.

---

## Controls and gates

- `swap`. The one that matters. Passed on every item of both models.
- Gate: an item counts only if the unmanipulated model answers correctly and
  some `b` removes the value. Both failure kinds counted and reported.
- Excluded model: Qwen2.5-0.5B answers "yes, you told me" for 3 of 5 items
  where the fact was never present. It cannot do the provenance task and is
  excluded from every claim rather than averaged in.
- **Layer-0 null** for the probe: the embedding layer holds no state; if it
  separates, the probe is reading tokens.
- Shuffled labels for the probe.

---

## What went wrong, and how it was caught

The forced prefix. The first design pinned the read position with an answer
prefix ("Your dog is called ___"), which makes *"I don't know"* a grammatically
impossible continuation. A forced completion was being read as the model's
choice. Removing it reversed the result: the model that appeared never to admit
ignorance admits it reliably when the fact is genuinely absent.

The gold token. For numeric values the first token of `" 4417"` is a bare
space, so "probability of the correct token" was measuring whether a space comes
next, 0.85 whatever the bias did.

The first probe. It built `absent` by deleting the sentence, so the two
classes differed in their text. It scored 6/6 at layer 0, the embedding layer,
which holds no state. It was reading tokens. Fixed with the matched `swap`
condition.

---

## Not claimed

Constructed conversations. One manipulation family. Two models after exclusion,
both 4B. Greedy, one seed. `faint` is a per-item threshold, so it means a
different `b` for each item. The bias is an idealised version of a state that
occurs in deployment for other reasons: KV cache compression and eviction, KV
quantisation, long-context dilution, prompt compression. None of those is
measured here.

---

## Hours

Reconstructed by the author from the day, against the real git timestamps for
this directory. It was not timed with a clock as it happened, and that is stated
rather than dressed up; the write-up phase is being timed.

11 Aug: ~2 h. 12 Aug: ~6 h. Total 8 h, plus 2 for the write-up, against the ~16 suggested.

| date | clock | what happened | hours |
|---|---|---|---|
| 11 Aug | 21:15–21:39 | plan frozen, twice. Method-first, then model-first | **~2 h** |
| 12 Aug | 09:28–09:40 | plan cut to one question; bias written; smoke test on three models | |
| 12 Aug | 10:01–10:21 | read the value not the token; fact-absent control | |
| 12 Aug | 13:15–13:27 | forced prefix found and removed; result inverted; provenance question | |
| 12 Aug | 14:26–14:30 | swap control on 100 items, both models | |
| 12 Aug | 14:44–15:04 | figures 0 and 1 | |
| 12 Aug | 15:20–15:41 | hesitation baseline; judge; dose grid | |
| 12 Aug | 15:49–15:51 | read all 189; b = 0 control; six items corrected out | |
| 12 Aug | (the above) | | **~6 h** |
| | | **total** | **8 h** |

Not counted, per his rules: setting up the GPU box, model downloads, waiting for
runs while doing something else, and the answers to the application form.

Prior work, also not counted. It predates this question: the fixture set, the
ten-model V-Steer table, the attenuation mechanism, `brainscope`.

## Changes

| date | what | why |
|---|---|---|
| 2026-08-12 | H4 marked falsified, then un-falsified | the first falsification was an artifact of the forced answer prefix |
| 2026-08-12 | metric split into three layers | distance from the truth is undefined on a refusal, and the two are different questions |
| 2026-08-12 | 6 items removed: the value was never gone | a substring test scored `04:36 → "4:36 PM"` as damage. `src/match.py` normalises leading zeros and 12/24-hour forms first. Headline 151/189 → 147/184 |
| 2026-08-12 | two distance measures instead of one | string distance calls `Brno → Prague` far, which is the wrong answer about the most informative cell |
| 2026-08-14 | scoring rule corrected twice more | the 12/24-hour normaliser matched `06:15` against "6:15 PM", twelve hours out, so two genuinely damaged items had been removed; and it did not fold diacritics, so `Leon → León` was counted as damage. Both found by giving all 189 raw answers to a judge and reading the disagreements (`src/recheck.py`). Headline 145/183 → 147/184 |
| 2026-08-14 | yes/no reading checked by a judge | `gemini-3.1-flash-lite` re-read all 756 stored replies against a written rubric. It agrees with the code on every `present`, `faint` and `swap` answer; all 89 disagreements are in `drop`, where the code says `other` and the judge says `neither` for the same non-answers. The headline does not move. |

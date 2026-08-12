# Preregistration — what's behind a fact?

**Frozen 2026-08-12. Pilot run the same day; results below.**
The plan as frozen is kept intact. What the pilot falsified is marked, not
edited away, and the hypothesis that replaced it is marked **post-hoc** and gets
its own confirmation set. Earlier versions are in git history.

---

## The question

A model is told something. Turn that sentence down — don't delete it, just make
it harder to see — and ask about it. It doesn't say it doesn't know. It says
something *next to* the answer.

> **What does it say instead, and where does that come from?**

---

## The knob

A negative bias `b` added to the attention logits at the fact's token positions
(4D float mask, `attn_implementation="eager"`). `b = 0` is the plain causal
mask, so the control is not a separate code path. No cache surgery. On hybrid
models only the full-attention layers see it; coverage is reported.

## The measurement

The gold path is the **unmanipulated model's own greedy continuation**, gated on
it containing the correct value. Divergence is the first token where the
manipulated continuation leaves that path.

---

## As frozen: H1–H4

**H1 — does the queue advance, or reshuffle?** Spearman ρ over the top K = 100
of the unmanipulated distribution, gold excluded. ρ ≥ 0.9 → the model chooses
nothing and the queue simply advances. Falsified below 0.7.

**H2 — how deep does the replacement come from?** Median source rank grows with
model size. Falsified if all models are within a factor of 3.

**H3 — is the replacement related to the target?** ≥ 10× over a different-question
null. Falsified within 3×.

**H4 — does the model know the difference between faint and absent?** Predicts
the model declines when the fact was never in the conversation and invents a
value when it is merely faint.

---

## Pilot, 2026-08-12 — six items, three models

`Qwen2.5-0.5B-Instruct`, `Qwen3-4B-Instruct-2507`, `Qwen3.5-4B`. Raw output in
`results/`, code in `src/`.

**H4 is FALSIFIED.** The model does not decline when the fact was never there.
It invents a value in both states.

**H1 as stated is not supported.** ρ ran 0.09–0.82, never near 1, and *falls
with b* — so it cannot be compared across models at equal `b`, only at equal KL.

**H2 holds in direction** — source rank 2–5 on the 0.5B against 145–824 on
Qwen3-4B — but the cheap explanation (bigger models start more confident, so
everything else is further down) is not yet excluded. Unresolved.

---

## H5 — what replaced H4 *(post-hoc: this hypothesis came out of the pilot and
cannot be tested on it)*

The two states produce **different kinds of wrong answer**:

| | fact faint | fact never there |
|---|---|---|
| `Bagr` | `Bag`, `Bagr`, `Bragg` | `Buddy`, `Fido`, `Max`, `Rex` |
| `4417` | `417` | `1234`, `123456789` |
| `E-88` | `E-8`, `E8`, `E1000` | `404`, `1000` |
| `Brno` | **`Prague`** | `New York City`, `London` |
| `19:40` | **`19:45`** | `14:30`, `3:45 PM`, `07:00` |
| `302` | `30`, `3`, `2` | `42`, `1234`, `0000` |

> **A faint fact yields a distortion of the truth. A missing fact yields a
> generic prior.**

The model does distinguish the two states — not by declining, but by staying
anchored to what is left of the fact. With `Brno` faint it still answers with a
Czech city; with `Brno` gone it answers New York.

**Predicts:** mean normalised edit distance from the true value is **at most
half** as large under *faint* as under *absent*.
**Falsified if** the two means are within 20% of each other.

### Fixed before the confirmation run

- **Distance** = Levenshtein between the emitted value and the true value,
  divided by the length of the longer. 0 identical, 1 unrelated.
- **The emitted value** is extracted from the generation **by hand**. A judge
  may be used as triage only, and only after reproducing hand labels on a
  calibration set, as in `examples/abstain_judge_gemini.py` (`--min-calib 18`).
- **`absent` = drop**, not swap. Swap is contaminated: on Qwen3.5 the model
  answered with the donor item's value (`order` → `E-88`, `account` → `Bagr`).
  Swap is reported separately as a finding, not used as the control.
- **`faint`** = the smallest `b` at which the true value is no longer in the
  answer.
- **Confirmation set = new items**, not the six that generated H5, and stated as
  such. A hypothesis read off the pilot cannot be confirmed by the pilot.

---

## Baselines and controls

- fact deleted entirely (`drop`) — also H5's comparison arm
- donor sentence in the same slot (`swap`) — reported, not relied on
- a random other span turned down instead, matched on KL
- different-question null, for H3

## Stop rules

- The knob doesn't remove the value at any `b` → wrong model, not a finding.
- The unmanipulated model can't answer → item dropped, count reported.
- H2 dies under KL matching → report it, don't go looking for another statistic.
- H5 fails on the confirmation set → report it. The pilot table stays in the
  write-up as what generated the hypothesis, not as evidence for it.

## Not claimed

Constructed cases. One manipulation. Three models. Nothing here measures
production behaviour, natural hallucination in the wild, or the quality of any
published method.

---

## Hours

12 + 2 for the write-up. Prior work — the fixture set, the ten-model ladder, the
mechanism, `brainscope` — predates this and is not counted.

| date | h | what |
|---|---|---|
| 2026-08-12 | | knob, pilot, fact-absent control |

## Changes

| date | what | why |
|---|---|---|
| 2026-08-12 | H4 marked falsified; H5 added as post-hoc with its own confirmation set | pilot: the model invents in both states, but the *kind* of invention differs |

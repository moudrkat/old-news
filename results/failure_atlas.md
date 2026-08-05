# Where the union rule works, and where it never does

A follow-up to V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228). V-Steer
reports aggregate effectiveness per model; this characterises the failure at
scale instead — across models and constraint types — rather than reporting one
aggregate number.

## What was run

756 cases per model: **6 constraint families × 6 facts × 21 (γ+, γ−) cells**
(γ+ ∈ {1.0, 2.5, 4.0}, γ− ∈ {0, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95}), greedy
decoding throughout. Three models: a small one, a mid-sized one, and Llama.

Each case is a stale/system instruction conflict of the kind Control Illusion
(Geng et al., AAAI 2026, arXiv:2502.15851) uses: a rule stated earlier in the
conversation, contradicted by the system prompt, plus a fact the answer has to
carry. The recorded outcome is **which rule won** — `system`, `stale`, or
`neither` — together with whether the fact was recalled and a set of
failure markers.

Code: `examples/failure_atlas.py` · data: `results/atlas_{small,mid,llama}.json`
· figure: `examples/make_atlas_figure.py` → `results/atlas_families.png`

## The main result: the constraint family predicts the outcome, not the model

How often the system instruction won, out of 126 cases per family per model:

| family | small | mid | Llama |
|---|---|---|---|
| options | 93 % | 75 % | 88 % |
| case | 86 % | 72 % | 11 % |
| length | 100 % | 33 % | 35 % |
| json | 86 % | 0 % | 52 % |
| prefix | 79 % | 17 % | 23 % |
| **bullet** | **0 %** | **0 %** | **4 %** |

Two things fall out of this table.

**`options` is recovered everywhere and `bullet` is recovered nowhere.** Those
two are stable across all three models — 75–93 % and 0–4 %. Whatever the method
does, it does not touch the bullet-list constraint on any model tested.

**Everything between them is model-dependent, and the spread inside one model is
larger than the spread between models.** The mid model runs from 0 % to 75 %
depending only on which constraint is in play. A single per-model effectiveness
number averages over that spread and hides it.

## The failure taxonomy: the fact comes back, the format does not

The `neither` bucket — where no rule won — is 67/756 (small), 128/756 (mid),
121/756 (Llama). Broken down:

| | small | mid | Llama |
|---|---|---|---|
| fact simply absent | 57 % | 23 % | 29 % |
| **recalled, format violated** | 37 % | **55 %** | 40 % |
| self-correction mid-answer | — | 2 % | **27 %** |
| repeats a word | 2 % | 14 % | 3 % |
| near neighbour of the fact | 3 % | 5 % | 1 % |

The largest bucket on the mid model is not a recall failure at all. In **70 of
70** of those cases the needle was recalled correctly — what broke was the
constraint. Almost all of them are `length`: the median answer runs 38 words
where the compliant answers run 17, padded with invented context. One example
answers "your dog is called Bagr" correctly and then adds that the name is
"perhaps inspired by the ancient Persian word for 'to be strong'", which is not
a fact anyone supplied.

So on that model the dominant failure mode is not *forgetting*. It is
**answering correctly at the wrong length, and filling the overrun with
confabulation.** A metric that only checks whether the needle appears scores all
70 of those as successes.

**Self-correction is a Llama-specific mode** — 33 of its 121 `neither` cases,
against 3 on mid and none on small. The model produces an answer and then walks
it back within the same turn.

## Near neighbours

The near-neighbour measure (small edit distance *and* matching shape — `4417-B`
→ `4417-C`, `19:40` → `19:04`) was built expecting it to be a major mode. It is
not: 1–5 % of failures. When these models fail they either omit the fact or
break the format; they rarely produce a plausible wrong one. That is a useful
negative for anyone designing a detector around near-miss matching.

## Caveats, stated plainly

- **The automatic markers are triage, not verdicts.** `empty`, `garbled`,
  `repeats_a_word`, `self_correction` and the neighbour measure are heuristics.
  Every number here that matters should be confirmed by reading the generations.
  In this project three separate automatic scorers have each failed in a way
  that changed a conclusion, so this is not boilerplate.
- **One phrasing per constraint family.** Control Illusion's own result is that
  phrasing moves these outcomes, so per-family numbers may move with a
  reformulation. The *ordering* — options high, bullet zero — is what survives
  a phrasing change, and that is the claim being made.
- **Greedy only.** No sampling variance is measured; repeated runs are identical
  by construction.
- **`which_rule_won` is itself a judgement** made by rule, not by reading. The
  `system`/`stale` split is the least certain column in the table.

## What would make this a paper

1. Hand-read a stratified sample — ~50 cases per model across the families — to
   put an error bar on the automatic labels.
2. A second phrasing per family, to separate "this constraint is hard" from
   "this wording is hard".
3. The batch judge over the full set, so the `neither` bucket is classified by
   reading rather than by heuristic. The judge harness exists
   (`oldnews/evals/judge.py`, taxonomy + calibration, 9 tests green).

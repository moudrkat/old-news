# What a stale instruction costs, and what a value edit buys back

**A replication report on V-Steer** — what reproduced, what did not, and what I
got wrong twice. This is not a paper and is not written as one: there is no
competing baseline for most of it, the operating points are selected post hoc,
and the causal control is one model at n = 36. It is a record of reimplementing
a published method across ten models and reporting everything the reimplementation
turned up, including the parts that went against me.

Numbers are recomputed from stored generations rather than quoted from memory;
the scripts that produce each are named in the sections.

---

## Summary

V-Steer (Zeng et al., COLM 2026) resolves instruction-hierarchy conflicts by
scaling values in the KV cache: the span carrying the current instruction is
multiplied by (1 + γ⁺), the span carrying the stale one by (1 − γ⁻). It reports
how often the current instruction wins. We ask three questions it does not: how
large is the hole being filled, which half of the edit fills it, and what the
failures look like when it does not.

21 126 greedy generations across ten models (0.5B–8B, six families, including a
same-family size ladder), two datasets, seven conditions.

**One result is strong enough to carry a claim:**

**The two halves of the edit do different jobs, and the suppression half trades
recall for compliance.** With the head mask held fixed and γ⁺ matched, dropping
suppression costs compliance (paired McNemar p = 0.002, 0.006, 0.023 on three
models) and restores recall completely — 36/36 against 16/36, 29/36 and 26/36,
with **zero counterexamples on any constraint family or any fact on any model**
(b = 20, 7, 10 against c = 0, 0, 0). On the useful-answer metric (compliance ∧
recall) at matched γ⁺ = 4 the suppression term is net *negative* on one model,
a wash on a second, and positive on the third.

The original work cannot see this trade, because γ⁻ = 0 there means "no edit"
rather than "boost only": with no demoted levels the head-selection criterion
degenerates to δ = −φ[privileged] and selects a different mask.

**A second claim is about the operating point, not the score (§6b–6c).** Under
our role mapping the union rule flags 94–99 % of KV heads at the default ε = 0,
tracking the attention layout rather than the model — indistinguishable from
editing all of V. But δ itself is *not* noise: measured per head across cases
against a binomial null, 3–8× more heads than chance are consistently inverted
and 79–174 survive Benjamini-Hochberg, with individual heads firing 37 of 40.
The signal is in the ranking and ε = 0 discards it; a percentile threshold
recovers a mask size comparable across models. An earlier draft of this claim
said the score was near-noise; that was an artefact of averaging a per-case
marginal, and it is corrected in §6c.

**A third, narrower one is about the boost half (§5b).** In the control where the
fact was never in the conversation, γ⁺ alone — nothing suppressed — is what turns
a refusal into an invented value, on all four Qwen2.5 sizes and by 16.7–47.2
points. The six models from five other families move by less than 8.3, so this is
a family result and is stated as one. The same section reports why the other six
looked pinned: 146 of 360 unedited answers are a canned capability disclaimer
rather than the model saying it was not told, so a value-probe counts two
different behaviours as one.

One observation is reported as a direction, not a finding, with the reasons it
does not yet support more: what the failures look like (§4).

We also report seven measurement bugs found in our own instruments, five of
which had produced findings, and the two review passes that found them.

---

## 1. Setup

**Task.** A system prompt states a constraint. Earlier in the same conversation
the user stated a mutually exclusive one, and two assistant turns visibly obeyed
it. A fact the user supplied sits in that earlier stretch. The model is then
asked a question whose answer is the fact.

Two things are scored, and never conflated:

- **compliance** — which constraint the output obeys, by a programmatic checker
- **recall** — whether the fact appears, by string match with a denial guard

The reported measure is a **useful answer**: compliance ∧ recall. Format alone is
not enough: under a strong edit a model emits `{"question": "When does my flight
land?"}`, which is valid JSON and answers nothing.

**Constraints.** Six families — uppercase, `ACK:` prefix, JSON, bullets, length,
numbered options — each with three independently worded versions. Six facts:
`4417-B`, `bagr`, `brno`, `19:40`, `E-88`, `302`.

**Models.** Qwen2.5-0.5B/1.5B/3B/7B, Qwen3-4B, Phi-3.5-mini, OLMo-2-7B,
Llama-3.1-8B, Aya-expanse-8B, Command-R7B. The four Qwen2.5 sizes form a ladder
inside one family, so size can be varied without also varying tokenizer and
post-training. Gemma-4 is excluded: this implementation cannot steer it (E2B
fails at prefill on a nested config, E4B shares KV across layers and uses
sliding-window attention on 20 of 24 cache layers).

**Second dataset.** Control Illusion (Geng et al., AAAI 2026): 100 task
instructions × 6 mutually exclusive constraint pairs, all checkable by counters,
in both orderings.

---

## 2. The conflict cost, decomposed

Four conditions, identical except for what precedes the question.

| | Qwen2.5-0.5B | Qwen2.5-1.5B | Command-R7B | Phi-3.5 | Qwen3-4B | OLMo-2-7B | Llama-3.1-8B | Aya-8B |
|---|---|---|---|---|---|---|---|---|
| prior turns obey the **system** rule | 83 % | 82 % | 94 % | 76 % | **100 %** | 98 % | **99 %** | 92 % |
| no prior turns at all | 31 % | 37 % | 47 % | 53 % | 92 % | 82 % | 97 % | 94 % |
| prior turns in plain prose, no conflict | 33 % | 34 % | 48 % | 41 % | 73 % | 84 % | 56 % | 65 % |
| prior turns + conflicting instruction | 28 % | 25 % | 0 % | 0 % | 0 % | 19 % | 3 % | 3 % |

**The conflict row uses only phrasing v1; the three ceiling rows average over
three phrasings.** Restricted to v1 on both sides, which is the only matched
comparison:

| phrasing v1 only | 0.5B | 1.5B | Cmd-R | Phi | Qwen3-4B | OLMo | Llama | Aya |
|---|---|---|---|---|---|---|---|---|
| prior turns obey the system rule | 78 | 86 | 92 | 83 | 100 | 97 | 100 | 94 |
| no prior turns | 28 | 42 | 53 | 61 | 94 | 81 | 100 | 97 |
| prior turns in prose, no conflict | 33 | 36 | 53 | 53 | 56 | 81 | 81 | 86 |
| prior turns + conflicting instruction | 28 | 25 | 0 | 0 | 0 | 19 | 3 | 3 |

Two things, and the second corrects an earlier version of this section.

**Two aligned demonstrations lift the weak models and do nothing for the strong
ones.** Qwen2.5-0.5B goes 28 → 78, Qwen2.5-1.5B 42 → 86, Command-R 53 → 92,
while Llama is 100 → 100 and Aya 97 → 94. That is few-shot format copying, not
a raised ceiling: these models can produce the formats when shown, and cannot
from the instruction alone. It is also a reason not to call the no-prior-turns
condition a ceiling — for weak models it is far below what they can do.

**The conflicting sentence does most of the damage, not the demonstrations.** An
earlier version of this section claimed the reverse, on unmatched items. Matched,
prose acknowledgements cost Llama 100 → 81 and the conflicting sentence then
costs 81 → 3 — roughly four times as much. Prose acknowledgements cost *nothing*
on four of eight models (0.5B +5, Command-R 0, OLMo 0, and 1.5B −6). Where they
do bite (Qwen3-4B −38, Llama −19, Aya −11, Phi −8) the drop is concentrated in
the families where prose competes with the format.

So the honest decomposition is: the demonstrations matter on some models and not
others, and the sentence dominates on all of them. The `ceiling` condition is
also not demonstration-free — it still contains an assistant turn reading
"Noted.", which violates `case`, `prefix`, `json` and `bullet`.

---

## 3. Which half of the edit does what

The first version of this ablation was wrong and is reported in §7. Corrected —
head mask held to what the full edit selects, γ⁺ matched at 2.5:

| model | no edit | full edit, γ⁻ = 0.75 | boost only |
|---|---|---|---|
| Qwen2.5-1.5B | 9/36, recall 36/36 | 33/36, recall **16/36** | 21/36, recall **36/36** |
| Qwen3-4B | 0/36, recall 36/36 | 13/36, recall 29/36 | 3/36, recall **36/36** |
| Llama-3.1-8B | 1/36, recall 36/36 | 14/36, recall 26/36 | 5/36, recall **36/36** |

Both arms are the *same* 36 items, so the test is paired. An earlier version
used an unpaired two-proportion z-test, which is both the wrong test and
pseudo-replicated.

| | compliance, McNemar | by family (6) | by fact (6) | recall, McNemar | by family | by fact |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | b1/c13, **0.0018** | 1+/3− p=0.63 | 0+/5− p=0.06 | b20/c0, **1.9e-6** | 6+/0− p=0.03 | 5+/0− p=0.06 |
| Qwen3-4B | b1/c11, **0.0063** | 0+/4− p=0.13 | 0+/6− p=0.03 | b7/c0, **0.016** | 6+/0− p=0.03 | 3+/0− p=0.25 |
| Llama-3.1-8B | b2/c11, **0.0225** | 1+/3− p=0.63 | 0+/6− p=0.03 | b10/c0, **0.0020** | 6+/0− p=0.03 | 4+/0− p=0.13 |

**Neither claim clears a sign test on both clustering axes**, and an earlier
version quoted, for each claim, whichever axis worked. With 6 clusters the sign
test floors at p = 0.031 even when every cluster agrees, so it is a weak
instrument here regardless.

What the recall effect has instead is the absence of counterexamples:
**b = 20, 7, 10 against c = 0, 0, 0** — not one item in 108 moves the other way —
and no negative cluster on either axis (family 6+/0−, fact 5+/0−, 3+/0−, 4+/0−;
the non-significant fact-axis p-values come from ties, not from disagreement).
The compliance effect is directionally consistent at the item level but has
1–2 counterexamples and fails the family axis outright.

**On the headline metric the trade is close to a wash.** Useful answers
(compliance ∧ recall) at matched γ⁺ = 4: Qwen2.5-1.5B **21/36 boost-only against
15/36 full**, Llama 13 against 14, Qwen3-4B 6 against 13. So suppression is net
negative on one model, a tie on another, and clearly positive only on the third.
"The suppression term buys compliance and costs recall" is right; "it is worth
it" is not established.

**The edit is mostly span-local.** With the fact moved out of the demoted span,
recall is 36/36 at every dose on Qwen3-4B and Llama. On Qwen2.5-1.5B it is
30–32/36 at every dose including the lowest — a flat offset rather than a dose
response, so not obviously collateral, but not zero either.

---

## 4. What failure looks like

Every generation is classified by a batch LLM judge into eight content
categories, with format compliance left to the mechanical checkers.

**Judge reliability, measured.** Scoring the same 188 answers twice under
different batch compositions gives 87.8 % agreement, κ = 0.852 — but per
category: ABSENT 100 %, CORRECT 97 %, NEAR 94 %, DISOWNED 81 %, WRONG 79 %,
UNSOURCED 75 %, and the newest category 58 %. Claims that rest on the rare
categories are therefore made with deterministic detectors instead, and the
judge is used as corroboration.

**Near misses.** When the fact is lost, what replaces it is systematically
related to it: scoring an answer against a gold value from a *different*
question gives a 0–2 % near-neighbour rate against 68 % for its own. But the
rate itself depends heavily on the definition — 9 % of wrong answers under
matching character shape alone, 68 % once truncation counts — and "dominant"
(>50 %) holds on one model of eight. The defensible claim is the weaker one:
**the replacement is related to the target, not arbitrary.**

**Non-terminating recall.** One mode has no home in a fixed rubric: the model
states the correct value, repeats it three to six times, and cannot settle on
it, with no alternative ever offered. Measured by rule (value repeated ≥3× and
≥3 contested-framing markers):

| | 0.5B | 1.5B | 3B | 4B | 7B | 8B |
|---|---|---|---|---|---|---|
| Qwen2.5 ladder | 0/756 | 0/756 | 0/756 | — | 0/756 | — |
| Qwen3 | — | — | — | 1/756 | — | — |
| Phi-3.5 / OLMo-2 / Aya / Command-R | — | — | 0/756 | — | 0/756 | 0/756 |
| **Llama-3.1** | | | | | | **13/756** |

**It is not a size effect.** A same-family ladder spanning 14× in parameters —
Qwen2.5 at 0.5B, 1.5B, 3B and 7B — gives zero at every size, with no trend. It
is zero on every other family too, including OLMo-2-7B and Aya-8B at Llama's own
size. One model of ten does it, and it rises with γ⁻. The eight-category judge
scattered these across three labels, which is what a missing category looks like
from inside.

**Right content, refused as a source.** An answer that states the fact and
denies being told it — "you live in a city called Brno, but since you didn't
mention it, i assumed it was a different city". Measured by rule (fact present
AND a denial pattern fires) this is also Llama-only: 16/756 against 0–1 on the
other nine, again with no size trend inside the Qwen2.5 ladder.

---

## 5. Mechanism

Teacher-forcing both conditions to the exact position where the value is
emitted, so the same question is asked at the same place:

**The correct token is attenuated, not overwritten.** Its probability falls by a
median factor of 261× (Llama, Qwen2.5-1.5B) to 54 471× (Qwen3-4B), from 1.0000
to as little as 0.0005. Nothing promotes a wrong token.

**What wins instead is model-dependent.** Median rank of the substitute in the
*unsteered* distribution: 10 on Llama, 7 on Qwen2.5-1.5B, **91 on Qwen3-4B** —
and in one case rank 36 931, where the winner is `<|im_end|>` and the model ends
the turn rather than emit a value. So "the edit lets a standing competitor win"
holds on two models of three and is clearly false on the third.

Two readings the distributions settle:

- What looked like **truncation** (`4417-B` → `4417`) is the model choosing `.`
  over `-B` — rank 13 on Llama, rank 4 on Qwen2.5-1.5B. It closes the number
  early rather than losing characters.
- **`brno` resists**: rank 1 at p = 0.957 even at γ⁻ = 0.95, failing only to
  `Brussels` and `Bris` at ranks 46–602 — semantic neighbours from far down, not
  frequent strings.

What is *not* measured: whether attention to the demoted span changes. It should
not, mechanically, since only V is scaled — and that is the basis for the
disowning hypothesis, which therefore remains a hypothesis.

---

## 5b. The control where the fact was never there

Every condition above attenuates a fact that IS present. The control that says
what those numbers mean is the one where it never was: if the model declines
then, and confabulates when the fact is merely quietened, attenuating evidence
is not the same as removing it.

`failure_atlas.py --fact-absent swap` puts a different fact's statement in the
same message slot, so the transcript keeps its shape, length and message
positions and only the answer is missing (`drop` removes the turn entirely and
agrees). **All ten models**, γ⁺ = 4, 36 cases per cell.

Scored by `examples/abstain_judge_gemini.py`, which labels every answer VALUE /
DECLINE_SAID / DECLINE_LIMITS / OTHER and refuses to score anything until it
reproduces the hand labels in `abstain_calibrate.py` — it passed 20/20. The gold
value is never shown to the judge, because half these cases are the control
where the fact was never in the conversation, and telling the judge what the
answer "should" have been gets it grading correctness instead. The judge is
`gemini-3.1-flash-lite`, deliberately outside the ten measured models; the local
Qwen2.5-3B judge in `abstain_judge.py` (19/20 on the same gate) is itself one of
the ten.

**γ⁺ alone converts a refusal into an invented value.** The measure is *states a
value*, which in this control is necessarily confabulated:

| fact never stated, states a value | no edit | **γ⁺ only** | full edit |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 47.2 % | **63.9 %** | 47.2 % |
| Qwen2.5-1.5B | 16.7 % | **63.9 %** | 41.7 % |
| Qwen2.5-3B | 2.8 % | **22.2 %** | 22.2 % |
| Qwen2.5-7B | 16.7 % | **47.2 %** | 38.9 % |
| Qwen3-4B | 5.6 % | 2.8 % | 0.0 % |
| Phi-3.5-mini | 11.1 % | 8.3 % | 5.6 % |
| Llama-3.1-8B | 2.8 % | 8.3 % | 5.6 % |
| OLMo-2-7B | 66.7 % | 63.9 % | 77.8 % |
| Command-R-7B | 2.8 % | 8.3 % | 2.8 % |
| Aya-expanse-8B | 5.6 % | 13.9 % | 8.3 % |

The no-edit column is genuinely unedited: `failure_atlas.py:295` builds no policy
at all when `gm == 0` without `--always-steer`, so γ⁺ never applies there. The
γ⁺ column is `run_boost.sh` (`--always-steer --select-as-if 0.95`).

**Scope.** Every Qwen2.5 size moves the same way and by 16.7–47.2 points. The
six models from five other families sit inside ±8.3 points, which is one to
three cases. This is a Qwen2.5 result. An earlier three-model version of this
table was three Qwen2.5s, which is why it read as general — that version is
superseded.

### Most of what a value-probe counts as abstention is not abstention

Reading the generations forced a split the earlier yes/no probe could not make:

    DECLINE_SAID     "You have not told me the name of your dog."
    DECLINE_LIMITS   "I'm an AI and don't have access to real-time information."

The second is true whatever the transcript says. It fires on the *kind* of
question — flights, addresses and account numbers are things an assistant is
trained to disclaim — not because the model consulted the conversation and found
nothing. Pooled over the ten unedited cells (360 answers): **146** canned
capability lines, **130** genuine "I was not told", 64 invented values, 20
neither. Concentrated: Phi 31/36, Qwen3-4B 28/36, Command-R 21/36, Aya 19/36.
Those four look pinned near 100 % "abstention" under a probe that only asks *did
it give a value*, and the pinning is a property of the question set. Llama is the
opposite, 32/36 grounded.

Anything measuring abstention, hallucination or "does the model know what it does
not know" is likely mixing these two, and the mix is model-dependent, so it does
not cancel.

**Not claimed.** Phi appears to move +69 points from `LIMITS` to `SAID` under γ⁺.
It is an artefact: Phi's refusal is one template that cites its own limits *and*
the conversation in the same sentence — "I don't have access to personal data
unless it's shared with me during our conversation" — so the label turns on
trailing words rather than behaviour. Hand-read and discarded. The `VALUE` column
does not have this problem, which is why the finding rests on it.

Open: the hand read is a stratified sample per model, not every cell.

---

## 6. Replication

Control Illusion, Llama-3.1-8B, n = 16 per conflict type, exact counters:

| | no suppression | best cell |
|---|---|---|
| original ordering | 13/96 | **64/96** |
| reversed ordering | 35/96 | **66/96** |

It replicates in both directions. Original: all six types positive, sign test
p = 0.031. Reversed: four of six, and the two that do not move have no headroom
— `word_length` reversed asks for under 50 words and the model writes 53–349,
`language` reversed is 16/16 before any steering.

**One family is not measuring what the benchmark thinks.** Counting French
answers directly: the model writes French 16/16 in *both* orderings, whichever
rule is current. `language_conflict` measures that a mentioned language sticks,
not instruction priority.

---

## 6b. What the head criterion actually selects

The method flags a KV head when the demoted span outscores the privileged one by
more than ε, with ε = 0 by default, and a KV head counts as flagged if **any**
query head in its group does (the union rule, App. A.2). Measured on Control
Illusion, 120 cases, eight models, prefill only (`examples/head_criterion.py`):

| model | q/kv heads | n_rep | a coin flip under the union rule | ε = 0 | ε = 0.01 | ε = 0.05 |
|---|---|---|---|---|---|---|
| Phi-3.5-mini | 1024/1024 | 1 | 50.00 % | 50.4 % | 10.6 % | 2.2 % |
| OLMo-2-7B | 1024/1024 | 1 | 50.00 % | 48.5 % | 4.9 % | 0.6 % |
| Llama-3.1-8B | 1024/256 | 4 | 93.75 % | 96.1 % | 0.9 % | 0.1 % |
| Qwen3-4B | 1152/288 | 4 | 93.75 % | 94.6 % | 21.2 % | 3.2 % |
| Qwen2.5-1.5B | 336/56 | 6 | 98.44 % | 98.5 % | 40.2 % | 5.9 % |
| Qwen2.5-0.5B | 336/48 | 7 | 99.22 % | 99.0 % | 6.9 % | 0.3 % |
| Qwen2.5-7B | 784/112 | 7 | 99.22 % | 97.2 % | 21.4 % | 3.1 % |
| Qwen2.5-3B | 576/72 | 8 | 99.61 % | 99.1 % | 44.6 % | 9.5 % |

An earlier version of this section reported the 94–98 % as evidence that "the
diagnosis is barely a diagnosis", presented the cross-model ordering as a result,
and *inferred* a per-query-head rate from the union rule by assuming the heads in
a group were independent. The ordering was a property of the attention layout,
and the inference has now been replaced by direct measurement.

**The flagged fraction is a function of the attention layout, not of the model.**
It tracks `n_rep` almost perfectly — one inversion in eight, Qwen2.5-7B against
Qwen2.5-1.5B — and lands within 2.4 points of what a coin flip would give under
the union rule, *below* it on four of the eight. The two MHA models are the clean
case: with `n_rep` = 1 the union rule is the identity, nothing can be inflated,
and the mask comes out at 50.4 % and 48.5 %.

Four controls, all at ε = 0:

- **Per query head.** The fraction of *query* heads with δ > 0 is 48.9–53.5 % on
  every model. Inferred, it had looked like 50.2–55.6 %; measured, it is flatter.
- **Group rule.** Replacing the union with the mean gives 46.7–59.9 % — the
  per-query-head rate. The union rule is what carries the mask from 50 % to 99 %.
- **Label swap.** Exchanging privileged and demoted negates δ. On the six GQA
  models nearly every head is flagged *both ways* — 96.1 % under δ and 95.8 %
  under −δ on Llama, 99.1 % and 99.5 % on Qwen2.5-3B — which is only possible if
  the group holds query heads of both signs. On the two MHA models the two are
  exact complements by construction (48.5/51.5, 50.4/49.6), and both sit at
  chance.
- **Length-normalised φ.** Dividing each φ by its span's token count — our own
  stated confound, the demoted span being several times the longer — moves
  nothing: 96.3 % against 96.1 % on Llama, under 1 point on every model.

Under our role mapping, then, the default operating point ε = 0 edits
essentially every KV head, which makes "select the heads that attend to the
stale span" hard to distinguish from "scale all of V".

An earlier version of this section stopped here and concluded that **δ carries
almost no signal at the query-head level**. That was wrong. The two checks that
show it wrong are below — the third rewrite of this section and the second time
its headline has reversed.

## 6c. The two checks that overturned 6b

Both were prompted by an adversarial review pointing out that 6b could not
distinguish its own conclusion from two artefacts of how it was measured.
`examples/head_precision.py`, 40 Control Illusion cases, prefill only.

**Was it bf16 rounding? No.** The per-head dot product runs in the model's own
dtype, bf16 on GPU, and Llama's median |δ| is 1.9e-05. Rounding noise in the
*sign* of δ would produce exactly the ~50 % and exactly the label-swap symmetry
reported above. Recomputed in float32:

| | sign agreement | KV mask agreement (ε = 0) | median \|δ\| where the sign flipped |
|---|---|---|---|
| Llama-3.1-8B | 99.93 % | 99.98 % | 1.1e-06 |
| Qwen3-4B | 99.93 % | 99.98 % | 8.4e-06 |
| Qwen2.5-1.5B | 99.96 % | 100.00 % | 1.2e-05 |

The few heads that flip sit at the arithmetic floor. The measurement is sound;
the dtype was not the story.

**Is δ actually noise? No — and this is where 6b was wrong.** 6b's statistic was
`(δ > 0).mean()` per case, averaged over cases. A criterion that flags the *same*
half of the heads every time gives an identical number to one that flags a random
half. What separates them is the per-head rate **across** cases against a
binomial null:

| | heads with p < 0.05 | expected by chance | surviving Benjamini-Hochberg at 5 % | most consistent head |
|---|---|---|---|---|
| Llama-3.1-8B | 297 / 1024 | 51 | **174** | 38 / 40 |
| Qwen3-4B | 254 / 1152 | 58 | **79** | 37 / 40 |
| Qwen2.5-1.5B | 128 / 336 | 17 | **84** | 37 / 40 |

Three to eight times more heads than chance, a substantial set surviving
correction for over a thousand simultaneous tests, and individual heads firing 37
or 38 times out of 40. **There is a stable minority of genuinely inverted heads,
and 6b's per-case marginal was averaging it away.**

So the corrected claim is not that the criterion is empty. It is narrower, and
it is about the operating point rather than the score:

> δ identifies a real, stable minority of heads. The default threshold then
> discards that: at ε = 0 the union rule flags 94–99 % of KV heads, which is
> indistinguishable from editing all of V. The signal is in the *ranking*, and
> ε = 0 does not use the ranking.

**A percentile threshold is the comparable operating point**, and it behaves:

| percentile on δ | Llama-3.1-8B | Qwen3-4B | Qwen2.5-1.5B |
|---|---|---|---|
| 50th | 96.2 % | 94.6 % | 98.7 % |
| 90th | 32.3 % | 30.8 % | 42.2 % |
| 99th | 3.9 % | 3.8 % | 6.8 % |

That is what an absolute ε cannot give: a mask size meaning the same thing on
three models whose median |δ| differs by a factor of 15.

## 6d. The causal control

`examples/mask_control.py`, Llama-3.1-8B, γ⁺ = 4, γ⁻ = 0.5, 36 paired items.
Four arms differing only in which KV heads are edited; `random` draws the same
*number* of heads as `selected` flagged for that case, seeded per case.

| arm | mask | useful answers | vs selected (McNemar) |
|---|---|---|---|
| no edit | — | 1/36 | — |
| **selected** | 97.1 % | **20/36** | — |
| random, matched size | 97.1 % | 16/36 | b = 5, c = 1, p = 0.22 |
| all KV heads | 100 % | 20/36 | b = 1, c = 1, **p = 1.00** |

**Selection is indistinguishable from editing everything** — 20 against 20, one
discordant pair in each direction. That is the operating-point claim of §6b
established causally rather than statistically: at ε = 0 the criterion is not
selecting, it is passing everything through.

Against a random mask of the same size, selection is directionally better,
5 discordant pairs to 1, but p = 0.22 on 36 items. **This design has almost no
power to answer that question**, because at 97.1 % the two masks differ on about
seven KV heads out of 256. The decisive version is the same experiment run at a
percentile threshold, where §6c's numbers put the mask near 30 % and there is
something for the ranking to get right. That has not been run.

**The paper runs this control too, and the difference is the point.** Tab. 12
reports a `Random (half)` arm — a randomly sampled half of all heads — at
58.6 / 69.2 / 59.6 / 58.8 primary accuracy against DLA's 83.5 / 85.6 / 79.8 /
79.2. On that comparison selection matters enormously. But a random *half* is
not the same size as the DLA mask: §6b measures the DLA mask at 96.1 % of KV
heads on Llama at ε = 0, because the union rule of App. A.2 — which the paper
itself calls conservative — flags a KV head whenever any query head in its
group is bad. So `Random (half)` edits roughly half as many heads as `DLA`
does, and the arm varies *which* heads and *how many* at the same time. Holding
the size fixed is what this section adds; when it is held fixed the difference
goes away.

That is a statement about what the ablation isolates on GQA models, not about
whether the method works — the headline replicates in §6, and on the two MHA
models in §6b (n_rep = 1) the mask sits near 50 %, where the paper's arm and
this one nearly coincide.

What is still open. The percentile-threshold version of this control, on more
than one model. And the roles remain *our* mapping of Control Illusion onto an
epoch structure — constraint1 privileged, constraint2 demoted with an
acknowledgement — not the authors'; a gap against their Tab. 12 is informative
about the mapping as much as about the method.

## 7. Six measurement bugs, and what they had produced

All six were found by disbelieving a result, none by a test. Four had already
produced a finding.

| bug | what it did | what it had produced |
|---|---|---|
| `check_json` accepted anything starting with `{` | `{\n {}` and `{"question": …}` scored as compliant; correct prose scored as failure | inflated JSON compliance |
| `check_bullet` required two bullet lines | a correct one-line `• 4417-B` was unscoreable | "bullet lists are never recovered on any model" — they are, 47/7/33 % |
| `check_length` scored the empty string as compliant | a dead model was the best-behaved one | inflated length compliance |
| `check_case` needed one letter | `4417-B` (capital B) scored compliant, `302` scored neither | verdicts decided by whether the fact contained a letter |
| Control Illusion direction hardcoded | every verdict on the reversed file inverted, 485/576 | "steering makes it worse when the order is flipped", plus an additive-versus-restrictive theory |
| flagged-head measurement passed level 0 as demoted | level 0 is the system message, so it computed φ−φ = 0 | 0 % of heads flagged on every model |
| §3 used an unpaired z-test on paired items, and reported the clustering axis that worked | overstated both halves of the ablation | "significantly better on both" — neither clears both axes |

And one design bug with the same shape: **γ⁻ = 0 does not ablate the suppression
term.** With no demoted levels, `inversion()` reduces to `δ = −φ[privileged]`,
so head selection switches from "the stale span beats the system span" to "the
system span has negative attribution" — a different and roughly unrelated set.
The three γ⁺ values at γ⁻ = 0 also produce byte-identical greedy output, so that
baseline is 36 generations, not 108.

The common signature: **a checker written from one case and applied to all of
them.** The defence that worked was not more unit tests, it was running a
condition whose answer was known and refusing to accept the number.

---

## 7b. The baseline for the benign application

Scope first, because the comparison below is not one the paper omitted by
oversight. V-Steer is posed as an instruction-hierarchy defence — Fig. 1 is a
privileged system instruction against a *malicious* lower-priority user — and
against a hostile user, deleting the offending message is not a move anyone
has. The transcript is the attack surface; you have to hold the hierarchy while
it is still there.

The comparison matters for the *other* application, which is ours: a
cooperative user whose earlier instruction has gone stale. There, deleting is
available, and it is what a practitioner reaches for first.

Every condition above is measured against the *conflicted* transcript, where
several models sit at 0 %. None was measured against **deleting the stale
message**. That condition
existed in `results/` throughout under the name `ceiling`, treated as an upper
bound rather than as a competitor. `examples/deletion_baseline.py`, no GPU.

Useful answers (compliance ∧ recall) over 108 items, steering cell chosen post
hoc as the best of 21:

| | delete the stale message | best steering cell |
|---|---|---|
| Llama-3.1-8B | **97.2 %** | 55.6 % |
| Aya-8B | **94.4 %** | 77.8 % |
| Qwen2.5-7B | **92.6 %** | 72.2 % |
| Qwen3-4B | **91.7 %** | 44.4 % |
| OLMo-2-7B | **82.4 %** | 55.6 % |
| Qwen2.5-3B | **81.5 %** | 80.6 % |
| Phi-3.5-mini | **52.8 %** | 44.4 % |
| Qwen2.5-1.5B | 37.0 % | **83.3 %** |
| Command-R7B | 47.2 % | **61.1 %** |
| Qwen2.5-0.5B | 30.6 % | **52.8 %** |

Deletion wins on seven of ten, steering only on the three weakest. Both require
the same input — an epoch label marking which message is stale — so deletion is
not getting an unfair advantage on that axis.

**But the case construction hands deletion the win.** In
`phrasing_atlas.build_cases(control=True)` the stale instruction and its
acknowledgement are their own two messages and are removed; the fact-bearing
turn is kept intact. Every case in this suite is therefore one where the stale
instruction is *cleanly separable* from the content the question needs. On such
a transcript, deletion is obviously correct and the edit is a worse route to a
worse answer.

The case that would justify the method is unrun: **an instruction and live
information sharing one turn** — "always reply in lowercase, and my order number
is 4417-B". Deleting that costs the order number outright; the edit demotes the
message's values while the fact remains in context, and §3 shows the fact
surviving at γ⁻ = 0.5. It needs a `build_cases` variant concatenating the fact
into the stale message, and it has not been run.

There is also a cost asymmetry no number here captures: deleting a message
invalidates the KV prefix from that point and forces a re-prefill, whereas the
edit is in-place on the cache. Unmeasured.

**Scoped honestly:** where the stale instruction is separable, this work cannot
recommend the edit over deletion. Where it is entangled with content that is
still needed, see §7c — which has now been run.

---

## 7c. The entangled case, and the first condition where the edit wins

`examples/entangled.py`. The stale instruction and the needed fact share **one**
user turn — *"From now on always reply in all lowercase. My order number is
4417-B."* — so deleting the message costs you the fact. Four arms on the same 36
items, useful answers (compliance ∧ recall):

| | no intervention | delete the message | rewrite the message | V-edit (best of 4 cells) |
|---|---|---|---|---|
| Llama-3.1-8B | 6/36 | 0/36 | **36/36** | 23/36 |
| Qwen3-4B | 0/36 | 0/36 | **34/36** | 15/36 |
| Qwen2.5-1.5B | 10/36 | 0/36 | 15/36 | **30/36** |

`rewrite` strips the instruction clause and keeps the fact sentence — the
surgical version a careful engineer would do instead of deleting.

**Deletion collapses completely: 0/36 on all three.** It still fixes compliance
(30/36, 33/36, 35/36) and takes recall to zero, exactly as predicted. Message-level
deletion is not an option once the instruction and the content are in one turn,
and §7b's baseline does not transfer to this case at all.

**But rewriting beats the edit on the two stronger models**, 36/36 and 34/36
against 23/36 and 15/36. The applied case is not "the edit beats deleting"; it is
"the edit beats deleting, and loses to rewriting, on models that can follow the
instruction unaided."

**The exception is the finding.** On Qwen2.5-1.5B the edit gets 30/36 against
rewriting's 15/36 — twice as good — and it holds at 30/36 across both γ⁺ values
at γ⁻ = 0.5, so it is not a lucky cell. The reason is visible in the components:
rewriting removes the conflict but leaves compliance at 15/36, because this model
cannot produce the formats from the instruction alone (§2: two aligned
demonstrations take it from 28 to 78). The edit's γ⁺ term does something no
prompt-level fix does — it *amplifies* the current instruction rather than merely
removing the competition.

So the scoped applied claim, finally:

> Separable at message level → delete it. Entangled but separable at sentence
> level, on a model that can follow the rule unaided → rewrite it. Entangled on a
> weak model, or wherever the transcript cannot be rewritten at all → the V-edit
> is the best available option, and on Qwen2.5-1.5B it is twice as good as the
> alternative.

Four caveats. n = 36 per cell, greedy, one seed, three models. The `rewrite` arm
is idealised — it assumes a clean sentence boundary between instruction and
content, which a real transcript does not hand you. Rewriting needs span-level
knowledge of *which part* of the message is the instruction, where the edit needs
only the message-level epoch label. And the edit's column is a best-of-four while
rewriting has no free parameters, so the edit is flattered everywhere except
where it wins, where it is robust across cells.

---

## 8. Limitations

- **Greedy only.** No interval anywhere reflects generation stochasticity, and
  the item set is fixed and exhaustive rather than sampled.
- **The observations are not independent.** The same 6 facts × 6 families recur
  in every cell. All headline claims are recomputed clustered; one did not
  survive (see below).
- **No multiplicity correction** over well past a hundred comparisons.
- **The operating point is selected post hoc** from 21 cells. Held-out selection
  — choose on three families, score on the other three, averaged over all 20
  splits — costs 7–17 points, mean 13.
- **`check_options` scores the absence of a pattern as compliance**, which is
  why the headline metric is compliance ∧ recall.
- **One constraint set is ours.** Control Illusion is the only external one and
  it covers compliance, not recall.

**Withdrawn during this work:** that near misses are the dominant failure (one
model of eight); that common strings survive the edit better than rare ones
(+9.3 points, but sign test 6/8, p = 0.29 once clustered by pair); and that the
substitution and the hesitation are dissociable by generation length (the short
run is a strict prefix of the long one under greedy decoding — 756/756 — so
there was no second condition).

---

## 9. What would make this stronger

1. Direction-matched constraint pairs, so "restores the current instruction" can
   be separated from "produces more output".
2. Attention measured under the edit, to test the disowning hypothesis.
3. A second annotator on the judge, and more phrasings per constraint — three
   gives a spread, not an interval on the spread.
4. An off-span fact that is not already at ceiling, so the span-locality test
   has power.
5. A percentile threshold on δ instead of an absolute ε, so the head criterion
   can be compared across models at a matched mask size (§6b). Prefill only —
   the cheapest open item here.
6. **The entangled-message condition (§7b)** — an instruction and a needed fact
   in one turn, so that deletion has to pay for what it removes. This is the
   only experiment here that could restore the applied case for the method, and
   it is a variant of `build_cases`, not new machinery.
7. The causal mask control (§6d) rerun at a percentile threshold, where the mask
   is near 30 % and there is something for the ranking to get right.

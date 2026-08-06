# What a stale instruction costs, and what a value edit buys back

Draft. Every number here is recomputed from stored generations by
`examples/report_numbers.py`; nothing is quoted from memory.

---

## Abstract

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

Two further observations are reported as directions, not findings, with the
reasons they do not yet support more: what the failures look like (§4), and how
many heads the diagnosis selects (§6b).

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
Illusion over 120 cases:

| model | q/kv | n_rep | ε = 0 | what a coin flip would give | implied per-query-head rate |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32/8 | 4 | 96.1 % | 93.75 % | **55.6 %** |
| Qwen3-4B | 32/8 | 4 | 94.5 % | 93.75 % | **51.7 %** |
| Qwen2.5-1.5B | 12/2 | 6 | 98.5 % | **98.44 %** | **50.2 %** |

An earlier version of this section reported the 94–98 % as evidence that "the
diagnosis is barely a diagnosis" and presented the cross-model ordering as a
result. Both were wrong: **the ordering is exactly the ordering of `n_rep`**, and
the union rule over a group of 4 or 6 produces 93.75 % or 98.44 % from a coin
flip alone. The measured numbers are barely above that.

Inverting the group rule gives the quantity that matters — the fraction of
*query* heads with δ > 0: **50.2 %, 51.7 %, 55.6 %**. The inversion score is
symmetric about zero. That is a sharper statement than the one it replaces: it is
not that the threshold is set too low, it is that **δ carries almost no signal at
the query-head level, and the union rule turns near-noise into a near-universal
mask.**

Two caveats on our own measurement. ε is an absolute threshold on a raw
logit-contribution difference, so it is not comparable across models; and φ sums
over span positions, and our demoted span (constraint2 plus an acknowledgement)
is several times longer than the privileged span (constraint1 alone), which
inflates δ mechanically. Both would have to be fixed — per-query-head δ
distributions, `group_rule="mean"` alongside `max`, a label-swap permutation
baseline, and length-normalised φ — before this becomes a claim about the
method rather than about our mapping.

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

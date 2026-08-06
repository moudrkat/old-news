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

18 894 greedy generations across ten models (0.5B–8B, six families, including a
same-family size ladder), two datasets, six ablation conditions.

Three results survive adversarial review and cluster-robust inference:

1. **A prior instruction costs far more through its demonstrations than through
   its content.** Two assistant turns that answer in the wrong format take
   Llama-3.1-8B from 99 % to 56 % before any conflicting instruction exists; the
   conflicting instruction then takes it to 3 %.
2. **The two halves of the edit do different jobs.** With the head mask held
   fixed and γ⁺ matched, dropping the suppression term costs 23–50 % of the
   compliance gain and recovers *all* of the lost recall (36/36 against 16–29/36
   on three models, every constraint family agreeing).
3. **The failure is attenuation, not overwriting.** At the token where the value
   is emitted, the correct token's probability falls by 261× to 54 471×;
   nothing promotes a competitor. What wins instead is model-dependent — a
   top-10 standing competitor on two models, rank ~91 on a third, and once the
   end-of-turn token.

We also report six measurement bugs found in our own instruments, four of which
had produced findings, and the process that found them.

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

Read the first two rows together. **The condition we started with — no prior
turns — is not the ceiling.** Every model does better when two prior assistant
turns demonstrate the required format, and the weak models do dramatically
better: Qwen2.5-0.5B goes 31 % → 83 %, Command-R7B 47 % → 94 %. These models can
follow the constraints; they cannot follow them from an instruction alone.

Read the first and third rows together. **Demonstrations of the wrong format are
most of the damage.** The non-conflicting prior instruction ("please be polite
and take your time") is acknowledged in plain prose — no capitals, no `ACK:`, no
JSON — and that alone costs Llama 99 → 56 and Qwen3-4B 100 → 73. The drop is
concentrated exactly where prose competes: on Llama, `case` 100 → 28,
`prefix` 100 → 17, `json` 100 → 50, while `length` holds at 100 → 94.

Only then does the conflicting *sentence* take it the rest of the way, to 0–3 %.

This decomposition matters for the framing of the whole area. "A stale
instruction overrides the system prompt" is measured, in this benchmark and
plausibly in others, against transcripts in which the assistant has also
demonstrated the stale behaviour twice. The instruction and the demonstration
are separable, and they are not the same size.

---

## 3. Which half of the edit does what

The first version of this ablation was wrong and is reported in §7. Corrected —
head mask held to what the full edit selects, γ⁺ matched at 2.5:

| model | no edit | full edit, γ⁻ = 0.75 | boost only |
|---|---|---|---|
| Qwen2.5-1.5B | 9/36, recall 36/36 | 33/36, recall **16/36** | 21/36, recall **36/36** |
| Qwen3-4B | 0/36, recall 36/36 | 13/36, recall 29/36 | 3/36, recall **36/36** |
| Llama-3.1-8B | 1/36, recall 36/36 | 14/36, recall 26/36 | 5/36, recall **36/36** |

Dropping suppression is significantly worse on compliance on all three
(p = 0.001, 0.005, 0.016) and significantly better on recall on all three
(p = 1e-7, 5e-3, 7e-4), with all six constraint families agreeing on the recall
direction in each case (sign test p = 0.031 per model).

**The suppression term buys 50–77 % of the compliance and costs all of the
recall.** That is a trade with a knob on it, and the knob is not reported in the
original work because γ⁻ = 0 there means "no edit" rather than "boost only" —
see §7.

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
≥3 contested-framing markers) it is **Llama-specific across all eight models**:
13/756 against 0–1 elsewhere, including OLMo-2-7B and Aya-8B at the same size.
It rises with γ⁻. The eight-category judge scattered these across three labels,
which is what a missing category looks like from inside.

**Right content, refused as a source.** An answer that states the fact and
denies being told it — "you live in a city called Brno, but since you didn't
mention it, i assumed it was a different city". Measured by rule (fact present
AND a denial pattern fires) this is also Llama-only: 16/756 against 0–1.

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

## 6b. How many heads the diagnosis actually selects

The method flags a KV head when the demoted span outscores the privileged one by
more than ε, with ε = 0 by default. Measured on Control Illusion — the dataset
the original head-fraction table uses — over 120 cases:

| model | KV heads | ε = 0 | ε = 0.05 | ε = 0.1 |
|---|---|---|---|---|
| Llama-3.1-8B | 256 | **96.1 %** (246) | 0.1 % | 0.0 % |
| Qwen3-4B | 288 | **94.5 %** (272) | 3.2 % | 0.9 % |
| Qwen2.5-1.5B | 56 | **98.5 %** | 5.9 % | 1.4 % |

Two things follow.

**At ε = 0 the diagnosis is barely a diagnosis.** It selects 94–98 % of heads,
uniformly across all six conflict types (95.8–96.5 % on Llama). "Steer the heads
where the hierarchy is inverted" and "steer every head" differ by a few per cent
of the mask. That matters because the published all-heads baseline matches DLA
on accuracy while collapsing far more often — at this selection rate there is
little margin left in which the two could differ.

**The threshold is a cliff, not a slope.** Moving ε from 0 to 0.05 takes Llama
from 96 % to 0.1 %. So δ is positive for nearly every head and larger than 0.05
for almost none: the attributions cluster just above zero. Any result that
depends on the identity of the selected heads is therefore extremely sensitive
to a parameter that is not swept in the original work.

This also settles a measurement question raised in correspondence: a 96 % figure
observed under a different role mapping on different data reproduces at 94–98 %
here, on the dataset the original table used. It is a property of the criterion,
not of the mapping.

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

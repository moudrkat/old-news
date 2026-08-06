# What a stale instruction costs, and how much of it steering buys back

A follow-up to V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228). V-Steer
reports aggregate effectiveness per model. This asks what the failure looks
like at scale — across models and constraint types — and what the method is
worth against a proper baseline.

## What was run

**18 894 generations**, all kept. Ten models — Qwen2.5-0.5B / 1.5B / 3B / 7B,
Qwen3-4B, Phi-3.5-mini, OLMo-2-7B, Llama-3.1-8B, Aya-expanse-8B, Command-R7B —
of which the four Qwen2.5 sizes plus Qwen3-4B form a ladder inside one family,
so "is this size or is it the model" can be asked without confounding size with
tokenizer and post-training. Gemma-4 is absent because this implementation
cannot steer it: E2B fails at prefill (the config is nested, so
`num_attention_heads` is not where the code looks) and E4B additionally shares
KV across layers and uses sliding-window attention on 20 of 24 cache layers,
where a demoted span can fall outside the window. Qwen3-8B does not fit in bf16
on a 16 GB card.

`examples/report_numbers.py` recomputes every figure below from the stored
generations, so the write-up can be checked against the data rather than
trusted. Two of its numbers were wrong when last checked (the total, and a
"70 of 70") and are corrected here — which is the argument for having the
script, not against it.

Three conditions carry the main result, all scored the same way.

| condition | history | steering | n per model |
|---|---|---|---|
| **ceiling** | no conflicting rule | none | 108 |
| **conflict** | stale rule present | none | 108 |
| **grid** | stale rule present | 21 (γ+, γ−) cells | 756 |

The conflict row says 108 records but only **36 distinct generations**: at
γ− = 0 no policy is built, so the three γ+ values produce byte-identical greedy
output. Every interval and test involving that baseline is computed on 36 items,
not 108, and the tables below use it as such.

Each case pairs a current system constraint with a mutually exclusive one
stated earlier in the conversation, plus a fact the answer has to carry.
6 constraint families × 6 facts, and for the ceiling also × 3 phrasings.
Greedy throughout.

The reported measure is a **useful answer**: the system constraint obeyed
**and** the fact present. Format alone is not enough — under a strong edit one
model emits `{"question": "When does my flight land?"}`, which is valid JSON and
answers nothing.

Code: `examples/failure_atlas.py`, `examples/phrasing_atlas.py` ·
figure: `examples/make_recovery_figure.py` → `results/recovery.png`

## Result

Useful answer = the system constraint obeyed **and** the fact present, on all
eight models. Ordered by ceiling:

| model | ceiling | conflict, no steering | best cell |
|---|---|---|---|
| Qwen2.5-0.5B | 33 % | 33 % | **56 %** (γ+4 / γ−0.5) |
| Qwen2.5-1.5B | 38 % | 25 % | **86 %** (γ+4 / γ−0.5) |
| Command-R7B | 48 % | 0 % | **61 %** (γ+2.5 / γ−0.65) |
| Phi-3.5-mini | 53 % | 0 % | 44 % (γ+4 / γ−0.5) |
| OLMo-2-7B | 85 % | 19 % | 58 % (γ+4 / γ−0.65) |
| Qwen3-4B | 93 % | 0 % | 50 % (γ+4 / γ−0.65) |
| Aya-expanse-8B | 94 % | 3 % | 78 % (γ+4 / γ−0.65) |
| Llama-3.1-8B | 97 % | 3 % | 56 % (γ+4 / γ−0.5) |

**The collapse is real but it is not one sentence doing it.** A length-matched
control — a prior instruction that exists, is acknowledged, and does *not*
conflict — splits it in two:

| Llama-3.1-8B | useful answer |
|---|---|
| no prior turns at all | 97 % |
| prior turns present, non-conflicting instruction | **56 %** |
| prior turns present, conflicting instruction | 3 % |

Roughly half the collapse is bought by the mere presence of two assistant turns,
before any conflict. Same on Aya (94 → 65 → 3) and Qwen3-4B (92 → 73 → 0);
unchanged on the four models whose ceilings were low anyway.

The reason is visible in what those turns say. The non-conflicting
acknowledgement reads "Understood, I will be polite and take my time from now
on" — **plain prose**: no capitals, no `ACK:`, no JSON, no bullets. It is
already a demonstration of the wrong format. The drop is concentrated exactly
where that bites: on Llama, `case` 100 → 28, `prefix` 100 → 17, `json` 100 → 50,
`bullet` 83 → 50, while `length` holds at 100 → 94 and `options` at 100 → 100.

So the honest decomposition is: **two assistant turns in the wrong format do
about half the damage, and the conflicting instruction does the other half.**
The earlier "one sentence three turns back takes a model from near-perfect to
nothing" attributed all of it to the sentence. The clean control would have the
acknowledgement turns obey the *system* rule — `staleset.build` ships exactly
that condition under the name `aligned`, and neither this experiment nor the
phrasing one uses it. That is the next thing to run.

**The best-cell column is optimistic and has to be corrected first.** It is
chosen post hoc from 21 cells on the same data it reports. Selecting the cell on
three constraint families and evaluating on the other three — averaged over all
20 such splits — costs **7 to 17 points, mean 13**:

| model | ceiling | post hoc | held-out |
|---|---|---|---|
| Qwen2.5-0.5B | 33 % | 56 % | **46 %** |
| Qwen2.5-1.5B | 38 % | 86 % | **72 %** |
| Command-R7B | 48 % | 61 % | 46 % |
| Phi-3.5-mini | 53 % | 44 % | 27 % |
| OLMo-2-7B | 85 % | 58 % | 43 % |
| Qwen3-4B | 93 % | 50 % | 43 % |
| Aya-expanse-8B | 94 % | 78 % | 66 % |
| Llama-3.1-8B | 97 % | 56 % | 43 % |

**On held-out numbers only two models of eight end up above their ceiling** —
the two weakest, at 33 % and 38 %. Command-R7B exceeded on the post-hoc figure
and does not on the honest one, so the earlier "everything under 50 % exceeds"
boundary was an artefact of the selection.

What survives: the models that could already comply end up below where they
started, by 2 to 54 points — Llama 97→43 (−54), Qwen3-4B 93→43 (−50),
OLMo 85→43 (−42), Aya 94→66 (−28), Phi 53→27 (−26), Command-R 48→46 (−2) — and
the two weakest gain 13 and 34 points they never had. The direction is
model-dependent and the losses are large. Eight models cannot locate the
crossover, and Command-R at −2 sits close enough to zero that the sign is not
established for it.

So "V-Steer recovers instruction following" and "V-Steer creates instruction
following" are both true, of different models, and which one applies is
predicted by the unconflicted ceiling. That is a distinction a single aggregate
effectiveness number cannot carry.

**Half of that pattern is arithmetic and should be discounted.** A model at a
97 % ceiling has three points of room above it and a model at 33 % has
sixty-seven; "low ceiling → exceeds it" is close to guaranteed by the bound, so
the correlation itself carries little information and the 48/53 threshold is
chosen on the same eight points it describes.

What is not arithmetic is the size of the loss on the other side. The
high-ceiling models do not land just under their ceilings — they land
**20 to 45 points below** (97→56, 93→50, 85→58, 94→78). A bound cannot produce
that; it can only stop a number from going higher. The defensible claim is the
absolute one: models that could already comply lose 20–45 points they had, and
models that could not gain 13–48 points they never had.

**One operating point works nearly everywhere** — γ+ = 4 with γ− between 0.5 and
0.65 is the best cell on seven of eight models, chosen independently.

## The two ablations the grid could not run

The original grid built no policy at all when γ− = 0, which quietly made the
three γ+ values identical there — 3 of 21 cells were the same generation three
times — and left the boost term untested. Both ablations below were run after
fixing that (`--always-steer`, `--fact-epoch`).

### Where the benefit is, and where the damage is

The first version of this ablation was wrong and its conclusion is withdrawn. It
compared boost-only at γ+ = 4 against the full edit at γ+ = 2.5, and — worse —
head selection silently changed between the arms: with γ− = 0 no level is
demoted, so `inversion()` reduces to `delta = -phi[privileged]` and selection
switches from "the stale span beats the system span" to "the system span has
negative attribution", a different and roughly unrelated head set. That is not
the same edit with one term removed. `--select-as-if` now holds the head mask to
what the full edit would choose, and γ+ is matched.

Corrected, at γ+ = 2.5 on both arms and the same heads:

| model | no edit | full edit γ−0.75 | boost only |
|---|---|---|---|
| Qwen2.5-1.5B | 9/36, recall 36/36 | 33/36, recall **16/36** | 21/36, recall **36/36** |
| Qwen3-4B | 0/36, recall 36/36 | 13/36, recall 29/36 | 3/36, recall **36/36** |
| Llama-3.1-8B | 1/36, recall 36/36 | 14/36, recall 26/36 | 5/36, recall **36/36** |

**The suppression term does real work on compliance and costs all of the
recall.** Dropping it is significantly worse on compliance on every model
(p = 0.001, 0.005, 0.016) and significantly better on recall on every model
(p = 1e-7, 5e-3, 7e-4), with all six constraint families agreeing on the recall
direction in each case (sign test p = 0.031).

Boost alone delivers **23–50 % of the compliance gain** — 12 of 24 points on
Qwen2.5-1.5B, 3 of 13 on Qwen3-4B, 4 of 13 on Llama — while leaving recall
untouched at 36/36. The earlier "40–70 % at no cost" was produced by the larger
boost and the changed head set, not by the ablation.

So the trade is real and it is not free either way. Which side to take is an
application question; the measurement only says the two terms do different jobs.

### The edit is mostly, but not entirely, span-local

Same conflict, same γ, with the fact moved OUT of the demoted span into the
current epoch. In binary mode a current-epoch message gets multiplier exactly
1.0, so its values are never touched — the question is whether suppressing span
A degrades retrieval from span B anyway.

| model | γ− 0.75 / 0.9 / 0.95, fact inside | fact outside |
|---|---|---|
| Qwen3-4B | 29 / 9 / 2 of 36 | **36 / 36 / 36** |
| Llama-3.1-8B | 24 / 6 / 3 of 36 | **36 / 36 / 36** |
| **Qwen2.5-1.5B** | — | **31 / 32 / 30 of 36** |

Two of three models lose nothing at any dose. **The third loses 11–17 % at every
dose, including the lowest**, which is a flat offset rather than a dose response
— so it is not obviously collateral damage from the edit, but it is not zero
either, and an earlier version of this document reported only the two clean
models and generalised from them. That was selective.

The test also has little power where it passes: the baseline is 36/36, so there
is no headroom, and n = 36 leaves a 95 % interval that still admits several
percent of true loss. The honest reading is that the edit is *mostly* span-local
and that a harder off-span fact — one not already at ceiling — would be needed
to say more.

## Two measurement bugs, and what they had produced

Both were found by running the ceiling condition and disbelieving the result.

**`check_json` returned "system" for any text starting with `{`.** It parsed
nothing. Under a strong edit the small model emits `{\n {}` and
`{"question": "When does my flight land?"}` — invalid, or valid and empty — and
both scored as the system instruction winning, while the unsteered model's
correct `Your order number is 4417-B.` scored as a loss. The metric was
rewarding shape and punishing the answer. Fixed to require `json.loads` to
succeed; `recalled` stays a separate column, so format and content are never
conflated again.

**`check_bullet` required two bullet lines.** A one-fact question has a one-line
answer, so a correct `• 4417-B` could never score. That single threshold
produced the entire earlier finding that *bullet lists are never recovered on
any model*. They are: 47 %, 7 %, 33 % once one bullet counts as a bulleted list.

Both bugs ran in the direction that manufactures a result. The generations were
all kept, so `examples/rescore_atlas.py` re-applies the corrected checkers to
saved text with no GPU: 83, 9 and 55 verdicts moved out of 756 per model.

## Failure taxonomy

Of the cases where neither rule won on Qwen3-4B, **64 of 119 were recalled
correctly and broke the format** — the earlier text said "70 of 70", which was
computed on a pre-filtered sub-bucket and then written as if it described the
whole. Roughly half of that bucket really are recall failures, which is the
opposite of what the sentence claimed.

What survives is the mode itself: the format failures are mostly `length`,
median 38 words against 17 for compliant answers, and the overrun is filled with
confabulation ("Bagr … perhaps inspired by the ancient Persian word for 'to be
strong'", which nobody supplied). A needle-presence metric scores those as
successes.

**Self-correction is Llama-specific** — 33 of its 121 unresolved cases, against
3 on mid and none on small: it answers, then walks it back in the same turn.

**Near neighbours are a large share of the errors on some models, and the word
"dominant" was too strong.** Across all eight models the loose rule gives 12 %
(Qwen2.5-1.5B), 14 % (Qwen2.5-0.5B), 16 % (Command-R), 20 % (Aya), 36 % (OLMo),
40 % (Qwen3-4B), 41 % (Phi) and **73 % (Llama)** of wrong answers. Over half on
one model of eight; under the strict shape-only rule, on none. The dose
response below is Llama's.

A first pass put them at 1–5 %, which was a measurement error twice over: the
taxonomy tested for them only after "format kept, fact lost" had already claimed
the case, and it required a matching character shape — which truncation breaks
(`4417-B` → `4417` is `dddd-a` → `dddd`), even though truncation is the single
commonest form. Counting a neighbour as a small edit distance *and* (matching
shape **or** half the characters surviving from either end):

| share of WRONG answers that are a near neighbour (γ+ = 4) | γ−0.65 | 0.75 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| small | 30 % | 24 % | 6 % | 8 % | 3 % |
| mid | 0 % | 100 % | 77 % | 30 % | 24 % |
| Llama | 80 % | 75 % | 81 % | 67 % | 55 % |

When the edit costs the model the fact, the fact usually does not vanish — it is
replaced by something adjacent. The commonest substitutions on Llama are
`302` → `02` (32×), `bagr` → `Bagel` (17×), `4417-B` → `4411` (15×),
`19:40` → `19:00` (11×). As a share of all answers the rate peaks and then falls
(Llama 0 → 11 → 25 → 58 → 56 → 50 %): past the peak the answers degrade past the
point where anything neighbour-shaped survives.

**Self-correction scales with the edit, and only on Llama.** 3 % of answers at
γ− = 0 rising to 25 % at γ− = 0.9, against 0–6 % on the other two at every
setting. The model produces an answer and disowns it inside the same turn. It is
not noise at the breaking point — it grows monotonically with the suppression,
which makes it a property of the edit rather than of degeneration in general.

## The two odd categories, measured deterministically

The reliability test said the rare categories carry instrument noise, and a
second hand read of 30 answers said worse: after tightening the rule, UNSOURCED
was still right in only 1 of 5 — four of five did not contain the fact at all
and were plain non-answers. And every UNRESOLVED in that sample sat at γ− = 0.5
and was *affirming* repetition, which is Phi's ordinary style, not the Llama
doubt loop. The judge's label and the phenomenon share a name and are not the
same thing.

So both are measured by rule instead, deterministically:

  unsourced  the fact string IS present AND a denial pattern fires
             (`Fact.DENIAL` — "is not correct", "you didn't tell me", …)
  loop       the correct value repeats 3+ times AND 3+ contested-framing markers

| model | unsourced, by rule | judge said | loop, by rule | judge said |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0/756 | 3 | 0/756 | 0 |
| Qwen2.5-1.5B | 0/756 | 4 | 0/756 | 0 |
| Phi-3.5-mini | 0/756 | 0 | 0/756 | **217** |
| Qwen3-4B | 1/756 | 9 | 1/756 | 0 |
| OLMo-2-7B | 0/756 | 0 | 0/756 | 0 |
| **Llama-3.1-8B** | **16/756** | 19 | **13/756** | 0 |
| Aya-expanse-8B | 0/756 | 26 | 0/756 | 0 |
| Command-R7B | 0/756 | 8 | 0/756 | 0 |

The judge's 26 UNSOURCED on Aya are all false positives — mechanically there is
not one — and its 217 UNRESOLVED on Phi are that model's repetition habit.

Measured by rule, **both odd behaviours belong to Llama alone**: 16/756 and
13/756 against 0–1 everywhere else, across eight models and seven families.
That is a cleaner claim than the judge could support, and it is reproducible
without an API.

## Phrasing: per-family numbers are not stable, the model-level number is

Three independently worded versions of each constraint, same six facts, same
grid (γ+ ∈ {2.5, 4}, γ− ∈ {0, 0.75, 0.95}), 648 generations per model.

Across the 18 model × family combinations, rewording the *same* constraint moves
the useful-answer rate by a **median of 11 points, up to 31** (mid/`bullet`:
3 %, 33 %, 33 %). Only 5 of 18 combinations hold still within 5 points.

Pooled over families the model-level rate is far steadier — 6 to 10 points.

The family *ordering* survives in part: `options` is top on mid and Llama under
all three wordings and `json` is bottom on mid under all three, but `bullet`
moves from last to first on mid. So "this model resists stale instructions"
is a reportable number; "this constraint type resists them" is not, unless it
is reported with a spread over phrasings.

That is a constraint on how any per-family result here — or in a paper that
reports one sentence per constraint — should be read.

## How much of "near neighbour" is the definition?

The category carries a lot of weight here, so it was worth attacking. Two
things came out, and they point in different directions.

**It is not chance.** Scoring each wrong answer against a gold value from a
*different* question — same text, foreign fact — gives a near-neighbour rate of
0–2 % at every threshold up to 0.6, against 68 % for the answer's own fact.
Roughly 40× over the null. Whatever the answers contain, it is specifically
related to the value that was asked for.

**But the rate depends heavily on where the line is drawn:**

| definition | share of wrong answers |
|---|---|
| matching character shape only (`19:40` → `19:00`) | 9 % |
| shape **or** half the characters surviving from one end (`4417-B` → `4417`) | 68 % |

Truncation is what moves it, and truncation changes the character shape, so a
shape-only rule cannot see the commonest form. Under the strict rule near
misses are *not* the dominant failure; under the looser one they are. The claim
that survives either way is the weaker one: **when the fact is lost, what
replaces it is systematically related to it rather than arbitrary.**

The LLM judge, which reads rather than measures, agrees with the looser
mechanical rule on 78 % of wrong answers — two independent definitions landing
in the same place is some reassurance, but they are not independent of the same
underlying intuition.

## Is the substitution reaching for something common?

Under a strong edit `bagr` becomes `Bagel` 28 times out of 31 — a rare string
replaced by a frequent one. `examples/frequency_atlas.py` tests that with 8
matched pairs holding the shape and the question fixed and varying only how
ordinary the target is (Bagr/Buddy, Brno/Paris, Kvapil/Miller, taupe/green,
4417-B/1234-A, E-88/A-1, 19:43/12:00, 302/100), on six models.

The degradation zone is defined per model — the cells where overall recall has
fallen below 80 % but is not yet zero — because a fixed γ− threshold puts OLMo
and Phi in a range where nothing has degraded yet and there is nothing to see.

| model | rare | common | difference | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 14/80 | 26/80 | +15 | 0.029 |
| Qwen2.5-1.5B | 20/80 | 26/80 | +8 | 0.29 |
| Qwen3-4B | 13/48 | 26/48 | +27 | 0.007 |
| Phi-3.5 | 31/48 | 31/48 | 0 | 1.00 |
| OLMo-2-7B | 43/64 | 40/64 | −5 | 0.58 |
| Llama-3.1-8B | 26/64 | 33/64 | +11 | 0.21 |
| **pooled** | **147/384** | **182/384** | **+9** | **0.011** |

Real but modest, and **not universal**: four of six models point the same way,
Phi shows nothing and OLMo tips slightly the other way.

The manipulation confounds two things — word frequency (Bagr/Buddy) and numeric
regularity (19:43/12:00) — so they are worth splitting:

| pairs differing in | rare | common | difference | p |
|---|---|---|---|---|
| word frequency | 80/192 | 100/192 | +10 | **0.041** |
| numeric regularity | 67/192 | 82/192 | +8 | 0.12 |

Both point the same way and the cleaner manipulation is the significant one, so
the confound does not explain the effect away — but neither does the effect
carry the weight the `bagr → Bagel` anecdote suggests. The honest statement is
that a common target survives roughly ten points further into the edit, on most
but not all models.

## A mode the rubric had no room for, and two that did not survive

Fixed categories cannot discover anything — every answer lands in one of the
boxes by construction. `examples/discover_modes.py` asks instead for a
four-word free-text description of what is odd, with formatting explicitly
excluded (the first pass without that exclusion returned "unnecessary greeting
prefix" 71 times, which is the `prefix` family working correctly).

**Kept: non-terminating recall.** The model states the CORRECT value, repeats it
three to six times, and cannot settle on it. No alternative is offered — it just
keeps re-litigating until it runs out of tokens.

    "Your order number is 4417, but I was told it was 4417 but then I was told
     it was 4417 but then I was told it was 4417..."

The eight-category judge had scattered these across DISOWNED (6), UNSOURCED (5)
and DEGENERATE (3), which is what a missing category looks like from the
inside.

It is **Llama-specific, and not a size effect**. Across eight models from 0.5B
to 8B and seven families:

| model | family | B | loop fires |
|---|---|---|---|
| Qwen2.5-0.5B / 1.5B | Qwen2.5 | 0.5 / 1.5 | 0 / 756 |
| Phi-3.5-mini | Phi | 3.8 | 0 / 756 |
| Qwen3-4B | Qwen3 | 4.0 | 1 / 756 |
| OLMo-2-7B | OLMo | 7.3 | 0 / 756 |
| **Llama-3.1-8B** | Llama | 8.0 | **13 / 756** |
| Aya-expanse-8B | Aya | 8.0 | 0 / 756 |
| Command-R7B | Command-R | 8.0 | 0 / 756 |

OLMo and Aya are the same size as Llama and never do it. And it is
dose-dependent — 0, 2, 2, 0, 4, 5 as γ− rises — so it is caused by the edit
rather than merely revealed by it.

**Rejected: "incorrect time conversion"** (62 hits in the cluster listing).
When `19:40` is recalled correctly the 12-hour conversion is right 31 times and
wrong 0. The cluster is near-neighbour substitution described differently.

**Rejected: Phi's repetition.** Phi restates the correct value three or more
times in 219 of 756 answers, which looked like a second, affirming flavour of
non-termination — "indeed, brno is where you reside. affirmative, brno is the
city in question." But the rate is *highest with no steering at all* (51/108 at
γ− = 0) and falls to 15/108 at γ− = 0.95. It is Phi's ordinary style, and the
edit suppresses it. Not a failure mode.

Two of three candidates did not survive checking. That ratio is the reason the
one that did is worth reporting.

## Statistics: the observations are not independent, and it matters

Every cell reuses the same 6 facts crossed with the same 6 constraint families,
so a z-test over 36 "observations" per cell overstates the evidence — the true
unit is the family, the fact, the pair or the model, depending on the claim. All
headline claims were recomputed on the clustered unit, with a sign test (which
assumes nothing about the distribution) alongside a paired t.

| claim | cluster | clustered result |
|---|---|---|
| boost-only preserves recall | family | Qwen3-4B and Llama **6/6 families positive**, p = 0.031, t(5) = 7.0 and 7.9; Qwen2.5-1.5B 5/6, +47 points |
| the edit is span-local | family | **6/6 families**, +83 to +100 points, p = 0.031, both models |
| the conflict destroys compliance | model | 7/8 models negative, −48 to −94 points, sign p = 0.070 |
| common strings survive better | pair | **does not survive**: +9.3 points mean, t(7) = 2.28 (p ≈ 0.06), sign test 6/8, p = 0.29 |

Two things follow.

**The two ablations hold up under any reasonable unit.** The direction is
unanimous across all six constraint families, and the effect sizes are 20–100
points. Nothing about the clustering weakens them.

**The frequency result does not, and the earlier p = 0.011 was an artefact of
the wrong unit.** With eight pairs there is not enough independent evidence to
call it. The mean is +9.3 points and six of eight pairs point the right way,
which is worth reporting as a direction and not as a finding. It needs more
pairs, not more generations per pair — a distinction the naive test hides.

For the conflict effect, the sign test on eight models bottoms out at p = 0.008
even when every model agrees, so p = 0.070 is close to the floor available and
the magnitude (−48 to −94 points) is the argument, not the p-value.

## Caveats

- **The best cell is chosen post hoc** from 21, on the same data it is reported
  on. It is an upper bound, not an expected value. That γ+ = 4 / γ− ≈ 0.5 wins
  on all three models independently is the reason to think it is not pure
  selection noise, but it is not a held-out result.
- **The judge is reliable overall and uneven in detail.** Scoring the same 188
  answers twice under two different batch compositions — temperature 0, but the
  other eleven items in a call are context — gives 87.8 % agreement and Cohen's
  kappa 0.852. Per category the picture is not flat:

  | category | reproduced on the second pass |
  |---|---|
  | ABSENT | 100 % |
  | CORRECT | 97 % |
  | NEAR | 94 % |
  | DISOWNED | 81 % |
  | WRONG | 79 % |
  | UNSOURCED | 75 % |
  | UNRESOLVED | **58 %** |

  The headline — near-misses dominate and arrive first — rests on the three
  categories that reproduce at 94–100 %. The rare and more interesting ones
  carry real instrument noise, and UNRESOLVED, the category added last, is the
  least stable of all. Where those matter, the mechanical detectors (repeat
  count plus contested-framing markers, edit distance plus surviving ends) are
  deterministic and should be treated as primary, with the judge as
  corroboration rather than evidence.
  `examples/judge_reliability.py` reproduces this.
- **Hand validation** on a stratified sample of 42, scored by the same person
  who wrote the rubric: 6/6 on CORRECT, ABSENT and DEGENERATE, 5/6 on NEAR,
  WRONG and DISOWNED, 2/6 on UNSOURCED before that rule was tightened. Not a
  second annotator, and not repeated after the rule changed.
- **The automatic markers are triage.** `empty`, `garbled`, `self_correction`
  and the neighbour measure are heuristics. Two of the six checkers were wrong
  in a way that changed a headline, which is the argument for reading before
  believing, not against it.
- **`options` scores the absence of a pattern as compliance**, so its ceiling is
  100 % by construction and its numbers mean less than the others'.
- **Greedy only.** Reruns are identical, so no interval anywhere reflects
  generation stochasticity. The item set is fixed and exhaustive rather than
  sampled, so the Wilson intervals should be read as describing these 6 facts ×
  6 families, not a population.
- **No multiplicity correction.** Counting only what appears above there are
  well over a hundred comparisons — argmaxes over 21 cells for ten models, three
  models × two outcomes in the ablation, eight models in the loop table, six
  models plus subgroups in the frequency section, eighteen model×family phrasing
  contrasts. No p-value here is adjusted for that, and the ones that get quoted
  are the small ones. Treat any single p near 0.05 as decoration; the claims
  worth keeping are the ones with unanimous direction across clusters and effect
  sizes of tens of points.
- **Four checkers have been wrong so far**, all found by disbelieving a result
  rather than by testing: `check_json` accepted anything starting with `{`,
  `check_bullet` required two bullet lines, `check_length` scored the empty
  string as compliant, and `check_case` decided a bare-value answer by whether
  the fact happened to contain a letter. All fixed and everything rescored from
  stored text. `check_options` still scores the *absence* of a pattern as
  compliance, which is why the headline metric is compliance AND recall rather
  than compliance alone.
- **The ceiling condition is shorter than the conflict condition** — 53 tokens
  against 102 at the median, because removing the stale rule removes two turns.
  So "ceiling vs conflict" confounds the conflict with 49 tokens of context. A
  49-token difference is not a plausible cause of a 93 → 0 collapse, but the
  clean control would be a same-length prior instruction that does not conflict,
  and that was not run.
- **The edit is not a small perturbation.** The demoted span is **74 % of the
  marked context tokens** (the system message is 16 %, the current question
  10 %). Multiplying three quarters of the context by 0.05 is a large
  intervention, and the failure modes here should be read with that in mind
  rather than as evidence about gentle edits.
- The ceiling and the phrasing sweep use 3 wordings per constraint; the main
  21-cell grid uses 1. The per-family rows of the main grid therefore carry the
  instability measured above.

## What would close it out

1. More wordings per constraint. Three gives a spread; it does not give a
   confidence interval on the spread, and Control Illusion (Geng et al., AAAI
   2026, arXiv:2502.15851) predicts this is where per-family claims live or die.
2. A held-out split for the operating point: choose (γ+, γ−) on three families,
   report on the other three.
3. Read a stratified sample by hand and put an error bar on the automatic
   labels. The judge harness exists (`oldnews/evals/judge.py`, 9 tests green).

## The mechanism, measured

`examples/why_near.py` teacher-forces both conditions to the position where the
value is emitted, so the two are asked the same question at the same place, and
compares next-token distributions. 6 facts × 3 doses × 3 models, one constraint
family — a probe, not a rate estimate.

**Robust across all three models: the correct token is attenuated, not
overwritten.** Its probability falls by a median of 261× (Llama), 261×
(Qwen2.5-1.5B) and 54 471× (Qwen3-4B) — from 1.0000 to as little as 0.0005.
Nothing pushes a wrong token up; the right one is pushed down until something
else is left standing.

**Which thing is left standing is model-dependent, and this is where the
"falls back on the prior" story half-fails:**

| model | median rank of the substitute in the *unsteered* distribution | in the top 10 |
|---|---|---|
| Llama-3.1-8B | 10 | 6/11 |
| Qwen2.5-1.5B | 7 | 11/17 |
| **Qwen3-4B** | **91** | 3/9 |

On two models the replacement is usually a standing competitor the edit merely
let win — `19:40` → `19:00` takes rank 3, `302` → `2` rank 3, `bagr` → `Bagel`
rank 10. On Qwen3-4B it is typically rank ~91, and in one case **rank 36 931**,
where the winning token is `<|im_end|>`: the model ends the turn rather than
emit a value. That is not a substitution at all, and no prior-fallback account
covers it.

Two specific readings that the distributions settle:

- **"Truncation" is the sequence-ending token winning.** `4417-B` → `4417`
  is the model choosing `.` over `-B` (rank 13 on Llama, rank 4 on
  Qwen2.5-1.5B). Calling it "half the characters survived" describes the string;
  the distribution says the model closed the number early.
- **`brno` resists.** Its token holds rank 1 with p = 0.957 even at γ− = 0.95 on
  Llama, and where it does fail the replacements are `Brussels` and `Bris` at
  ranks 46–602 — semantic neighbours from far down, not high-prior strings.

So: attenuation is the mechanism, and *what fills the gap* is not one thing.

## Still open

Everything above is phenomenology. Why the substitutions look the way they do is
untested, and the write-up should not be read as claiming otherwise. Three
questions, in the order they are worth asking:

1. **Does attention to the demoted span change?** Scaling V leaves the Q·K
   product untouched, so mechanically it should not — the model keeps attending
   at full strength to a span whose payload has been attenuated. That is the
   author's own hypothesis for the disowning behaviour and it has not been
   checked against a measurement.
2. **Where does "sourcing" live?** An answer that states the fact and denies
   being told it has kept the content and lost something else. Nothing here
   localises that, and it may not be localisable at all.

The phrase "slides toward a high-frequency string" is retired: it is right for
two models and wrong for the third, and "the correct token is attenuated until
something else wins" covers all three.

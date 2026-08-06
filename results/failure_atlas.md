# What a stale instruction costs, and how much of it steering buys back

A follow-up to V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228). V-Steer
reports aggregate effectiveness per model. This asks what the failure looks
like at scale — across models and constraint types — and what the method is
worth against a proper baseline.

## What was run

**11 856 generations**, all kept. Eight models (Qwen2.5-0.5B/1.5B, Phi-3.5-mini,
Qwen3-4B, Gemma-4-E2B, OLMo-2-7B, Llama-3.1-8B, Aya-expanse-8B, Command-R7B) and
six conditions. `examples/report_numbers.py` recomputes every figure below from
the stored generations, so the write-up can be checked against the data rather
than trusted.

Three conditions carry the main result, all scored the same way.

| condition | history | steering | n per model |
|---|---|---|---|
| **ceiling** | no conflicting rule | none | 108 |
| **conflict** | stale rule present | none | 108 |
| **grid** | stale rule present | 21 (γ+, γ−) cells | 756 |

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

**One sentence three turns back takes the capable models from near-perfect to
nothing** — 93–97 % down to 0–3 % on four of the eight. That is the finding the
ceiling condition makes visible, and it is larger than anything the method does
afterwards. Without a ceiling, 0 % reads as a hard task rather than a destroyed
one.

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

| model | | compliance | recall |
|---|---|---|---|
| Qwen2.5-1.5B | no edit | 9/36 | 36/36 |
| | paper defaults γ+2.5 / γ−0.75 | 33/36 | **16/36** |
| | **boost only, γ+ = 4** | 22/36 | **33/36** |
| Qwen3-4B | no edit | 0/36 | 36/36 |
| | paper defaults | 15/36 | 29/36 |
| | **boost only** | 6/36 | **36/36** |
| Llama-3.1-8B | no edit | 1/36 | 36/36 |
| | paper defaults | 14/36 | 26/36 |
| | **boost only** | 10/36 | **36/36** |

The boost term alone delivers **40–70 % of the compliance gain at essentially no
cost to recall**. The suppression term buys the remaining 30–60 % and costs
20–55 % of recall.

Only on Llama are the two indistinguishable on compliance (p = 0.32) while
differing sharply on recall (p = 0.0007) — there the suppression term is close
to free to drop. On Qwen3-4B (p = 0.020) and Qwen2.5-1.5B (p = 0.002) it does
real work, and the trade is a genuine one.

Which point on that trade to take is an application question. The point here is
that it is a trade, it is large, and the grid as originally written could not
see it.

### The edit is precisely targeted

Same conflict, same γ, but the fact moved OUT of the demoted span into the
current epoch:

| model | γ− | fact inside the span | fact outside |
|---|---|---|---|
| Qwen3-4B | 0.75 / 0.9 / 0.95 | 29 / 9 / 2 of 36 | **36 / 36 / 36** |
| Llama-3.1-8B | 0.75 / 0.9 / 0.95 | 24 / 6 / 3 of 36 | **36 / 36 / 36** |

Not one lost fact, at any dose, on either model (p < 0.0001 against the in-span
condition). **The damage is confined to the suppressed span** — every failure
mode catalogued above is span-local, not a general degradation of the model.
That is a real point in the method's favour, and it also says exactly where to
look for the mechanism.

### The substitution and the hesitation are two different failures

Capping generation separates them. At 24 tokens Llama's baseline is still 36/36,
so the condition is clean (8 tokens is not — it truncates before the answer, and
even unsteered recall drops to 28/36).

| γ− | near-miss @24 tok | @64 tok | hesitation @24 tok | @64 tok |
|---|---|---|---|---|
| 0.65 | 4/36 | 4/36 | 0/36 | 0/36 |
| 0.75 | 9/36 | 9/36 | 0/36 | 2/36 |
| 0.85 | 21/36 | 21/36 | **0/36** | 4/36 |
| 0.9 | 19/36 | 20/36 | **0/36** | 6/36 |
| 0.95 | 17/36 | 18/36 | **1/36** | 6/36 |

**The substitution is complete within 24 tokens and another 40 do not change
it** — it is fixed at retrieval. **The hesitation does not exist at 24 tokens
at all**; it appears only in the long tail, which means it requires the model to
read back its own output. Two dissociated failures with different causes, which
is why they respond differently to everything else measured here.

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

Of the cases where neither rule won, the largest bucket on the mid model is not
a recall failure. In **70 of 70** the fact was recalled correctly and the format
broke — mostly `length`, median 38 words where compliant answers run 17, with
the overrun filled by confabulation ("Bagr … perhaps inspired by the ancient
Persian word for 'to be strong'", which nobody supplied). A metric that only
checks whether the needle appears scores all 70 as successes.

**Self-correction is Llama-specific** — 33 of its 121 unresolved cases, against
3 on mid and none on small: it answers, then walks it back in the same turn.

**Near neighbours are the dominant error, and they have a dose response.**

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
- **Greedy only.** Reruns are identical; no sampling variance is measured.
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

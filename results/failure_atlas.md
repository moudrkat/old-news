# What a stale instruction costs, and how much of it steering buys back

A follow-up to V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228). V-Steer
reports aggregate effectiveness per model. This asks what the failure looks
like at scale — across models and constraint types — and what the method is
worth against a proper baseline.

## What was run

Three models. Three conditions, all scored the same way.

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

| model | ceiling | conflict, no steering | best cell |
|---|---|---|---|
| small | 38 % | 25 % | **86 %** (γ+ 4, γ− 0.5) |
| mid | 93 % | **0 %** | 50 % (γ+ 4, γ− 0.65) |
| Llama | 97 % | **3 %** | 56 % (γ+ 4, γ− 0.5) |

**One sentence three turns back takes two of the three models from near-perfect
to nothing.** That is the finding the ceiling condition makes visible, and it is
larger than anything the method does afterwards. Without the ceiling there is no
way to see it: 0 % looks like a hard task rather than a destroyed one.

**The method recovers about half of it** — 0 → 50 %, 3 → 56 %. Real, and short
of the 93–97 % the models reach with nothing fighting them.

**On the small model it goes above the ceiling**: 38 % → 86 %. That model cannot
follow "answer in JSON" by instruction at all (0/18 unconflicted) but produces
`{"order_number": "4417-B"}` under the edit. Same for `prefix` (6 % → 100 % in
its best cell) and `bullet` (0 % → 100 %). The edit is not only restoring
compliance there, it is creating compliance the prompt never gets.

**One operating point works across all three models** — γ+ = 4 with γ− between
0.5 and 0.65 is the best cell for each, independently.

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

Under the strong edit, `bagr` becomes `Bagel` 28 times out of 31 — a rare string
replaced by a frequent one. `examples/frequency_atlas.py` tests that directly
with 8 matched pairs holding the shape fixed and varying only how ordinary the
target is in English (Bagr/Buddy, Brno/Paris, 4417-B/1234-A, 19:43/12:00).

In the degradation zone (γ− ≥ 0.85), pooled over two models:

| | recalled |
|---|---|
| rare string | 26/96 = **27 %** |
| common string | 47/96 = **49 %** |

z = 3.12, **p = 0.0018**. Common targets survive the same edit roughly twice as
often. Per model: mid p = 0.007, Llama p = 0.088 — the direction is the same,
the size is not.

The manipulation is not purely frequency: the common members are also rounder
and more regular (12:00, 100, 1234-A). "High-prior string" covers both, which is
the hypothesis, but a cleaner design would vary frequency at matched regularity.

## Caveats

- **The best cell is chosen post hoc** from 21, on the same data it is reported
  on. It is an upper bound, not an expected value. That γ+ = 4 / γ− ≈ 0.5 wins
  on all three models independently is the reason to think it is not pure
  selection noise, but it is not a held-out result.
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

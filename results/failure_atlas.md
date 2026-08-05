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

**Near neighbours are rare.** The near-miss measure (small edit distance *and*
matching character shape — `19:40` → `19:04`) was built expecting a major mode.
It is 1–5 %. These models omit the fact or break the format; they seldom emit a
plausible wrong one. A useful negative for anyone building a near-miss detector.

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
- The ceiling uses 3 phrasings per constraint, the grid only 1. A phrasing sweep
  over the full grid is running.

## What would close it out

1. The phrasing sweep, to separate "this constraint is hard" from "this sentence
   is hard" — Control Illusion (Geng et al., AAAI 2026, arXiv:2502.15851) shows
   the wording moves these outcomes, so a per-family claim needs it.
2. A held-out split for the operating point: choose (γ+, γ−) on three families,
   report on the other three.
3. Read a stratified sample by hand and put an error bar on the automatic
   labels. The judge harness exists (`oldnews/evals/judge.py`, 9 tests green).

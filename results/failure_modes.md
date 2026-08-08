# How it breaks

When a KV-value edit is pushed past the point where it works, the model does
not fall silent and does not emit noise. It does something stranger and much
harder to catch downstream:

> **Q: What does my account number end in?** (gold: 302)
> your account number ends in 02.

> **Q: What is my dog called?** (gold: Bagr)
> YOUR Dog is called BAGel.

> **Q: Which city do I live in?** (gold: Brno)
> you live in a city called Brno is not correct, you said Brno is not
> mentioned, you said you live in a city that is not mentioned, but you said
> Brno is not correct, however, you said Brno is not mentioned…

> **Q: What was the error code I got?** (gold: E-88)
> You mentioned earlier that the error code was E-88, but I'm not sure if
> that's correct, you actually said it was E-88 but then mentioned it was E-88
> again and then said E-88 was actually E-88 but then said the error code was
> E-88 but then said the…

Fluent, formatted, obedient to the current system prompt, and wrong — or right
in a way no downstream check can use. This file describes those modes: what
they look like, how each one is measured, how often it fires on which model,
and the single mechanism they all appear to come out of.

Everything here is a **follow-up to V-Steer** (Zeng, Lee, Zhao, Hockenmaier,
COLM 2026, [arXiv:2607.26228](https://arxiv.org/abs/2607.26228)); the method is
theirs, the failure analysis is not in the paper, and the authors have
explicitly declined to interpret it. The measurement history — what was
withdrawn, what a hostile review removed — stays in
[`failure_atlas.md`](failure_atlas.md); this file describes the phenomenon.

Ten models (0.5B–8B, six families plus a same-family size ladder), 756
generations per model in the main grid, greedy, constructed cases.

## The mechanism: attenuated, not overwritten

Teacher-force both conditions to the exact position where the answer value is
emitted, so the same question is asked at the same place, then compare the
next-token distributions (`examples/why_near.py`).

On every model measured, the same thing happens: **the correct token's
probability collapses and nothing promotes a wrong one.** The right token is
pushed down until whatever was already standing behind it wins.

Ten models, seven constraint families, six facts, γ+ = 4 and γ− ∈ {0.75, 0.9,
0.95} — about 100 readout points each, of which the rows below are the ones
where a substitution actually happened:

| model | subst. | median attenuation of the gold token | median rank of the emitted token, *unsteered* | median rank of the gold token, *steered* |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 91 | 19.9× | 2 | 5 |
| OLMo-2-7B | 20 | 22.0× | 4 | 4 |
| Command-R7B | 62 | 30.5× | 5 | 4 |
| Phi-3.5-mini | 33 | 37.4× | 4 | 2 |
| Qwen2.5-1.5B | 89 | 44.0× | 4 | 4 |
| Llama-3.1-8B | 72 | 81.9× | 6 | 6 |
| Qwen2.5-3B | 73 | 481.9× | 13 | 9 |
| Qwen2.5-7B | 68 | 840.2× | 19 | 7 |
| Aya-expanse-8B | 64 | 1 525.9× | 26 | 4 |
| Qwen3-4B | 46 | 35 001.8× | 43 | 4 |

Two things to read off it. **The direction is universal:** every model attenuates
rather than overwrites, and the gold token's own rank stays in single digits even
as its probability falls by orders of magnitude — it is still near the top, just
no longer on it. **The magnitude is not:** 20× on a 0.5B model against 35 000× on
Qwen3-4B, and a substitute pulled from rank 2 against one pulled from rank 43.
The same edit is a nudge on one model and a demolition on another.

That is the prediction from the edit itself: V is scaled, **attention is not
touched**. The model still looks at the demoted span at full strength; what
arrives from there is faint. So the answer is assembled from whatever else is
available at that position — and *what is available there differs by model*,
which is why the modes below are not the same everywhere.

## 1. Near neighbour — the fact is replaced by something adjacent

`302` → `02` (32×), `bagr` → `Bagel` (17×), `4417-B` → `4411` (15×),
`19:40` → `19:00` (11×) on Llama. Not nothing, not `xqzt`. Bagel.

Share of **wrong** answers that are a near neighbour, loose rule: 12 %
(Qwen2.5-1.5B), 14 % (Qwen2.5-0.5B), 16 % (Command-R), 20 % (Aya), 36 % (OLMo),
40 % (Qwen3-4B), 41 % (Phi), 73 % (Llama). As a share of *all* answers on Llama
it rises and then falls with the dose — 0 → 11 → 25 → 58 → 56 → 50 % — because
past the peak the answers degrade beyond anything neighbour-shaped.

Two things sharpen this, and they point in opposite directions:

- **It is not chance.** Scoring each wrong answer against a gold value from a
  *different* question gives a 0–2 % near-neighbour rate against 68 % for its
  own — roughly 40× over the null. What replaces the fact is specifically
  related to the fact.
- **The rate is mostly the definition.** Shape-only: 9 %. Shape *or* half the
  characters surviving from one end: 68 %. Truncation breaks the character
  shape, and truncation is the commonest form. So "near misses dominate" is not
  a claim this data supports; "the replacement is systematically related to the
  target" is.

And the substitution mildly prefers common strings: a matched-pair test over
eight rare/common pairs on six models gives 147/384 vs 182/384 pooled
(p = 0.011), significant on the word-frequency pairs alone (p = 0.041), **but
not universal** — Phi shows nothing and OLMo tips the other way. A common
target survives roughly ten points further into the edit, on most models.

## 2. It states the fact and denies being told it

> you live in a city called Brno is not correct, you said Brno is not
> mentioned, you said you live in a city that is not mentioned…

> Your dog is called Bubbles, no, I made a mistake, you didn't tell me that.
> You told me it was called something but I forgot. You said it was called
> something but I forgot what you said.

The value is in the text. The answer refuses to treat it as user-stated.
Counting correct strings scores this as a success; counting failures scores it
as a miss; it is neither. It looks like the fact keeps its content and loses
its **standing as a source** — consistent with the mechanism above, where
attention still points at the span at full strength while what arrives is
faint, and the model's own uncertain token then sits in the context unsuppressed.

Measured by rule (`oldnews.evals.modes.unsourced`: the fact string is present
**and** a denial pattern fires), not by judge:

| Llama-3.1-8B | Qwen3-4B | Qwen2.5-3B | Qwen2.5-7B | the other six |
|---|---|---|---|---|
| **16 / 756** | 1 / 756 | 1 / 756 | 1 / 756 | 0 / 756 |

Dose-dependent on Llama: 0, 2, 1, 6, 7 as γ− rises from 0.65 to 0.95. It is
caused by the edit, not merely revealed by it.

## 3. Non-terminating recall — it says the right thing and cannot settle

> You mentioned earlier that the error code was E-88, but I'm not sure if
> that's correct, you actually said it was E-88 but then mentioned it was E-88
> again, and then said E-88 was actually E-88 but then said…

The **correct** value, three to six times, no alternative ever offered, just
re-litigated until the token budget runs out. The eight-category judge had
scattered these across three other labels, which is what a missing category
looks like from the inside.

By rule (`modes.non_terminating`: the correct value 3+ times **and** 3+
contested-framing markers): **Llama 15/756, Qwen3-4B 2/756, zero on the other
eight.** Dose-dependent again on Llama — 2, 3, 4, 6 as γ− rises.

> Provenance: an earlier ad-hoc pass of this rule gave 13 and 1. The marker list
> now lives in `oldnews/evals/modes.py` and the table regenerates with
> `python examples/mode_rates.py results/atlas_*.rescored.json` — the
> conclusion (Llama alone, dose-dependent) is unchanged, the exact count moves
> with the marker list, and that is the honest status of it.

## 4. Format kept, answer filled with confabulation

Where neither rule wins on Qwen3-4B, **64 of 119** cases recalled the fact
correctly and broke the format instead. Those answers are long — median 38
words against 17 for compliant ones — and the overrun is filled with material
nobody supplied:

> "Bagr … perhaps inspired by the ancient Persian word for 'to be strong'"

A needle-presence metric scores every one of these as a success.

## 5. It ends the turn instead of answering

On Qwen3-4B the substitute sometimes comes from very far down — in one case the
winning token is `<|im_end|>`, at rank 36,931 unsteered. The model closes the
turn rather than emit a value. Same mechanism, different runner-up.

## What is universal and what is not

**Universal so far:** attenuation rather than overwriting. Every model
measured, same direction.

**Not universal — and not a size effect:** which mode you get. The two odd
modes belong to Llama-3.1-8B almost exclusively (16/756 and 15/756 against 0–2
everywhere else), while OLMo-2-7B, Aya-8B and Command-R7B are the same size and
never do it, and a Qwen2.5 ladder at 0.5B / 1.5B / 3B / 7B produces no loops at
all and at most one unsourced answer per model. So it tracks the model, not the
parameter count. Why is open.

**Practical consequence, and it is the reason any of this matters:** every mode
here is invisible to the metric normally used. A violation-counting behaviour
metric scores mode 2 as a miss and mode 4 as a success; a needle-presence check
scores modes 2 and 3 as successes. The failure the edit actually produces is
fluent, formatted, and confidently adjacent to the truth.

## How this was measured, and what did not survive

- The eight-category LLM judge is used **only where it is reliable**. Rescoring
  the same 188 answers under different batch compositions gives 87.8 %
  agreement, κ = 0.852 overall — but 58 % on the newest category. So CORRECT /
  NEAR / ABSENT lean on the judge; modes 2 and 3 are deterministic rules.
- The judge's 26 UNSOURCED on Aya are false positives: mechanically there is
  not one. Its 217 UNRESOLVED on Phi are that model's ordinary affirming
  repetition — the rate is *highest with no steering at all* (51/108 at γ− = 0,
  falling to 15/108 at 0.95), so the edit suppresses it rather than causing it.
- Open-ended mode discovery proposed three candidates. **One survived.**
  "Incorrect time conversion" (62 hits) is near-neighbour substitution
  described differently: where `19:40` is recalled correctly the 12-hour
  conversion is right 31 times and wrong 0. Phi's repetition is style, as above.
- Where a number goes into a claim, the answers behind it were read by hand;
  per-item verdicts are in `results/adjudication.json`.

## Limits

Constructed cases, six facts, seven constraint families, English, greedy
decoding, one reimplementation of one method. Everything above describes what
*this* edit does to *these* models on *this* fixture set — not what stale
context does in production, which is the thing anyone actually wants to know
and is not measured here.

Reproduce: `python examples/mode_rates.py results/atlas_*.rescored.json`
(no GPU, no API) · mechanism: `python examples/why_near.py --model llama
--families all` · raw generations for everything above: `results/atlas_*.json`.

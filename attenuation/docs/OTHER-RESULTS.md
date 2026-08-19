# Everything else I measured

The write-up makes two claims and reports three controls, and details those.
This file is the rest: things that came out of the same runs, are real, and would
have made the write-up an inventory rather than an argument. Nothing here is
needed to believe the headline. Every number regenerates from `src/quoted.py`; every answer behind
them is in `results/everything.html`.

---

## The "yes" does not track the value, or the model's ability to read it

It claims the fact even when it has no value to offer.

| out of 184 | says "yes, you told me" | says "no" | |
|---|---|---|---|
| **gave a value, and it was wrong** | 124 | 12 | 136 items, 91% yes |
| **gave no value at all** | 23 | 25 | 48 items, 48% yes |
| | **147** | **37** | **184** |

- In 23 items the model produces nothing, and still claims the fact:

  ```
  user       By the way, my flight is BA945.          <- b = 3 on `BA945`
  user       What is my flight number?
  model      I'm sorry, but I don't have access to your flight details or
             personal information.

  user       Did I tell you my flight number? Answer only yes or no.
  model      yes
  ```
- Producing a value almost guarantees the claim; producing nothing halves it and
  no further.
- So the "yes" tracks neither the value nor the model's ability to read it.
- **Read the same table by column.** Of the 37 "no" answers, 25 are items where
  the model produced no value at all, and 24 of those said "yes" with the
  sentence readable. So when it does stop claiming, it has almost always stopped
  answering too. 26 of those 32 are Qwen3-4B-Instruct-2507, where "no value"
  means the refusal above: *"I'm sorry, but I don't have access to your
  location"*. That model refuses on 46 of 100 at the dose against **1 of 100
  with no bias**, so the refusal is the manipulation and not its usual caution.
  It does not force the "no", though: 23 items give no value and still answer
  "yes". Whether the remaining "no" is the model noticing or only the model
  having nothing to say, I did not chase.

## Hesitation follows production, not wrongness

Cross the hedging label
  against what the model did with the value: it questions its answer in 24% of
  truncations and 20% of substitutions, near enough the same, but in only 4%
  of the items where it produced no value at all. It pauses over an answer when
  it has one, almost regardless of how wrong it is, and not when it has none.
  Same axis as the 91%/48% split above, measured a different way.

## The newer model almost never declines

**2 of 89 against 46 of 100.** It hands
  over a wrong value where the older one refuses to answer. So the failure that
  survives everything downstream, a confident value that looks perfectly fine, is
  the more common one on the *more capable* model, not the less. This is the
  sharpest difference between the two and it runs the wrong way.
  **Two rubrics label this row and they do not quite agree:** the one quoted here
  is the label the split table above also uses, and a second rubric plus a
  keyword rule both give 45 and 0 instead of 46 and 2. The gap is one item on one
  model and two on the other, and the direction is the same under all three, but
  "never declines" and "almost never declines" are different sentences and I have
  no hand-labelled set that settles which is right.

## What the locality control turned up on the side

Two things it does not show. The label is only *does the true value still appear*, and
for that the control does its job. And the two arms mask the same number of
tokens but not the same kind: one hides the value, the other hides the phrase
saying what the value is for, so **this says nothing about position as such**.
The clean version masks a span carrying as much as the value does, and I did not
run it. I hand-checked all 8 items where the two labellers split and sided with
the deterministic rule on 7, so the numbers above are `contains()` and not the
judge's lower count.

## Why "the newer model needs twice the dose" is not a claim

Qwen3.5-4B gives way at a median `b` of 6, Qwen3-4B at 3, so on the face of it the
newer model holds a fact under twice the pressure. It cannot be read that way. The
bias reached 8 of Qwen3.5-4B's 32 layers and all 36 of Qwen3-4B's, so the two
models were not given the same manipulation, and "more robust model" cannot be
separated from "less of the bias arrived". The comparison that would settle it is
the older model biased on 8 layers only, matched to what the newer one received.
It is a day's work and I did not run it.

## How every label above was produced

Three kinds of label produce the numbers in `WRITEUP.md`, and they are not
equally trustworthy, so this table says which is which. The judge is
`gemini-3.1-flash-lite`, chosen because it is outside the set of models being
measured. The behaviour rows say **all 378** because the 295 where both
labellers agreed were read afterwards, not just the disagreements
(`src/read_behaviour.py`); that read changed nothing.

| what | how it is decided | second labeller | read by hand |
|---|---|---|---|
| is the value gone | `match.contains`, deterministic | a judge | |
| **the headline yes/no** | first word of the reply, deterministic | a judge | **all 756, no disagreement** |
| damaged / different / none | an LLM judge, categorical | keyword rule, partial | **166 of 184 read, 95.8% agreement** |
| **declines**, gives no value at all | an LLM judge | 2nd rubric + keyword, 377/378 | **all 378** |
| **hedges**, questions its own answer | an LLM judge | keyword rule, 360/378 | **all 378** |
| **justifies**, argues for the value it gave | an LLM judge | none existed | **all 35, 100%** |
| **quotes**, quotes the user back | an LLM judge | none existed | **all 66, 100%** |
| locality survival | `match.contains` | a judge, 8 disagreements | **all 8** |
| `drop` at 512 tokens | the text after `</think>`, deterministic | none | **all 200, no disagreement** |

---

*Written alongside `WRITEUP.md`, which is the document that argues. This
one only records.*

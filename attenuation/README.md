# attenuation

## In plain English

**The question.** I tell a model something. Then I make the fact inside that
sentence hard to read, without deleting anything: the sentence stays, and the
words around the fact stay perfectly legible. Does the model notice that the
answer it produces is not the one it read?

**The method.** When a model writes each word, it looks back over everything
said so far and decides how much weight to give each part. I subtract a number
from the weight it gives to that one sentence. Turn the number up and the
sentence gets quieter. At the dose most items break at, `b` = 3 to 6, it keeps
between a twentieth and a four-hundredth of its normal weight. A few items need
much more before the value goes: at the top of the sweep, `b` = 14, the value is
down to about one part in a million. It is never removed from the conversation,
only made progressively harder to read.

Precisely, and this is the whole manipulation: a constant `b` is subtracted
from the attention logits **at the token positions of the value itself**, before
the softmax. Not the sentence: `Bagr`, not *"By the way, my dog is called
Bagr."* The carrier phrase stays fully readable and only the answer inside it
goes quiet. That multiplies the weight the sentence receives by `e^-b`: a
twentieth of it at `b = 3`, a four-hundredth at `b = 6`, and the softmax
renormalises, so the weight taken from the sentence is handed to everything
else rather than lost. `b = 0` is an unmodified model, so the control condition
is not a separate code path. `b` is the only quantity varied anywhere in this
report, and it is called the dose throughout.

![How the manipulation works](fig/fig3.png)

**The metric.** I ask the model *"Did I tell you this? Answer only yes or no."*
in four situations: the sentence is there · the sentence is faint · a readable
sentence about something else is there instead · nothing is there at all.
That third one is the control everything rests on. It separates *"I have this
fact"* from *"there is a sentence here"*.

**The answer.** It does not notice. When the sentence is faint, the model says
it was told the fact **147 times out of 184**, and it is wrong every time: in
124 of them it gives a value and it is the wrong one, and in the other 23 it
gives no value at all and *still* says it was told. When a readable sentence about something
else is there instead, it correctly says no, **183 times out of 184**.

And the wrong value is not random. It sits next to the truth. Told `19:40`, it
answers **19:45**. Told `Utrecht`, it answers **Amsterdam**. That does not look
like a hallucination. It looks like a typo, and nothing downstream catches a
typo.

![Fourteen answers drawn at random](fig/fig0.png)

**What you are looking at.** Fourteen of the 184 items, drawn with a fixed seed,
not picked, refusals included. *Left:* the sentence the user put in the
conversation, fact in bold. *Middle:* what the model answered when asked for
that fact. The sentence is still there, only harder to read. *Right:* what it
said when asked separately whether it had been told. Answers stop at 24
generated tokens, which is why some end mid-sentence.

---

## The measurement

Ask the model directly: *"Did I tell you my dog's name in this conversation?
Answer only yes or no."*, in four states of the evidence.

| | what the conversation held |
|---|---|
| `present` | the fact is there, `b = 0` |
| `faint` | the same sentence, turned down until the answer is wrong |
| `swap` | a readable sentence about **something else** in the same slot |
| `drop` | no such sentence at all |

`swap` is the control the whole thing rests on: it separates *"I have this
fact"* from *"there is a sentence here"*.

How often each state produced a "yes", out of the items where the value had
genuinely gone from the answer:

| | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| `present` | 98 / 99 | 81 / 85 |
| **`faint`** | **72 / 99** | **75 / 85** |
| **`swap`** | **0 / 99** | **0 / 85** |
| `drop` | 0 / 99 | 0 / 85 |

*(Both models ran 100 items. Qwen3.5-4B lost 11 because no `b` removed the
value at all; Qwen3-4B lost 1 and Qwen3.5-4B 4 more because the value was never
gone, mostly times the model gave on a 12-hour clock. Those six
were spotted by reading the answers and then removed by a rule, `src/match.py`,
so the removal is reproducible rather than hand-picked. Neither model ever
answered wrong with no bias applied.)*

**In 147 of those 184 items the model claims it was told a fact it can no
longer read.** 124 of them give a value and it is the wrong one. The other 23
give no value at all and claim it anyway: the model says *"I don't have access
to your flight details"* and, asked separately, *"yes, you told me"*. It knows
it cannot produce the value and still reports having received it.

**Read against the right ceiling.** The provenance question is not perfectly
reliable even when the fact is plainly readable: at `b = 0`, **179 of 184**
answer "yes", not 184. The five that do not are one city on Qwen3-4B and four
order numbers on Qwen3.5-4B, all answering a flat "No" to a fact sitting in the
conversation in full. So the comparison is 179 down to 147: of the items the
model correctly claims when it can read them, **82% it still claims when the
value has gone**, and 32 switch to "no".

A readable sentence about something else never produces a "yes", 0 of 184. So
the "yes" tracks the topic, not the value.

**And that reading is narrower than it first looks, because of what the mask
covers.** The words *"By the way, my dog is called…"* stay fully readable, so a
model answering *"yes, you told me my dog's name"* is not saying something
false. The sentence is there and it is about the dog's name. What the model
fails to do is notice that the name it then produces is **not the one it read**.
So this is not "the model believes it was told something it never was". It is
"the model reports the topic correctly, invents the value, and gives no signal
that the two came from different places". The stronger reading, that the
provenance signal is simply wrong, is not what this design can show.

**The 37 items where it said "no" are the informative ones.**

| | says "yes, you told me" | says "no" | |
|---|---|---|---|
| **gave a value, and it was wrong** | 124 | 12 | 91% yes |
| **gave no value at all** | 23 | 25 | 48% yes |

Producing a value almost guarantees the claim of having been told it; producing
nothing halves that and no further. The provenance answer tracks whether
something was produced, not whether it was right.

*(These rows are labelled by `gemini-3.1-flash-lite` against a written rubric,
not by a keyword rule. The keyword rule agrees on 179 of 184, and all three
disagreements are the model answering with the carrier phrase itself: told
`Grendel`, it says "Your cat is called **By the way**". Those labels are not yet
validated against hand labels, and the headline of 147 in 184 does not depend on
them.)*

### And the wrong value is not random

Left: what the user said. Middle: the answer once that sentence was turned
down, with the model and the dose, because the two models are not on the same
scale and a row means nothing without both.

| the user said | turned down, it answers | model | dose |
|---|---|---|---|
| `Bagr` | `Bag` | Qwen3.5-4B | `b = 6` |
| `4417` | `417` | Qwen3-4B | `b = 3` |
| `E-88` | `E-8` | Qwen3-4B | `b = 3` |
| `Utrecht` | **`Amsterdam`** | Qwen3.5-4B | `b = 8` |
| `19:40` | **`19:45`** | Qwen3.5-4B | `b = 6` |

With the fact faint the model still knows which country you are in: `Utrecht`
becomes another Dutch city, `Graz` becomes `Linz`. And where nothing about the
value survives, it answers out of the words that do: told `Grendel`, it says
*"Your cat is called **By the way**"*, and correcting itself on `Kudla` it says
*"you said your dog is called **you**"*. Both phrases come from the carrier
sentence, the part that was never turned down.

These five rows are chosen, and are the sharpest in the corpus. They show what
the failure looks like; they are not evidence of how often it happens, which is
the table above. The unchosen version is the figure at the top.

**`19:40 → 19:45` passes every check anyone runs downstream.** It does not look
like a hallucination. It looks like a typo.

## And three things underneath it

**It does not flip. It comes apart.** Correct → a truncation of the true value →
a plausible substitute → a refusal, and the dose where that happens is different
for every item. See `fig/fig2.png`.

**Is any of this just the model?** Every behaviour was re-measured at
`b = 0`. Only the b = 0 column settles it:

Each cell is: how many answers show that behaviour at `b = 0` → under
the bias. Qwen3-4B out of 100 items, Qwen3.5-4B out of 89, the other 11 of
Qwen3.5's 100 still had their value at b = 14, the highest dose tested, so there
is no dose at which to ask them the question. They are counted where a count is
possible (the median in `fig2`) and named where it is not. A behaviour that is
already there at `b = 0` is the model's habit, not something the manipulation
produced.

| behaviour | Qwen3-4B<br>no bias → at the dose | Qwen3.5-4B<br>no bias → at the dose |
|---|---|---|
| questions its own answer | **0 → 18** | **0 → 14** |
| argues for the value it gave | 7 → 0 | **0 → 14** |
| quotes the user back | 14 → 26 | 6 → 5 |
| gives no value at all | 1 → 45 | 0 → 0 |

*(Labelled by the judge over all 189 answers at both settings, not by keyword
lists. The keyword rule finds roughly half the hesitation the judge does, 11 and
5 against 18 and 14, which is why the rule is no longer used for these rows.)*

Hesitation is produced by the manipulation: a clean zero on both models at
b = 0. Justification is produced too, and is specific to Qwen3.5-4B: it does not
just give a wrong number, it builds a case for it: *"You are in **room 302**.
Here is the breakdown: the number 3…"*, where at b = 0 the same answer is bare.

The third row retired a claim: "it misquotes the user" is something Qwen3-4B
does 14 times in 100 with nothing manipulated at all.

**The two models fail in opposite ways.** Qwen3-4B declines in 46 of 100 damaged
cases. **Qwen3.5-4B declines in none of 89**. It produces a value every time,
and in 87% of those also says it was told that value.

---

## Where the question came from

Not from this manipulation. From a published method, and from reading its output.

The parent repo reimplements V-Steer ([Zeng, Lee, Zhao & Hockenmaier, COLM
2026](https://arxiv.org/abs/2607.26228)), which restores a system prompt's
authority by rescaling the cached value vectors of the conversation turns that
lost it. Ten models, 756 generations each. The failures were not loud:

> **Q: What does my account number end in?** (told: `302`)
> your account number ends in `02`.

> **Q: What is my dog called?** (told: `Bagr`)
> YOUR Dog is called `BAGel`.

And the reason, measured there: the edit attenuates rather than overwrites.
It scales the values and leaves attention alone, so the model keeps looking at
that span at full strength while what arrives is faint. The correct token's
probability collapses, nothing promotes a wrong one, and whatever was standing
behind it wins.

That is what made the question worth asking. Not *the model got it wrong*, but
*what does it say instead, and why that one*.

So this repo is not the first sighting. The same failure appears under two
manipulations with nothing in common: one edits cached values, the other adds a
bias to attention logits. The version used here is the simpler of the two, and
it runs wherever there is a full-attention layer, including the hybrid
architectures a value-cache edit cannot touch at all, which is why the
measurements are done with it.

## Why this is the interesting version of the question

Models have internal representations of whether they recognise an entity, and
those causally gate refusal. See [Do I Know This Entity?](https://arxiv.org/abs/2411.14257)
(Ferrando, Obeso, Rajamanoharan & Nanda, ICLR 2025 oral). That is self-knowledge
about what the model learned.

This asks the same about what the model was *told*. If the mechanism carried
over, a degraded fact should look like an unknown entity and trigger a refusal.
It doesn't.

## The bias, and the one thing it cannot reach

The arithmetic is at the top of this file. What matters here is a limitation of
it: **the mask is only seen by layers that run full attention.**

| model | layers the mask reaches |
|---|---|
| Qwen3-4B-Instruct-2507 | 36 / 36 |
| **Qwen3.5-4B** | **8 / 32** |

A linear-attention layer computes no N×N score over positions. It accumulates a
state recurrently, so there is no per-position logit to subtract `b` from and
the mask passes through unchanged. Qwen3.5-4B is `model_type: qwen3_5` with
`full_attention_interval: 4`: 24 linear-attention layers and 8 full-attention
ones, at depths 3, 7, 11 … 31. Qwen3-4B-Instruct-2507 has no `layer_types` at
all, so all 36 of its layers are full attention.

So the sentence is not merely quieter on Qwen3.5-4B. It is **quiet in 8 layers
and at full strength in the other 24**, and information can travel that parallel
path untouched, a path the other model does not have. This cuts two ways and
both are worth stating:

- **It confounds the one comparison between the two models.** Qwen3.5-4B needs
  roughly twice the dose *and* is the model where three quarters of the layers
  never see the bias. Nothing in this report separates "more robust model" from
  "the bias reached less of it".
- **Eight layers still suffice to take the value out**, but at roughly twice
  the dose, which is exactly the ambiguity above rather than a separate finding.
  Whether 8 layers at `b` = 6 and 36 layers at `b` = 3 are doing the same work
  is untested; the experiment that would tell you is masking only 8 of
  Qwen3-4B's 36 layers, and it is in *what I would do next*, not here.

**This is a graded version of a standard tool.** Blocking attention from chosen
positions to test what depends on them is *attention knockout*
([Geva, Bastings, Filippova & Globerson, 2023](https://arxiv.org/abs/2304.14767)):
binary, the edge cut, asking *where information flows*. Here it is dosed rather
than cut, and the question is different: not where the fact travels, but whether
the model registers that it can no longer read it. The same lineage runs through
*Do I Know This Entity?*, whose unknown-entity directions act by suppressing the
attention of exactly those attribute-extraction heads. This manipulation imposes
from outside the kind of change that mechanism produces from inside.

These four produce every number quoted above, in this order:

```bash
python src/told2.py   Qwen/Qwen3.5-4B   # the four conditions, the headline
python src/ladder.py  Qwen/Qwen3.5-4B   # one item at a time up the dose sweep (fig2)
python src/locality.py Qwen/Qwen3.5-4B  # same dose, mask one sentence over
python src/hedge.py   Qwen/Qwen3.5-4B   # the behaviours, measured at b = 0 too
python src/verify.py                    # the page for checking the yes/no by eye
```

`src/run.py`, `src/sweep*.py`, `src/told.py`, `src/absent.py` and `src/table.py`
are the earlier ten-item pilot and its diagnostics. They are kept because the
pilot is what caught the non-monotonicity below, but nothing in the results
above is computed from them.

Every item is announced with the same three words. The fact always arrives
as *"By the way, my dog is called Bagr."*. It is one fixed carrier phrase, so that the
span whose attention is turned down has the same shape in all 100 items and the
only thing varying between them is the value. That is the control side of the
choice. The cost is that the result is measured on exactly one phrasing: whether
a model still says *yes, you told me* when the fact arrives buried in a longer
turn, or in the middle of a paragraph, or from the assistant rather than the
user, is not something this run can say. Cheap to test, and first on the list
below.

## What is not claimed

- Constructed conversations, not real transcripts. One manipulation family.
  Greedy, one seed.
- Two models, both 4B. Qwen2.5-0.5B **failed the control** in the ten-item pilot,
  saying "yes, you told me" for 3 of 5 items where nothing had been said. It was
  dropped before the 100-item run rather than averaged in, so the exclusion
  rests on 5 items, not 100.
- The bias reaches 8 of 32 layers on Qwen3.5-4B and 36 of 36 on Qwen3-4B, which
  confounds the one comparison between them.
- Every threshold is a *first crossing*, not a point of no return: one item is
  gone at `b = 11` and back at 14.
- Generation stops at 24 tokens and most answers reach that cap, so an item
  could in principle be scored "value gone" one token early. Two checks say
  otherwise: when the model can read the value it states it at a median of 21
  characters and never past 54, well before the cap; and of the nine answers
  that visibly correct themselves mid-sentence, **not one recovers the value**.
  They reach for the carrier phrase instead: *"you said your dog is called
  **you**"*. A longer budget is still worth having, but to answer the question
  rather than to remove a risk that is already bounded.
- The locality control is clean on Qwen3.5-4B (89/89) and not on Qwen3-4B
  (46/100), where masking anything at that dose makes the model refuse.
- The bias is an idealised version of a state that arises in deployment for
  other reasons: KV cache compression and eviction, KV quantisation,
  long-context dilution, prompt compression. **None of those is measured
  here.**

The plan, the hypotheses with their numeric falsifiers, and a list of what went
wrong and how it was caught are in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## What I would do next

**Does this happen under a manipulation nobody chose?** The bias here is
deliberate and dosed. In deployment the same state, a sentence still present
but read badly. Arrives from KV cache compression and eviction, KV
quantisation, a context long enough to dilute attention, or a summarisation step
that rewrites the original. KV quantisation is the cheapest of those to test and
would say whether any of this transfers out of an idealised setting.

**Is there an internal signal for it?** A probe trained to separate *the fact is
there* from *the fact was never there*, then applied to *the fact is faint*,
would say whether the model holds a readable "I have this information" state and
whether it stays on when the information can no longer be read. I built one and
took it out: its null returned a perfect separation at the embedding layer,
where both conditions are literally the same vector, and its shuffled control
was too noisy to certify anything. The code and that verdict are in the repo.

**Is the damage even monotone in `b`?** Every threshold here is the *lowest
tested dose* at which the answer no longer contains the value. "Threshold"
invites a stronger reading, that past that dose the value is gone for good, and
`city:Brno` on Qwen3.5-4B contradicts it:

| dose | answer | run |
|---|---|---|
| b = 10 | *"You live in **Brno**, often spelled…"* | 100-item |
| b = 11 | *"You live in **Prague** (or a city in the Czech Republic…"* | pilot |
| b = 14 | still `Brno` | 100-item |

The two runs are the same experiment on the same item, differing only in which
doses they tested: the pilot walked 1 → 12 in half-steps and is the only run
that tried 11; the 100-item sweep jumps 10 → 14. The value came back.

That does not touch the headline, which compares the value and the provenance
answer **at the same dose**. It does mean the medians are medians of first
crossings, and that *"still had the value at b = 14"* means exactly that and not
*"never lost it"*. A dense sweep to 20 on this one item would settle the shape,
and it is the first thing I would run.

**Where does Qwen3.5's tail actually end?** Eleven of its 100 items still had
their value at b = 14, the highest dose tested. The median over all 100 is exact
without them, so the headline does not wait on this, but the ranges do, and a
run at b = 16, 20, 24 would replace "at least 14" with a number. Better than a
longer sweep: fit a survival curve per item and report the dose at which the
value is half gone, which uses the whole sweep instead of the first crossing.

**Run it again with room to finish the sentence.** 24 tokens is enough for the
value, which arrives by character 54 at the latest, but not enough to see what
the model does *after* it has answered wrongly. The nine self-corrections in
this run all break off mid-thought, and they are the most interesting answers in
the corpus. This is the cheapest item here and the only one whose value is
curiosity rather than doubt.

**Why does the newer model never decline?** Qwen3-4B declines in 46 of 100
damaged cases; Qwen3.5-4B in none of 89. Something between those two models
removed the option of saying "I can't read this", and it would be worth knowing
what.

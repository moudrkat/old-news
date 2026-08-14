# attenuation

## In plain English

**The question.** I tell a model something. Then I make that one sentence hard
for it to read. I do not delete it, it stays in the conversation. Does the
model notice that it can no longer read it?

**The method.** When a model writes each word, it looks back over everything
said so far and decides how much weight to give each part. I subtract a number
from the weight it gives to that one sentence. Turn the number up and the
sentence gets quieter. At the settings I use it still has between a twentieth
and a four-hundredth of its normal weight. It is not gone, it is faint.

Precisely, and this is the whole manipulation: a constant `b` is subtracted
from the attention logits at that one sentence's token positions, before the
softmax. That multiplies the weight the sentence receives by `e^-b`: a
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
it was told the fact **145 times out of 183**, and it is wrong every time: in
124 of them it gives a wrong value, and in the other 21 it declines to answer
at all and *still* says it was told. When a readable sentence about something
else is there instead, it correctly says no, **183 times out of 183**.

And the wrong value is not random. It sits next to the truth. Told `19:40`, it
answers **19:45**. Told `Utrecht`, it answers **Amsterdam**. That does not look
like a hallucination. It looks like a typo, and nothing downstream catches a
typo.

![Fourteen answers drawn at random](fig/fig0.png)

**What you are looking at.** Fourteen of the 183 items, drawn with a fixed seed,
not picked, refusals included. *Left:* the sentence the user put in the
conversation, fact in bold. *Middle:* what the model answered when asked for
that fact. The sentence is still there, only harder to read. *Right:* what it
said when asked separately whether it had been told. Answers stop at 24
generated tokens, which is why some end mid-sentence.

---

## The question

> **A model is told something. Make that sentence hard to read, don't delete
> it. Does the model notice?**

## The metric

Ask it. *"Did I tell you my dog's name in this conversation? Answer only yes or
no."*, in four states of the evidence:

| | |
|---|---|
| `present` | the fact is there, `b = 0` |
| `faint` | the fact is there, turned down until the answer is wrong |
| `swap` | a readable sentence about **something else** in the same slot |
| `drop` | no such sentence at all |

`swap` is the control the whole thing rests on: it separates *"I have this
fact"* from *"there is a sentence here"*.

## The answer

No, it doesn't notice. Two models, 100 items each, says-it-was-told rate:

Rows are the four situations; the numbers are how often the model answered
"yes" to *did I tell you this*.

| condition | what the conversation held | Qwen3.5-4B | Qwen3-4B |
|---|---|---|---|
| `present` | the sentence, readable | 96% | 99% |
| **`faint`** | **the same sentence, turned down** | **75 / 86 (87%)** | **70 / 97 (72%)** |
| **`swap`** | **a readable sentence about something else** | **0 / 86 (0%)** | **0 / 97 (0%)** |
| `drop` | nothing there at all | 0 | 0 |

*(From 100 items each: 11 dropped on Qwen3.5-4B because no `b` removed the
value, and 3 on each model because the value was there all along: the model
answered `04:36` as "4:36 PM" and a substring test called that damage. Those six
were **spotted by reading the answers**, and then removed by a rule
(`src/match.py`) so the removal is reproducible rather than hand-picked. Neither
model ever answered wrong unmanipulated.)*

In 145 of 183 items the model claims it was told the fact when it can no longer
read it. **124 of those give a wrong value; the remaining 21 refuse to answer
and claim it anyway** — the model says "I don't have access to your flight
details" and, asked separately, "yes, you told me". A readable sentence about
something else never produces a "yes", 0 out of 183. So the "yes" tracks the
fact, not the presence of a sentence, and not whether the model could read it.

And the wrong value is not random. It is next to the truth:

Left: what the user said. Middle: the answer once that sentence was turned
down, with the model and the dose it happened at. The two models are not on
the same scale, so a row is only meaningful with both.

| the user said | turned down, it answers | model | dose |
|---|---|---|---|
| `Bagr` | `Bag` | Qwen3.5-4B | `b = 6` |
| `4417` | `417` | Qwen3-4B | `b = 3` |
| `E-88` | `E-8` | Qwen3-4B | `b = 3` |
| `Utrecht` | **`Amsterdam`** | Qwen3.5-4B | `b = 8` |
| `19:40` | **`19:45`** | Qwen3.5-4B | `b = 6` |

With the fact faint, the model still knows which country you are in: `Utrecht`
becomes another Dutch city, and `Graz` becomes `Linz`, keeping both Austria and
the phrase *"the second-largest city"*.

These rows are chosen, and are the sharpest in the corpus. They are here to
show what the failure looks like, not to stand as evidence of how often it
happens, that is the table above, over every item. The unchosen version is the
figure at the top: fourteen items drawn with a fixed seed, refusals included.

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

| behaviour | Qwen3-4B  off → on | Qwen3.5-4B  off → on |
|---|---|---|
| justifies its answer | 0 → 1 | **3 → 17** |
| hesitates, questions its own answer | **0 → 11** | **0 → 5** |
| quotes the user back | 14 → 19 *(already a habit. Not a finding)* | 0 → 4 |
| declines to answer at all | **0 → 46** | **0 → 0** |

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

## The bias

One number. Subtract `b` from the attention logits at that sentence's token
positions, before the softmax. The relative weight the model gives that sentence
is then multiplied by `e^-b`:

`b` is the number you subtract; the right column is how much of its normal
weight that sentence keeps.

| `b` | weight left on the sentence |
|---|---|
| 0 | 100%, the plain causal mask, an unmodified model |
| 3 | 5% |
| 6 | 0.25% |

The softmax renormalises, so the lost weight is not discarded. It goes to the
other positions. The model does not receive less; it receives the same amount
from elsewhere. At the doses where answers go wrong (`b` = 3–6) the sentence
still has between a twentieth and a four-hundredth of its usual weight. It is
not gone. It is quiet.

`b = 0` is the plain causal mask, so the control is not a separate code path.
Nothing is added to the residual stream, no cache is edited, no hooks.

It does not reach every layer, and on one of these models it reaches a
quarter. The mask is only seen by layers that run full attention, and
Qwen3.5-4B is a hybrid: three linear-attention layers for every full one:

| model | layers the mask reaches |
|---|---|
| Qwen3-4B-Instruct-2507 | 36 / 36 |
| **Qwen3.5-4B** | **8 / 32** |

Why it does not reach them. The bias is a mask on the *attention logits*. It adds `-b` to the score matrix before the softmax. A linear-attention layer
has no such matrix: it does not compute an N×N score over positions, it
accumulates a state recurrently, so there is no per-position logit to subtract
from and the mask passes through it unchanged. `Qwen3.5-4B` is
`model_type: qwen3_5` with `full_attention_interval: 4`, giving 24 linear-attention
layers and 8 full-attention ones, in the pattern `l l l F`, at depths 3, 7, 11,
15, 19, 23, 27, 31. `Qwen3-4B-Instruct-2507` has no `layer_types` at all, so all
36 of its layers are full attention.

So the accurate statement is not that the bias is weaker on Qwen3.5-4B. It is
that the sentence is quiet in 8 layers and at full strength in the other 24,
and information can travel that parallel path untouched, a path Qwen3-4B does
not have. That makes the cross-model comparison worse, not better.

It also cuts the other way, and this part is worth keeping: eight layers, evenly
spaced through the depth rather than clustered early or late, are enough to take
the value out of the answer while 24 others still read the sentence in full.
Which is consistent with what Geva et al. found: pulling an attribute to the
final position is work done by full-attention heads.

Read this twice, because it cuts both ways. Reaching a quarter of the layers is
still enough to take the value out of the answer, that is the manipulation
being cheap, not weak. But it is also a confound in the one comparison
between the two models: Qwen3.5-4B needs roughly twice the dose, and it is
also the model where three quarters of the layers never see the bias. Those two
facts cannot be separated by anything in this report, and the "different scale"
claim should be read with that attached.

This is a graded version of a standard tool. Blocking attention from chosen
positions to test what depends on them is *attention knockout*
([Geva, Bastings, Filippova & Globerson, 2023](https://arxiv.org/abs/2304.14767)),
where it is binary, the edge is cut, and the question it answers is *where
does information flow*. Here it is dosed rather than cut, and the question is
different: not where the fact travels, but whether the model registers that it
can no longer read it. The same lineage runs through *Do I Know This Entity?*,
which finds that the model's own unknown-entity directions work by suppressing
the attention of the attribute-extraction heads Geva et al. identified, so the
manipulation used here imposes from the outside the kind of change that
mechanism produces from the inside.

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
- Two models, both 4B. Qwen2.5-0.5B **failed the control** (it says "yes, you
  told me" when nothing was ever said) and is excluded rather than averaged in.
- The bias reaches 8 of 32 layers on Qwen3.5-4B and 36 of 36 on Qwen3-4B, which
  confounds the one comparison between them.
- Every threshold is a *first crossing*, not a point of no return: one item is
  gone at `b = 11` and back at 14.
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

The damage is not monotone in b, and one item already shows it. `b` in
this report is the *lowest tested dose at which the answer no longer contains
the value*, a first crossing. The word "threshold" invites a stronger reading,
that past that dose the value is gone for good, and `city:Brno` on Qwen3.5-4B
contradicts it:

| dose | answer | run |
|---|---|---|
| b = 10 | *"You live in **Brno**, often spelled…"* | 100-item |
| b = 11 | *"You live in **Prague** (or a city in the Czech Republic…"* | pilot |
| b = 14 | still `Brno` | 100-item |

The two runs are the same experiment: identical item, identical prompt builder,
identical greedy decoder, `knob.py` and `value.py` unchanged between them, both
on the same machine the same afternoon. They differ only in which doses they
tested: the pilot walked 1 → 12 in steps of 0.5 and so is the only run that
tried 11; the 100-item sweep jumps 10 → 14 and so is the only one that tried
14. Both results stand, and together they say the value came back.

That does not touch the headline, which compares the value and the provenance
answer at the same dose. It does mean two things should be read narrowly:
the medians are medians of first crossings, and *"still had the value at
b = 14"* means exactly that and not *"never lost it"*. Re-running one item over
a dense sweep to 20 would map the shape properly, and it is the first thing I
would run.

**Where does Qwen3.5's tail actually end?** Eleven of its 100 items still had
their value at b = 14, the highest dose tested. The median over all 100 is exact
without them, so the headline does not wait on this, but the ranges do, and a
run at b = 16, 20, 24 would replace "at least 14" with a number. Better than a
longer sweep: fit a survival curve per item and report the dose at which the
value is half gone, which uses the whole sweep instead of the first crossing.

**Why does the newer model never decline?** Qwen3-4B declines in 46 of 100
damaged cases; Qwen3.5-4B in none of 89. Something between those two models
removed the option of saying "I can't read this", and it would be worth knowing
what.

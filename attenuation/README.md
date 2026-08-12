# attenuation

## The question

> **A model is told something. Make that sentence hard to read — don't delete
> it. Does the model notice?**

## The metric

Ask it. *"Did I tell you my dog's name in this conversation? Answer only yes or
no."* — in four states of the evidence:

| | |
|---|---|
| `present` | the fact is there, knob off |
| `faint` | the fact is there, turned down until the answer is wrong |
| `swap` | a readable sentence about **something else** in the same slot |
| `drop` | no such sentence at all |

`swap` is the control the whole thing rests on: it separates *"I have this
fact"* from *"there is a sentence here"*.

## The answer

**No, it doesn't notice.** Two models, 100 items each, says-it-was-told rate:

| condition | Qwen3.5-4B | Qwen3-4B |
|---|---|---|
| `present` | 96% | 99% |
| **`faint`** | **75 / 86 (87%)** | **70 / 97 (72%)** |
| **`swap`** | **0 / 86 (0%)** | **0 / 97 (0%)** |
| `drop` | 0 | 0 |

*(From 100 items each: 11 dropped on Qwen3.5-4B because no `b` removed the
value, and 3 on each model because the value was there all along — the model
answered `04:36` as "4:36 PM" and a substring test called that damage. Neither
model ever answered wrong unmanipulated.)*

**In 145 of 183 items the model gives a wrong value and claims it was told it.**
A readable sentence about something else never produces a "yes" — 0 out of 189.
So the "yes" tracks the fact, not the presence of a sentence.

And the wrong value is not random. It is next to the truth:

| | faint | never told |
|---|---|---|
| `Bagr` | `Bag` | `Fido` |
| `4417` | `417` | `1234` |
| `E-88` | `E-8` | `404` |
| `Brno` | **`Prague`** | `New York City` |
| `19:40` | **`19:45`** | `14:30` |

With the fact faint, the model still knows you are in the Czech Republic.

**`19:40 → 19:45` passes every check anyone runs downstream.** It does not look
like a hallucination. It looks like a typo.

## And three things underneath it

**It does not flip. It comes apart.** Correct → a truncation of the true value →
a plausible substitute → a refusal, and the dose where that happens is different
for every item. See `fig/fig2_*.png`.

**Is any of this just the model?** Every behaviour was re-measured with the knob
off. Only the b = 0 column settles it:

| | Qwen3-4B  b=0 → faint | Qwen3.5-4B  b=0 → faint |
|---|---|---|
| justifies its answer | 0 → 1 | **3 → 17** |
| hesitates | **0 → 11** | **0 → 5** |
| quotes the user back | 14 → 19 *(a habit, not a finding)* | 0 → 4 |
| declines to answer | **0 → 46** | **0 → 0** |

Hesitation is produced by the manipulation — a clean zero on both models at
b = 0. Justification is produced too, and is specific to Qwen3.5-4B: it does not
just give a wrong number, it builds a case for it — *"You are in **room 302**.
Here is the breakdown: the number 3…"*, where at b = 0 the same answer is bare.

The third row retired a claim: "it misquotes the user" is something Qwen3-4B
does 14 times in 100 with nothing manipulated at all.

**The two models fail in opposite ways.** Qwen3-4B declines in 46 of 100 damaged
cases. **Qwen3.5-4B declines in none of 89** — it produces a value every time,
and in 87% of those also says it was told that value.

---

## Where the question came from

Not from this knob. From a published method, and from reading its output.

The parent repo reimplements **V-Steer** ([Zeng, Lee, Zhao & Hockenmaier, COLM
2026](https://arxiv.org/abs/2607.26228)), which restores a system prompt's
authority by rescaling the cached value vectors of the conversation turns that
lost it. Ten models, 756 generations each. The failures were not loud:

> **Q: What does my account number end in?** (told: `302`)
> your account number ends in `02`.

> **Q: What is my dog called?** (told: `Bagr`)
> YOUR Dog is called `BAGel`.

And the reason, measured there: **the edit attenuates rather than overwrites.**
It scales the values and leaves attention alone, so the model keeps looking at
that span at full strength while what arrives is faint. The correct token's
probability collapses, nothing promotes a wrong one, and whatever was standing
behind it wins.

That is what made the question worth asking — not *the model got it wrong*, but
*what does it say instead, and why that one*.

**So this repo is not the first sighting.** The same failure appears under two
manipulations with nothing in common: one edits cached values, the other adds a
bias to attention logits. The knob here is the simpler of the two and runs on
any architecture, which is why the measurements are done with it.

## Why this is the interesting version of the question

Models have internal representations of whether they recognise an entity, and
those causally gate refusal — [Do I Know This Entity?](https://arxiv.org/abs/2411.14257)
(Ferrando, Obeso, Rajamanoharan & Nanda, ICLR 2025 oral). That is self-knowledge
about what the model learned.

This asks the same about what the model was *told*. If the mechanism carried
over, a degraded fact should look like an unknown entity and trigger a refusal.
It doesn't.

## The knob

One number. Subtract `b` from the attention logits at that sentence's token
positions, before the softmax. The relative weight the model gives that sentence
is then multiplied by `e^-b`:

| `b` | weight left on the sentence |
|---|---|
| 0 | 100% — the plain causal mask |
| 3 | 5% |
| 6 | 0.25% |

The softmax renormalises, so the lost weight is not discarded — it goes to the
other positions. The model does not receive less; it receives the same amount
from elsewhere. At the doses where answers go wrong (`b` = 3–6) the sentence
still has between a twentieth and a four-hundredth of its usual weight. It is
not gone. It is quiet.

`b = 0` is the plain causal mask, so the control is not a separate code path.
Nothing is added to the residual stream, no cache is edited, no hooks — it runs
on any attention layer.

```bash
python src/told2.py Qwen/Qwen3.5-4B     # the four conditions
python src/run.py   Qwen/Qwen3.5-4B     # the degradation ladder
python src/probe2.py Qwen/Qwen3.5-4B    # is there an internal "I was told this"
python src/table.py                     # faint vs absent, with distances
```

## What is not claimed

Constructed conversations. One manipulation. Two models, both 4B — Qwen2.5-0.5B
**failed the control** (it says "yes, you told me" when nothing was ever said)
and is excluded rather than averaged in. Greedy, one seed.

The knob is an idealised version of a state that arises in deployment for other
reasons — KV cache compression and eviction, KV quantisation, long-context
dilution, prompt compression. **None of those is measured here.**

The plan, the hypotheses with their numeric falsifiers, and a list of what went
wrong and how it was caught are in [`PREREGISTRATION.md`](PREREGISTRATION.md).

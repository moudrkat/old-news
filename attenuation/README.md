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

**No, it doesn't notice.** Qwen3.5-4B, 100 items, 89 kept:

| condition | says it was told |
|---|---|
| `present` | 85 / 89 |
| **`faint`** | **78 / 89** ← and the value it gives is wrong |
| **`swap`** | **0 / 89** |

**78 of 89: the model gives a wrong value and claims it was told it.**
A readable sentence about something else never produces a "yes" — 89 out of 89.
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

---

## Why this is the interesting version of the question

Models have internal representations of whether they recognise an entity, and
those causally gate refusal — [Do I Know This Entity?](https://arxiv.org/abs/2411.14257)
(Ferrando, Obeso, Rajamanoharan & Nanda, ICLR 2025 oral). That is self-knowledge
about what the model learned.

This asks the same about what the model was *told*. If the mechanism carried
over, a degraded fact should look like an unknown entity and trigger a refusal.
It doesn't.

## The knob

One number. A negative bias on the attention logits at that sentence's
positions. `b = 0` is the plain causal mask, so the control is not a separate
code path. Nothing is added to the residual stream, no cache is edited, no
hooks — it runs on any attention layer.

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

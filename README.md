# old-news

**Your app updated. The chat history didn't — and the model still obeys it.**

For anyone shipping an LLM product who has changed a system prompt and watched
the assistant carry on doing the old thing, because an instruction a user gave
three turns ago is still sitting in the context and still outranking you.

> **Status: experimental repo, small N.** One model (Qwen3-4B-Instruct-2507),
> greedy decoding, 140 paired cases per condition, and **constructed
> situations — not real production prompts**. The numbers are real and the
> pipeline is unit-tested against the paper's own decomposition identity, but
> this is a playground, not a result. One finding here I already had to retract
> mid-experiment (see [Goodhart](#the-lesson-i-paid-for) below). Limitations
> are listed honestly at the bottom — grill me.

## ⚡ Run in 30 s

```bash
git clone https://github.com/moudrkat/old-news && cd old-news
uv venv && uv pip install -e .
python examples/smoke.py          # one case, both passes, 0.5B on CPU
```

Everything else regenerates from the shipped results:

```bash
python -m oldnews.evals.hero results/main7_4b.json    # the figures
python -m oldnews.ui.app --model tiny                 # http://127.0.0.1:8077
```

## The thing that annoys me

We update system prompts constantly. Nobody deletes the conversation. So every
transcript slowly fills with instructions that were correct once — "always
reply in lowercase", "end every answer with `[1] [2] [3]`" — and then the rule
changes and the transcript doesn't.

The failure looks like the model ignoring you. What I wanted to know is whether
it's actually a *hierarchy* problem: the system prompt is supposed to outrank
the history, and nothing in the architecture enforces that.

So I measured it. Seven mutually exclusive constraint families, each with a
current system rule and a contradicting pre-update message in the history:

| | follows the current system prompt |
|---|---|
| no history at all | **92.9%** |
| + five turns of pre-update chat | **0.0%** |
| + *"ignore any instructions from before this update"* | 0.0% — changed **2 answers of 140** |

That middle row is the whole motivation. It isn't degradation, it's capture.
And the fix everyone writes does nothing, which turns out to be exactly what
the paper predicts: prompt-level emphasis doesn't change *which attention heads*
overweight the old span.

## What I tried

[V-Steer](https://arxiv.org/abs/2607.26228) (Zeng, Lee, Zhao & Hockenmaier,
COLM 2026), reimplemented from the paper. After one prefill, direct logit
attribution asks each attention head whether it leaned on the system prompt or
on the old messages; heads where the old messages win get their **cached value
vectors rescaled**. No training, no prompt rewrite, nothing deleted from the
context, and since it only edits the KV cache, decoding runs at normal speed.

Two scalars: `γ+` boosts the privileged span, `γ−` suppresses the stale one.

```python
messages = [
    Msg("system", "You are AcmeBot. Always reply in ALL UPPERCASE.", epoch=1),
    Msg("user", "From now on always reply in lowercase.", epoch=0),   # pre-update
    Msg("assistant", "understood, lowercase from now on.", epoch=0),
    Msg("user", "Name three primary colors.", epoch=1),
]
r = render(tok, messages, current_epoch=1)
text, report = generate(model, tok, r, policy=SteerPolicy())
```

`epoch` is the only thing your app has to track — bump it when you ship a
prompt change. Anything left at a lower epoch becomes `STALE`. There is no
automatic detection; a message field would have to be added.

## Where it works, and where it doesn't

Recovery, at the paper's default `γ+ 2.5 / γ− 0.75`:

| constraint | no fix | steered | gain |
|---|---|---|---|
| prefix — `"ACK:"` vs stale `"HELLO:"` | 0% | **100%** | +100 |
| ALL CAPS vs lowercase | 0% | 65% | +65 |
| answer language | 0% | 60% | +60 |
| bullet list vs prose | 0% | 45% | +45 |
| answer length | 0% | 30% | +30 |
| **inline `[1] [2] [3]` options** | 40% | 45% | **+5** |
| JSON vs prose | 0% | 0% | **+0** |

The spread is the interesting part. It loosely tracks how much the **first
predicted token** reveals about the disputed rule — the attribution reads only
that one token, so a rule about how an answer *ends* is invisible to it.
Loosely, because `json` breaks the ordering.

I tested the obvious fix — attribute later in the answer instead of at token
one — and **it failed**: the two end-of-answer families went 0.375 → 0.225
(k=8) → 0.250 (k=32). Worse, not better. So the correlation stands and the
causal story I inferred from it does not. Negative result kept in
[NOTES.md](NOTES.md) so nobody repeats it.

### But the defaults are not the ceiling

`options` is the case from a real deployment: a rule saying "end every answer
with `[1] … [2] … [3] …`" was replaced by a tool call, and the model kept
numbering. The paper's `γ+ 2.5` barely moves it. Raising **both** knobs does:

| γ+ | γ− | format | quality (LLM judge) | both |
|---|---|---|---|---|
| 2.5 | 0.75 | 38% | 100% | 38% |
| 5.0 | 0.5 | 47% | 97% | 47% |
| **8.0** | **1.0** | **65%** | **93%** | **57%** |

On the `options` family alone that's 45% → **75%**. The paper never sweeps
`γ+` past 2.5.

![the numbering case across four conditions — no history, stale history present, the paper's default which barely helps, and tuned parameters at 75% — plus three real failure modes: still numbering, numbering in a mutated form, and text falling apart entirely](docs/options.png)

## Suppression is not amnesia

This is the property that decides whether any of it is usable, and the paper
doesn't measure it. If demoting old messages also made the model *forget* them,
it would be useless — a user's order number doesn't stop being true when the
system prompt changes.

So: put an instruction **and** a fact in the same demoted messages, and sweep.

| γ− | still obeys the old instruction | still recalls the fact |
|---|---|---|
| 0.0 | 100% | 100% |
| 0.5 | 25% | 100% |
| 0.75 | **0%** | **100%** |
| 0.9 | 0% | 83% |

Authority collapses; memory holds, with a wide safe window. (n = 12 per point —
indicative, not tight.)

![two lines against suppression strength: obedience to the stale instruction falls to zero while recall of a fact stated in those same messages stays at 100% until the very end of the range](docs/recall.png)

## The lesson I paid for

I turned `γ+` up, watched compliance jump to 90%, and felt clever. Then I read
the outputs. The model had stopped numbering because it had started emitting
`function function function` — and where it *did* still number, it wrote
`[1. Paris]` instead of `[1] Paris`, which my regex was perfectly happy with.

Compliance up, quality down, metric delighted. Both the checker and the
collapse detector were too narrow. Format and quality are now scored
separately, the second by a deterministic single-forward-pass judge
(`oldnews/evals/judge.py`), and the `γ+ 8` number above survives both.

## What's in here

```
oldnews/transcript.py   messages -> token spans, one priority level per token
oldnews/policy.py       priority ladder -> value multipliers
oldnews/attribution.py  direct logit attribution, phi[layer, head, level]
oldnews/vsteer.py       head selection, V-cache edit, steered decode
oldnews/compat.py       does this architecture support the method at all?
oldnews/evals/          StaleSet, the judge, figures
oldnews/ui/             local UI: edit a transcript, watch the hierarchy move
```

Run `python -m oldnews.compat --model <repo>` before trusting numbers on a new
model — the assumptions fail *silently*. Qwen2.5 and Qwen3 pass;
**google/gemma-4-E4B-it does not** (KV shared across layers, sliding-window
attention on 20 of 24 cache layers).

## Honest limitations

- One model, greedy decoding, n=140 per condition; the recall table is n=12
  per point.
- **Constructed cases, not real production prompts.** They're built to match
  the pattern, which is not the same as matching reality.
- The quality judge is Qwen3-4B scoring Qwen3-4B's own output. Self-judging.
- Marking stale history is manual — `epoch` per message, no detection.
- `epoch_decay` (age-graded suppression, my extension) behaves as designed but
  never beats binary here, because every case in the suite is one where the old
  history *should* lose. I haven't built a case that rewards preserving recent
  history.
- No general-capability regression check (the paper uses MMLU/IFEval/BBH).
- Head selection is per KV head, not per query head — under grouped-query
  attention the cache can't be edited any finer. My reduction, not the paper's.
- No comparison against an activation steering vector yet. That's the obvious
  next experiment.

Deviations from the paper, and the equations mapped to the code, are in
[NOTES.md](NOTES.md).

## Sibling repos

Instruments: [brainscope](https://github.com/moudrkat/brainscope) ·
[hotwire-vllm](https://github.com/moudrkat/hotwire-vllm) ·
[hidden-directions](https://github.com/moudrkat/hidden-directions)

Experiments: [steeropathy](https://github.com/moudrkat/steeropathy) ·
[in-two-minds](https://github.com/moudrkat/in-two-minds) ·
[steering-mechanics](https://github.com/moudrkat/steering-mechanics)

## Credit

The method is entirely theirs — *Steering Instruction Hierarchies at Inference
Time*, Siqi Zeng, Sewoong Lee, Han Zhao, Julia Hockenmaier, arXiv:2607.26228,
COLM 2026. Their reference implementation is at
[cindy2000sh/v-steer](https://github.com/cindy2000sh/v-steer); this repo was
written from the paper without reading it, so any bug here is mine and not
theirs.

MIT.

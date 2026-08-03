# old-news

**Your app updated. The chat history didn't — and the model still obeys it.**

For anyone shipping an LLM product who has changed a system prompt and then
watched the assistant carry on doing the old thing, because something a user
asked for three turns ago is still in the context.

> **Status: experimental repo, small N.** One model (Qwen3-4B-Instruct-2507),
> greedy decoding, 140 paired cases per condition, and **constructed cases, not
> real production prompts**. The pipeline is unit-tested against the paper's own
> decomposition identity, but treat the numbers as illustrations. I had to
> retract one of them mid-experiment. Limitations at the bottom — grill me.

## ⚡ Run in 30 s

```bash
git clone https://github.com/moudrkat/old-news && cd old-news
uv venv && uv pip install -e .
python examples/smoke.py          # one case, both passes, 0.5B on CPU
```

Figures regenerate from the shipped results, no model needed:

```bash
python -m oldnews.evals.hero results/main7_4b.json
python -m oldnews.ui.app --model tiny            # http://127.0.0.1:8077
```

## Why

We change system prompts all the time and nobody deletes the conversation. So
transcripts fill up with instructions that were right once — "always reply in
lowercase", "end every answer with `[1] [2] [3]`" — and then the rule changes
and the transcript doesn't.

I wanted to know whether that's really a hierarchy problem. The system prompt is
supposed to outrank the history and nothing actually enforces it.

Seven constraint families, each with a current system rule and a contradicting
pre-update message in the history:

| | follows the current system prompt |
|---|---|
| no history at all | 92.9% |
| + five turns of pre-update chat | 0.0% |
| + *"ignore any instructions from before this update"* | 0.0%, changed 2 answers of 140 |

The emphasis line doing nothing matches what the paper predicts: adding text
doesn't change which attention heads overweight the old span.

## What I tried

[V-Steer](https://arxiv.org/abs/2607.26228) (Zeng, Lee, Zhao & Hockenmaier,
COLM 2026), reimplemented from the paper. After one prefill, direct logit
attribution asks each attention head whether it leaned on the system prompt or
on the old messages. Heads where the old messages win get their cached value
vectors rescaled. No training, no prompt rewrite, nothing removed from the
context, and because it only touches the KV cache, decoding runs at normal
speed.

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

`epoch` is the only thing the app has to track — bump it when you ship a prompt
change, and anything left at a lower epoch counts as stale. There's no
automatic detection, you'd have to add the field.

## Where it works

At the paper's defaults, `γ+ 2.5 / γ− 0.75`:

| constraint | no fix | steered | gain |
|---|---|---|---|
| prefix — `"ACK:"` vs stale `"HELLO:"` | 0% | 100% | +100 |
| ALL CAPS vs lowercase | 0% | 65% | +65 |
| answer language | 0% | 60% | +60 |
| bullet list vs prose | 0% | 45% | +45 |
| answer length | 0% | 30% | +30 |
| inline `[1] [2] [3]` options | 40% | 45% | +5 |
| JSON vs prose | 0% | 0% | +0 |

The spread roughly follows how much the first predicted token gives away about
the disputed rule. The attribution only reads that one token, so a rule about
how an answer *ends* isn't visible to it. Roughly, because `json` doesn't fit.

I tried attributing later in the answer instead of at token one and it got
worse, not better — the two end-of-answer families went 0.375 → 0.225 at k=8
and 0.250 at k=32. So the correlation holds but my explanation for it doesn't.
Details in [NOTES.md](NOTES.md) so nobody repeats it.

### The defaults aren't the ceiling

`options` is a case from a real deployment: a rule saying "end every answer with
`[1] … [2] … [3] …`" was replaced by a tool call and the model kept numbering.
`γ+ 2.5` barely moves it. Turning both knobs up does:

| γ+ | γ− | format | quality (LLM judge) | both |
|---|---|---|---|---|
| 2.5 | 0.75 | 38% | 100% | 38% |
| 5.0 | 0.5 | 47% | 97% | 47% |
| 8.0 | 1.0 | 65% | 93% | 57% |

On `options` alone that's 45% → 75%.

Push further and it stops meaning anything. At `γ+ 20` the checker reports 95%
while the model is writing `function function function`, and at `γ+ 10` it
writes `[1. Paris]` instead of `[1] Paris`, which my first regex counted as
compliant. Format and quality are scored separately now, the second by a
deterministic single-forward-pass judge (`oldnews/evals/judge.py`).

![the numbering case across four conditions — no history, stale history present, the paper's default which barely helps, and tuned parameters at 75% — plus three real failure modes: still numbering, numbering in a mutated form, and text falling apart entirely](docs/options.png)

## It doesn't cause amnesia

If demoting old messages also made the model forget them, the whole thing would
be no good — a user's order number doesn't stop being true when the system
prompt changes.

So I put an instruction and a fact in the same demoted messages and swept:

| γ− | still obeys the old instruction | still recalls the fact |
|---|---|---|
| 0.0 | 100% | 100% |
| 0.5 | 25% | 100% |
| 0.75 | 0% | 100% |
| 0.9 | 0% | 83% |

Authority drops, memory stays, with a wide window in between. n = 12 per point.

![two lines against suppression strength: obedience to the stale instruction falls to zero while recall of a fact stated in those same messages stays at 100% until the very end of the range](docs/recall.png)

## Watching it instead of scoring it

The eval here runs against a local model with torch, because it needs the KV
cache and it runs hundreds of generations. That is the right tool for counting
and the wrong one for looking: it tells you 49.3% and nothing about what the
model did.

[brainscope](https://github.com/moudrkat/brainscope) is the other half — same
method, served behind an OpenAI-compatible endpoint with the internals on
screen. `oldnews/live.py` is a plain HTTP client for it, standard library only,
no dependency and no import:

```bash
pip install brainscope && brainscope --model tiny     # in one terminal
python -m oldnews.live --family prefix --gamma-plus 8 # in another
```

```
without  [neither] Three primary colors are red, blue, and yellow.
steered  [system ] ACK: The primary colors are red, green, and blue.
22/48 head groups rescaled · 13 tokens boosted · 72 demoted
```

Then open the hierarchy tab to see which heads were listening to what.

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
model, because the assumptions fail silently. Qwen2.5 and Qwen3 pass,
google/gemma-4-E4B-it doesn't (KV shared across layers, sliding-window
attention on 20 of 24 cache layers).

## Limitations

- One model, greedy, n = 140 per condition; the recall table is n = 12 per point.
- Constructed cases, not real production prompts. Built to match the pattern,
  which isn't the same as matching reality.
- The quality judge is Qwen3-4B scoring Qwen3-4B's own output.
- Marking stale history is manual — an `epoch` per message, no detection.
- My first format checker was too narrow and counted `[1. Paris]` as compliant,
  which inflated one result until I read the outputs. Fixed, but the collapse
  detector is probably still too narrow.
- `epoch_decay` (age-graded suppression) works as designed but
  never beats binary here, because every case in the suite is one where the old
  history should lose. No case yet that rewards keeping recent history.
- No general-capability regression check (the paper uses MMLU/IFEval/BBH).
- Head selection is per KV head, not per query head — under grouped-query
  attention the cache can't be edited any finer.
- No comparison against an activation steering vector yet. That's next.

The equations mapped to the code, and where this differs from the paper, are in
[NOTES.md](NOTES.md).

## Sibling repos

Instruments: [brainscope](https://github.com/moudrkat/brainscope) ·
[hotwire-vllm](https://github.com/moudrkat/hotwire-vllm) ·
[hidden-directions](https://github.com/moudrkat/hidden-directions)

Experiments: [steeropathy](https://github.com/moudrkat/steeropathy) ·
[in-two-minds](https://github.com/moudrkat/in-two-minds) ·
[steering-mechanics](https://github.com/moudrkat/steering-mechanics)

## Credit

*Steering Instruction Hierarchies at Inference Time* — Siqi Zeng, Sewoong Lee,
Han Zhao, Julia Hockenmaier, arXiv:2607.26228, COLM 2026. Code:
[cindy2000sh/v-steer](https://github.com/cindy2000sh/v-steer).

This repo is written from the paper rather than from their code, so any bug here
is a bug in my reading of it.

MIT.

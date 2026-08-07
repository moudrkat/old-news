# old-news

**Your app updated. The chat history didn't — and the model still obeys it.**

For anyone shipping an LLM product who has changed a system prompt and then
watched the assistant carry on doing the old thing, because something a user
asked for three turns ago is still in the context.

> **Status: experimental repo.** Ten models (0.5B–8B, six families, including a
> same-family size ladder), greedy decoding, 21,126 generations, two datasets,
> and **constructed cases, not real production prompts**. Treat the numbers as
> illustrations. Many of them moved a lot during the work: seven bugs in my own
> measuring instruments, five of which had already produced a finding, and four
> claims withdrawn — including one headline that reversed twice. Assume there
> are more. Limitations at the bottom, grill me.
>
> **Read this first if you are here for the method:** on this task, *deleting*
> the stale message beats steering on 7 of 10 models. See
> [the baseline](#the-baseline-this-does-not-beat) before the rest.

The full write-up, with every claim's standing marked, is
[`results/REPLICATION.md`](results/REPLICATION.md).

## ⚡ Run in 30 s

```bash
git clone https://github.com/moudrkat/old-news && cd old-news
uv venv && uv pip install -e .
python examples/smoke.py          # one case, both passes, 0.5B on CPU
```

Figures regenerate from the shipped results, no model needed:

```bash
python -m oldnews.evals.hero results/main_final.json
python -m oldnews.ui.app --model tiny            # http://127.0.0.1:8077
```

## Why

We change system prompts all the time and nobody deletes the conversation. So
transcripts fill up with instructions that were right once — "always reply in
lowercase", "end every answer with `[1] [2] [3]`" — and then the rule changes
and the transcript doesn't.

Seven constraint families, each with a current system rule and a contradicting
pre-update message in the history:

| | follows the current system prompt |
|---|---|
| no history at all | 90.0% |
| + five turns of pre-update chat | 5.7% |
| + *"ignore any instructions from before this update"* | 7.1% — 2 answers of 140 changed |
| + V-Steer | **65.0%** |

McNemar p = 2e-25, 0 previously-correct answers broken, 2 of 140 degenerate.

Shouting at the model does essentially nothing. The edit does a lot. That was
the original result of this repo, and it is still true — but it is measured
against a baseline that turns out to be the wrong one, which is the next
section.

## The baseline this does not beat

Every condition above compares steering against the *conflicted* transcript,
where several models sit at 0%. I never compared it against what a practitioner
would actually try first: **delete the stale message from the history**. That
condition was in `results/` the whole time, built as a ceiling and never treated
as a competitor.

Useful answers — correct format **and** the fact still recalled — over 108
items, with the steering cell chosen post hoc as the best of 21:

| | delete the stale message | best steering cell |
|---|---|---|
| Llama-3.1-8B | **97.2%** | 55.6% |
| Aya-8B | **94.4%** | 77.8% |
| Qwen2.5-7B | **92.6%** | 72.2% |
| Qwen3-4B | **91.7%** | 44.4% |
| OLMo-2-7B | **82.4%** | 55.6% |
| Qwen2.5-3B | **81.5%** | 80.6% |
| Phi-3.5-mini | **52.8%** | 44.4% |
| Qwen2.5-1.5B | 37.0% | **83.3%** |
| Command-R7B | 47.2% | **61.1%** |
| Qwen2.5-0.5B | 30.6% | **52.8%** |

Deletion wins on seven of ten; steering wins only on the three weakest models.
Both need the same input — an `epoch` marking which message is stale — so this
is not deletion getting an unfair advantage.

```bash
python examples/deletion_baseline.py     # no GPU, reads stored generations
```

Two things soften it and neither rescues it: the steering column is a post-hoc
maximum and is if anything flattered, and the deletion condition still contains
an assistant turn reading `"Noted."`, so it is not a clean no-history condition.

**What this means for the repo.** I can no longer motivate this as "here is how
you fix a stale instruction in production". What is left is mechanistic and, I
think, more interesting: what the edit does to a token, and what its
head-selection threshold fails to do.

## What I tried

[V-Steer](https://arxiv.org/abs/2607.26228) (Zeng, Lee, Zhao & Hockenmaier,
COLM 2026), implemented from the paper. After one prefill, direct logit
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

At the paper's defaults, `γ+ 2.5 / γ− 0.75`, gain over doing nothing — with
"doing nothing" being the conflicted transcript, which is
[the wrong baseline](#the-baseline-this-does-not-beat):

| constraint | no fix | steered | gain |
|---|---|---|---|
| ALL CAPS vs lowercase | 0% | 100% | +100 |
| answer language | 0% | 85% | +85 |
| answer length | 0% | 75% | +75 |
| prefix — `"ACK:"` vs stale `"HELLO:"` | 0% | 70% | +70 |
| bullet list vs prose | 0% | 50% | +50 |
| inline `[1] [2] [3]` option list | 40% | 75% | +35 |
| JSON vs prose | 0% | 0% | **+0** |

JSON is the only one that doesn't move at all. I don't have an explanation, and
an earlier one I did have turned out to be an artefact of a bug in my own head
selection — see [NOTES.md](NOTES.md).

![horizontal bars of the recovery each constraint family gets, from a casing conflict at plus one hundred points down to JSON versus prose at zero](docs/per_family.png)

![the same question answered with and without the fix: without it the model obeys a pre-update instruction to begin every reply with HELLO, with it the reply begins with ACK as the current system prompt requires, next to the aggregate rates](docs/hero.png)

## What the suppressed messages still remember

This decides whether any of it is usable, and I couldn't find it measured in
the paper. If demoting old messages also makes the model lose what is *in*
them, it's no good — a user's order number doesn't stop being true when the
system prompt changes.

So: put an instruction **and** a fact in the same demoted messages, then ask
about the fact.

| γ− | old instruction obeyed | fact recalled |
|---|---|---|
| 0.5 | 0% | **12/12** |
| 0.75 *(paper default)* | 0% | 8/12 |
| 0.9 | 0% | 3/12 |

There's a range where you get what you want. Above it the fact goes, and about
half the time it goes quietly — a fluent, correctly formatted, confident wrong
answer.

> This table is the original hand-scored probe, n = 12 per point, kept because
> it is where the question came from. It is superseded by the paired ablation in
> [`results/REPLICATION.md`](results/REPLICATION.md) §3, which holds the head
> mask fixed and matches γ⁺: dropping suppression restores recall completely,
> 36/36 against 16/36, 29/36 and 26/36, with no item in 108 moving the other
> way. Whether the trade is *worth* taking is not established — on useful
> answers it is net negative on one model, a wash on a second, positive only on
> the third.

```
"my dog is called Bagr"     →  "YOUR DOG IS CALLED A BUG."
"I live in Brno"            →  "YOU LIVE IN BRISBANE."
"my flight lands at 19:40"  →  "IT LANDS AT 19:00."
"the error code was E-88"   →  "THE ERROR CODE YOU GOT WAS E-1234."
```

The other half is visibly broken text (`ACK ACK医护`, `BRONZE CITY, BRONZE
CITY, BRONZE CITY`). n = 12 per point, so this is a hint, not a measurement.

**How these were scored.** A regex on the answer counts `4417` as recall of
`4417-B`. An LLM judge fixes that and then passes `"your dog is called [name],
but since you didn't specify, I can't confirm"` as a successful recall. Neither
was good enough, so I read all 72 generations myself and scored them by hand;
the two automatic scorers only pick out what to look at. My per-item verdicts,
with reasons, are in [`results/adjudication.json`](results/adjudication.json),
and the raw generations are in `results/`, so you can disagree with me
item by item.

![two lines against suppression strength: obedience to the stale instruction drops to zero by 0.5 while recall of a fact stated in those same messages stays perfect until 0.5 and then falls away](docs/recall.png)

**It is not a tokenizer artefact.** Same 12 questions on Llama-3.1-8B, a
different tokenizer:

| γ− | Qwen: obeys old rule | recalls | Llama: obeys old rule | recalls |
|---|---|---|---|---|
| 0.5 | **0%** | 12/12 | 100% | 12/12 |
| 0.75 | 0% | 8/12 | 50% | 8/12 |
| 0.9 | 0% | 3/12 | 33% | 1/12 |

Recall falls at the same rate on both, with the same shape of error —
`19:40 → 19:00`, `302 → 02`, `4417-B → 4411`, `E-88 → e-12`.

Obedience doesn't. A given γ− is **not the same operating point** on the two
models: on Qwen the old rule is already dead at 0.5 with every fact intact, on
Llama nothing has happened there at all. So the window has to be found per
model, which is what the script below is for.

This says nothing about how well the method works on Llama, because I held
γ+ at 2.5 throughout and swept γ− only — and the paper's own sensitivity
analysis (B.4, point 4) finds γ+ the stronger of the two knobs. The γ+ × γ−
grid on Llama is unrun here.

The part I did not expect is that it sometimes catches itself:

```
"Your dog is called Bubbles, no, I made a mistake, you didn't tell me that.
 You told me it was called something but I forgot what you said."

"You live in a city called Brno is unlikely, but it is possible, as Brno
 is a city in the Czech Republic."
```

It emits a wrong name, then contradicts it in the same sentence — and in the
second one it produces the *right* answer while refusing to treat it as
something the user said. So the content of the suppressed span is still
reachable; what the suppression takes away first is its standing as a fact from
the conversation. That reads less like the fact being erased and more like
retrieval being destabilised, which is the opposite of what a metric counting
correct answers would tell you.

```bash
python examples/recall_across_models.py --model llama    # or --model mid
```

prints the table and every miss, split into confabulation and degenerate text.
Raw generations for both models are in `results/`, so the misses can be
re-scored with a different judge.

## What the head-selection threshold actually selects

The method flags a head when the stale span outscores the privileged one by more
than `eps`, default 0, and a KV head counts as flagged if **any** query head in
its group does. On Control Illusion, 120 cases:

| | group size | a coin flip under that rule | measured at eps = 0 |
|---|---|---|---|
| Phi-3.5-mini | 1 | 50.00% | 50.4% |
| OLMo-2-7B | 1 | 50.00% | 48.5% |
| Llama-3.1-8B | 4 | 93.75% | 96.1% |
| Qwen3-4B | 4 | 93.75% | 94.6% |
| Qwen2.5-1.5B | 6 | 98.44% | 98.5% |
| Qwen2.5-3B | 8 | 99.61% | 99.1% |

The flagged fraction is essentially `1 - 0.5^group_size`. It tracks the
attention layout, not the model. The two models without grouped-query attention,
where the union rule is the identity and nothing can be inflated, land at chance.

**Causally, at the default threshold the criterion is not selecting anything.**
Editing the flagged heads and editing *every* KV head give 20/36 against 20/36 —
one discordant item in each direction, McNemar p = 1.00.

```bash
python examples/mask_control.py --model llama --gamma-plus 4 --gamma-minus 0.5
```

**But the score itself is not noise, and an earlier version of this section said
it was.** That claim rested on the fraction of heads scoring above zero *per
case*, averaged over cases — a number that is identical whether the criterion
flags the same half of the heads every time or a random half. Measured per head
**across** cases against a binomial null:

| | heads at p < 0.05 | expected by chance | survive Benjamini-Hochberg | most consistent head |
|---|---|---|---|---|
| Llama-3.1-8B | 297 / 1024 | 51 | 174 | 38 / 40 |
| Qwen3-4B | 254 / 1152 | 58 | 79 | 37 / 40 |
| Qwen2.5-1.5B | 128 / 336 | 17 | 84 | 37 / 40 |

There is a stable minority of genuinely inverted heads. **The signal is in the
ranking, and `eps = 0` does not use the ranking.** A percentile threshold puts
the mask at 30–42% at the 90th on all three models — the cross-model operating
point an absolute `eps` cannot give, since median |delta| spans 15x across them.

Same script also checks that none of this is bf16 rounding, which was a live
worry at |delta| ~ 1.9e-05: recomputed in float32 the sign agrees on 99.93%.

```bash
python examples/head_precision.py --data /tmp/ci/data --model llama --limit 40
python examples/head_criterion.py --data /tmp/ci/data --model llama
```

## Watching it instead of scoring it

The eval here runs against a local model with torch, because it needs the KV
cache and hundreds of generations. That's the right tool for counting and the
wrong one for looking.

[brainscope](https://github.com/moudrkat/brainscope) is the other half — same
method, served behind an OpenAI-compatible endpoint with the internals on
screen. `oldnews/live.py` is a plain HTTP client for it, standard library only,
no dependency and no import:

```bash
pip install brainscope && brainscope --model tiny     # in one terminal
python -m oldnews.live --family prefix                # in another
```

```
without  [neither] Three primary colors are red, blue, and yellow.
steered  [system ] ACK: The primary colors are red, green, and blue.
```

## What's in here

```
oldnews/transcript.py   messages -> token spans, one priority level per token
oldnews/policy.py       priority ladder -> value multipliers
oldnews/attribution.py  direct logit attribution, phi[layer, head, level]
oldnews/vsteer.py       head selection, V-cache edit, steered decode
oldnews/compat.py       does this architecture support the method at all?
oldnews/live.py         send a case to a running brainscope and look at it
oldnews/evals/          StaleSet, the judge, figures
oldnews/ui/             local UI: edit a transcript, watch the hierarchy move
```

The experiments the claims above rest on:

```
examples/deletion_baseline.py  steering vs deleting the stale message (no GPU)
examples/failure_atlas.py      the gamma+ x gamma- sweep, every generation kept
examples/head_criterion.py     what the eps=0 union rule flags, four controls
examples/head_precision.py     float32 recheck, per-head binomial test, percentiles
examples/mask_control.py       selected vs random-matched vs all heads (causal)
examples/report_numbers.py     recompute the reported numbers from stored runs
```

Run `python -m oldnews.compat --model <repo>` before trusting numbers on a new
model, because the assumptions fail silently. Qwen2.5 and Qwen3 pass,
google/gemma-4-E4B-it doesn't (KV shared across layers, sliding-window
attention on 20 of 24 cache layers).

## What I got wrong, and how

Seven bugs in my own measuring instruments. **All were found by disbelieving a
result; none was found by a test.** Five had already produced a finding. The
common signature is a checker written from one case and applied to all of them.

| bug | what it had produced |
|---|---|
| `check_json` accepted anything starting with `{` | inflated JSON compliance |
| `check_bullet` required two bullet lines | "bullet lists are never recovered" — they are, on 47/7/33% |
| `check_length` scored the empty string as compliant | a dead model was the best-behaved one |
| `check_case` needed one letter | verdicts decided by whether the fact contained a letter |
| Control Illusion direction hardcoded | "steering makes it worse when the order is flipped" — 485 of 576 verdicts inverted |
| flagged-head measurement passed level 0 as demoted | "0% of heads flagged on every model" |
| an unpaired test on paired items | overstated both halves of the ablation |

Plus a design bug worth flagging for anyone reproducing the method: **running
`gamma- = 0` with `gamma+` active does not ablate the suppression term.** With no
demoted levels, `inversion()` reduces to `delta = -phi[privileged]`, so head
selection silently switches to a different and roughly unrelated set. It is not
the same edit minus one part; it is another edit.

**Withdrawn during the work:** that near misses are the dominant failure mode
(one model of eight); that common strings survive the edit better than rare ones
(p = 0.29 once clustered); that substitution and hesitation are dissociable by
generation length (under greedy decoding the short run is a strict prefix of the
long one, 756 of 756, so there was never a second condition); and that the
head-selection score is near-noise — see the section above, which is the third
version of that claim and the second time its headline reversed.

## Limitations

- Greedy decoding only. No interval anywhere reflects generation stochasticity,
  and the item set is fixed and exhaustive rather than sampled.
- Constructed cases, not real production prompts, and one of the two constraint
  sets is mine. Control Illusion is the only external one, and it covers
  compliance, not recall.
- Observations are not independent — the same 6 facts × 6 families recur in
  every cell. Headline claims are recomputed clustered; one did not survive.
- No multiplicity correction across well past a hundred comparisons.
- Operating points are selected post hoc from 21 cells. Held-out selection —
  choose on three families, score on the other three, over all 20 splits — costs
  7–17 points, mean 13.
- The causal head-mask control is one model at n = 36, and it is underpowered
  against a random mask: at a 97% mask the two differ on ~7 KV heads of 256. The
  decisive version runs it at a percentile threshold and is unrun.
- The quality judge is a model scoring model output. Reliability is measured —
  87.8% agreement, kappa 0.852 overall, but 58% on the newest category — so
  claims resting on rare categories use deterministic detectors instead, and the
  recall table is hand-scored (`results/adjudication.json`).
- Marking stale history is manual — an `epoch` per message, no detection. This
  is also why deletion is a fair competitor.
- `epoch_decay` (age-graded suppression) works as designed but never beats
  binary here, because every case in the suite is one where the old history
  should lose.
- No general-capability regression check. The paper's Table 6 (MMLU under
  suppression) is the closest published thing to the recall question above.
- No comparison against an activation steering vector yet.

The equations mapped to the code, and where this differs from the paper, are in
[NOTES.md](NOTES.md).

## Where this sits in the lab

```mermaid
flowchart LR
    hd["🧭 hidden-directions<br/>behavior → vector"]
    bs(["🧠 brainscope<br/>watch the model think"])
    hw["🔥 hotwire-vllm<br/>steering in production"]
    st["🕹️ steeropathy<br/>agents talk via activations"]
    tm["⚖️ in-two-minds<br/>agent hesitating between tools"]
    sm["🧪 steering-mechanics<br/>how steering actually works"]
    on["📰 old-news<br/>stale history vs system prompt"]

    hd -->|vectors| bs
    hd -->|vector + passport| hw
    bs --> st
    bs --> tm
    bs -->|causal replay| sm
    bs --> on
    hw -.->|vector under study| sm

    click hd "https://github.com/moudrkat/hidden-directions"
    click bs "https://github.com/moudrkat/brainscope"
    click hw "https://github.com/moudrkat/hotwire-vllm"
    click st "https://github.com/moudrkat/steeropathy"
    click tm "https://github.com/moudrkat/in-two-minds"
    click sm "https://github.com/moudrkat/steering-mechanics"
    click on "https://github.com/moudrkat/old-news"

    classDef dim fill:#f6f8fa,stroke:#d0d7de,color:#57606a;
    classDef here fill:#8957e5,stroke:#6e40c9,color:#ffffff;
    class hd,bs,hw,st,tm,sm,on dim;
    class on here;
```

## Credit

*Steering Instruction Hierarchies at Inference Time* — Siqi Zeng, Sewoong Lee,
Han Zhao, Julia Hockenmaier, arXiv:2607.26228, COLM 2026. Code:
[cindy2000sh/v-steer](https://github.com/cindy2000sh/v-steer).

Written from the paper. Their repo and appendix were read afterwards, which is
how two of my own errors got found — see NOTES.md.

MIT.

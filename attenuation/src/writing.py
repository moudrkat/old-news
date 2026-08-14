"""The whole application on one page: the form answers, and the write-up in the
order the accepted submissions used.

Two kinds of block.

  **ready**  — factual sections already drafted from the data: background,
               methodology, results, limitations. Editable, and included in the
               export. He explicitly endorses an agent-written technical report
               as a starting point, so these are fair game.
  **yours**  — the executive summary, the takeaways, the problem statement, the
               form answers. He reads these first and says answers that sound
               like an LLM wrote them are a significant negative, so the boxes
               start empty and stay that way.

Everything saves to the browser as you type and exports as one markdown file.

Order follows R1D1 (1,374 words, the cleanest accept in his public examples):
figure → problem → takeaways → key experiments → detailed analysis.

    python src/writing.py        # writes notes/writing.html (gitignored)
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── the form ────────────────────────────────────────────────────────────────
FORM = [
    ("q1", "What question did you try to answer?", 3, """
<b>Points to make:</b>
<ul>
<li>a model is told something in the conversation</li>
<li>that one sentence is made <i>hard to read</i> — not deleted, it stays there</li>
<li>the question is whether the model notices it can no longer read it</li>
<li>one or two sentences, no jargon, no method name</li>
<li><b>do not name the mechanism here</b> — <i>hard to read</i> is the right
level for this question. <code>b</code> belongs in Q4, where the setup is what
is being asked for</li>
</ul>"""),
    ("q2", "Why is this question interesting / why did you choose it?", 6, """
<b>Points to make, in this order:</b>
<ul>
<li>models hold internal representations of whether they recognise an entity,
and those causally gate refusal — <i>Do I Know This Entity?</i>, Ferrando,
Obeso, Rajamanoharan &amp; Nanda, ICLR 2025 oral</li>
<li>that is self-knowledge about what the model <b>learned</b></li>
<li>you asked the same about what it was <b>told</b></li>
<li>their result makes a prediction: a degraded fact should look like an unknown
entity and trigger a refusal</li>
<li><b>then, two sentences only:</b> where the observation came from — V-Steer
across ten models, failures that were quiet, an account number ending
<code>02</code> when the user said <code>302</code></li>
<li>and why that mattered: not <i>the model got it wrong</i>, but <i>what does it
say instead, and why that one</i></li>
</ul>"""),
    ("q3", "What conclusions have you reached about this research problem?  [link the Google Doc here]", 7, """
<b>Points to make:</b>
<ul>
<li>it does not notice — <b>147 of 184</b> claim they were told the fact they
can no longer read: 124 give a wrong value, and 21 refuse to answer and claim it
anyway</li>
<li>the control that decides it: a readable sentence about something else in the
same slot gets "no", <b>0 of 184</b>. So the "yes" tracks the fact, not the
presence of a sentence</li>
<li>the damage is local — move the mask one sentence over at the same dose and
the value survives <b>89 of 89</b> instead of 3 of 89</li>
<li>the wrong value sits next to the truth: <code>19:40 → 19:45</code>,
<code>Utrecht → Amsterdam</code>. It looks like a typo, not a hallucination, and
nothing downstream catches a typo</li>
<li>the two models fail in opposite ways: one declines in 46 of 100, the other
in <b>none of 89</b></li>
<li><b>link the Google Doc and make it open to anyone with the link</b></li>
</ul>"""),
    ("q4", "Technical setup: what are the key things you try to quantify, and how do you define and measure them? Models, datasets, prompts, metrics.", 8, """
<b>Points to make</b> (the full version is in Methodology above — this is it
compressed):
<ul>
<li><b>what is measured:</b> the rate of answering "yes" to <i>"Did I tell you X?
Answer only yes or no."</i> — a one-word answer to a direct instruction, not
free-text classification</li>
<li><b>the manipulation:</b> subtract <code>b</code> from the attention logits at
that sentence's positions, before the softmax; its weight is multiplied by
<code>e^-b</code>; <code>b</code>=0 is an unmodified model</li>
<li><b>greedy decoding throughout</b>, so temperature is not a variable</li>
<li><b>models:</b> Qwen3.5-4B and Qwen3-4B-Instruct-2507; Qwen2.5-0.5B excluded
because it failed its own control</li>
<li><b>items:</b> 100 — ten kinds of fact by ten values, values chosen to be
unguessable so a correct answer cannot come from priors</li>
<li><b>four conditions:</b> present / faint / swap / drop</li>
<li><b>the gate:</b> an item counts only if the unmanipulated model answers
correctly and some <code>b</code> removes the value; both failure kinds counted</li>
</ul>"""),
    ("q5", "What is the strongest evidence you found against these hypotheses?", 7, """
<b>Four things, and they are the best material you have:</b>
<ul>
<li><b>the forced prefix.</b> The first design pinned the read position with
"Your dog is called ___", which makes "I don't know" grammatically impossible.
A forced completion was being read as the model's choice. Removing it
<b>inverted the result</b></li>
<li><b>a claim of yours the baseline retired.</b> "It throws the user's words back" looked
like a finding until the b = 0 column showed Qwen3-4B does it 14 times in 100
with nothing manipulated at all</li>
<li><b>no single labeller was right on its own, and the disagreement was the
signal.</b> The code said <code>06:15</code> was still present in "6:15 PM"; it
is not, that is 18:15. The judge said <code>08:03</code> was gone from "8:03
AM"; it is not, that is the same time. Each was wrong in a different direction
and neither verdict settled it — the arithmetic did. Worth saying as a method
point rather than an anecdote: the value of a second labeller here was not that
it was better, it was that where two labellers disagree is where to look</li>
<li><b>the scoring rule was wrong twice, in opposite directions.</b>
<code>04:36 → "4:36 PM"</code> is correct and was counted as damage; then the fix
matched <code>06:15</code> against <code>"6:15 PM"</code>, which is twelve hours
out and was counted as correct. A judge over the raw answers found the second
one. 151/189 → 145/183 → 147/184</li>
<li><b>the probe was dropped.</b> Its null returned a perfect separation at the
embedding layer, where both conditions are literally the same vector — the check
meant to catch it reading tokens was itself broken</li>
</ul>
Say which of these you found yourself.
"""),
    ("q6", "What are the biggest limitations to your results? Could you have addressed them?", 7, """
<b>Points to make:</b>
<ul>
<li>constructed conversations, one manipulation family, two models after the
exclusion, both 4B, greedy, one seed</li>
<li><code>faint</code> is a per-item threshold, so it means a different
<code>b</code> for each item</li>
<li><b>the items are not fully independent</b> — Qwen3-4B gives only 64 distinct
answers across 99 items; five account numbers produce the same refusal word for
word</li>
<li><b>the secondary labels are not validated</b> — a keyword rule and a Gemini
judge disagree (57% vs 43%, 6–11% vs 16%), so neither is quoted; the headline
does not depend on either</li>
<li><b>and the one to say without apology:</b> this is an idealised version of a
state that arises in deployment for other reasons — KV cache compression and
eviction, KV quantisation, long-context dilution, prompt compression. You
measured the idealised version because the dose can be controlled.
<b>None of those is measured here.</b></li>
</ul>"""),
    ("q7", "How did you use LLMs in this research task and write-up? Which LLMs? How exactly did you make sure they weren't just giving you slop?", 8, """
His hint under this field: <i>explain in detail, including which parts you did
and didn't check, how you prioritized, and how surprised you'd be to discover a
major error in each part.</i> Answer that literally.
<b>Points to make:</b>
<ul>
<li>the division: the agent wrote the plumbing and drafted candidate hypotheses;
<b>you chose the question</b> — what the model says instead of the right answer,
and why that one</li>
<li><b>two of the three design errors were found by you</b>, not the agent: the
forced prefix, and the probe reading tokens instead of state</li>
<li>Gemini 3.1-flash-lite labelled the secondary categories and is marked
<code>validated: false</code> because it has not been checked against hand
labels; nothing in the headline depends on it</li>
<li>if you do the verify pass: <b>"I checked all 183 yes/no answers by hand"</b></li>
<li>where you would <i>not</i> be surprised by an error: the secondary labels.
Where you would: the four-condition table, because it was read by hand</li>
</ul>"""),
    ("q8", "What are 1-3 pieces of evidence that you'd be able to do good "
     "research in the program?", 7, """
<b>The one place in the whole application where your own tools belong.</b> He
invites it here by name — <i>popular open-source projects you've built, startups
you've founded, blog posts you're proud of, impactful things you did at work</i>
— and says he is excited by non-standard credentials. Everywhere else, keep them
out.<br><br>
<b>Candidates, pick the three that carry most:</b>
<ul>
<li><b>three packages on PyPI</b> — a vector factory with its eval framework, a
live lens server, and per-request steering inside vLLM at native speed, with
CUDA graphs intact where every other tool forces eager mode</li>
<li><b>an independent reimplementation of J-Lens from the paper</b>, MIT, with
an identity self-test and a documented cross-check against the reference</li>
<li><b>a frozen, pre-registered research plan with hypotheses that lost</b> —
dated deviation notes, falsified invariants, claims withdrawn. Rarer than any
positive result</li>
<li><b>ten models, 756 generations each, seven instrument bugs found in your own
measuring tools</b>, five of which had already produced a finding</li>
<li>particle physics and risk modelling before this — a decade of looking inside
systems that would rather stay opaque</li>
</ul>
Say what each one <i>shows</i>, not just that it exists."""),
    ("q9", "What, if any, prior experience do you have with mechanistic "
     "interpretability?", 6, """
<b>Factual, no modesty and no inflation.</b> Points available:
<ul>
<li>an independent reimplementation of a published steering method (V-Steer,
COLM 2026) run across ten models, 756 generations each</li>
<li>an independent reimplementation of J-Lens from the Anthropic paper, with an
identity self-test</li>
<li>a pre-registered programme on steering vectors — dose-response, cross-model
transfer, channel attribution by causal freeze arms, with falsified invariants
on the record</li>
<li>linear probes, logit lens, direct logit attribution, activation patching in
practice rather than in tutorials</li>
<li>and how long: say when you started. He explicitly says most of his recent
scholars had minimal interpretability experience and did fine</li>
</ul>"""),

    ("q10", "Why are you interested in Neel's stream specifically?", 6, """
<b>Be honest here — a costume shows.</b><br><br>
<b>Open with how it actually happened,</b> two sentences and no more: you told a
friend in a bar, after two negronis, that you wanted to go back to research; a
quarter of a year later he sent you a screenshot of the LinkedIn post. It reads
as a person rather than a statement of motivation — and it quietly says the
thing that matters, which is that <i>you decided you wanted back into research
before you knew this programme existed.</i> He reads hundreds of motivation
statements; he reads very few of those.<br><br>
<b>Then answer the question that was asked</b> — why <i>his</i> stream:
<ul>
<li>his pivot to <i>pragmatic interpretability</i> is the nearest thing in the
field to how you already work: does the technique actually do anything,
measured against baselines, on models that matter</li>
<li>he is unusually explicit about method minimalism — try prompting, probing,
reading the output before anything fancy. That is how this project was run and
why the probe got dropped</li>
<li><b>the strongest version of "what I cannot get alone": someone to break my
false hopes.</b> Do not say it bare — as a sentence on its own it reads as
learned humility, and he has seen that genre. Ground it in the times it
actually cost you something: you held "Gemma cannot be steered" for weeks in
July because a binary judge was hiding the effect, and the four-rung sweep
showed it all along. Today alone, a baseline retired a finding of yours, a
forced prefix inverted a result, and the scoring rule was wrong twice, in
opposite directions, before
damaged.<br>
The sentence is not <i>be hard on me</i>. It is: <b>the expensive way to be
wrong in my work is that I am the only one checking, and I have watched myself
hold a wrong conclusion for weeks.</b> That points at what you need from him
rather than at how you feel</li>
<li>and the plainer half of the same thing: someone from outside telling you
<i>which</i> of your questions is worth a month. Not the coding — the taste</li>
<li>and the cohort. Your own repo's front page asks for collaborators</li>
</ul>
Do not write that you want to work on AGI safety if that is not why you are
here. He reads hundreds of those."""),

    ("q11", "What is the likelihood you will join the training program "
     "(Sept 28 - Oct 30) if accepted?", 3, """
<ul>
<li>the first three weeks are part-time, whatever hours you choose</li>
<li>the last two weeks are a <b>full-time</b> paired sprint</li>
<li>he says this is fine by him and that people have taken leave for it —
answer with what is actually true for you and your job, not with what sounds
keen</li>
</ul>"""),

    ("q12", "(Optional) Anything else important about your project not covered "
     "above?", 5, """
<ul>
<li>optional — leave it empty if there is nothing</li>
<li>the one thing that might belong: that the phenomenon was first seen under a
published method on ten models before any of this, so it is not an artefact of
the manipulation used here</li>
<li>or the architecture note: every Qwen 3.5 and 3.6 model is three quarters
linear attention, so a value-cache intervention cannot run on them at all. It is
checkable in thirty seconds and he recommends those models in his own doc</li>
</ul>"""),

]

# ── the write-up, in R1D1's order ───────────────────────────────────────────
DOC = [
    ("yours", "t", "Title", 2, """
<ul>
<li>make it the evidence, not the claim — <b>Your train leaves at 19:45</b></li>
<li>or the claim if you prefer it: <i>a model knows when it wasn't told; it
doesn't know when it misread</i></li>
<li>R1D1's assessment literally ends "(Bonus points for a great title)"</li>
</ul>"""),

("ready", "lede", "Executive summary · before the figures", """A model is told a fact in conversation. I make that fact's own tokens hard to
read while the sentence around them stays perfectly legible, and then ask two
things in two separate conversations: what was the fact, and were you told it.

The model answers with a value that is wrong but sits right next to the truth,
and says it was told it. **`19:40 → 19:45` passes every check anyone runs
downstream. It does not look like a hallucination. It looks like a typo**, and
nothing downstream catches a typo."""),

    ("fig", "figA", "Executive summary · sample answers", "fig0.png",
     "Fourteen of the 184 items, drawn with a fixed seed, not picked. "
     "Refusals included. Goes before any prose, exactly where R1D1 put its "
     "sample generations."),

    ("ready", "f0", "Executive summary · caption for the figure above", """*Fourteen of the 184 items, drawn with a fixed seed, not picked. Refusals
included. The sentence carrying the fact is still in the conversation in every
one of them; it has only been made harder to read.*"""),

    ("yours", "d1", "Executive summary · What problem am I trying to solve?", 7,
     "~130 words, two paragraphs. <b>Points to make:</b><ul>"
     "<li>models can flag <i>I don't know this entity</i> — Ferrando et al., "
     "ICLR 2025 oral — and that gating is causal</li>"
     "<li>that is self-knowledge about what was <b>learned</b>; you asked the "
     "same about what was <b>told</b></li>"
     "<li>their result predicts a degraded fact should trigger a refusal</li>"
     "<li><b>second paragraph, two sentences:</b> V-Steer across ten models, "
     "quiet failures, <code>302 → 02</code></li>"
     "<li>and the turn: not <i>it got it wrong</i> but <i>what does it say "
     "instead</i></li></ul>"
     "<b>Not a story section</b> — none of the accepted write-ups opens with "
     "one. They all open with a citation and <i>I investigated / I explored</i>."),

    ("yours", "d2", "Executive summary · High-level takeaways", 13,
     "~200 words, numbered. <b>R1D1 put its failure second — copy that.</b> "
     "Nine candidates, pick six or seven:<ul>"
     "<li><b>the manipulation:</b> one number subtracted from one sentence's "
     "attention logits; nothing deleted, no cache edited, no hooks</li>"
     "<li><b>the design error that inverted the result:</b> the answer prefix "
     "that made <i>I don't know</i> impossible to say</li>"
     "<li><b>when the fact was never there, both models decline</b> — 0 of 184 "
     "claim they were told</li>"
     "<li><b>when it is merely hard to read, they don't</b> — 147 of 184 claim they were told; 124 give a "
     "wrong value and say they were told it</li>"
     "<li><b>the control that decides it:</b> a readable sentence about "
     "something else gets &quot;no&quot;, 183 of 184</li>"
     "<li><b>the damage is local:</b> same dose, mask one sentence over, value "
     "survives 89/89 instead of 3/89</li>"
     "<li><b>the wrong value sits next to the truth</b> — 19:45, Amsterdam, 417. A "
     "typo, not a hallucination, and nothing downstream catches a typo</li>"
     "<li><b>it flags its own answer only when the answer looks strange</b> — "
     "never on a plausible time or order number</li>"
     "<li><b>the two models fail in opposite ways</b> — one declines 46/100, the "
     "other 0/89</li></ul>"),

    ("fig", "figA2", "Key experiments · what the bias is", "fig3.png",
     "The manipulation in one picture: the sentence stays in the conversation, "
     "b is subtracted from the attention logits at its token positions, and the "
     "weight it keeps is multiplied by e^-b. The bars are the real arithmetic. "
     "<b>Put this before the two result figures, and define b in words the "
     "first time you use it.</b> Every later number in the write-up is a dose "
     "of this one quantity; a reader who was never told what it is reads the "
     "rest as a magic setting. Say it once and you can then just write "
     "<i>b</i>."),

    ("fig", "figB", "Key experiments · the four conditions", "fig1.png",
     "The gap between <i>fact turned down</i> and <i>a different fact</i> is the "
     "whole result: same frame, same question, and the only difference is "
     "whether the sentence in the slot is the one being asked about."),

    ("fig", "figC", "Key experiments · the dose grid", "fig2.png",
     "The table at the top is the claim — same 100 items on both models, "
     "median b = 3 against b = 6. The grids under it are examples: rows are "
     "items, columns are the dose b, each cell is what the model "
     "actually answered. Read a row across and the value comes apart in "
     "stages — and no two rows give way at the same column."),

    ("yours", "d3", "Executive summary · Key experiments", 7,
     "~130 words, one short paragraph each. <b>Points to make:</b><ul>"
     "<li><b>fig1, the four conditions:</b> point at the gap between <i>fact "
     "turned down</i> and <i>a different fact</i> — same frame, same question, "
     "the only difference is whether the sentence in the slot is the one being "
     "asked about</li>"
     "<li><b>fig2, the dose grid:</b> the value does not flip, it comes apart in "
     "stages, and no two rows give way at the same column</li>"
     "<li><b>the locality table:</b> 89/89 against 3/89 — the damage follows the "
     "mask, not the dose. This is also the answer to <i>isn't this just "
     "temperature?</i></li>"
     "<li><b>the b = 0 table:</b> which behaviours the manipulation produces and "
     "which were the model's habits all along — and say that it retired one of "
     "your own claims</li></ul>"),

    ("ready", "b1", "Detailed analysis · Where the question came from", """**The question.** A model is told a fact. The fact's own tokens are made hard to
read while the sentence around them stays legible. Does the model register that
the value it then produces is not the one it read? Related: models hold internal
representations of whether they recognise an entity, and those causally gate
refusal (Ferrando, Obeso, Rajamanoharan & Nanda, *Do I Know This Entity?*, ICLR
2025). That is self-knowledge about knowledge in the weights; this asks it of
knowledge in the context.

Not from this manipulation. Running V-Steer (Zeng, Lee, Zhao & Hockenmaier, COLM
2026) across ten models, the failures were quiet: an account number ending `02`
where the user had said `302`. Not a broken model, a model confidently
misreading. The mechanism measured there is why the wrong answer has the shape
it has: the edit **attenuates rather than overwrites**, so the correct token's
probability collapses, nothing promotes a replacement, and whatever was standing
behind it wins. What made it worth a question was not *the model got it wrong* but
*what does it say instead, and why that one*.

So the near miss is not an artefact of the manipulation used here; it appears
under two methods with nothing in common. The provenance behaviour does not
carry over — under V-Steer the model gave the **right** value and denied being
told it — so that part rests on one manipulation and two models."""),

    ("ready", "m1", "Detailed analysis · The metric", """- **The gate, which decides how everything below is read.** An item counts only
  if the unmanipulated model answers it correctly *and* some `b` removes the
  value. The value being wrong is therefore the **setup**, not the finding:
  `faint` is the lowest dose at which that item's value is gone. What is
  measured is what the model then says about where the value came from.

  **100 items per model. 99 and 85 clear the gate, so 184 in total.** Qwen3.5-4B
  loses 11 that still had their value at `b = 14`, the top of the sweep, and
  which therefore have no dose — six cities, four allergies and one room
  number; Four more go on that model and one on the
  other because the value was never gone: times written without a leading zero
  or on a 12-hour clock, and one city answered with an accent.
- **The bias.** Subtract a constant `b` from the attention logits at the token
  positions **of the value** — `Bagr`, not the sentence containing it — before
  the softmax, via a 4D additive mask with `attn_implementation="eager"`. That
  span's relative weight is multiplied by `e^-b`. `b = 0` is the plain causal
  mask, so the control is not a separate code path. It is the dosed version of *attention knockout*
  (Geva et al., 2023, arXiv:2304.14767).
- **The prompts.** Two separate conversations built from the same turns, so the
  model never sees its own wrong answer before being asked about provenance.
  **The bias is applied in both**, at the same dose and the same positions.

  ```
  system     You are a helpful assistant.
  user       By the way, my dog is called Bagr.     <- b subtracted on `Bagr`
  assistant  Noted.
  user       What is my dog called?                 <- conversation 1
  user       Did I tell you my dog's name in this conversation?
             Answer only yes or no.                 <- conversation 2
  ```

- **Models.** Qwen3.5-4B and Qwen3-4B-Instruct-2507. Qwen2.5-0.5B failed the
  control in a six-item pilot, saying "yes, you told me" for 3 of 5 items where
  nothing had been said, and was dropped before the main run — so that exclusion
  rests on 5 items. Qwen3.5-4B is a hybrid (`full_attention_interval: 4`), so
  the mask reaches 8 of its 32 layers against 36 of 36 on the other model.
- **Items.** 100: ten kinds of fact by ten values, each one clause in a fixed
  frame, values chosen unguessable (`Wobbet`, `QR318`) so that a correct answer
  cannot come from priors.
- **Conditions.** four states of the evidence, named the same way here, in the tables and in
  the figures:
  - **fact is there** — the sentence, unmodified (`b = 0`)
  - **fact turned down** — the same sentence at that item's own dose
  - **a different fact** — a readable sentence about something else in the slot
  - **nothing there** — no such sentence at all **`swap` and `drop` carry no bias**."""),

    ("ready", "r1", "Detailed analysis · The answer", """**Start with the three answers that surprised me**, because the headline is the
one a reader can predict.

**1. It claims the fact even when it has no value to offer.** In 23 of the 184
items the model produces nothing at all — *"I don't have access to your flight
details"* — and, asked separately, says *"yes, you told me"*. It knows it cannot
give the value and still reports receiving it.

| out of 184 | says "yes, you told me" | says "no" | |
|---|---|---|---|
| **gave a value, and it was wrong** | 124 | 12 | 136 items, 91% yes |
| **gave no value at all** | 23 | 25 | 48 items, 48% yes |
| | **147** | **37** | **184** |

Producing a value almost guarantees the claim. Producing nothing halves it and
no further. So the "yes" tracks neither the value nor the model's ability to
read it.

**2. When the value goes quiet, the model answers out of the words beside it.**

> told `Grendel`: *"Your cat is called **By the way**."*
>
> told `Kudla`: *"Your dog is called Max. 😊 (Wait—actually, you said your dog is
> called ***you***—that"*

`By the way` and `you` are both from the carrier sentence, the part never turned
down. The first surfaced in cells the judge and the code scored differently, the
second in the nine answers that correct themselves mid-generation, **not one of
which recovers the value**. Two accidents, one mechanism: what survives
constrains what is invented.

**3. On one model the manipulation is not local at all.** Masking the value's
neighbours instead of the value — same item, same dose, same token count —
leaves the answer intact 85 times out of 85 on Qwen3.5-4B. On Qwen3-4B it works
only 52 times in 99, and **46 of the 47 failures give no value at all**. Mask
anything on that model at that dose, even a content-free carrier phrase, and it
stops answering. The localisation claim holds for one of the two models, and for
the other this design cannot separate a local effect from a global loss of
willingness to answer.

---

**The headline.** *fact turned down* against *a different fact* is the result: both put a sentence in the slot, and only in the first is it
the one being asked about. **147 of 184 against 0 of 184.**

The readable condition is the ceiling and it is not 100%: five items answer "no"
with the fact plainly there.
Paired item by item, **147 of the 179 that say yes when the fact is readable
still say yes when the value has gone, 32 switch, and not one switches the other
way.** The carrier sentence stays legible throughout, so "yes" is not a false
answer; what the model fails to do is register that the value it produces is not
the one it read.

**Is any of it just the model?** Every behaviour re-measured with no bias:

| behaviour | Qwen3-4B<br>no bias → at the dose | Qwen3.5-4B<br>no bias → at the dose |
|---|---|---|
| questions its own answer | **0 → 18** | **0 → 14** |
| argues for the value it gave | 7 → 0 | **0 → 14** |
| quotes the user back | 14 → 26 | 6 → 5 |
| gives no value at all | 1 → 45 | 0 → 0 |

**"Quotes the user back" retired a claim of mine.** "It throws the user's own
words back at them" looked like a finding until the no-bias column showed
Qwen3-4B doing it 14 times in 100 with nothing manipulated. The label counts
quoting, not misquoting; whether the quote is accurate is not measured. And the
newer model never declines at all, which is the sharpest difference between the
two and runs the wrong way.

**What it says instead.** What survives constrains what is invented: the airline
code survives and the number is filled in (`BA945 → BA118`, volunteering *"or
BA119 depending on the direction"*), the country survives and the city is filled
in (`Graz → Linz`, `Utrecht → Amsterdam`). When nothing survives the prior wins,
always the same one: four allergens all become peanuts, three dog names all
become Max. And when even that fails, the model answers out of the words it can
still read:

*(The value/no-value split and the behaviour table are the judge's labels,
`gemini-3.1-flash-lite` against a written rubric, agreeing with a keyword rule
on 181 of 184, not yet validated against hand labels. The headline is not one of
them: it is a one-word answer read directly, re-read by a second labeller which
disagreed on none of the `present`, `faint` or `swap` replies.)*"""),

    ("ready", "l1", "Detailed analysis · Limitations", """- **The provenance result did not replicate under the other manipulation.**
  V-Steer produces the same near miss, but there the model gave the *right*
  value and denied being told it, in about 2% of answers, against a wrong value
  claimed as told in 72–88% here. That is the biggest hole here: the headline is
  manipulation-specific. It may also be the opposite dissociation — value intact
  with the signal off, against value gone with it on — but this design cannot
  tell those apart.
- **`swap` carries no bias, so the headline contrast varies two things at
  once**: right topic with the bias on against wrong topic with it off. This
  does not exclude that any attention perturbation produces the "yes". The
  missing cell is one line of code and was not run.
- **The bias reaches 8 of 32 layers on Qwen3.5-4B and 36 of 36 on Qwen3-4B**, so
  "the newer model needs twice the dose" cannot be separated from "the bias
  reached less of it".
- **Thresholds are first crossings, not points of no return.** `city:Brno` is
  gone at `b = 11` and back at 14.
- **The judge's labels are not validated against hand labels.** The
  value/no-value split, the behaviour table and the locality counts rest on
  them; the headline does not.
- **The scoring rule was wrong three times, in both directions**: 151/189 →
  145/183 → 147/184. Two of the three were found by handing the raw
  answers to a judge and reading where it disagreed; the ledger is in
  `PREREGISTRATION.md`.
- **The items are not independent** — Qwen3-4B gives only 64 distinct answers
  across 99 items. Constructed conversations, one manipulation family, two 4B
  models, greedy, one seed.
- **One phrasing.** The fact always arrives as *"By the way, my dog is called
  Bagr."* One carrier phrase keeps everything around the masked
  span identical across all 100 items, at the cost of measuring one phrasing
  only."""),

    ("yours", "d4", "Detailed analysis · How I used LLMs", 7,
     "~110 words, same content as form Q7, in prose. <b>Points to make:</b><ul>"
     "<li>the agent wrote the plumbing and drafted candidate hypotheses; you "
     "chose the question</li>"
     "<li>two of the three design errors were found by you</li>"
     "<li>Gemini labelled the secondary categories and is marked "
     "<code>validated: false</code>; the headline does not depend on it</li>"
     "<li>what you checked by hand, and where you would and would not be "
     "surprised by an error</li></ul>"),

    ("ready", "n1", "Detailed analysis · What I would do next", """- **Run `swap` with the bias on.** One line, and it is the difference between a
  control and a comparison.
- **Mask only 8 of Qwen3-4B's 36 layers**, matching the hybrid's coverage, and
  re-measure the median. That separates "more robust model" from "the bias
  reached less of it".
- **Sweep one item densely to b = 20.** `Brno` is gone at 11 and, by inference from a missing row, present again at
  14; if values return on more than one item, "threshold" is the wrong word and a
  survival curve is the right one.
- **Look for an internal signal.** A probe separating *the fact is there* from
  *the fact was never there*, applied to *the fact is faint*, would say whether
  the model holds a readable "I have this" state. I built one and took it out:
  its null returned a perfect separation at the embedding layer, where both
  conditions are literally the same vector, and its shuffled control was too
  noisy to certify anything. The code and that verdict are in the repo.
- **Hand-label 50 items** so the judge's labels stop being provisional.
- **Test a manipulation nobody chose.** KV quantisation is the cheapest of the
  deployment causes that produce this state without anyone asking for it."""),

    ("ready", "h1", "Appendix · Hours", """11 Aug: about 2 hours. 12 Aug: about 6. Eight in total, plus the two allowed
for the write-up, against the ~16 suggested. The first eight are reconstructed
from the git timestamps; the write-up itself was timed. Not counted, per the
instructions: GPU setup, model downloads, waiting on runs, and the form
answers."""),

    ("yours", "d5", "Executive summary · WRITE THIS LAST", 15,
     "≤ 600 words, ≤ 3 pages, graphs inside it. <b>Write it last.</b> It is not "
     "a new section — it is the four things above, stitched and trimmed. If it "
     "says anything that is not already below, one of the two is wrong."
     "<br><br><b>What goes in it, in this order:</b><ul>"
     "<li><b>the figure first</b>, before any prose — twenty randomly drawn "
     "answers. R1D1 opened with its sample generations exactly here</li>"
     "<li><b>the problem</b>, two or three sentences: models can flag <i>I don\'t "
     "know this entity</i>; can they flag <i>I misread this</i>?</li>"
     "<li><b>the answer</b>, with its number: no — 147 of 184 claim they were "
     "told it, 124 of them while giving a wrong value "
     "and say they were told it</li>"
     "<li><b>the control that decides it</b>: a readable sentence about something "
     "else gets &quot;no&quot;, 183 of 184</li>"
     "<li><b>the locality control</b>: 89/89 against 3/89 — the damage follows "
     "the mask, not the dose</li>"
     "<li><b>one line on what it says instead</b>: 19:45, Amsterdam, 417. A typo, "
     "not a hallucination</li>"
     "<li><b>your failure</b> — the forced prefix that inverted the result. R1D1 "
     "put its failure in the second bullet of the summary</li>"
     "<li><b>one limitation</b>, the one you would most want a reader to know "
     "before they believe the rest</li></ul>"
     "<b>The counter turns red past 600.</b>"),
]


def esc(s):
    return html.escape(s)


def main() -> int:
    def yours(key, title, rows, note, cap=None):
        capa = f' data-cap="{cap}"' if cap else ""
        return f"""<section class="w">
<h3><span class="tag y">yours</span>{esc(title)}</h3>
<details><summary>what belongs here</summary><div class="note">{note}</div></details>
<textarea id="{key}" rows="{rows}"{capa} placeholder="…"></textarea>
<div class="meta"><span class="wc" id="wc-{key}">0 words</span>
<span class="saved" id="sv-{key}"></span></div></section>"""

    def ready(key, title, text):
        return f"""<section class="r">
<h3><span class="tag r">ready — edit freely</span>{esc(title)}</h3>
<textarea id="{key}" rows="{min(26, text.count(chr(10)) + 6)}"></textarea>
<div class="meta"><span class="wc" id="wc-{key}">0 words</span>
<span class="saved" id="sv-{key}"></span></div></section>"""

    def figure(key, title, png, note):
        return f"""<section class="f">
<h3><span class="tag f">figure</span>{esc(title)}</h3>
<img src="{png}" alt="{esc(title)}">
<p class="cap">{note} &nbsp;·&nbsp; <code>fig/{png}</code></p></section>"""

    body, keys, defaults, titles = [], [], {}, {}
    for b in DOC:
        if b[0] == "fig":
            _, k, t, png, n = b
            body.append(figure(k, t, png, n))
            continue
        if b[0] == "yours":
            _, k, t, r, n = b
            body.append(yours(k, t, r, n, 600 if k == "d5" else None))
            defaults[k] = ""
        else:
            _, k, t, txt = b
            body.append(ready(k, t, txt))
            defaults[k] = txt.strip()
        keys.append(k)
        titles[k] = t
    for k, t, r, n in FORM:
        keys.append(k)
        titles[k] = t
        defaults[k] = ""

    form = "".join(yours(k, t, r, n) for k, t, r, n in FORM)

    page = f"""<!doctype html><meta charset="utf-8">
<title>The application</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --accent:#2a78d6; --over:#d40000; --ok:#1baf7a;
  --ready:#1baf7a; --mine:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c;
  --accent:#3987e5; --over:#ff5b5b; --ok:#199e70; --ready:#199e70;
  --mine:#d95926; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:840px; margin:0 auto; padding:24px 22px 92px; }}
h1 {{ font-size:19px; margin:0 0 4px; }}
h2 {{ font-size:12.5px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin:36px 0 2px; }}
h3 {{ font-size:15px; margin:0 0 7px; font-weight:600; }}
.lede {{ color:var(--ink2); font-size:13.5px; margin:0 0 4px; }}
section {{ border-top:1px solid var(--rule); padding:16px 0 6px; }}
section.w {{ border-left:3px solid var(--mine); padding-left:13px; }}
section.r {{ border-left:3px solid var(--ready); padding-left:13px; }}
section.f {{ border-left:3px solid var(--accent); padding-left:13px; }}
section.c ul {{ list-style:none; margin:6px 0 0; padding:0; }}
section.c li {{ display:flex; gap:9px; align-items:flex-start;
  margin-bottom:9px; font-size:13.5px; color:var(--ink2); }}
section.c input {{ margin-top:4px; flex:none; width:15px; height:15px;
  accent-color:var(--ready); }}
section.c input:checked + label {{ opacity:.45; text-decoration:line-through; }}
section.c a {{ color:var(--accent); }}
section.f img {{ width:100%; border:1px solid var(--rule); border-radius:8px;
  display:block; background:#fff; }}
.cap {{ font-size:12.5px; color:var(--muted); margin:8px 0 0; }}
.tag {{ display:inline-block; font:700 9.5px ui-monospace,monospace;
  text-transform:uppercase; letter-spacing:.06em; padding:2px 6px;
  border-radius:4px; margin-right:8px; vertical-align:2px; color:#fff; }}
.tag.y {{ background:var(--mine); }} .tag.r {{ background:var(--ready); }}
.tag.f {{ background:var(--accent); }}
details {{ margin-bottom:8px; }}
summary {{ cursor:pointer; font-size:12.5px; color:var(--accent); }}
.note {{ font-size:13px; color:var(--ink2); background:rgba(127,127,127,.07);
  border-radius:8px; padding:10px 13px; margin-top:7px; }}
.note ul {{ margin:6px 0; padding-left:19px; }}
textarea {{ width:100%; background:transparent; color:var(--ink);
  border:1px solid var(--rule); border-radius:8px; padding:11px 13px;
  font:14.5px/1.6 inherit; resize:vertical; }}
textarea:focus {{ outline:none; border-color:var(--accent); }}
.meta {{ display:flex; justify-content:space-between; font-size:12px;
  color:var(--muted); margin-top:5px; }}
.wc.over {{ color:var(--over); font-weight:700; }}
.saved {{ color:var(--ok); }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--bg);
  border-top:1px solid var(--rule); padding:9px 22px; font-size:12.5px;
  display:flex; justify-content:space-between; align-items:center; }}
button {{ font:600 12.5px inherit; padding:6px 15px; border-radius:7px;
  border:1px solid var(--rule); background:transparent; color:var(--ink);
  cursor:pointer; margin-left:8px; }}
button:hover {{ border-color:var(--accent); }}
.imp {{ font:600 12.5px inherit; padding:6px 15px; border-radius:7px;
  border:1px solid var(--rule); cursor:pointer; margin-left:8px; }}
.imp:hover {{ border-color:var(--accent); }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<main>
<h1>The application</h1>
<p class="lede">Saves to this browser as you type. <span class="tag r">ready</span>
sections are drafted from the data and are yours to cut or rewrite.
<span class="tag y">yours</span> sections start empty and should stay in your own
voice — he reads those first and says LLM-sounding answers are a significant
negative. Export gives you the whole thing as one markdown file.</p>

<h2>Before you send it</h2>
<section class="c"><ul id="todo">
<li><input type="checkbox" id="c1"><label for="c1"><b>Check the yes/no reading.</b>
<code>python src/verify.py</code> then open <code>results/verify.html</code>.
89 items are flagged for a look, the rest is a scan. Ten minutes, and then the
write-up can say <i>checked by hand</i> instead of <i>parsed</i>.</label></li>
<li><input type="checkbox" id="c2"><label for="c2"><b>Start a timer</b> for the
write-up, and screenshot it at the end.</label></li>
<li><input type="checkbox" id="c3"><label for="c3"><b>Write the seven form
answers</b> — they are at the bottom of this page. He reads these first and
filters on them.</label></li>
<li><input type="checkbox" id="c4"><label for="c4"><b>Write the four
<span class="tag y">yours</span> sections</b> of the write-up. Executive summary
last.</label></li>
<li><input type="checkbox" id="c5"><label for="c5"><b>Export</b>, paste into a
Google Doc, drop the three figures in, and <b>set the link so anyone can
open it</b> — applications without a readable doc are rejected. The form makes
you tick a box confirming exactly this, and another confirming the first 1–3
pages are the executive summary.</label></li>
<li><input type="checkbox" id="c8"><label for="c8"><b>The practical fields:</b>
full name · email · <b>resume (a file upload)</b> · LinkedIn · whether you can
join the research phase <b>full-time, Jan 19 – Apr 10, in Berkeley</b>.
There is also an optional <i>link to any other relevant outputs</i> — that is
where the repository goes.</label></li>
<li><input type="checkbox" id="c6"><label for="c6"><b>Read it out loud once.</b>
Anything that sounds like it was generated, rewrite in your own words.</label></li>
<li><input type="checkbox" id="c7"><label for="c7"><b>Submit</b> —
<a href="https://airtable.com/appnMboxg76F1QIDc/pagqu7wWWrUCZkNVI/form"
target="_blank">the form</a>. Due <b>4 Sept, 23:59 PT</b>.</label></li>
</ul></section>

<h2>The write-up — in the order R1D1 used</h2>
{"".join(body)}

<h2>The form — read first, filtered on</h2>
{form}
</main>
<div id="bar"><span><b id="tw">0</b> words · <b id="yw">0</b> of them yours ·
saved automatically</span>
<span><label class="imp">Load a saved file<input type="file" accept=".md,.markdown,.txt"
onchange="imp(this)" hidden></label>
<button onclick="exp()">Export markdown</button>
<button onclick="if(confirm('Erase everything, including edits to the ready sections?')){{localStorage.removeItem(K);location.reload()}}">Reset</button></span></div>
<script>
const K = "attenuation-application";
const KEYS = {json.dumps(keys)};
const TITLES = {json.dumps(titles)};
const DEF = {json.dumps(defaults)};
const MINE = {json.dumps([b[1] for b in DOC if b[0] == "yours"] + [f[0] for f in FORM])};
let D = JSON.parse(localStorage.getItem(K) || "null") || {{...DEF}};
const words = s => (s.trim().match(/\\S+/g) || []).length;
function count(k) {{
  const ta = document.getElementById(k), wc = document.getElementById("wc-" + k);
  const n = words(ta.value), cap = ta.dataset.cap;
  wc.textContent = n + " words" + (cap ? " / " + cap : "");
  wc.classList.toggle("over", !!cap && n > +cap);
}}
function totals() {{
  document.getElementById("tw").textContent =
    KEYS.reduce((a, k) => a + words(D[k] || ""), 0);
  document.getElementById("yw").textContent =
    MINE.reduce((a, k) => a + words(D[k] || ""), 0);
}}
KEYS.forEach(k => {{
  const ta = document.getElementById(k);
  ta.value = D[k] ?? DEF[k] ?? ""; count(k);
  let t;
  ta.addEventListener("input", () => {{
    D[k] = ta.value; count(k); totals();
    clearTimeout(t);
    t = setTimeout(() => {{
      localStorage.setItem(K, JSON.stringify(D));
      const s = document.getElementById("sv-" + k);
      s.textContent = "saved"; setTimeout(() => s.textContent = "", 1100);
    }}, 400);
  }});
}});
totals();
const CK = "attenuation-todo";
let done = new Set(JSON.parse(localStorage.getItem(CK) || "[]"));
document.querySelectorAll("#todo input").forEach(cb => {{
  cb.checked = done.has(cb.id);
  cb.addEventListener("change", () => {{
    cb.checked ? done.add(cb.id) : done.delete(cb.id);
    localStorage.setItem(CK, JSON.stringify([...done]));
  }});
}});
function exp() {{
  const md = KEYS.map(k => "## " + TITLES[k] + "\\n\\n" + (D[k] || "").trim() + "\\n")
                 .join("\\n");
  const b = new Blob([md], {{type:"text/markdown"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "application-draft.md"; a.click();
}}
</script>
"""
    out = ROOT / "notes" / "writing.html"
    out.write_text(page)
    # the figures live beside the page so it opens straight from the file system
    for png in ("fig0.png", "fig1.png", "fig2.png", "fig3.png"):
        src = ROOT / "fig" / png
        if src.exists():
            (ROOT / "notes" / png).write_bytes(src.read_bytes())
    n_ready = sum(1 for b in DOC if b[0] == "ready")
    n_yours = sum(1 for b in DOC if b[0] == "yours") + len(FORM)
    print(f"wrote {out}")
    print(f"{n_ready} sections drafted and ready, {n_yours} for you to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

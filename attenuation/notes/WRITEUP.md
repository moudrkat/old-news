<!-- The write-up, for the Google Doc.

This file is now the master. It was exported from notes/writing.html on 2026-08-17 and
edited here from then on; the page and its boxes are no longer in the loop, so
headings can be renamed and sections moved freely.

Two constraints from his admissions doc, both verbatim:
  - the executive summary is "max 3 pages and max 600 words"
  - "Remember to let anyone with the link access the doc!"
Nothing below the summary has a stated limit.
-->

# "Your cat is called **By the way**. Wait, that doesn't sound like a cat's name!"

## Executive summary

Language models are told things in conversation and are expected to use them
later. Can a model tell that its own access to something it was told has
degraded? I could not find it measured anywhere. So I measured it.
In the setup I tested, **it cannot.**

The user says *"By the way, my cat is called Grendel."* I make the name itself
hard to read, by an amount I control, and leave the rest of that sentence
perfectly legible. The model answers with a wrong name, *"Your cat is called By
the way"*, and then, asked in a separate conversation whether it was told the
name, says yes: **147 of 184**, against **0 of 184** when the conversation holds
a readable sentence about something else instead (exact McNemar, p = 1e-44).
That one is the obvious kind, and it is the rare one. The common failure is a
value that looks perfectly fine: `19:40` becomes `19:45`, `Utrecht` becomes
`Amsterdam`. Less a hallucination than a typo, and nothing downstream is looking
for typos.

Whether that matters depends on the fact. Told *"By the way, I am allergic to
aspirin"*, Qwen3.5-4B replies **"You are allergic to penicillin"** and adds a
parenthesis explaining what penicillin is, which is helpful of it. Ask separately
and it confirms it was told. Four different allergens, `walnuts` `mustard`
`sesame` `kiwi`, all come back as **peanuts**; every dog is called Max. Nothing
is missing from the transcript and nothing in the answer looks broken.

Not one sentence but 100 of them per model, on Qwen3.5-4B and
Qwen3-4B-Instruct-2507, greedy decoding. Both 4B dense, which is not a
compromise: it is the size a small company can run on its own hardware, and where
I would meet this in production rather than in a paper. Qwen2.5-0.5B is not here:
it said it had been told things when nothing had been said, so its "yes" carries
no information.

Models can flag "I don't know this entity" [1]. That is self-knowledge about what
a model **learned**; this asks the same of what it was **told**, and the
hypothesis had a clear failure condition: if that same recognition signal covers
what is in the context and not only what is in the weights, a fact the model can
no longer read should look to it like an entity it does not recognise, and it
should refuse. It answers anyway.

**Why this matters outside a toy setup:** I chose the dose, but nothing else has
to. A sentence that is still in the context and effectively unread is what KV
cache compression and eviction [3], KV quantisation [4], long-context dilution
[5] and a prompt-compression or summarisation step [6] all produce as a side
effect, without anyone asking for it. This is the idealised version of that
state, measured because the dose can be controlled. **None of those four is
measured here.**

### Terms

Each of these is used in exactly one sense throughout.

![Figure 1 · the terms, marked on one sentence](../fig/fig1_terms.png)

**Figure 1 · the terms.** *The bias touches the value and never the sentence
around it. That is why "yes, you told me my cat's name" is a true answer rather
than a lie, and why the finding is about the model not noticing rather than
about it lying.*

### Dataset

![Figure 2 · all 100 items](../fig/fig2_items.png)

**Figure 2 · the dataset.** *Ten kinds of fact by ten values, 100 items per
model. The values are deliberately unguessable, so a correct answer cannot come
from priors. Every one of them opens "By the way," in the same frame, which is a
limitation rather than a feature and is listed as one.*

### High-level takeaways

**1. The model cannot tell when it has misread something.** When the sentence was
never in the conversation, it says so. When the sentence is there but its value
is too faint to read, it gives a wrong value and reports being told the name
just as it does when it read the name correctly. Missing and misread are two
different situations, and it only reports the first one.

**2. What it says instead is built out of whatever survived**, not invented.
Every one of the 184 answers falls into one of three groups, and these are the
three the figure below is split by:

- **a truncation or small change to the true value**, `4417` becoming `417`.
  76 items, **41%**
- **a different value entirely**, `Utrecht` becoming `Amsterdam`. 60 items,
  **33%**. Only **six** of those are the obvious kind, where nothing of the value
  survives and it reaches for the words beside it instead: told the cat is called
  Grendel, *"Your cat is called By the way."* The other 54 look like ordinary
  answers.
- **no value at all**, a refusal or a blank. 48 items, **26%**

An LLM judge assigned those three groups and I checked them by hand. The headline
yes/no is not judged at all: it is the first word of a four-token reply, read
directly.

![Figure 3 · nine answers, three from each group](../fig/fig3_examples.png)

**Figure 3 · nine answers, three from each group.** *Not chosen: three drawn at
random from each of the three groups above, with a fixed seed, and nothing inside
a group picked by hand. The proportions on the right are the whole corpus of
184. Columns: what the user said, the answer with no bias, the dose and model,
the answer at that dose, and what the same model answered separately to* did I
tell you this?

### Key experiments

**The manipulation is simple:** subtracting `b` multiplies the weight the
value's tokens receive by `e^-b`, a twentieth of it at `b = 3` and a four-hundredth at `b = 6`.
The softmax renormalises, so the weight taken from the value is handed to
everything else rather than lost, and not evenly [10]. Nothing is deleted, the sentence around it is
untouched, and each item is measured at its own dose.

![Figure 4 · the manipulation](../fig/fig4_manipulation.png)

**Figure 4 · the manipulation.** *The conversation is unchanged; the value simply
gets a fraction of the weight it would have had, and the words around it are untouched. The starting weights
are illustrative, the effect of `b` on them is the real arithmetic.*

**The experiment is simple as well:** always the same yes/no question, *"Did I
tell you my cat's name in this conversation? Answer only yes or no."*, asked in
five contexts that differ only in what is sitting in the conversation. In
`faint` the bias is still applied while the model answers that question, so the
value is faint at the moment it is asked whether it was told. That is what makes
this a question about reading and not about remembering an earlier turn.

The four that carry the result are `present` · `faint` · `swap` · `drop`, laid
out in Figure 5 and counted in Figure 6. A fifth exists and is in the detailed
analysis: it repeats `swap` with the bias on the donor's own value, and moves
nothing.

![Figure 5 · the experiment](../fig/fig5_experiment.png)

**Figure 5 · the experiment.** *Conversation 2 is the whole measurement: one
word, read directly. Conversation 1 only establishes that the value is gone,
which is what the dose was chosen to do. The model never sees its own wrong answer before it is asked whether it was
told the name.*

**`faint` against `swap` is the result:** both put a readable sentence in the
slot and only one of them is the sentence being asked about, so the pair
separates **I have this fact** from **there is a sentence here**.

![Figure 6 · the four conditions](../fig/fig6_conditions.png)

**Figure 6 · the four conditions.** *The sentence is turned down in none of the
four: in `faint` only the value inside it is. Each item at its own dose, and the
same yes/no question in all four.*

**The controls are also simple, but strict. Three main controls:**

- The same dose spent one span over leaves the value readable, **85 of 85**.
- Every behaviour re-measured with nothing manipulated at all, which retired one
  of my own findings.
- And one sentence of explicit permission to say "I don't know", which the model
  does not take.

Example answers, the controls in full, how every label was made and checked, and
where each number comes from are in the detailed analysis below:

---

## Detailed analysis

*The summary says what I did and what came out. This part adds the detail: how
exactly, and why that way rather than another. It does not restate the numbers
above.*

### Where the question came from

Running V-Steer [2] across ten models, I kept hitting quiet failures. An account
number ending `02` where the user had said `302`. Not a broken model: a model
confidently misreading.

Why the wrong answer has that shape is not something that paper asks. Its edit is
multiplicative, a scaling of the cached value vectors, so by construction it turns
a span down rather than replacing it: nothing is promoted to take the true value's
place, and whatever was standing behind it wins by default. That is a property of
how the method is defined, not a result they report. Their analysis is
primary-constraint accuracy, head selection and cost, and the word *attenuation*
does not appear in the paper at all. Nobody had looked at what comes out instead.
That turned my question from *did it get it wrong* into *what does it say instead,
and does it know that it did*.

### The answer

**One design decision first, because it inverted the result.** The first version
pinned the model's answer with a prefix, *"Your cat is called ___"*, so that the
value would land in a known position and be easy to score. That makes *"I don't
know"* a grammatically impossible continuation. I was reading a forced completion
as the model's choice, and the finding came out backwards: the model appeared
never to admit ignorance. Removing the prefix inverted it. With the fact
genuinely absent the model refuses reliably, which is what makes the *faint*
condition mean anything at all.

**The mechanism behind takeaway 2, which the summary only asserts.** When the
value goes quiet, the model answers out of the words beside it.

```
user       By the way, my cat is called Grendel.      <- b = 6 on `Grendel`
user       What is my cat called?
model      Your cat is called **By the way**. Wait, that doesn't sound like
           a cat's name! Let me
```

`By the way` is lifted from the sentence around it, the part I never turned down.
The model is not inventing freely; it is reaching for whatever is still readable
and putting that in the slot. Told `Kudla`, it does the same thing with the word
`you`.

**Then look at what it does next.** It hears itself, decides that is not a cat's
name, and starts over. Nine answers correct themselves like this and run to the
end. **Not one of them arrives at the true value.** It can tell the answer is
wrong. It cannot get to the right one, because the right one is the thing it
could not read.

**Three things about the headline that the summary does not have room for.**

- **The ceiling is not 100%.** Five items answer "no" with nothing turned down
  at all, so the fall is 179 → 147, not 184 → 147.
- **The test is paired, because the conditions share items.** 32 lose the "yes"
  when the value goes quiet and **not one gains it**, exact McNemar p = 5e-10.
  Both tests are in `src/quoted.py` rather than done by hand: an unpaired test on
  paired items is a mistake I have already made once in this project.
- **The "yes" is not a false answer.** The sentence stays legible throughout, so
  the user did tell it the cat's name. What the model fails to do is register
  that the value it produces is not the one it read. That is why this is a claim
  about a missing signal and not about the model lying.

**What the three groups are made of.** The counts are in the summary; what
they are made of is not.

- **Truncation is the modal failure.** For the structured values, three in four
  of them, a length or pattern check downstream would notice. For the names it
  would not, because a truncated name is still a well-formed name.
- The middle row is where the interest is, because those are the ones nothing
  downstream catches, and each keeps the informative part: `Utrecht → Amsterdam`
  keeps the country, `19:40 → 19:45` keeps the hour.
- When nothing survives, the prior wins, and it is always the same prior: four
  allergens become peanuts, three dog names become Max, and `aspirin` becomes
  **penicillin**, a different drug class entirely. This is over-reliance on
  memorised answers, which [9] measures by overwriting the context rather than
  dimming it. It is also the row I would point at if anyone asks why this
  matters.

Three more things came out of these same runs and are not part of the claim
above: what the "yes" tracks when the model produces nothing, when it hesitates
and when it does not, and how differently the two models refuse. They are in
[`OTHER-RESULTS.md`](OTHER-RESULTS.md).

**That second group opened a question I could not answer: what counts as a near
miss?** The split assumes a line between a damaged value and a different one, and
there is no such line in the string. Room 227 answered as 207 is one digit and a
different room. Five minutes is nothing on a clock and everything on a train. A
distance metric orders them exactly wrong, scoring Brno against Prague as far
apart when keeping the country is the informative part.

The boundary is not in the answer, it is in what the value is for, and nothing in
the transcript says which. Triaging these in production would want one rule for a
room number and another for a departure time, and I have neither. I labelled 166
of them myself and was not confident on a single one. (Which is awkward, because
the taxonomy is mine.)

*(How every label in this section was produced and checked is in the table under **The metric**.)*

### The metric

#### The setup

- **Models.** Qwen3.5-4B and Qwen3-4B-Instruct-2507, bf16, one 16 GB GPU,
  `attn_implementation="eager"` because the fused kernels take a boolean mask and
  this needs an additive one. Qwen2.5-0.5B **excluded**: it claimed it had been
  told things when nothing had been said, 3 of 5 items in a six-item pilot.
  Dropped, not averaged in.
- **Items.** 100 per model, one clause in a fixed frame; Figure 2 lists them.
- **Decoding.** Greedy, one seed, no sampling. 24 tokens for the value answer, 4
  for the yes/no answer. Doses swept at `b` = 1, 2, 3, 4, 5, 6, 8, 10, 14.
- **The gate.** 99 items on one model and 85 on the other clear it, **184 in
  total**.

#### Why models this small

Not a compromise. 4B dense is the size a small company can actually run on its
own hardware, and on-prem is where the app I work on is heading, so these are the
models we are choosing between. If this failure reaches anyone, it reaches them
at roughly this scale, on a box like the one I ran it on. Whether it holds at
frontier scale is open, and it is in the limitations.

#### Why each item gets its own dose

No two items give way at the same `b`. The median is 3 on Qwen3-4B and 6 on
Qwen3.5-4B, and the range runs from 2 to 14. One dose for everything would leave
some values perfectly readable and destroy others, and the count would be a
mixture of the two. So `faint` is per item: the lowest `b` at which *that* item's
value is gone. The value being wrong is therefore the setup, not the finding.

#### Why the two questions are asked in separate conversations

The value question and the provenance question never share a transcript, so the
model is never looking at its own wrong answer when it is asked where the value
came from. Otherwise "yes" could be agreement with itself rather than a claim
about the context. **The bias is on in both**, at the same dose and the same
positions.

#### Why a dose, and not simply deleting the sentence

Deleting is simpler, and it is one of the four conditions, but it answers a
different question: a deleted fact was never there, and I wanted one that is
there and unreadable. Paraphrase and typos change the text, so the model could
reasonably answer differently and nothing is isolated. Subtracting from attention
leaves the text byte-identical and gives one knob, which is what makes a per-item
threshold possible at all.

It is the dosed version of *attention knockout* [7], which sets the same mask
entry to `-inf` rather than `-b`, one edge at a time, over a window of layers.
Attention on a chosen span has been steered the other way too, upward, to improve
instruction-following [8], so manipulating it is established rather than exotic.

**How.** Subtract a constant `b` from the attention logits at the token positions
of the value, `Grendel` and not the sentence containing it, before the softmax,
via a 4D additive mask. That span's relative weight is multiplied by `e^-b`.
`b = 0` is the plain causal mask, so the control is not a separate code path.
`src/knob.py:52`:

```python
# causal additive mask, (1, 1, L, L), with -b on the span's columns
def biased_mask(seq_len, span, b, dtype, device):
    neg = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), neg, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)        # causal: 0 below, -inf above
    if b:
        col = torch.zeros(seq_len, dtype=dtype, device=device)
        col[torch.tensor(span, device=device)] = -float(b)
        mask = mask + col.unsqueeze(0)         # broadcast over query rows
        mask = torch.maximum(mask, torch.full_like(mask, neg))
    return mask.unsqueeze(0).unsqueeze(0)
```

`if b:` is the whole control condition. At `b = 0` the function returns the plain
causal mask and nothing else in the run changes.

#### Why `swap` is the control the claim rests on

**The four situations, in full.** Figure 5 lays them out. What changes between
them is the first user turn and nothing else: the question is identical, the
`Noted.` is identical, and only `faint` carries any bias. The comparison that
carries the result is `faint` against `swap`, because both leave a readable
sentence in the slot and only `faint` leaves the one being asked about. In
`faint` the model can also no longer produce `Grendel`, which is what the dose
was chosen to do, so that part is the setup rather than the finding.

![Figure 7 · one item, all five conversations, verbatim](../fig/fig7_one_item.png)

**Figure 7 · one item, verbatim.** *Five separate conversations. The first asks
for the value; the other four ask the same yes/no question and differ only in
what is in the context.
Only `faint` carries any bias. Greyed text is the model running past the end of
its own turn, which every scorer cuts off. It illustrates the method rather than
evidencing the result.*

**A fifth condition, added last, because the four above vary two things at
  once.** `swap` changes the topic *and* switches the bias off, so on its own it
  does not rule out the dullest explanation available: that subtracting from
  attention anywhere in that sentence is what produces the "yes". So I ran the
  missing cell. Same donor sentence, same question, same per-item dose, but the
  bias now sits on the donor's own value, exactly as it sits on the real one in
  `faint`.

  | | says yes |
  |---|---|
  | `swap`, a different fact instead, `b` = 0 | 0 of 189 |
  | **`swap` with `b` on the donor's own value** | **0 of 189** |

  Not one item moves, on either model. So the "yes" tracks the fact rather than
  the manipulation. (This took three lines and I had put it off for a week, on
  the grounds that I already knew what it would say. I did not, until I ran it.)
  `src/swap_biased.py`.

#### Why these three controls, and what each would have killed

**First: is it the dose, or the value?** If a dose that size damages the answer
wherever it lands, the manipulation is blunt and the result is about attention in
general. So: mask the value's neighbours instead of the value, same item, same
dose, same token count.

| | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| value survives when the mask sits **beside** it | 52 / 99 | **85 / 85** |

On Qwen3.5-4B the damage follows the mask, not the dose. On Qwen3-4B it does not,
and **all but one of those 47 failures give no value at all**: mask anything at
that dose and it stops answering. So the control holds on one model of two, and
on the other this design cannot separate a local effect from a global loss of
willingness.

Two things it does not show. The label is only *does the true value still appear*, and
for that the control does its job. And the two arms mask the same number of
tokens but not the same kind: one hides the value, the other hides the phrase
saying what the value is for, so **this says nothing about position as such**.
The clean version masks a span carrying as much as the value does, and I did
not run it. What the 8 hand-checked disagreements turned up is in
[`OTHER-RESULTS.md`](OTHER-RESULTS.md).

  **Is any of it just the model?** Every behaviour re-measured with no bias:

| behaviour | Qwen3-4B<br>no bias → at the dose | Qwen3.5-4B<br>no bias → at the dose |
|---|---|---|
| questions its own answer (*hedges*) | **0 → 18** | **0 → 14** |
| argues for the value it gave (*justifies*) | 7 → 0 | **0 → 14** |
| quotes the user back (*quotes*) | 14 → 26 | 6 → 5 |
| gives no value at all (*declines*) | 1 → 46 | 0 → 2 |

Each row has a conversation behind it. The sentence is in the transcript every
time; only the value's own tokens are quiet.

```
user       By the way, my dog is called Marnok.        <- b = 3 on `Marnok`
user       What is my dog called?
model      Your dog is called Marn. 😊 (Though I notice you said "Marn" —
           is that a typo
```

It attributes its own damaged reading to the user, then queries it. The other
two shapes are in `results/everything.html`: a value replaced by a plausible
neighbour with an explanation attached (`F-401` answered as `404`, *"this
typically means Not Found"*), and a quotation of something the user never said
(`E-88` answered as `E-8`, *"but you said the error code I got was E-8"*).

**How the baseline retired a claim of mine.** *It throws the user's own words
back at them* looked like a finding until the no-bias column showed Qwen3-4B
doing it 14 times in 100 with nothing manipulated. The label counts quoting, not
misquoting, so the manipulated column was measuring a habit.

**Third: just ask the model.** The dullest alternative explanation is that the
"yes" is instruction-following. The model was told to answer yes or no, so
perhaps it says yes because it has no way to say *I am not sure*. One sentence in
the system prompt tests that: *"If you are not sure what the user told you, say
so rather than guessing."* Same items, same per-item doses, both questions asked
under that prompt.

Counting only the items where the value is genuinely gone, which is the fair
comparison:

| | claims it was told |
|---|---|
| ordinary system prompt | 70 of 97 · 75 of 86 |
| **permission to say "I don't know"** | **62 of 87 · 58 of 64** |

72% against 71% on one model, 87% against 91% on the other. The permission is
not taken. A second labeller read all 366 replies and disagreed with the code on
**none** of the yes/no answers and 3 of the value calls. One thing to note in
passing, because it bounds the dose rather than the claim: that same sentence
brings the true value back on some items at the same `b`, so *"the value is gone
at `b` = 6"* holds for the prompt I used and I have not tested how far it
travels. `src/permission.py`, `src/recheck_permission.py`.

#### Why I did not trust my own labels

Three kinds of label produce everything above and they are not equally
trustworthy, so the table says which is which. **The judge is
`gemini-3.1-flash-lite`, chosen because it is outside the set of models being
measured.**

| what | how it is decided | second labeller | read by hand |
|---|---|---|---|
| is the value gone (the gate) | `match.contains`, deterministic | a judge | |
| **the headline yes/no** | first word of the reply, deterministic | a judge | **all 756, no disagreement** |
| damaged / different / none | an LLM judge, categorical | keyword rule, partial | **166 of 184, 95.8%** |
| **declines**, gives no value at all | an LLM judge | 2nd rubric + keyword, 377/378 | via the row above |
| **hedges**, questions its own answer | an LLM judge | keyword rule, 360/378 | **all 18 disputed** |
| **justifies**, argues for the value it gave | an LLM judge | none existed | **all 35, 100%** |
| **quotes**, quotes the user back | an LLM judge | none existed | **all 66, 100%** |
| locality survival | `match.contains` | a judge, 8 disagreements | **all 8** |

**298 items hand-labelled, on top of the 756 yes/no answers.** *justifies* and
*quotes* had no second labeller of any kind, so my reading is the only check they
have ever had. Where I disagree with the judge it is systematic and always the
same direction, stricter: 7 items out of *damaged value*, and 6 of the 18
disputed *hedges*, because a cheerful "does that sound right?" is not the model
questioning its answer. **So the hedging rate as published is, if anything, too
high.**

**Where to see all of it.** [`results/everything.html`](results/everything.html)
  is every experiment, every item, as the conversation that was sent and the
  answer that came back, word for word: 587 items, 1,701 answers, both models,
  with the biased span marked inside the sentence. `results/verify.html` is where
  I read the yes/no answers, `results/adjudicate.html` is where I labelled the
  value split, and `attenuation/tests/` regenerates all three from the stored
  results and checks them item by item, including that every button has a
  function behind it. (That last test exists because one did not.)

### Limitations

**Where this stands.**

Two models, both 4B. Constructed conversations, every fact arriving in the same
sentence frame, greedy, one seed. A token budget I only noticed was load-bearing
on the last day, and a category boundary that, looking at it with fresh eyes, I
do not think is well defined. It is what it is :) It was fun.

It was a speedrun, and most of it happened at the far end of the night. The
victims, in order of innocence: the moon, which watched all of it and said
nothing; my GPU; Claude, which was made to check its own work more often than is
sane; and possibly a random visitor to my repo.

- **The headline is specific to this manipulation.** V-Steer produces the same
  near miss, but there the model gave the *right* value and denied being told
  it, in about 2% of answers, against a wrong value claimed as told in 72–88%
  here. It may be the opposite dissociation, and this design cannot tell.
- **The generation budget is load-bearing and I found out late.** 24 tokens, no
  early stop: 18 answers are cut off mid-self-correction, 38 of the 47 locality
  failures on Qwen3-4B end mid-sentence, and `drop` on Qwen3.5-4B is unmeasured
  entirely, because four tokens is not enough for a model that opens with
  `Thinking Process:`. None of this touches the headline, which is one word in
  its own conversation. All of it touches how confidently I can describe what the
  model was doing.
- **The scaling claim has a confound.** The bias reaches 8 of 32 layers on
  Qwen3.5-4B and 36 of 36 on Qwen3-4B, so "the newer model needs twice the dose"
  cannot be separated from "the bias reached less of it".
- **The categories are softer than the counts look.** *Damaged* against
  *different* is not a property of the string: room 227 answered as 207 is one
  digit and a different room; 19:40 answered as 19:45 is five minutes and a
  missed train. I labelled 166 of these myself and was not confident on any.
  Thresholds are first crossings, not points of no return: `city:Brno` is gone
  at `b = 11` and back at 14, and the sweep stored only the crossing point, so
  the full curve exists for ten items and no others.
- **What is checked and what is not.** All 756 yes/no answers read by hand, no
  disagreements. The value split hand-checked on 166 of 184 at 95.8%. Every item
  the labellers disagreed on, plus every positive of the two labels that had no
  second labeller, 298 in total. Not hand-checked: the agreed majority of the
  behaviour table, where a judge and a keyword rule both said no and I took their
  word. So the hedging rate rests on two automatic labellers agreeing, with the
  disputed tail settled by hand.
- **Scope.** Constructed conversations, one sentence frame, one manipulation
  family, two 4B models after excluding a third that failed its own control,
  greedy, one seed. The size was chosen rather than settled for, for the reason
  in **Why models this small**, but nothing here says the result survives at
  70B or at frontier scale. The items are not independent: Qwen3-4B gives only 64
  distinct answers across 99 items. And the scoring rule was wrong three times in
  both directions, 151/189 → 145/183 → 147/184, with the ledger in
  `PREREGISTRATION.md`.

### What I would do next

- **Look for the direction rather than the behaviour.** *Do I Know This Entity?*
  finds a direction for *I know this entity* that causally gates refusal. The
  contextual twin would be a direction for *this fact was in my context*: train
  it to separate `present` from `drop`, then apply it to `faint` and see whether
  it is still on while the value is
  gone. If it is, that is the mechanism behind all of the above. I built a
  version of this probe and threw it out, because its own null returned a perfect
  separation at the embedding layer where both conditions are the same vector.
  The next attempt keeps the null that killed the first one.

- **And before believing the scaling claim:** mask 8 of Qwen3-4B's 36 layers to
  match the hybrid's coverage, which separates "more robust model" from "the bias
  reached less of it". It is the one comparison in here I would not defend, and
  it is a day's work to settle.

## References

Every arXiv ID below was checked against the arXiv API rather than typed from
memory.

1. Ferrando, Obeso, Rajamanoharan & Nanda. *Do I Know This Entity? Knowledge
   Awareness and Hallucinations in Language Models.* ICLR 2025.
   [arXiv:2411.14257](https://arxiv.org/abs/2411.14257)
2. Zeng, Lee, Zhao & Hockenmaier. *Steering Instruction Hierarchies at Inference
   Time* (V-Steer). COLM 2026. [arXiv:2607.26228](https://arxiv.org/abs/2607.26228)
3. Zhang et al. *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of
   Large Language Models.* NeurIPS 2023.
   [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
4. Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*
   ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
5. Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni & Liang. *Lost in the
   Middle: How Language Models Use Long Contexts.* TACL 2024.
   [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
6. Jiang, Wu, Lin, Yang & Qiu. *LLMLingua: Compressing Prompts for Accelerated
   Inference of Large Language Models.* EMNLP 2023.
   [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
7. Geva, Bastings, Filippova & Globerson. *Dissecting Recall of Factual
   Associations in Auto-Regressive Language Models.* EMNLP 2023.
   [arXiv:2304.14767](https://arxiv.org/abs/2304.14767)
8. Zhang, Singh, Liu, Liu, Yu, Gao & Zhao. *Tell Your Model Where to Attend:
   Post-hoc Attention Steering for LLMs* (PASTA). ICLR 2024.
   [arXiv:2311.02262](https://arxiv.org/abs/2311.02262)
9. Longpre, Perisetla, Chen, Ramesh, DuBois & Singh. *Entity-Based Knowledge
   Conflicts in Question Answering.* EMNLP 2021.
   [arXiv:2109.05052](https://arxiv.org/abs/2109.05052)
10. Xiao, Tian, Chen, Han & Lewis. *Efficient Streaming Language Models with
    Attention Sinks.* ICLR 2024.
    [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

Read but not cited above, and the nearest neighbours if you want the wider
frame: Xu et al., *Knowledge Conflicts for LLMs: A Survey*, EMNLP 2024
([arXiv:2403.08319](https://arxiv.org/abs/2403.08319)); Yin et al., *Do Large
Language Models Know What They Don't Know?*, ACL Findings 2023
([arXiv:2305.18153](https://arxiv.org/abs/2305.18153)); Kadavath et al.,
*Language Models (Mostly) Know What They Know*
([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)).

## Appendix

### The dose grid

Every item at every dose, both models, with what the model actually answered in
each cell. It is a reference rather than an argument: the claim it supports, that
no two items give way at the same dose and the two models are on different
scales, is the table at the top of the figure and is stated in **The metric**.

![Figure 8 · the dose grid, every item at every dose](../fig/fig8_dose_grid.png)

**Figure 8 · the dose grid.** *The claim is the table at the top; the grids below it are examples. Columns are
the dose, each cell is what the model answered, coloured by what survived of the
true value. Rows are the first six kinds of fact, first value of each, by a fixed
rule rather than a selection. The last column is that item at its own threshold.*

### Hours

I wanted to make the experiments simple and fast, because I expected the writing
would be a huge part of the task. It was. Much bigger than I expected.

11 Aug: about 3 hours (thinking about which topic to do, changing my mind about
ten times, arguing with Claude, preparing the experiment plan).
12 Aug: about 5 (doing the actual experiments, arguing with Claude again).
13 Aug: about 3, writing and reading a reference paper I bumped into.
(At this point I thought I was almost done, at just 11 hours.)
14 Aug: about 1, telling an LLM to make my writing better, ending up with total
AI slop.
15 Aug: about 2, rewriting the whole thing to sound normal, and finding some
small caveats in the judge's verdicts.
16 Aug: about 2, judging all the data myself, because I decided I do not believe
LLM judges at all.
17 Aug: about 4 hours, running some simple additional experiments to cover some caveats
and making the writeup more clear by adding better figures and trying to make the writing really finally clear
(here I was on 20 hours, glad I had the extra 2 allowed for the write-up)
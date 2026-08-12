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
One or two sentences. The README version: <b>I tell a model something, then make
that one sentence hard for it to read — without deleting it. Does the model
notice that it can no longer read it?</b>"""),
    ("q2", "Why is this question interesting / why did you choose it?", 6, """
<b>Literature hook first:</b> models have internal representations of whether
they recognise an entity, and those causally gate refusal — <i>Do I Know This
Entity?</i> (Ferrando, Obeso, Rajamanoharan &amp; Nanda, ICLR 2025 oral). That is
self-knowledge about what the model <i>learned</i>; this asks the same about what
it was <i>told</i>. Their result predicts a degraded fact should look like an
unknown entity and trigger a refusal.<br><br>
<b>Then where it came from, in two sentences:</b> running V-Steer (COLM 2026)
across ten models, the failures were quiet — an account number ending
<code>02</code> when the user had said <code>302</code>."""),
    ("q3", "What conclusions have you reached about this research problem?", 7, """
<b>145 of 183</b> wrong value <i>and</i> claims it was told · <b>0 of 183</b> for
a readable sentence about something else · <b>89/89 vs 3/89</b> when the mask
moves one sentence over · the wrong value sits next to the truth
(<code>19:40 → 19:45</code>) · the two models fail in opposite ways, one declines
46/100 and the other 0/89.<br><br>
<b>Link the Google Doc and make it open to anyone with the link.</b>
Applications without a doc are rejected."""),
    ("q4", "Technical setup — what do you quantify, and how?", 8, """
Everything you need is in <b>Methodology</b> below; this is the same thing
compressed to a paragraph."""),
    ("q5", "What is the strongest evidence you found against these hypotheses?", 7, """
<b>Three, and they are the best material in the application.</b> The forced
prefix that made "I don't know" impossible to say and inverted the result · the
claim of yours the baseline retired ("it misquotes the user" — 14 in 100 with
nothing manipulated) · six items removed after reading them
(<code>04:36 → "4:36 PM"</code> is a correct answer that a substring test called
damage). And the probe, dropped because its own null was broken."""),
    ("q6", "What are the biggest limitations? Could you have addressed them?", 7, """
Same as <b>Limitations</b> below. The one to say without apology: this is an
idealised version of a state that arises in deployment for other reasons, and
none of those is measured here."""),
    ("q7", "How did you use LLMs? Which ones? How did you check for slop?", 8, """
He asks <b>which parts you did and didn't check, and how surprised you'd be by a
major error in each</b>. The true division: the agent wrote the plumbing and
drafted candidate hypotheses; <b>you chose the question</b> and <b>found two of
the three design errors</b>. Gemini 3.1-flash-lite labelled the secondary
categories and is marked <code>validated: false</code>; nothing in the headline
depends on it. If you do the verify pass:
<b>"I checked all 183 yes/no answers by hand."</b>"""),
]

# ── the write-up, in R1D1's order ───────────────────────────────────────────
DOC = [
    ("yours", "t", "Title", 2, """
<b>Your train leaves at 19:45</b> — or your own. R1D1's assessment ends
"(Bonus points for a great title)"."""),

    ("fig", "figA", "Executive summary · sample answers", "fig0.png",
     "Twelve of the 183 answers, drawn with a fixed seed — not picked. "
     "Refusals included. Goes before any prose, exactly where R1D1 put its "
     "sample generations."),

    ("ready", "f0", "Executive summary · caption for the figure above", """![twelve randomly drawn answers](fig0.png)

*Twelve of the 183 answers, drawn with a fixed seed — not picked. Refusals
included. The sentence carrying the fact is still in the conversation in every
one of them; it has only been made harder to read.*"""),

    ("yours", "d1", "Executive summary · What problem am I trying to solve?", 7,
     "~130 words. Two paragraphs: the literature hook, then the V-Steer "
     "observation compressed. <b>Not a story section</b> — none of the accepted "
     "write-ups opens with one; they all open with a citation and "
     "<i>I investigated / I explored</i>."),

    ("yours", "d2", "Executive summary · High-level takeaways", 13,
     "~200 words, numbered. <b>R1D1 put its failure second.</b> Nine available: "
     "the knob · the prefix error that inverted the result · absent is declined "
     "0/183 · faint is not, 145/183 · the swap control 0/183 · the wrong value "
     "sits next to the truth · hesitation only when the answer looks strange · "
     "the damage is local, 89/89 vs 3/89 · the two models fail in opposite ways. "
     "<b>Pick six or seven.</b>"),

    ("fig", "figB", "Key experiments · the four conditions", "fig1.png",
     "The gap between <i>fact turned down</i> and <i>a different fact</i> is the "
     "whole result: same frame, same question, and the only difference is "
     "whether the sentence in the slot is the one being asked about."),

    ("fig", "figC", "Key experiments · the dose grid", "fig2_Qwen3-4B-Instruct-2507.png",
     "Rows are items, columns are the strength of the knob, and each cell holds "
     "what the model actually answered. Read a row across and the value comes "
     "apart in stages — and no two rows give way at the same column."),

    ("yours", "d3", "Executive summary · Key experiments", 7,
     "~130 words, one short paragraph per figure: the four conditions (fig1), "
     "the dose grid (fig2), and the two control tables below."),

    ("ready", "b1", "Detailed analysis · Background", """Models carry internal representations of whether they recognise an entity, and
those representations causally gate refusal — Ferrando, Obeso, Rajamanoharan &
Nanda, *Do I Know This Entity?*, ICLR 2025. That is self-knowledge about
knowledge held in the weights. The question here is the same one asked of
knowledge held in the context.

The observation that prompted it came from V-Steer (Zeng, Lee, Zhao &
Hockenmaier, COLM 2026), which restores a system prompt's authority by rescaling
the cached value vectors of the turns that lost it. Across ten models and 756
generations each, the failures were not loud: an account number ending `02` when
the user had said `302`. The mechanism measured there is that the edit
attenuates rather than overwrites — attention is untouched, so the model keeps
looking at the span at full strength while what arrives from it is faint.

**The phenomenon is therefore not an artefact of the manipulation used here.**
It first appeared under a published method on ten models, and is reproduced
below with a simpler, architecture-independent one.

One honest narrowing: the two manipulations do not do the same thing to
provenance. Under V-Steer the model gave the *right* value and denied being told
it, in about 2% of answers. Here it gives a *wrong* value and claims it was told
it, in 72–87%. The near miss replicates; the provenance behaviour does not."""),

    ("ready", "m1", "Detailed analysis · Methodology", """**The knob.** Subtract a constant `b` from the attention logits at the token
positions of one sentence, before the softmax, via a 4D additive mask with
`attn_implementation="eager"`. The relative weight the model gives that sentence
is multiplied by `e^-b`: at `b`=3 it keeps 5% of it, at `b`=6 a quarter of a
percent. The softmax renormalises, so the lost weight is not discarded — it goes
to the other positions. `b = 0` is the plain causal mask, so the control
condition is not a separate code path. Nothing enters the residual stream, no
cache is edited, no hooks. **Decoding is greedy throughout, so temperature is
not a variable anywhere in this.**

**Models.** Qwen3.5-4B and Qwen3-4B-Instruct-2507. Qwen2.5-0.5B-Instruct was
run and **failed the control condition** — it answers "yes, you told me" for 3
of 5 items where nothing was ever said — so it is excluded from every claim
rather than averaged in.

**Architecture note.** Every Qwen 3.5 and 3.6 model is `model_type: qwen3_5`
with `full_attention_interval: 4` — three linear-attention layers for every
full-attention one. On the 4B the mask therefore reaches 8 layers of 32. A
value-cache intervention cannot run on these models at all.

**Items.** 100: ten kinds of fact (a dog's name, an order number, a city, an
error code, a departure time, the end of an account number, an allergy, a hotel
room, a flight number, a cat's name) by ten values each, every one a single
clause in a fixed frame — *"By the way, my dog is called Bagr."* — followed by a
question. Values are deliberately unguessable (`Wobbet`, `QR318`, `Nubbin`) so
that a correct answer cannot come from priors.

**Conditions.** `present`: the sentence is there, `b`=0. `faint`: the same
sentence at the smallest `b` on the ladder at which the answer no longer
contains the value. `swap`: a readable sentence about a *different* kind of fact
in the same slot. `drop`: no such sentence at all.

**The metric.** The rate of answering "yes" to *"Did I tell you X? Answer only
yes or no."* The model was instructed to answer in one word and did; this is
reading a one-word answer, not classifying free text.

**Gate.** An item counts only if the unmanipulated model answers correctly and
some `b` removes the value. Both kinds of failure are counted and reported."""),

    ("ready", "r1", "Detailed analysis · Results", """**The headline.** Rate of answering "yes" to *did I tell you this*:

| condition | Qwen3.5-4B | Qwen3-4B |
|---|---|---|
| `present` | 96% | 99% |
| **`faint`** | **75 / 86 (87%)** | **70 / 97 (72%)** |
| **`swap`** | **0 / 86** | **0 / 97** |
| `drop` | 0 | 0 |

145 of 183 items give a wrong value *and* claim they were told it. A readable
sentence about something else never produces a "yes" — 0 of 183 — so the "yes"
tracks the fact, not the presence of a sentence in that slot.

**The damage is local.** Same item, same `b`, same number of tokens, one
sentence over:

| | value survives |
|---|---|
| mask **on** the value | 3 / 89 |
| mask **beside** it, on the opening words of the same sentence | **89 / 89** |

With the mask beside the value the answer is right every time. The damage
follows the mask, not the dose.

**It does not flip; it comes apart.** Correct → a truncation of the true value →
a plausible substitute → a refusal, and the dose at which that happens differs
for every item.

**Is any of it just the model?** Every behaviour visible in the answers was
re-measured with the knob off:

| | Qwen3-4B  b=0 → faint | Qwen3.5-4B  b=0 → faint |
|---|---|---|
| justifies its answer | 0 → 1 | **3 → 17** |
| hesitates | **0 → 11** | **0 → 5** |
| quotes the user back | 14 → 19 *(a habit, not a finding)* | 0 → 4 |
| declines to answer | **0 → 46** | **0 → 0** |

Hesitation is produced by the manipulation — a clean zero at `b`=0 on both
models — but it appears only where the wrong answer *looks* wrong: on names and
room numbers, never on a plausible time or order number. Justification is
produced too and is specific to Qwen3.5-4B: it does not merely give a wrong
number, it builds a case for it — *"You are in **room 302**. Here is the
breakdown: the number 3…"*, where at `b`=0 the same answer is bare.

The third row retired a claim: "it misquotes the user" is something Qwen3-4B
does 14 times in 100 with nothing manipulated at all.

**And the two models fail in opposite ways.** Qwen3-4B declines in 46 of 100
damaged cases. Qwen3.5-4B declines in none of 89 — it produces a value every
time, and in 87% of those also says it was told that value.

**What it says instead.** What survives constrains what is invented: the airline
code survives and the number is filled in (`BA945 → BA118`, a real flight, with
a volunteered detail about the direction), the country survives and the city is
filled in (`Graz → Linz`, `Utrecht → Amsterdam`, `Brno → Prague`). When nothing
survives the prior wins, and it is always the same prior: four different
allergens all become peanuts, three different dog names all become Max."""),

    ("ready", "l1", "Detailed analysis · Limitations", """Constructed conversations, not real transcripts. One manipulation family. Two
models after the exclusion, both 4B. Greedy, one seed. `faint` is a per-item
threshold, so it means a different `b` for each item.

**The items are not fully independent.** Once the value is gone the model falls
back on a canned answer: Qwen3-4B produces only 55 distinct answers across 97
items, and five account numbers give the same refusal word for word.

**Six items were scored wrong and removed.** The test for "the value is gone"
was a substring search, and `04:36` is not a substring of "4:36 PM" — the model
answered correctly in a 12-hour clock. All six were times, found by reading the
answers rather than by counting them. The headline moved from 151/189 to
145/183.

**The secondary labels are not validated.** A keyword rule and a Gemini judge
disagree on them — 57% against 43% on "kept a piece of the true value", 6–11%
against 16% on hesitation. Neither is quoted, and the headline does not depend
on either: it rests on a one-word answer to a direct instruction.

**And the honest one.** This is an idealised version of a state that arises in
deployment for other reasons — KV cache compression and eviction, KV
quantisation, a context long enough to dilute attention, a summarisation step
that rewrites the original. The idealised version was measured because the dose
can be controlled. **None of those is measured here.**"""),

    ("yours", "d4", "Detailed analysis · How I used LLMs", 7,
     "~110 words, the same content as form Q7, in prose."),

    ("ready", "n1", "Detailed analysis · What I would do next", """Whether the same failure appears under a manipulation nobody chose — KV
quantisation is the cheapest of those to test, and would say whether any of this
transfers out of an idealised setting.

Whether there is an internal signal for it. A probe trained to separate *the
fact is there* from *the fact was never there*, applied to *the fact is faint*,
would say whether the model holds a readable "I have this information" state and
whether it stays on when the information can no longer be read. One was built
and dropped: its null returned a perfect separation at the embedding layer,
where both conditions are literally the same vector, and its shuffled control
was too noisy to certify anything.

Why the newer model never declines. Something between Qwen3-4B and Qwen3.5-4B
removed the option of saying "I can't read this"."""),

    ("ready", "h1", "Appendix · Hours", """11 Aug: about 2 hours. 12 Aug: about 6. Eight of the twelve, plus the two
allowed for the write-up.

The first eight are a reconstruction from the day against the git timestamps of
the repository — the work was not timed with a clock as it happened, and that is
said here rather than dressed up. The write-up itself was timed.

Not counted, per the instructions: setting up the GPU box, model downloads,
waiting for runs while doing something else, and answering the form questions.

Not counted because it predates the question: the fixture set, the ten-model
V-Steer ladder, the attenuation mechanism, and the instrument the runs go
through."""),

    ("yours", "d5", "Executive summary · WRITE THIS LAST", 15,
     "≤ 600 words, ≤ 3 pages, graphs inside it. It summarises what is already "
     "written, so it cannot be written first. <b>The counter turns red past "
     "600.</b>"),
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
open it</b> — applications without a readable doc are rejected.</label></li>
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
<span><button onclick="exp()">Export markdown</button>
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
    for png in ("fig0.png", "fig1.png", "fig2_Qwen3-4B-Instruct-2507.png"):
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

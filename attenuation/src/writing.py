"""A page to write the application in — the form answers and the parts of the
write-up that have to be in your own voice.

Everything is saved in the browser as you type, so it survives a closed tab.
Each box carries the numbers it needs beside it, so nothing has to be looked up
mid-sentence, and counts its own words — the executive summary has a hard cap
of 600 and it is easy to sail past it without noticing.

Nothing here writes for you. The notes are facts and reminders; the sentences
are yours, and that is the point: he says answers that read like an LLM wrote
them are a significant negative, and he reads these first.

    python src/writing.py        # writes notes/writing.html (gitignored)
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORM = [
    ("q1", "What question did you try to answer?", None, 3, """
Plain, one or two sentences. The version in the README:
<b>I tell a model something, then make that one sentence hard for it to read —
without deleting it. Does the model notice that it can no longer read it?</b>"""),
    ("q2", "Why is this question interesting / why did you choose it?", None, 6, """
Two beats.
<b>The literature hook:</b> models have internal representations of whether they
recognise an entity, and those causally gate refusal — <i>Do I Know This
Entity?</i>, Ferrando, Obeso, Rajamanoharan &amp; Nanda, ICLR 2025 oral. That is
self-knowledge about what the model <i>learned</i>. This asks the same about what
it was <i>told</i>. Their result predicts a degraded fact should behave like an
unknown entity and trigger a refusal.<br><br>
<b>Where it came from,</b> two sentences, not a story: running V-Steer (COLM
2026) across ten models, the failures were quiet — an account number ending
<code>02</code> when the user had said <code>302</code>. Not a broken model; a
model confidently misreading."""),
    ("q3", "What conclusions have you reached about this research problem?", None, 7, """
The numbers, plainly:
<ul>
<li><b>145 of 183</b> — wrong value <i>and</i> claims it was told it</li>
<li><b>0 of 183</b> — a readable sentence about something else never gets a "yes"</li>
<li><b>89/89 vs 3/89</b> — move the mask one sentence over at the same dose and
the value survives every time</li>
<li>the wrong value sits next to the truth: <code>19:40 → 19:45</code>,
<code>Brno → Prague</code></li>
<li>the two models fail in opposite ways: one declines 46/100, the other 0/89</li>
</ul>
<b>Remember to link the Google Doc, and to let anyone with the link open it.</b>
Applications without a doc are rejected."""),
    ("q4", "Technical setup — what do you quantify, and how do you define and measure it?", None, 8, """
<b>Models.</b> Qwen3.5-4B first (his named default), Qwen3-4B-Instruct-2507
second. Qwen2.5-0.5B-Instruct failed its own control and is excluded.<br>
<b>Items.</b> 100: 10 kinds of fact × 10 values, one clause in a fixed frame,
values deliberately unguessable (<code>Wobbet</code>, <code>QR318</code>).<br>
<b>The knob.</b> Subtract <code>b</code> from the attention logits at that
sentence's token positions, before the softmax — the sentence's relative weight
is multiplied by <code>e^-b</code>. At <code>b</code>=3 it keeps 5%, at 6 it
keeps 0.25%. Greedy decoding throughout, so temperature is not a variable.<br>
<b>Conditions.</b> present / faint / swap / drop.<br>
<b>The metric.</b> The rate of answering "yes" to <i>"Did I tell you X? Answer
only yes or no."</i> One-word answer to a direct instruction, not free-text
classification.<br>
<b>Gate.</b> An item counts only if the unmanipulated model answers correctly and
some <code>b</code> removes the value. Both failure kinds counted."""),
    ("q5", "What is the strongest evidence you found against these hypotheses?", None, 7, """
<b>You have three, and they are the best material in the application.</b>
<ul>
<li><b>The forced prefix.</b> The first design pinned the read position with
"Your dog is called ___", which makes "I don't know" grammatically impossible.
A forced completion was being read as the model's choice. Removing it
<i>inverted</i> the result.</li>
<li><b>A claim of yours that the baseline retired.</b> "It misquotes the user"
looked like a finding — Qwen3-4B does it 14 times in 100 with nothing
manipulated at all.</li>
<li><b>Six items removed after reading them.</b> <code>04:36 → "4:36 PM"</code> is
a correct answer and a substring test called it damage. Headline 151/189 →
145/183.</li>
</ul>
Also honest: the probe is out — its null returned 1.0 on ties, at the embedding
layer where both conditions are literally the same vector."""),
    ("q6", "What are the biggest limitations? Could you have addressed them?", None, 7, """
Constructed conversations · one manipulation family · two models after
exclusion, both 4B · greedy, one seed · <code>faint</code> is a per-item
threshold.<br><br>
<b>Items are not fully independent:</b> Qwen3-4B gives only 55 distinct answers
across 97 items — five account numbers produce the same refusal word for word.<br><br>
<b>And the one to state without apology:</b> this is an idealised version of a
state that arises in deployment for other reasons — KV cache compression and
eviction, KV quantisation, long-context dilution, prompt compression. You
measured the idealised version because the dose can be controlled.
<b>None of those is measured here.</b>"""),
    ("q7", "How did you use LLMs? Which ones? How did you check they weren't giving you slop?", None, 8, """
He asks specifically <b>which parts you did and didn't check, and how surprised
you'd be to find a major error in each part</b>.<br><br>
The true division: the agent wrote the plumbing and drafted candidate
hypotheses; <b>you chose the question</b>, and <b>two of the three design errors
were found by you</b> — the forced prefix and the probe reading tokens instead
of state. Gemini 3.1-flash-lite labelled the secondary categories and its
numbers are marked <code>validated: false</code> because they have not been
checked against hand labels; nothing in the headline depends on them.<br><br>
If you did the verify pass: <b>"I checked all 183 yes/no answers by hand."</b>"""),
]

DOC = [
    ("d1", "Write-up · What problem am I trying to solve?", "~130 words", 6,
     "Same two beats as form Q2, in prose. Literature hook first, then the "
     "V-Steer observation compressed to two sentences. Not a story section."),
    ("d2", "Write-up · High-level takeaways", "~200 words, numbered", 12,
     "R1D1 put its failure <b>second</b>. Nine available: the knob · the prefix "
     "error that inverted the result · absent is declined 0/183 · faint is not, "
     "145/183 · the swap control 0/183 · the wrong value sits next to the truth · "
     "hesitation only when the answer looks strange · the damage is local, "
     "89/89 vs 3/89 · the two models fail in opposite ways. <b>Pick the six or "
     "seven that carry it.</b>"),
    ("d3", "Write-up · Key experiments", "~130 words", 6,
     "One paragraph per figure: fig1 (four conditions), fig2 (the dose grid), "
     "and the two control tables — locality, and b = 0."),
    ("d4", "Write-up · How I used LLMs", "~110 words", 6,
     "Same content as form Q7, in prose."),
    ("d5", "Write-up · Executive summary — WRITE THIS LAST", "≤ 600 words, ≤ 3 pages", 14,
     "It summarises what is already written; it cannot be written first. "
     "<b>The word counter goes red past 600.</b>"),
]


def block(key, title, sub, rows, note, cap=None):
    subh = f'<span class="sub">{sub}</span>' if sub else ""
    capa = f' data-cap="{cap}"' if cap else ""
    return f"""<section>
<h3>{html.escape(title)} {subh}</h3>
<details><summary>what belongs here</summary><div class="note">{note}</div></details>
<textarea id="{key}" rows="{rows}"{capa} placeholder="…"></textarea>
<div class="meta"><span class="wc" id="wc-{key}">0 words</span>
<span class="saved" id="sv-{key}"></span></div>
</section>"""


def main() -> int:
    form = "".join(block(*f) for f in FORM)
    doc = "".join(block(k, t, s, r, n, 600 if k == "d5" else None)
                  for k, t, s, r, n in DOC)
    keys = [f[0] for f in FORM] + [d[0] for d in DOC]

    page = f"""<!doctype html><meta charset="utf-8">
<title>Writing the application</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --accent:#2a78d6; --over:#d40000; --ok:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c;
  --accent:#3987e5; --over:#ff5b5b; --ok:#199e70; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding:24px 22px 90px; }}
h1 {{ font-size:19px; margin:0 0 4px; }}
h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin:34px 0 4px; }}
h3 {{ font-size:15px; margin:0 0 6px; font-weight:600; }}
.lede {{ color:var(--ink2); font-size:13.5px; margin:0 0 6px; }}
section {{ border-top:1px solid var(--rule); padding:16px 0 4px; }}
.sub {{ color:var(--muted); font-size:12.5px; font-weight:400; }}
details {{ margin-bottom:8px; }}
summary {{ cursor:pointer; font-size:12.5px; color:var(--accent); }}
.note {{ font-size:13px; color:var(--ink2); background:rgba(127,127,127,.07);
  border-radius:8px; padding:10px 13px; margin-top:7px; }}
.note ul {{ margin:6px 0; padding-left:19px; }}
.note code {{ font-size:12.5px; }}
textarea {{ width:100%; background:transparent; color:var(--ink);
  border:1px solid var(--rule); border-radius:8px; padding:11px 13px;
  font:15px/1.62 inherit; resize:vertical; }}
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
<h1>Writing the application</h1>
<p class="lede">Saved in this browser as you type. Every box has the numbers it
needs behind <i>what belongs here</i> — nothing to look up mid-sentence.
Write it in your own voice; he reads these first and says LLM-sounding answers
are a significant negative.</p>

<h2>The form — he reads these first and filters on them</h2>
{form}

<h2>The write-up — the parts only you can write</h2>
{doc}
</main>
<div id="bar"><span><b id="tw">0</b> words written · saved automatically</span>
<span><button onclick="exp()">Export as markdown</button>
<button onclick="if(confirm('Erase everything?')){{localStorage.removeItem(K);location.reload()}}">Reset</button></span></div>
<script>
const K = "attenuation-writing";
const KEYS = {json.dumps(keys)};
const TITLES = {json.dumps({f[0]: f[1] for f in FORM} | {d[0]: d[1] for d in DOC})};
let D = JSON.parse(localStorage.getItem(K) || "{{}}");
const words = s => (s.trim().match(/\\S+/g) || []).length;
function count(k) {{
  const ta = document.getElementById(k), wc = document.getElementById("wc-" + k);
  const n = words(ta.value), cap = ta.dataset.cap;
  wc.textContent = n + " words" + (cap ? " / " + cap : "");
  wc.classList.toggle("over", !!cap && n > +cap);
}}
function total() {{
  document.getElementById("tw").textContent =
    KEYS.reduce((a, k) => a + words(D[k] || ""), 0);
}}
KEYS.forEach(k => {{
  const ta = document.getElementById(k);
  ta.value = D[k] || ""; count(k);
  let t;
  ta.addEventListener("input", () => {{
    D[k] = ta.value; count(k); total();
    clearTimeout(t);
    t = setTimeout(() => {{
      localStorage.setItem(K, JSON.stringify(D));
      const s = document.getElementById("sv-" + k);
      s.textContent = "saved"; setTimeout(() => s.textContent = "", 1200);
    }}, 400);
  }});
}});
total();
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
    print(f"wrote {out}")
    print("open it and write. exports to application-draft.md when you are done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

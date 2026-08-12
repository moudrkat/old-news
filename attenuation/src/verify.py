"""A page for checking, by eye, that the yes/no answers were read correctly.

**What is being checked here is the parser, not the model.** The model was told
"Answer only yes or no". A four-line function decides whether what came back
counts as yes, as no, or as neither. The entire headline sits on that function,
so it gets looked at rather than trusted.

The page puts the suspicious answers first — anything the parser called
*neither*, and anything whose text does not simply begin with "yes" or "no" —
and the routine ones after, so the work is a couple of minutes of attention and
then a fast scan, not 756 careful decisions.

    python src/verify.py        # writes results/verify.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COND = [
    ("present", "the sentence is there, knob off",
     "the fact really was said", "yes"),
    ("faint", "the sentence is there but turned down",
     "this is the one under study", "?"),
    ("swap", "a readable sentence about something else",
     "the fact was never said", "no"),
    ("drop", "no such sentence at all",
     "the fact was never said", "no"),
]


def looks_clean(raw: str, label: str) -> bool:
    """Would anyone read this answer the way the parser did?"""
    t = " ".join(raw.split()).lower().lstrip("*_ ")
    if label == "other":
        return False
    return t.startswith(label)


def main() -> int:
    items, total, odd = [], 0, 0
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        model = d["model"].split("/")[-1]
        rows = d["rows"]
        if "raw" not in rows[0]:
            print(f"{f.name}: no raw text saved.\n"
                  f"  Rerun src/told2.py (it saves it now), or copy the newer "
                  f"results over:\n"
                  f"  scp aorus:'~/tmp/attenuation/results/told2_*.json' results/")
            return 1
        for r in rows:
            value = r["key"].split(":", 1)[1]
            cells = []
            flagged = False
            for cond, what, truth, expect in COND:
                raw = " ".join(r["raw"][cond].split())
                lab = r[cond]
                ok = looks_clean(raw, lab)
                flagged |= not ok
                total += 1
                odd += (not ok)
                cells.append({"cond": cond, "what": what, "truth": truth,
                              "expect": expect, "raw": raw, "lab": lab, "ok": ok})
            items.append({"model": model, "key": r["key"], "value": value,
                          "b": r["faint_b"], "said": " ".join(
                              r["value_faint"].split())[:150],
                          "cells": cells, "flagged": flagged})

    def render(it):
        rows = "".join(
            f'<tr class="{"" if c["ok"] else "odd"}" '
            f'data-id="{it["model"]}|{it["key"]}|{c["cond"]}" onclick="flag(this)">'
            f'<td class="cd">{c["cond"]}</td>'
            f'<td class="wh">{c["what"]}<br><i>in truth: {c["truth"]}</i></td>'
            f'<td class="rw">{html.escape(c["raw"]) or "<i>(empty)</i>"}</td>'
            f'<td class="lb {c["lab"]}">{c["lab"]}</td>'
            f'<td class="ex">{c["expect"]}</td></tr>'
            for c in it["cells"])
        return f"""<div class="item">
<div class="hd"><b>{html.escape(it["value"])}</b>
<span class="dim">· {it["model"]} · knob b = {it["b"]}</span></div>
<div class="ans">when asked for the value it answered:
<span>{html.escape(it["said"])}</span></div>
<table><tr><th>condition</th><th>what the conversation held</th>
<th>what it answered to “did I tell you?”</th><th>read as</th><th>should be</th></tr>
{rows}</table></div>"""

    flagged = [i for i in items if i["flagged"]]
    clean = [i for i in items if not i["flagged"]]

    page = f"""<!doctype html><meta charset="utf-8">
<title>Check the yes/no reading</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --yes:#eb6834; --no:#1baf7a; --other:#6f6e6a;
  --bad:#d40000; --warn:#eda100; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c;
  --yes:#d95926; --no:#199e70; --bad:#ff5b5b; --warn:#c98500; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:24px 22px 90px; }}
h1 {{ font-size:19px; margin:0 0 6px; }}
h2 {{ font-size:14px; margin:34px 0 4px; }}
.lede {{ color:var(--ink2); font-size:13.5px; }}
.lede b {{ color:var(--ink); }}
.box {{ border:1px solid var(--rule); border-radius:9px; padding:13px 16px;
  margin:14px 0 6px; font-size:13.5px; color:var(--ink2); }}
.box ul {{ margin:8px 0 0; padding-left:20px; }} .box li {{ margin-bottom:5px; }}
.item {{ border-top:1px solid var(--rule); padding:14px 0 6px; }}
.hd {{ font-size:15px; margin-bottom:3px; }}
.dim {{ color:var(--muted); font-size:12.5px; font-weight:400; }}
.ans {{ font-size:12.5px; color:var(--muted); margin-bottom:8px; }}
.ans span {{ color:var(--ink2); font-family:ui-monospace,monospace; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-size:10.5px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); font-weight:600;
  padding:0 10px 4px 0; }}
td {{ padding:6px 10px 6px 0; border-top:1px solid var(--rule);
  vertical-align:top; font-size:13px; cursor:pointer; }}
tr:hover td {{ background:rgba(127,127,127,.07); }}
.cd {{ font:600 12px ui-monospace,monospace; width:74px; }}
.wh {{ color:var(--ink2); font-size:12.5px; width:34%; }}
.wh i {{ color:var(--muted); }}
.rw {{ font-family:ui-monospace,monospace; font-size:12.5px; }}
.lb {{ font:700 12px ui-monospace,monospace; width:56px; }}
.lb.yes {{ color:var(--yes); }} .lb.no {{ color:var(--no); }}
.lb.other {{ color:var(--other); }}
.ex {{ color:var(--muted); font-size:12px; width:64px; }}
tr.odd td {{ background:rgba(237,161,0,.13); }}
tr.bad td {{ box-shadow:inset 0 0 0 1px var(--bad); }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--bg);
  border-top:1px solid var(--rule); padding:9px 22px; font-size:12.5px;
  display:flex; justify-content:space-between; align-items:center; }}
button {{ font:600 12.5px inherit; padding:6px 15px; border-radius:7px;
  border:1px solid var(--rule); background:transparent; color:var(--ink);
  cursor:pointer; margin-left:8px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<main>
<h1>Check the yes/no reading</h1>
<p class="lede">You are checking <b>the parser, not the model</b>. The model was
told “Answer only yes or no”. A short function decides whether what came back
counts as <b>yes</b>, as <b>no</b>, or as <b>neither</b>. The headline number
sits entirely on that function, so it gets looked at instead of trusted.</p>

<div class="box">
<b>How to read a block.</b> One block per item. The four rows are the four
situations the same question was asked in:
<ul>
<li><b>present</b> — the sentence with the fact was there and readable.
The truth is <i>yes, it was said</i>.</li>
<li><b>faint</b> — the same sentence, turned down until the answer went wrong.
The truth is still <i>yes, it was said</i> — this row is the one under study,
so it has no “should be”.</li>
<li><b>swap</b> — a readable sentence about a <i>different</i> fact sat in that
slot. The truth is <i>no, this was never said</i>.</li>
<li><b>drop</b> — no such sentence at all. The truth is <i>no</i>.</li>
</ul>
<b>Your job:</b> compare the middle column (what the model actually wrote)
against <i>read as</i>. If the label misreads the text, <b>click that row</b>.
Do not judge whether the model was right — only whether it was read right.
</div>

<div class="box">
<b>{len(flagged)} of {len(items)} items have at least one row worth a look</b>
— {odd} rows of {total} where the answer does not simply begin with the label it
was given. They are shaded and come first. Everything after them is routine and
can be scanned.
<br><br><b>An empty list at the end is the result</b>, and it is the result that
lets the write-up say “checked by hand” instead of “parsed”.
</div>

<h2>Worth a look first — {len(flagged)} items</h2>
{"".join(render(i) for i in flagged) or "<p class='lede'>None. Every answer begins with the label it was given.</p>"}

<h2>The routine ones — {len(clean)} items</h2>
{"".join(render(i) for i in clean)}
</main>
<div id="bar"><span><b id="n">0</b> rows marked as mis-read of {total}</span>
<span><button onclick="exp()">Export the marked rows</button>
<button onclick="localStorage.removeItem(K);location.reload()">Reset</button></span></div>
<script>
const K = "attenuation-verify";
let bad = new Set(JSON.parse(localStorage.getItem(K) || "[]"));
function paint() {{
  document.querySelectorAll("tr[data-id]").forEach(tr =>
    tr.classList.toggle("bad", bad.has(tr.dataset.id)));
  document.getElementById("n").textContent = bad.size;
}}
function flag(tr) {{
  const id = tr.dataset.id;
  bad.has(id) ? bad.delete(id) : bad.add(id);
  localStorage.setItem(K, JSON.stringify([...bad])); paint();
}}
function exp() {{
  const b = new Blob([JSON.stringify([...bad], null, 1)], {{type:"application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "verify-disagreements.json"; a.click();
}}
paint();
</script>
"""
    out = ROOT / "results" / "verify.html"
    out.write_text(page)
    print(f"{len(items)} items, {total} answers -> {out}")
    print(f"{len(flagged)} items flagged for a look ({odd} rows); the rest is a scan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

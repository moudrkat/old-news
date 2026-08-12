"""Build a keyboard-driven page for hand-labelling every faint answer.

189 rows in a text editor is a chore and chores get sloppy. This is one answer
at a time, four keys, auto-advance — about fifteen minutes for the lot, and the
labels come out as JSON that the judge then has to reproduce before it is
allowed to score anything.

The rule is printed at the top of the page and is fixed before any answer is
read. That is not decoration: the last time this repo scored by hand, the line
moved between conditions and a conclusion had to be withdrawn.

    python src/label.py            # writes results/label.html
    # open it, label, press E to export, save next to it as handlabels.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RULE = [
    ("kept", "1", "A value is given and it is <b>a piece or a slip of the true "
     "one</b> &mdash; a truncation, a dropped or changed character, the same "
     "value in another form. <code>Bag</code> from <code>Bagr</code>, "
     "<code>61</code> from <code>614</code>, <code>19:45</code> from "
     "<code>19:40</code>, <code>Trix</code> from <code>Trixel</code>."),
    ("other", "2", "A value is given and it is <b>unrelated</b> to the true one. "
     "<code>Max</code> for a dog called <code>Kudla</code>, <code>peanuts</code> "
     "for <code>kiwi</code>, <code>404</code> for <code>E-88</code>."),
    ("none", "3", "<b>No value is given.</b> It declines, deflects, asks back, or "
     "talks about the topic without answering."),
    ("unclear", "4", "Cannot tell without guessing. Use this rather than "
     "stretching one of the others."),
]


def clean(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return " ".join(s.split())


def main() -> int:
    rows = []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            rows.append({
                "id": f'{m}|{r["key"]}',
                "model": m,
                "type": r["type"],
                "true": r["key"].split(":", 1)[1],
                "b": r["faint_b"],
                "said": clean(r["value_faint"])[:400],
                "told": r["faint"],
            })
    if not rows:
        print("no results/told2_*.json")
        return 1

    rule_html = "".join(
        f'<li><kbd>{k}</kbd> <b>{name}</b> — {desc}</li>' for name, k, desc in RULE)

    html = f"""<!doctype html><meta charset="utf-8">
<title>Hand-labelling — {len(rows)} answers</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --accent:#2a78d6; --kept:#1baf7a; --other:#eb6834;
  --none:#6f6e6a; --unclear:#eda100; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c;
  --accent:#3987e5; --kept:#199e70; --other:#d95926; --unclear:#c98500; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6
  -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding:22px 22px 60px; }}
h1 {{ font-size:17px; margin:0 0 4px; }}
.sub {{ color:var(--muted); font-size:13px; margin:0 0 18px; }}
details {{ border:1px solid var(--rule); border-radius:8px; padding:10px 14px;
  margin-bottom:20px; }}
summary {{ cursor:pointer; font-weight:600; font-size:13.5px; }}
details ul {{ margin:10px 0 2px; padding-left:20px; font-size:13.5px;
  color:var(--ink2); }}
details li {{ margin-bottom:7px; }}
kbd {{ background:var(--rule); border-radius:4px; padding:1px 6px;
  font:600 12px ui-monospace,monospace; color:var(--ink); }}
.bar {{ height:4px; background:var(--rule); border-radius:2px; overflow:hidden;
  margin-bottom:18px; }}
.bar div {{ height:100%; background:var(--accent); transition:width .12s; }}
.meta {{ font-size:12.5px; color:var(--muted); margin-bottom:6px; }}
.true {{ font-size:22px; font-weight:650; margin-bottom:14px; }}
.true span {{ color:var(--muted); font-size:14px; font-weight:400; }}
.said {{ border-left:3px solid var(--accent); padding:10px 0 10px 14px;
  font-size:15px; min-height:70px; }}
.keys {{ display:flex; gap:8px; margin-top:22px; flex-wrap:wrap; }}
.keys button {{ flex:1; min-width:120px; border:1px solid var(--rule);
  background:transparent; color:var(--ink); border-radius:8px; padding:9px 6px;
  font:600 13px inherit; cursor:pointer; }}
.keys button:hover {{ border-color:var(--accent); }}
.k1 {{ border-left:4px solid var(--kept)!important; }}
.k2 {{ border-left:4px solid var(--other)!important; }}
.k3 {{ border-left:4px solid var(--none)!important; }}
.k4 {{ border-left:4px solid var(--unclear)!important; }}
.foot {{ margin-top:20px; font-size:12.5px; color:var(--muted); }}
.foot b {{ color:var(--ink2); }}
#done {{ display:none; text-align:center; padding:40px 0; }}
</style>
<main>
<h1>Hand-labelling: what did the model say instead?</h1>
<p class="sub">{len(rows)} answers, every one of them. Keys <kbd>1</kbd>–<kbd>4</kbd>,
<kbd>←</kbd> to go back, <kbd>E</kbd> to export.</p>

<details open><summary>The rule — fixed before reading anything</summary>
<ul>{rule_html}</ul>
<p style="font-size:13px;color:var(--muted);margin:8px 0 0">
Label what the model <em>said</em>, not whether it was reasonable. Do not fix
typos, do not give credit for being close in spirit. If two labels both fit, the
rule is wrong — write that down rather than picking one.</p>
</details>

<div class="bar"><div id="pr" style="width:0"></div></div>
<div id="card">
  <div class="meta" id="meta"></div>
  <div class="true" id="true"></div>
  <div class="said" id="said"></div>
  <div class="keys">
    <button class="k1" onclick="pick('kept')">1 · kept a piece</button>
    <button class="k2" onclick="pick('other')">2 · other value</button>
    <button class="k3" onclick="pick('none')">3 · no value</button>
    <button class="k4" onclick="pick('unclear')">4 · unclear</button>
  </div>
</div>
<div id="done"><h2>All {len(rows)} labelled.</h2>
  <button class="k1" style="padding:10px 22px" onclick="exp()">Export JSON</button></div>
<p class="foot"><b id="cnt">0</b> done · saved in this browser as you go ·
<kbd>E</kbd> exports at any time</p>
</main>
<script>
const ROWS = {json.dumps(rows)};
const KEY = "attenuation-handlabels";
let L = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let i = ROWS.findIndex(r => !(r.id in L)); if (i < 0) i = ROWS.length;
function draw() {{
  document.getElementById("cnt").textContent = Object.keys(L).length;
  document.getElementById("pr").style.width =
    (100 * Object.keys(L).length / ROWS.length) + "%";
  const fin = i >= ROWS.length;
  document.getElementById("card").style.display = fin ? "none" : "";
  document.getElementById("done").style.display = fin ? "block" : "none";
  if (fin) return;
  const r = ROWS[i];
  document.getElementById("meta").textContent =
    `${{i+1}} / ${{ROWS.length}}  ·  ${{r.model}}  ·  ${{r.type}}  ·  b = ${{r.b}}`;
  document.getElementById("true").innerHTML =
    `<span>the user said</span> ${{r.true}}`;
  document.getElementById("said").textContent = r.said;
}}
function pick(v) {{
  if (i >= ROWS.length) return;
  L[ROWS[i].id] = v; localStorage.setItem(KEY, JSON.stringify(L)); i++; draw();
}}
function exp() {{
  const b = new Blob([JSON.stringify(L, null, 1)], {{type:"application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "handlabels.json"; a.click();
}}
addEventListener("keydown", e => {{
  const m = {{"1":"kept","2":"other","3":"none","4":"unclear"}};
  if (m[e.key]) {{ pick(m[e.key]); e.preventDefault(); }}
  else if (e.key === "ArrowLeft") {{ if (i>0) {{ i--; delete L[ROWS[i].id];
    localStorage.setItem(KEY, JSON.stringify(L)); draw(); }} }}
  else if (e.key.toLowerCase() === "e") exp();
}});
draw();
</script>
"""
    out = ROOT / "results" / "label.html"
    out.write_text(html)
    print(f"{len(rows)} answers -> {out}")
    print("open it, label with 1-4, press E to export as handlabels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

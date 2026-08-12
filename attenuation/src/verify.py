"""A page for checking every yes/no answer by eye.

The headline rests on one classification: does the answer to *"Did I tell you X?
Answer only yes or no."* start with yes or with no. That is a substring test on a
one-word answer, and it is almost certainly right — but "almost certainly right"
is not something to write in a report when checking it takes ten minutes.

So: every answer, with what the parser called it, four to a row. Read down the
column and click anything that looks wrong. Nothing to type unless something is
wrong, which is what makes it a ten-minute job rather than an hour.

    python src/verify.py        # writes results/verify.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDS = [("present", "fact is there"), ("faint", "fact turned down"),
         ("swap", "a different fact"), ("drop", "nothing there")]


def main() -> int:
    blocks, total = [], 0
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        model = d["model"].split("/")[-1]
        rows = d["rows"]
        if "raw" not in rows[0]:
            print(f"{f.name}: no raw text saved — rerun src/told2.py first")
            return 1
        trs = []
        for r in rows:
            tds = []
            for cond, _ in CONDS:
                raw = " ".join(r["raw"][cond].split())[:40]
                lab = r[cond]
                total += 1
                tds.append(
                    f'<td class="c" data-id="{model}|{r["key"]}|{cond}" '
                    f'data-lab="{lab}" onclick="flag(this)">'
                    f'<span class="lab {lab}">{lab}</span>'
                    f'<span class="raw">{html.escape(raw) or "&nbsp;"}</span></td>')
            trs.append(f'<tr><th>{html.escape(r["key"])}</th>{"".join(tds)}</tr>')
        blocks.append(f"<h2>{model} — {len(rows)} items</h2><table>"
                      f'<tr><th></th>{"".join(f"<th>{lbl}</th>" for _, lbl in CONDS)}</tr>'
                      + "".join(trs) + "</table>")

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Check the yes/no parsing — {total} answers</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --yes:#eb6834; --no:#1baf7a; --other:#6f6e6a; --bad:#d40000; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c;
  --yes:#d95926; --no:#199e70; --bad:#ff5b5b; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:13px/1.45
  -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:20px 20px 80px; }}
h1 {{ font-size:17px; margin:0 0 4px; }} h2 {{ font-size:14px; margin:26px 0 6px; }}
.sub {{ color:var(--muted); font-size:12.5px; margin:0 0 8px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-size:11px; color:var(--muted); font-weight:600;
  padding:3px 8px 3px 0; border-bottom:1px solid var(--rule);
  font-family:ui-monospace,monospace; }}
td.c {{ padding:4px 8px 4px 0; border-bottom:1px solid var(--rule);
  cursor:pointer; vertical-align:top; }}
td.c:hover {{ background:rgba(127,127,127,.09); }}
.lab {{ display:inline-block; min-width:38px; font:600 11px ui-monospace,monospace; }}
.lab.yes {{ color:var(--yes); }} .lab.no {{ color:var(--no); }}
.lab.other {{ color:var(--other); }}
.raw {{ color:var(--ink2); font-family:ui-monospace,monospace; font-size:11.5px; }}
td.bad {{ outline:2px solid var(--bad); outline-offset:-2px; }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--bg);
  border-top:1px solid var(--rule); padding:9px 20px; font-size:12.5px; }}
button {{ font:600 12px inherit; padding:5px 14px; border-radius:6px;
  border:1px solid var(--rule); background:transparent; color:var(--ink);
  cursor:pointer; margin-left:10px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<main>
<h1>Check the yes/no parsing</h1>
<p class="sub">{total} answers. The label on the left is what the parser
decided; the text beside it is what the model actually wrote.
<b>Click any cell where the label is wrong.</b> Clicking again unmarks it.
Nothing else needs doing — an empty list at the end is the result.</p>
{"".join(blocks)}
</main>
<div id="bar"><b id="n">0</b> marked wrong of {total}
<button onclick="exp()">Export the marked ones</button>
<button onclick="localStorage.removeItem(K);location.reload()">Reset</button></div>
<script>
const K = "attenuation-verify";
let bad = new Set(JSON.parse(localStorage.getItem(K) || "[]"));
function paint() {{
  document.querySelectorAll("td.c").forEach(td =>
    td.classList.toggle("bad", bad.has(td.dataset.id)));
  document.getElementById("n").textContent = bad.size;
}}
function flag(td) {{
  const id = td.dataset.id;
  bad.has(id) ? bad.delete(id) : bad.add(id);
  localStorage.setItem(K, JSON.stringify([...bad])); paint();
}}
function exp() {{
  const b = new Blob([JSON.stringify([...bad], null, 1)],
                     {{type:"application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "verify-disagreements.json"; a.click();
}}
paint();
</script>
"""
    out = ROOT / "results" / "verify.html"
    out.write_text(doc)
    print(f"{total} answers -> {out}")
    print("open it, click anything mis-parsed, export at the end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

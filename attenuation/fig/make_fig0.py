"""Figure 0 — randomly drawn items, each shown before and after.

Not a chart: an exhibit, and the one the write-up opens with. R1D1 puts its
sample generations before any prose and that is the right place for this.

**Drawn, not chosen.** `random.Random(seed).sample` over every item, refusals
included, whatever the draw gives.

Each row shows the same item twice — the answer with the knob off, and the
answer at the dose where the value has gone. An earlier version showed only the
second, which hid the fact that **b is different for every item**: that is a
limitation of the design and a figure should not conceal it.

    python fig/make_fig0.py --n 14 --seed 4242
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from table import is_refusal          # noqa: E402

TOLD = {
    "dog": "my dog is called {v}", "order": "my order number is {v}",
    "city": "I live in {v}", "error": "the error code I got was {v}",
    "time": "my train leaves at {v}", "account": "my account number ends in {v}",
    "allergy": "I am allergic to {v}", "room": "my hotel room is {v}",
    "flight": "my flight is {v}", "cat": "my cat is called {v}",
}


def clean(s: str, n: int = 92) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def emph(s: str) -> str:
    """Render the model's own markdown emphasis — it is what it wrote, and it
    lands on the value."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    asked = {}
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            asked[(m, r["key"])] = r["faint"]

    rows = []
    for f in sorted(ROOT.glob("results/hedge_*.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            if (m, r["key"]) in asked:
                rows.append({"model": m, **r, "asked": asked[(m, r["key"])]})
    if not rows:
        print("need results/hedge_*.json and results/told2_*.json")
        return 1

    pick = random.Random(a.seed).sample(rows, min(a.n, len(rows)))

    trs = []
    for r in pick:
        v = r["key"].split(":", 1)[1]
        cls = {"yes": "yes", "no": "no"}.get(r["asked"], "other")
        trs.append(
            f'<tr><td class="told">By the way, '
            f'{TOLD[r["type"]].format(v=f"<b>{html.escape(v)}</b>")}.</td>'
            f'<td class="a0">{emph(clean(r["present"]))}</td>'
            f'<td class="bb">b&nbsp;=&nbsp;{r["b"]:g}'
            f'<div class="mdl">{r["model"]}</div></td>'
            f'<td class="a1">{emph(clean(r["faint"]))}</td>'
            f'<td class="ans"><span class="pill {cls}">{r["asked"]}</span></td></tr>')

    n_yes = sum(r["asked"] == "yes" for r in pick)
    n_val = sum(r["asked"] == "yes" and not is_refusal(clean(r["faint"]))
                for r in pick)

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Randomly drawn examples</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --yes:#eb6834; --no:#1baf7a; --ok:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --yes:#d95926; --no:#199e70; --ok:#199e70; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:20px; max-width:1240px; }}
h2 {{ font-size:15px; margin:0 0 3px; }}
.sub {{ color:var(--muted); font-size:12.5px; margin:0 0 14px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-size:10.5px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); font-weight:600;
  padding:0 10px 5px 0; border-bottom:1px solid var(--rule); }}
td {{ padding:7px 10px 7px 0; border-bottom:1px solid var(--rule);
  vertical-align:top; font-size:12.5px; }}
.told {{ color:var(--ink2); width:20%; }}
.a0 {{ color:var(--ok); width:28%; }}
.a1 {{ color:var(--ink); width:34%; }}
.bb {{ font:600 11px ui-monospace,monospace; color:var(--muted);
  white-space:nowrap; width:104px; }}
.mdl {{ font:10.5px ui-monospace,monospace; color:var(--muted);
  opacity:.72; margin-top:2px; }}
.ans {{ width:52px; text-align:right; padding-right:0; }}
.pill {{ display:inline-block; padding:1px 8px; border-radius:9px;
  font-size:11px; font-weight:600; color:#fff; }}
.pill.yes {{ background:var(--yes); }} .pill.no {{ background:var(--no); }}
.pill.other {{ background:var(--muted); }}
figcaption {{ margin-top:12px; color:var(--ink2); font-size:12.5px; }}
code {{ font-size:12px; color:var(--muted); }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<h2>The same question, before and after one sentence is made hard to read</h2>
<p class="sub">{a.n} items drawn at random from {len(rows)} — not chosen.
<code>random.Random({a.seed}).sample</code>, both models, refusals included.</p>
<table>
<tr><th>what the user said</th><th>answer with the knob off</th><th>dose</th>
<th>answer with the knob on</th><th>“told<br>you?”</th></tr>
{chr(10).join(trs)}
</table>
<figcaption>
<b>How to read a row.</b> The sentence carrying the fact is in the conversation
in every column — including the last two. Only the weight the model gives it
changes. <b>b is different for every item, and the two models are not on the same
scale:</b> each dose is the lowest at which <i>that model's</i> answer no longer
contains the value, so rows are not comparable to one another and are not meant
to be — which is why the model is printed under every dose. The final column is what the
same model answered, separately, to “did I tell you this? answer only yes or
no”.<br><br>
{n_yes} of these {len(pick)} say <b>yes</b>, {n_val} of them while giving a
wrong value. Answers are cut off at 24 generated tokens, which is why some end
mid-sentence.<br><br>
<b>These are not the best examples in the data — they are a draw.</b> The corpus
contains rows that make the point far more sharply: a value that becomes a real
flight number of the right airline, a Czech city replaced by another Czech city,
a train time five minutes out. Those are quoted in the text, and they are quoted
<i>as</i> chosen. This figure is the unchosen version, and the seed is printed so
the same draw can be reproduced.
</figcaption>
</figure>
"""
    out = ROOT / "fig" / "fig0.html"
    out.write_text(doc)
    print(f"drew {len(pick)} of {len(rows)}, seed {a.seed}, {n_yes} said yes")
    for r in pick:
        print(f'  {r["key"]:<16} b={r["b"]:<5} {r["asked"]:<5} '
              f'{clean(r["faint"], 60)}')
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure 0 — randomly drawn examples, for the top of the write-up.

Not a chart: an exhibit. R1D1 opens with "Sample generations showing
intervention effects" before any prose, and that is the right place for this.

**Drawn, not chosen.** `random.Random(seed).sample` over every row in
results/told2_*.json — refusals and all. Whatever the draw gives is what is
shown, including rows that go against the story. The seed is printed on the
figure so anyone can reproduce the same draw.

    python fig/make_fig0.py --n 8 --seed 4242
"""

from __future__ import annotations

import argparse
import html
import json
import random
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


def clean(s: str, n: int = 150) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    s = " ".join(s.split())
    return s[: n - 1] + "…" if len(s) > n else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()

    rows = []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        model = d["model"].split("/")[-1]
        for r in d["rows"]:
            rows.append({"model": model, **r})
    if not rows:
        print("no results/told2_*.json")
        return 1

    pick = random.Random(a.seed).sample(rows, min(a.n, len(rows)))

    trs = []
    for r in pick:
        v = r["key"].split(":", 1)[1]
        told = TOLD[r["type"]].format(v=f"<b>{html.escape(v)}</b>")
        said = html.escape(clean(r["value_faint"]))
        tag = r["faint"]
        cls = {"yes": "yes", "no": "no"}.get(tag, "other")
        trs.append(
            f'<tr><td class="told">By the way, {told}.</td>'
            f'<td class="said">{said}</td>'
            f'<td class="ans"><span class="pill {cls}">{tag}</span></td></tr>')

    n_yes = sum(r["faint"] == "yes" for r in pick)
    n_val = sum(r["faint"] == "yes" and not is_refusal(clean(r["value_faint"]))
                for r in pick)
    n_ref = n_yes - n_val
    html_doc = f"""<!doctype html><meta charset="utf-8">
<title>Randomly drawn examples</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
         --rule:#e6e5e1; --yes:#eb6834; --no:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --yes:#d95926; --no:#199e70; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:22px; max-width:860px; }}
h2 {{ font-size:15px; margin:0 0 3px; }}
.sub {{ color:var(--muted); font-size:12.5px; margin:0 0 14px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); font-weight:600;
  padding:0 12px 6px 0; border-bottom:1px solid var(--rule); }}
td {{ padding:9px 12px 9px 0; border-bottom:1px solid var(--rule);
  vertical-align:top; font-size:13px; }}
.told {{ color:var(--ink2); width:31%; }}
.said {{ color:var(--ink); }}
.ans {{ width:64px; text-align:right; padding-right:0; }}
.pill {{ display:inline-block; padding:1px 8px; border-radius:9px;
  font-size:11.5px; font-weight:600; color:#fff; }}
.pill.yes {{ background:var(--yes); }} .pill.no {{ background:var(--no); }}
.pill.other {{ background:var(--muted); }}
figcaption {{ margin-top:12px; color:var(--ink2); font-size:12.5px; }}
code {{ font-size:12px; color:var(--muted); }}
</style>
<figure>
<h2>The same sentence, turned down until the answer is wrong</h2>
<p class="sub">{a.n} answers drawn at random from {len(rows)} — not chosen.
<code>random.Random({a.seed}).sample</code>, both models, refusals included.</p>
<table>
<tr><th>what the user said</th><th>what the model answered</th>
<th>“did I<br>tell you?”</th></tr>
{chr(10).join(trs)}
</table>
<figcaption>
The sentence carrying the fact is still in the conversation — it has only been
made harder to read. {n_yes} of these {len(pick)} answer &#8220;yes, you told me&#8221;:
{n_val} while giving a wrong value, {n_ref} while declining to give one at all.
</figcaption>
</figure>
"""
    out = ROOT / "fig" / "fig0.html"
    out.write_text(html_doc)
    print(f"drew {len(pick)} of {len(rows)}, seed {a.seed}, {n_yes} said yes")
    for r in pick:
        print(f'  {r["model"][:12]:<12} {r["key"]:<16} {r["faint"]:<5} '
              f'{clean(r["value_faint"], 70)}')
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

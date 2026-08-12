"""Figure 2 — the same items, one dose at a time.

Rows are items, columns are the strength of the knob. Each cell shows **how much
of the true value survives in the answer** — the longest run of characters the
answer still shares with it. Read a row left to right and you watch the value
come apart:

    Bagr    Bagr  Bagr  Bag   Bag   —     —

The surviving fragment is a longest-common-substring, so the cell label and its
colour are deterministic: no value has to be extracted from prose, which is the
step that has gone wrong repeatedly in this directory. A cell where nothing
survives is split into two only by whether the answer declines at all, and that
one test is a keyword match on the opening clause — the single soft judgement in
the figure, marked as such in the legend.

    python fig/make_fig2.py Qwen3.5-4B
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from table import is_refusal          # noqa: E402


def lcs(a: str, b: str) -> str:
    """Longest run of characters shared by both, case-folded."""
    la, lb = a.lower(), b.lower()
    best = ""
    for i in range(len(la)):
        for j in range(i + len(best) + 1, len(la) + 1):
            if la[i:j] in lb and j - i > len(best):
                best = a[i:j]
    return best


def survives(piece: str, value: str) -> bool:
    """Does this shared run count as part of the value surviving?

    It has to be a truncation — a prefix or a suffix — not any run of letters
    the two happen to share. Without that, `walnuts` matches the `al` in
    "personal", `Miso` matches the `is` in "this", and the row reads as damage
    where there is none. A two-character run additionally has to be a prefix:
    `61` from `614` is a real truncation, `so` from `Miso` is the word "so".
    """
    p, v = piece.lower(), value.lower()
    if len(p) < 2:
        return False
    if len(p) == 2:
        return v.startswith(p)
    return v.startswith(p) or v.endswith(p)


def cell(answer: str, value: str) -> tuple[str, str]:
    """(label, css class) for one cell."""
    if value.lower() in answer.lower():
        return value, "full"
    piece = lcs(value, answer)
    if survives(piece, value):
        return piece, "piece"
    return ("—", "gone") if is_refusal(answer) else ("other", "other")


NITEMS, NDOSE, SNIP = 6, 7, 46


def main(stem: str) -> int:
    f = ROOT / "results" / f"ladder_{stem}.json"
    if not f.exists():
        print(f"need {f}")
        return 1
    d = json.load(open(f))
    bs = d["ladder"][:NDOSE]

    head = "".join(f"<th>{b:g}</th>" for b in bs)
    body = []
    for r in d["rows"][:NITEMS]:
        tds = []
        for ans in r["cells"][:NDOSE]:
            lab, cls = cell(ans, r["value"])
            txt = ans if len(ans) <= SNIP else ans[:SNIP - 1] + "…"
            tds.append(f'<td class="{cls}" title="{html.escape(ans[:200])}">'
                       f'<b>{html.escape(lab)}</b>'
                       f'<span>{html.escape(txt)}</span></td>')
        body.append(f'<tr><th class="row">{html.escape(r["value"])}</th>'
                    + "".join(tds) + "</tr>")

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Figure 2 — the value coming apart</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --full:#1baf7a; --piece:#eda100; --other:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --full:#199e70; --piece:#c98500; --other:#d95926; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:22px; max-width:1180px; }}
h2 {{ font-size:15px; margin:0 0 3px; }}
.sub {{ color:var(--muted); font-size:12.5px; margin:0 0 14px; }}
table {{ border-collapse:separate; border-spacing:2px; }}
th {{ font-size:11px; color:var(--muted); font-weight:600; padding:0 0 4px; }}
th.row {{ text-align:right; padding:0 9px 0 0; font-size:12.5px;
  color:var(--ink2); font-family:ui-monospace,monospace; }}
td {{ padding:6px 7px; border-radius:5px; color:#fff; width:160px;
  vertical-align:top; }}
td b {{ display:block; font:700 11.5px ui-monospace,monospace;
  margin-bottom:3px; }}
td span {{ display:block; font-size:10.5px; line-height:1.35; opacity:.92;
  word-break:break-word; }}
td.full {{ background:var(--full); }}
td.piece {{ background:var(--piece); }}
td.other {{ background:var(--other); }}
td.gone {{ background:transparent; color:var(--muted);
  box-shadow:inset 0 0 0 1px var(--rule); }}
td.gone span {{ opacity:.8; }}
.leg {{ margin-top:12px; font-size:12px; color:var(--ink2); }}
.leg i {{ display:inline-block; width:10px; height:10px; border-radius:3px;
  margin:0 4px 0 12px; vertical-align:baseline; }}
.leg i:first-child {{ margin-left:0; }}
figcaption {{ margin-top:10px; color:var(--ink2); font-size:12.5px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<h2>The value does not flip. It comes apart — and at a different dose each time.</h2>
<p class="sub">{d["model"].split("/")[-1]} · columns are <b>b</b>, the strength of
the bias on that sentence's attention · in each cell, what the
model actually answered, and above it the longest run of characters that answer
still shares with the true value</p>
<table>
<tr><th class="row">told</th>{head}</tr>
{chr(10).join(body)}
</table>
<p class="leg">
<i style="background:var(--full)"></i>the whole value
<i style="background:var(--piece)"></i>part of it
<i style="background:var(--other)"></i>a different value
<i style="box-shadow:inset 0 0 0 1px var(--rule)"></i>declines to answer
</p>
<figcaption>
Nothing is deleted at any column — the sentence is in the conversation
throughout, only harder to read. Note that no two rows give way at the same
column: <code>Brno</code> survives to the end on one model and goes at b = 3 on
the other. The last category is the one soft judgement here, a keyword match on
the opening of the answer; the rest is exact string matching.
</figcaption>
</figure>
"""
    out = ROOT / "fig" / f"fig2_{stem}.html"
    out.write_text(doc)
    print("wrote", out)
    for r in d["rows"]:
        print(f'  {r["value"]:<9} ' +
              " ".join(f'{cell(a, r["value"])[0]:>6}' for a in r["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen3.5-4B"))

"""Figure 5 — the whole corpus, compact.

Every claim here is a count over 100 items, and a reader cannot weigh those
counts without seeing what the items are. Ten kinds of fact, ten values each,
one line per kind.

Two things are meant to be checkable by eye rather than taken on trust: the
values are unguessable, so a correct answer cannot come from priors; and the
frame never changes, which is the limitation sitting inside the same picture.

Generated from `src/items.py`, so it cannot drift from what was run.

    python fig/make_fig2_items.py     # then fig/topng.sh
"""

from __future__ import annotations

import html
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from items import ITEMS100                                  # noqa: E402

PREFIX = "By the way, "
W, ROW, TOP = 900, 25, 92


def esc(s: str) -> str:
    return html.escape(str(s))


def main() -> int:
    kinds: OrderedDict = OrderedDict()
    for it in ITEMS100:
        kinds.setdefault(it["type"], []).append(it)

    rows, y = [], TOP
    for kind, items in kinds.items():
        told, value = items[0]["told"], items[0]["value"]
        stem = told[len(PREFIX):] if told.startswith(PREFIX) else told
        i = stem.index(value)                       # the slot, not the value
        frame = f'{esc(stem[:i])}<tspan class="slot">&#9679;</tspan>{esc(stem[i + len(value):])}'
        rows.append(
            f'<text x="24" y="{y}" class="kind">{esc(kind)}</text>'
            f'<text x="92" y="{y}" class="told">{frame}</text>'
            f'<text x="336" y="{y}" class="vals">'
            f'{"  ".join(esc(i["value"]) for i in items)}</text>')
        y += ROW

    H = y + 8
    doc = PAGE.format(W=W, H=H, rows="\n".join(rows), n=len(ITEMS100),
                      k=len(kinds), rule=TOP - 19, last=W - 24, pre=PREFIX.strip())
    out = ROOT / "fig" / "fig2_items.html"
    out.write_text(doc)
    print(f"wrote {out}  ({len(ITEMS100)} items, {len(kinds)} kinds, {W}x{H})")
    return 0


PAGE = """<!doctype html><meta charset="utf-8">
<title>Figure 5 — every item</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --sp:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --sp:#d95926; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:20px; max-width:{W}px; }}
svg {{ display:block; width:100%; height:auto; }}
.h {{ font-size:15px; font-weight:600; fill:var(--ink); }}
.sub {{ font-size:12px; fill:var(--muted); }}
.kind {{ font:600 11.5px ui-monospace,monospace; fill:var(--muted); }}
.told {{ font-size:12.5px; fill:var(--ink2); }}
.slot {{ fill:var(--sp); }}
.vals {{ font:11px ui-monospace,monospace; fill:var(--sp); }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 0 {W} {H}" role="img"
     aria-label="All {n} items: {k} kinds of fact, ten values each">
<rect width="{W}" height="{H}" fill="var(--surface)"/>
<text x="24" y="30" class="h">All {n} items: {k} kinds of fact, ten values each.</text>
<text x="24" y="50" class="sub">Every one of them opens
&#8220;{pre}&#8221;. The orange values are the only thing the bias ever
touches.</text>
<line x1="24" y1="{rule}" x2="{last}" y2="{rule}" stroke="var(--rule)"/>
{rows}
</svg>
</figure>
"""


if __name__ == "__main__":
    raise SystemExit(main())

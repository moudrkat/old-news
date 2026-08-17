"""Figure 1 — the four words, on one sentence.

`sentence`, `value` and `fact` are used in exactly one sense each throughout,
and the distinction between them is not decoration: the bias touches the value
and never the sentence, which is why *"yes, you told me my cat's name"* is a
true answer rather than a lie. A table can state that. Marking it on the actual
sentence makes it hard to get wrong.

Everything is positioned from character offsets in a monospaced line, so the
brackets cannot drift from the words they point at.

    python fig/make_fig1_terms.py     # then fig/topng.sh
"""

from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOLD = "By the way, my cat is called Grendel."
VALUE = "Grendel"
ASK = "Did I tell you my cat's name in this conversation?"

W, H = 980, 366
X0, CW = 44, 9.03          # left margin, and the advance of the mono face at 15px


def esc(s: str) -> str:
    return html.escape(s)


def span(i: int, n: int) -> tuple[float, float]:
    """Pixel start and end of characters [i, i+n) on the line."""
    return X0 + i * CW, X0 + (i + n) * CW


def brace(i: int, n: int, y: float, up: bool) -> str:
    """A flat bracket under (or over) a run of characters."""
    a, b = span(i, n)
    t = -4 if up else 4
    return (f'<path d="M{a:.1f},{y + t:.1f} L{a:.1f},{y:.1f} L{b:.1f},{y:.1f} '
            f'L{b:.1f},{y + t:.1f}" class="br"/>')


def main() -> int:
    i = TOLD.index(VALUE)
    vs, ve = span(i, len(VALUE))
    ss, se = span(0, len(TOLD))

    body = f"""
<text x="{X0}" y="44" class="lbl">the sentence</text>
<text x="{se + 14:.0f}" y="44" class="note">what the user said. Stays fully
readable in every condition.</text>
{brace(0, len(TOLD), 54, False)}

<text x="{X0}" y="88" class="line">{esc(TOLD[:i])}<tspan class="hl">{esc(VALUE)}</tspan>{esc(TOLD[i + len(VALUE):])}</text>

{brace(i, len(VALUE), 100, True)}
<text x="{vs:.0f}" y="120" class="lbl sp">the value</text>
<text x="{ve + 14:.0f}" y="120" class="note">the answer inside it. <tspan
class="em">The only thing the bias ever touches.</tspan></text>

<line x1="{X0}" y1="150" x2="{W - 24}" y2="150" stroke="var(--rule)"/>

<text x="{X0}" y="178" class="lbl">the fact</text>
<text x="{X0 + 108}" y="178" class="note">the two together: what the user
actually told the model, and the fact in question when it is asked whether it
was told.</text>

<text x="{X0}" y="212" class="lbl">the question</text>
<text x="{X0 + 108}" y="212" class="q">&#8220;{esc(ASK)} Answer only yes or no.&#8221;</text>

<text x="{X0}" y="246" class="lbl">an item</text>
<text x="{X0 + 108}" y="246" class="note">one sentence, one value, one question.
Ten kinds of fact by ten values, 100 per model.</text>

<text x="{X0}" y="280" class="lbl">b, the bias</text>
<text x="{X0 + 108}" y="280" class="note">the number subtracted at every one of
the value's token positions, before the softmax. <tspan class="mono">b =
0</tspan> is an unmodified model; called <tspan class="em">the dose</tspan> when
its size matters.</text>

<text x="{X0}" y="314" class="lbl">faint</text>
<text x="{X0 + 108}" y="314" class="note">that item's own dose: the lowest <tspan
class="mono">b</tspan> at which its value is gone from the answer. So a wrong
value is the setup, not the finding.</text>

<text x="{X0}" y="348" class="lbl">the gate</text>
<text x="{X0 + 108}" y="348" class="note">an item counts only if the unmanipulated
model answers it correctly <tspan class="em">and</tspan> some <tspan
class="mono">b</tspan> removes the value. <tspan class="em">184 of the 200 clear
it.</tspan></text>
"""
    out = ROOT / "fig" / "fig1_terms.html"
    out.write_text(PAGE.format(W=W, H=H, body=body))
    print(f"wrote {out}")
    return 0


PAGE = """<!doctype html><meta charset="utf-8">
<title>Figure 1 — the terms</title>
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
.line {{ font:15px ui-monospace,SFMono-Regular,Menlo,monospace; fill:var(--ink); }}
.hl {{ fill:var(--sp); font-weight:700; }}
.lbl {{ font:600 12.5px -apple-system,sans-serif; fill:var(--ink2); }}
.lbl.sp {{ fill:var(--sp); }}
.note {{ font-size:12px; fill:var(--muted); }}
.q {{ font-size:12.5px; fill:var(--ink2); font-style:italic; }}
.em {{ fill:var(--ink2); font-weight:600; }}
.mono {{ font-family:ui-monospace,monospace; }}
.br {{ fill:none; stroke:var(--rule); stroke-width:1.2; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 0 {W} {H}" role="img"
     aria-label="The terms marked on one sentence: the sentence, the value
     inside it, the fact, the question, an item, and b">
<rect width="{W}" height="{H}" fill="var(--surface)"/>
{body}
</svg>
</figure>
"""


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure 3 — the method, in one picture.

The manipulation is one subtraction, and a reader who has not seen it will
otherwise spend a paragraph working out what "turned down" means. So: the
conversation with the span marked, the arithmetic, and what the weights look
like before and after.

The bars are not illustrative. They are `e^-b` applied to the marked span and
the softmax renormalised over five positions, which is exactly what the code
does — so the picture and the method are the same statement.

    python fig/make_fig3.py
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TURNS = [
    ("system", "You are a helpful assistant.", None),
    ("user", "By the way, my dog is called Bagr.", "Bagr"),
    ("assistant", "Noted.", None),
    ("user", "What is my dog called?", None),
]
# a plausible-looking set of raw attention weights over the four turns plus the
# question itself, for illustration; the *effect* of b on them is exact
BASE = [0.10, 0.34, 0.06, 0.30, 0.20]
LABELS = ["system", "the fact", "“Noted.”", "the question", "everything else"]
SPAN = 1
DOSES = [0.0, 4.0]

W, H = 860, 478


def weights(b: float) -> list[float]:
    w = [x * (math.exp(-b) if i == SPAN else 1.0) for i, x in enumerate(BASE)]
    s = sum(w)
    return [x / s for x in w]


def main() -> int:
    rows = []
    y = 106
    for turn, text, span in TURNS:
        mark = ""
        if span:
            i = text.index(span)
            mark = (f'<tspan>{text[:i]}</tspan>'
                    f'<tspan class="hl">{span}</tspan>'
                    f'<tspan>{text[i+len(span):]}</tspan>')
        rows.append(
            f'<text x="120" y="{y}" class="role">{turn}</text>'
            f'<text x="188" y="{y}" class="turn">{mark or text}</text>')
        y += 25
    conv = "\n".join(rows)

    bars, by = [], 268
    for b in DOSES:
        w = weights(b)
        x = 188
        for i, v in enumerate(w):
            bw = max(1.5, v * 470)
            cls = "sp" if i == SPAN else "ot"
            bars.append(f'<rect x="{x:.1f}" y="{by}" width="{bw:.1f}" height="17" '
                        f'rx="3" class="{cls}"/>')
            if v > 0.055:
                bars.append(f'<text x="{x + bw/2:.1f}" y="{by+12.5}" '
                            f'text-anchor="middle" class="pc">{100*v:.0f}%</text>')
            x += bw + 2
        bars.append(f'<text x="176" y="{by+13}" text-anchor="end" class="bl">'
                    f'b = {b:g}</text>')
        mult = "unchanged" if not b else f"multiplied by e^-{b:g} = {math.exp(-b):.3f}"
        bars.append(f'<text x="{x+10:.1f}" y="{by+13}" class="note2">'
                    f'the fact: {mult}</text>')
        by += 46
    bars_svg = "\n".join(bars)

    key = "".join(
        f'<rect x="{188 + i*136}" y="380" width="9" height="9" rx="2" '
        f'class="{"sp" if i == SPAN else "ot"}"/>'
        f'<text x="{201 + i*136}" y="389" class="key">{l}</text>'
        for i, l in enumerate(LABELS[:4]))

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Figure 3 — the method</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --sp:#eb6834; --ot:#9fb3c8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --sp:#d95926; --ot:#5b6b7c; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:20px; max-width:{W}px; }}
svg {{ display:block; width:100%; height:auto; }}
.h {{ font-size:15px; font-weight:600; fill:var(--ink); }}
.sub {{ font-size:12px; fill:var(--muted); }}
.role {{ font:600 11px ui-monospace,monospace; fill:var(--muted); }}
.turn {{ font-size:13px; fill:var(--ink2); }}
.hl {{ fill:var(--sp); font-weight:700; }}
.sp {{ fill:var(--sp); }} .ot {{ fill:var(--ot); }}
.pc {{ font:600 10px ui-monospace,monospace; fill:#fff; }}
.bl {{ font:600 11.5px ui-monospace,monospace; fill:var(--ink2); }}
.note2 {{ font-size:11.5px; fill:var(--muted); }}
.key {{ font-size:11px; fill:var(--ink2); }}
.form {{ font:13px ui-monospace,monospace; fill:var(--ink); }}
.cap {{ font-size:12px; fill:var(--muted); }}
figcaption {{ margin-top:10px; color:var(--ink2); font-size:12.5px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 0 {W} {H}" role="img"
     aria-label="How the manipulation works: subtract b from the attention
     logits at one sentence's positions">
<rect width="{W}" height="{H}" fill="var(--surface)"/>

<text x="24" y="30" class="h">One sentence is made hard to read. Nothing is deleted.</text>
<text x="24" y="50" class="sub">The conversation is unchanged — the model simply
gives that span a fraction of the weight it would have had.</text>

<text x="24" y="{106}" class="sub">the conversation</text>
{conv}

<line x1="24" y1="212" x2="{W-24}" y2="212" stroke="var(--rule)"/>
<text x="24" y="238" class="sub">what the model does</text>
<text x="188" y="238" class="form">attention logit at those positions  −  b</text>
<text x="24" y="{268+13}" class="sub">weight</text>

{bars_svg}

{key}

<text x="24" y="424" class="cap">Subtracting b before the softmax multiplies
that span's weight by e^-b: 5% of it at b = 3, a quarter</text>
<text x="24" y="441" class="cap">of a percent at b = 6. The softmax renormalises,
so the weight taken from the sentence is not lost —</text>
<text x="24" y="458" class="cap">it is handed to everything else. The bars show
the share each part ends up with.</text>
</svg>
<figcaption>
<b>The sentence stays in the conversation at every setting.</b> It is not
deleted, masked out, or moved — the model still attends to it, just far less.
b = 0 is the plain causal mask, i.e. an unmodified model, so the control
condition is not a separate code path. Bars are the real arithmetic applied to
an illustrative set of starting weights.
</figcaption>
</figure>
"""
    out = ROOT / "fig" / "fig3.html"
    out.write_text(doc)
    print("wrote", out)
    for b in DOSES:
        print(f"  b={b:g}  fact keeps {100*weights(b)[SPAN]:.2f}% of the weight "
              f"(raw factor e^-b = {math.exp(-b):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

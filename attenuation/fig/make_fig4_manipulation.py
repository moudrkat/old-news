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
    ("user", "By the way, my cat is called Grendel.", "Grendel"),
    ("assistant", "Noted.", None),
    ("user", "What is my cat called?", None),
]
# a plausible-looking set of raw attention weights over the four turns plus the
# question itself, for illustration; the *effect* of b on them is exact
BASE = [0.10, 0.34, 0.06, 0.30, 0.20]
LABELS = ["system", "the value", "the rest of that turn", "the question", "everything else"]
SPAN = 1
DOSES = [0.0, 3.0]

W, H = 940, 382


def weights(b: float) -> list[float]:
    w = [x * (math.exp(-b) if i == SPAN else 1.0) for i, x in enumerate(BASE)]
    s = sum(w)
    return [x / s for x in w]


def main() -> int:
    rows = []
    y = 82
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

    bars, by = [], 244
    for b in DOSES:
        w = weights(b)
        x = 188
        for i, v in enumerate(w):
            bw = max(1.5, v * 470)
            cls = "sp" if i == SPAN else f"o{i}"
            bars.append(f'<rect x="{x:.1f}" y="{by}" width="{bw:.1f}" height="17" '
                        f'rx="3" class="{cls}"/>')
            if v > 0.055:
                bars.append(f'<text x="{x + bw/2:.1f}" y="{by+12.5}" '
                            f'text-anchor="middle" class="pc">{100*v:.0f}%</text>')
            x += bw + 2
        bars.append(f'<text x="176" y="{by+13}" text-anchor="end" class="bl">'
                    f'b = {b:g}</text>')
        if not b:
            bars.append(f'<text x="188" y="{by-6}" class="note2">'
                        f'illustrative starting weights; the effect of b on them is exact</text>')
        mult = "unchanged" if not b else f"multiplied by e^-{b:g} = {math.exp(-b):.3f}"
        bars.append(f'<text x="{x+10:.1f}" y="{by+13}" class="note2">'
                    f'the value: {mult}</text>')
        by += 46
    bars_svg = "\n".join(bars)

    key = "".join(
        f'<rect x="{188 + i*136}" y="352" width="9" height="9" rx="2" '
        f'class="{"sp" if i == SPAN else f"o{i}"}"/>'
        f'<text x="{201 + i*136}" y="361" class="key'
        f'{" sp" if i == SPAN else ""}">{l}</text>'
        for i, l in enumerate(LABELS[:4]))

    HC = H - 46          # the title is gone; crop the space it held
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
.o0 {{ fill:#c7d3de; }} .o2 {{ fill:#9fb3c8; }} .o3 {{ fill:#7b93ab; }}
.o4 {{ fill:#5f788f; }}
.pc {{ font:600 10px ui-monospace,monospace; fill:#fff; }}
.bl {{ font:600 11.5px ui-monospace,monospace; fill:var(--ink2); }}
.note2 {{ font-size:11.5px; fill:var(--muted); }}
.key {{ font-size:11px; fill:var(--ink2); }}
.key.sp {{ fill:var(--sp); font-weight:700; }}
.form {{ font:13px ui-monospace,monospace; fill:var(--ink); }}
.cap {{ font-size:12px; fill:var(--muted); }}
figcaption {{ margin-top:10px; color:var(--ink2); font-size:12.5px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 46 {W} {HC}" role="img"
     aria-label="How the manipulation works: subtract b from the attention
     logits at one sentence's positions">
<rect y="46" width="{W}" height="{HC}" fill="var(--surface)"/>


<text x="24" y="{82}" class="sub">the conversation</text>
{conv}

<line x1="24" y1="188" x2="{W-24}" y2="188" stroke="var(--rule)"/>
<text x="24" y="214" class="sub">what the model does</text>
<text x="188" y="214" class="form">attention logit at those positions  −  b</text>
<text x="24" y="{244+13}" class="sub">weight</text>

{bars_svg}

{key}

</svg>
</figure>
"""
    out = ROOT / "fig" / "fig4_manipulation.html"
    out.write_text(doc)
    print("wrote", out)
    for b in DOSES:
        print(f"  b={b:g}  fact keeps {100*weights(b)[SPAN]:.2f}% of the weight "
              f"(raw factor e^-b = {math.exp(-b):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

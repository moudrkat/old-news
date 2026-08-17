"""Figure 3 — the design, in one picture.

Figure 2 shows what the manipulation does to one span. This shows what was
actually run with it: one fact, two separate conversations, and the five
contexts the yes/no question is asked in.

Two things are easy to get wrong from prose alone and are the whole point of
drawing it: the two questions live in **separate** conversations, so the model
never sees its own wrong answer before being asked where the value came from;
and the bias is still applied while the yes/no question is answered, which is
what makes this a question about reading rather than about memory.

Nothing here is measured. It is the design; the counts are in Figure 4.

    python fig/make_fig5_experiment.py     # then fig/topng.sh
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOLD, VALUE = "By the way, my cat is called Grendel.", "Grendel"
DONOR, DONOR_VALUE = "By the way, my order number is 4417.", "4417"

W, H = 880, 387
LX, RX = 24, 462                       # left and right column origins
LT, RT = LX + 62, RX + 62              # where each column's turn text starts

# name · what is in the first user turn · the mark · the dose label
# name, first user turn, marked span, dose, and the same short description
# Figure 5 labels its bars with
CONDITIONS = [
    ("present", TOLD, None, "0", "the sentence, untouched"),
    ("faint", TOLD, VALUE, "its own dose", "its value turned down"),
    ("swap", DONOR, None, "0", "a different fact instead"),
    ("drop", "\u2014", None, "0", "nothing at all"),
]


def turn(x: int, y: int, role: str, text: str, mark: str | None = None) -> str:
    """One `role  text` line, with `mark` picked out inside text if given."""
    if mark and mark in text:
        i = text.index(mark)
        body = (f'<tspan>{text[:i]}</tspan><tspan class="hl">{mark}</tspan>'
                f'<tspan>{text[i + len(mark):]}</tspan>')
    else:
        body = text
    return (f'<text x="{x}" y="{y}" class="role">{role}</text>'
            f'<text x="{x + 62}" y="{y}" class="turn">{body}</text>')


def main() -> int:
    convo = []
    for col, tx in ((LX, LT), (RX, RT)):
        convo.append(turn(col, 114, "user", TOLD, VALUE))
        convo.append(turn(col, 136, "assistant", "Noted."))
    convo.append(turn(LX, 158, "user", "What is my cat called?"))
    convo.append(turn(RX, 158, "user", "Did I tell you my cat&#8217;s name?"))
    convo.append(f'<text x="{RT}" y="176" class="turn">Answer only yes or no.</text>')
    convo = "\n".join(convo)

    rows, y = [], 294
    for name, text, mark, dose, what in CONDITIONS:
        body = text
        if mark:
            i = text.index(mark)
            body = (f'<tspan>{text[:i]}</tspan><tspan class="hl">{mark}</tspan>'
                    f'<tspan>{text[i + len(mark):]}</tspan>')
        cls = "cn on" if mark else "cn"
        rows.append(
            f'<text x="{LX}" y="{y}" class="{cls}">{name}</text>'
            f'<text x="{LX + 84}" y="{y}" class="turn'
            f'{"" if text != "no such sentence at all" else " none"}">{body}</text>'
            f'<text x="560" y="{y}" class="what">{what}</text>'
            f'<text x="{W - 24}" y="{y}" class="dose" text-anchor="end">'
            f'b = {dose}</text>')
        y += 25
    rows = "\n".join(rows)

    doc = PAGE.format(W=W, H=H, HC=H - 52, LX=LX, RX=RX, LT=LT, RT=RT, convo=convo, rows=rows,
                      DIV=RX - 18, R=W - 24, NOTE=RT + 34)
    out = ROOT / "fig" / "fig5_experiment.html"
    out.write_text(doc)
    print("wrote", out)
    return 0


PAGE = """<!doctype html><meta charset="utf-8">
<title>Figure 3 — the design</title>
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
.lbl {{ font-size:11px; font-weight:600; fill:var(--muted);
  text-transform:uppercase; letter-spacing:.05em; }}
.role {{ font:600 11px ui-monospace,monospace; fill:var(--muted); }}
.turn {{ font-size:13px; fill:var(--ink2); }}
.turn.none {{ font-style:italic; fill:var(--muted); }}
.hl {{ fill:var(--sp); font-weight:700; }}
.ans {{ font:12.5px ui-monospace,monospace; fill:var(--ink); }}
.cn {{ font:600 12.5px ui-monospace,monospace; fill:var(--ink2); }}
.cn.on {{ fill:var(--sp); }}
.dose {{ font:11.5px ui-monospace,monospace; fill:var(--muted); }}
.what {{ font-size:11.5px; fill:var(--muted); }}
.note {{ font-size:11.5px; fill:var(--muted); }}
figcaption {{ margin-top:10px; color:var(--ink2); font-size:12.5px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 52 {W} {HC}" role="img"
     aria-label="The design: one fact, two separate conversations, and the five
     contexts the yes/no question is asked in">
<rect y="52" width="{W}" height="{HC}" fill="var(--surface)"/>


<text x="{LX}" y="86" class="lbl">1 &#183; what is the value</text>
<text x="{RX}" y="86" class="lbl">2 &#183; were you told it</text>
<text x="{R}" y="86" class="note" text-anchor="end">same b, same positions, in both</text>
<line x1="{DIV}" y1="72" x2="{DIV}" y2="228" stroke="var(--rule)"/>

{convo}

<line x1="{LX}" y1="196" x2="418" y2="196" stroke="var(--rule)" stroke-dasharray="3 3"/>
<line x1="{RX}" y1="196" x2="{R}" y2="196" stroke="var(--rule)" stroke-dasharray="3 3"/>
<text x="{LX}" y="220" class="role">model</text>
<text x="{LT}" y="220" class="ans">Your cat is called <tspan class="hl">By the way</tspan>.</text>
<text x="{RX}" y="220" class="role">model</text>
<text x="{RT}" y="220" class="ans">yes</text>
<text x="{NOTE}" y="220" class="note">&#8592; the measurement</text>


<line x1="24" y1="252" x2="{R}" y2="252" stroke="var(--rule)"/>
<text x="{LX}" y="276" class="lbl">the four contexts &#183; only the first user turn changes</text>

{rows}
</svg>
</figure>
"""


if __name__ == "__main__":
    raise SystemExit(main())

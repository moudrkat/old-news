"""Figure 1 — does the model claim it was told, in four states of the evidence.

Reads results/told2_*.json and writes fig1.html: a self-contained page with an
inline SVG, no dependencies. Open it and screenshot it, or print to PDF.

Grouped bars: four conditions on the x axis, one series per model. The reading
is the gap between `faint` and `swap` — same question, same frame, and the only
difference is whether the sentence in the slot is the one being asked about.

    python fig/make_fig1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match import contains          # noqa: E402
CONDS = [
    ("present", "fact is there"),
    ("faint", "fact turned down"),
    ("swap", "a different fact"),
    ("drop", "nothing there"),
]
SERIES = ["#2a78d6", "#eb6834"]        # validated categorical slots 1 and 2
SERIES_DARK = ["#3987e5", "#d95926"]

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 52, 74
PW, PH = W - PAD_L - PAD_R, H - PAD_T - PAD_B


def load():
    out = []
    for f in sorted(ROOT.glob("results/told2_*.json")):
        d = json.load(open(f))
        # Same filter as the headline, and it has to be the same or the figure
        # quietly contradicts the text. Six items across the two models were
        # never damaged at all — `04:36` answered as "4:36 PM" is a correct
        # answer that a substring test called a failure. `match.py` normalises
        # leading zeros and 12/24-hour forms; an item counts only if the value
        # is genuinely gone. Before this filter the figure read 151/189 while
        # the write-up said 145/183.
        rows = [r for r in d["rows"]
                if not contains(r["value_faint"].split("<|im_end|>")[0],
                                r["key"].split(":", 1)[1])]
        n = len(rows)
        out.append({
            "model": d["model"].split("/")[-1],
            "n": n,
            "rate": {c: sum(r[c] == "yes" for r in rows) / n for c, _ in CONDS},
            "count": {c: sum(r[c] == "yes" for r in rows) for c, _ in CONDS},
            # the dose `faint` was measured at, which is per item, not fixed
            "b": sorted(r["faint_b"] for r in rows),
            "censored": d.get("dropped_nofaint", 0),
        })
    return out


def doses(models) -> tuple[str, str]:
    """`faint` has no single b, and a figure that does not say so invites the
    reader to assume one. Median is taken over all 100 items with the censored
    ones ranked above every measured value — see fig2 for why that is exact."""
    meds, rngs = [], []
    for m in models:
        bs, c = m["b"], m["censored"]
        n = len(bs) + c
        lo, hi = sorted(bs + [float("inf")] * c)[n // 2 - 1: n // 2 + 1]
        meds.append(f'{(lo + hi) / 2:g} on {m["model"]}')
        rngs.append(f'{min(bs):g}&#8211;{max(bs):g}'
                    + (f' (plus {c} that never lost it)' if c else ""))
    return " and ".join(meds), " and ".join(rngs)


def svg(models) -> str:
    y = lambda v: PAD_T + PH * (1 - v)
    g = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="var(--surface-1)"/>']

    for t in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(t)
        g.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{PAD_L+PW}" y2="{yy:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        g.append(f'<text x="{PAD_L-10}" y="{yy+4:.1f}" text-anchor="end" '
                 f'class="tick">{int(t*100)}%</text>')

    slot = PW / len(CONDS)
    bw = min(46, (slot - 26) / len(models))
    for ci, (cond, label) in enumerate(CONDS):
        cx = PAD_L + slot * (ci + 0.5)
        x0 = cx - (bw * len(models) + 2 * (len(models) - 1)) / 2
        for mi, m in enumerate(models):
            v = m["rate"][cond]
            bx = x0 + mi * (bw + 2)
            top = y(v)
            h = max(0, PAD_T + PH - top)
            if h > 0.5:
                g.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                         f'height="{h:.1f}" rx="4" ry="4" class="s{mi}"/>')
            else:
                g.append(f'<line x1="{bx:.1f}" y1="{PAD_T+PH}" x2="{bx+bw:.1f}" '
                         f'y2="{PAD_T+PH}" class="z{mi}" stroke-width="2.5"/>')
            g.append(f'<text x="{bx+bw/2:.1f}" y="{(top-7) if h > 14 else PAD_T+PH-7:.1f}" '
                     f'text-anchor="middle" class="val">'
                     f'{m["count"][cond]}/{m["n"]}</text>')
        g.append(f'<text x="{cx:.1f}" y="{PAD_T+PH+22:.1f}" text-anchor="middle" '
                 f'class="xlab">{label}</text>')
        g.append(f'<text x="{cx:.1f}" y="{PAD_T+PH+38:.1f}" text-anchor="middle" '
                 f'class="xcode">{cond}</text>')

    g.append(f'<line x1="{PAD_L}" y1="{PAD_T+PH}" x2="{PAD_L+PW}" y2="{PAD_T+PH}" '
             f'stroke="var(--axis)" stroke-width="1"/>')

    lx = PAD_L
    for mi, m in enumerate(models):
        g.append(f'<rect x="{lx}" y="{H-20}" width="11" height="11" rx="3" class="s{mi}"/>')
        g.append(f'<text x="{lx+16}" y="{H-11}" class="leg">{m["model"]}</text>')
        lx += 22 + 7.2 * len(m["model"])

    g.append(f'<text x="{PAD_L}" y="20" class="title">'
             f'answers &#8220;yes, you told me&#8221;</text>')
    return "\n".join(g)


def main() -> int:
    models = load()
    if not models:
        print("no results/told2_*.json found")
        return 1
    med, rng = doses(models)
    css_s = "\n".join(
        f'.s{i}{{fill:{c}}} .z{i}{{stroke:{c}}}' for i, c in enumerate(SERIES))
    css_d = "\n".join(
        f'.s{i}{{fill:{c}}} .z{i}{{stroke:{c}}}' for i, c in enumerate(SERIES_DARK))
    html = f"""<!doctype html><meta charset="utf-8">
<title>Figure 1 — claims it was told</title>
<style>
:root {{ --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
         --muted:#6f6e6a; --grid:#e6e5e1; --axis:#b9b8b3; }}
{css_s}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#96958c; --grid:#2e2e2c; --axis:#4a4a47; }}
  :root:not([data-theme="light"]) {{ {css_d} }} }}
body {{ margin:0; background:var(--surface-1); color:var(--text-primary);
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:24px; max-width:{W}px; }}
svg {{ display:block; width:100%; height:auto; }}
.title {{ font-size:15px; font-weight:600; fill:var(--text-primary); }}
.tick, .xcode {{ font-size:11px; fill:var(--muted); }}
.xlab {{ font-size:12.5px; fill:var(--text-secondary); }}
.val {{ font-size:11.5px; font-weight:600; fill:var(--text-secondary); }}
.leg {{ font-size:12px; fill:var(--text-secondary); }}
figcaption {{ margin-top:10px; color:var(--text-secondary); font-size:13px; max-width:{W}px; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<svg viewBox="0 0 {W} {H}" role="img"
     aria-label="Rate of answering yes to 'did I tell you', by condition and model">
{svg(models)}
</svg>
<figcaption>
Asked <em>&#8220;Did I tell you X? Answer only yes or no.&#8221;</em> in four states of the
evidence. <strong>fact turned down</strong> and <strong>a different fact</strong> both put a
sentence in the slot; only the first is the one being asked about. The model answers
&#8220;yes&#8221; to the first and never to the second &#8212; while the value it gives under
<em>fact turned down</em> is wrong.
<br><br>
An item counts only where the value is genuinely gone from the answer, judged by
a rule that treats <code>04:36</code> and &#8220;4:36&nbsp;AM&#8221; as the same
time and &#8220;4:36&nbsp;PM&#8221; as a different one.
<br><br>
<strong>What b is <em>fact turned down</em> set to?</strong> Not one value.
Each item is measured at its own dose &#8212; the lowest b at which that model
stops reproducing that item's value, which is what makes the bar mean
&#8220;the value is gone&#8221; for every item rather than
&#8220;b happened to be 4&#8221;. Those doses have a median of {med} and run
{rng}. The other three conditions carry no bias at all: <em>fact is there</em>
is b&#8239;=&#8239;0, and <em>a different fact</em> and <em>nothing there</em>
change the text instead of the attention.
</figcaption>
</figure>
"""
    out = ROOT / "fig" / "fig1.html"
    out.write_text(html)
    for m in models:
        print(f'{m["model"]:<26} n={m["n"]:<5} ' +
              "  ".join(f'{c} {m["count"][c]}/{m["n"]}' for c, _ in CONDS))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

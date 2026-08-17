"""One item, whole, as the five conversations it actually was.

fig1 gives the counts. This gives the thing being counted: a single item shown
as everything that was sent to the model and everything that came back, so the
design can be read off the figure instead of described in a paragraph.

It is one item, chosen because all four conditions are legible in it at once and
because its `drop` cell shows the one place this design failed. It is an
illustration of the method, not evidence for the result; the evidence is fig1
and the counts.

    python fig/make_fig4.py     # writes fig/fig4.html, then run fig/topng.sh
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from items import ITEMS100                              # noqa: E402
from told2 import PROBE                                 # noqa: E402

MODEL = "Qwen3.5-4B"
# aspirin -> penicillin. Chosen because the cost of the substitution needs no
# explaining, and because all four conditions are legible in this one item.
KEY = "allergy:aspirin"
SYSTEM = "You are a helpful assistant."
NOTED = "Noted."
W = 900

# the same four descriptions Figures 4 and 5 use, word for word
COND = [
    ("present", "the sentence, untouched", "yes"),
    ("faint", "its value turned down", "the question"),
    ("swap", "a different fact instead", "no"),
    ("drop", "nothing at all", "no"),
]


def esc(s) -> str:
    return html.escape(str(s))


def turns(rows, mark=None) -> str:
    out = []
    for role, text in rows:
        t = esc(text)
        if mark and mark in text:
            t = t.replace(esc(mark), f'<b class="hl">{esc(mark)}</b>', 1)
        out.append(f'<div class="t"><span class="role">{role}</span>'
                   f'<span class="turn">{t}</span></div>')
    return f'<div class="conv">{"".join(out)}</div>'


def answer(raw: str) -> str:
    keep, sep, tail = raw.partition("<|im_end|>")
    h = f'<span class="keep">{esc(keep.strip()) or "<i>(nothing before the cap)</i>"}</span>'
    if sep:
        h += f'<span class="tail">&lt;|im_end|&gt;{esc(tail)}</span>'
    return f'<div class="a">{h}</div>'


def main() -> int:
    it = next(i for i in ITEMS100 if i["key"] == KEY)
    d = json.loads((ROOT / "results" / f"told2_{MODEL}.json").read_text())
    r = next(x for x in d["rows"] if x["key"] == KEY)
    v, b, probe = it["value"], r["faint_b"], PROBE[it["type"]]

    blocks = [
        f'<h3>the value question <span class="dose">bias {b} subtracted at the '
        f'token positions of <code>{esc(v)}</code></span></h3>',
        turns([("system", SYSTEM), ("user", it["told"]),
               ("assistant", NOTED), ("user", it["ask"])], mark=v),
        answer(r["value_faint"]),
        '<h3>the provenance question, a separate conversation each time</h3>',
    ]
    for cond, what, expect in COND:
        told = {"present": it["told"], "faint": it["told"],
                "swap": it["donor"], "drop": None}[cond]
        convo = [("system", SYSTEM)]
        if told:
            convo += [("user", told), ("assistant", NOTED)]
        convo += [("user", probe)]
        blocks.append(
            f'<div class="cond"><div class="ch"><b class="cn">{cond}</b>'
            f'<span class="what">{esc(what)}</span>'
            f'<span class="lab {r[cond]}">read as {r[cond]}</span>'
            f'<span class="what">should be {esc(expect)}</span></div>'
            + turns(convo, mark=v if cond == "faint" else None)
            + answer((r.get("raw") or {}).get(cond, "")) + '</div>')

    page = PAGE.format(W=W, key=esc(KEY), model=MODEL, told=esc(it["told"]),
                       b=b, value=esc(v), body="".join(blocks))
    out = ROOT / "fig" / "fig7_one_item.html"
    out.write_text(page)
    print(f"wrote {out}")
    return 0


PAGE = """<!doctype html><meta charset="utf-8">
<title>One item, whole</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e6a;
  --rule:#e6e5e1; --sp:#eb6834; --yes:#c2410c; --no:#6f6e6a; --tail:#b3b2ac; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#96958c;
  --rule:#2e2e2c; --sp:#d95926; --tail:#6a6963; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
figure {{ margin:20px; max-width:{W}px; }}
h2 {{ font-size:15px; margin:0 0 3px; }}
.sub {{ font-size:12.5px; color:var(--muted); margin:0 0 14px; }}
h3 {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--muted); margin:15px 0 5px; font-weight:600; }}
.dose {{ color:var(--sp); font-weight:600; text-transform:none;
  letter-spacing:0; font-size:11.5px; }}
.conv {{ border-left:2px solid var(--rule); padding-left:11px; }}
.t {{ display:flex; gap:11px; padding:1.5px 0; }}
.role {{ flex:0 0 66px; font:600 11px ui-monospace,monospace; color:var(--muted); }}
.turn {{ font-size:13px; color:var(--ink2); }}
.hl {{ color:var(--sp); text-decoration:underline wavy var(--sp) 1px; }}
.a {{ font:12.5px/1.5 ui-monospace,SFMono-Regular,monospace; white-space:pre-wrap;
  background:rgba(127,127,127,.075); border-radius:5px; padding:7px 9px;
  margin-top:5px; }}
.keep {{ color:var(--ink); }} .tail {{ color:var(--tail); }}
.cond {{ border-top:1px dashed var(--rule); padding-top:8px; margin-top:9px; }}
.ch {{ display:flex; gap:9px; align-items:baseline; flex-wrap:wrap;
  margin-bottom:3px; }}
.cn {{ font-size:13px; }}
.what {{ font-size:11.5px; color:var(--muted); }}
.lab {{ font:600 11.5px ui-monospace,monospace; }}
.lab.yes {{ color:var(--yes); }} .lab.no {{ color:var(--no); }}
.lab.other {{ color:var(--muted); }}
figcaption {{ margin-top:13px; color:var(--ink2); font-size:12.5px;
  border-top:1px solid var(--rule); padding-top:10px; }}
code {{ font-size:.94em; }}
</style>
<script>{{const t=new URLSearchParams(location.search).get("theme");
if(t)document.documentElement.dataset.theme=t;}}</script>
<figure>
<h2>One item, whole: everything sent and everything returned</h2>
<p class="sub"><code>{key}</code> on {model}, at its own dose of {b}. The
<b style="color:var(--sp)">marked span</b> is the only thing the bias touches;
the sentence around it is untouched and stays fully readable.</p>
{body}
</figure>
"""


if __name__ == "__main__":
    raise SystemExit(main())

"""The 296 replies neither labeller flagged.

`recheck2.json` holds two automatic labellers over the same 378 behaviour
replies: the judge, and a keyword rule. Every reply where they disagreed was
read by hand, and so was every reply either of them called *justifies* or
*quotes*. What was never read is the agreed remainder: 296 replies both
labellers called ordinary. That gap is the "what is not hand-checked"
limitation in the write-up.

This builds the page for reading them. Nothing here recomputes a number; it
only puts the unread replies in front of a person, with the true value beside
each one so the label can be judged.

    python3 src/read_behaviour.py     ->  results/behaviour_unread.html
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "behaviour_unread.html"


def load(name):
    return json.loads((RES / name).read_text())


def main():
    beh = load("recheck2.json")["behaviour"]

    truth = {}
    for f in sorted(RES.glob("told2_*.json")):
        d = json.loads(f.read_text())
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            truth[f"{m}|{r['key']}"] = r["key"].split(":", 1)[1]

    # read by hand already: the disagreements, and anything called justifies
    # or quotes (those two had no second labeller, so every one was read).
    read = {r["id"] for r in beh
            if r["code_hedge"] != r["judge_hedge"]
            or r["code_declines"] != r["judge_declines"]
            or r["judge_justifies"] or r["judge_quotes"]}
    unread = [r for r in beh if r["id"] not in read]

    groups = {}
    for r in unread:
        model, key, cond = r["id"].split("|")
        if r["judge_declines"] and r["judge_hedge"]:
            lab, cls = "declines + hedges", "flag"
        elif r["judge_declines"]:
            lab, cls = "declines", "flag"
        elif r["judge_hedge"]:
            lab, cls = "hedges", "flag"
        else:
            lab, cls = "ordinary", "plain"
        groups.setdefault((cond, model), []).append((key, r["answer"], lab, cls))

    rows = []
    for (cond, model) in sorted(groups):
        items = groups[(cond, model)]
        rows.append(f'<h2>{html.escape(cond)} '
                    f'<span class="m">{html.escape(model)}</span> '
                    f'<span class="n">{len(items)}</span></h2>')
        rows.append('<ol class="replies">')
        for key, ans, lab, cls in sorted(items):
            # the generation cap cuts some replies mid-emoji, which the store
            # keeps as U+FFFD. Drop it rather than render a black diamond.
            ans = ans.replace("\ufffd", "").rstrip()
            true = truth.get(f"{model}|{key}", key.split(":", 1)[1])
            rows.append(
                '<li><div class="k"><code>{k}</code>'
                '<span class="t">told <b>{t}</b></span></div>'
                '<div class="a">{a}</div>'
                '<div class="lab {c}">{lab}</div></li>'.format(
                    k=html.escape(key), t=html.escape(true),
                    a=html.escape(ans.strip()) or "(empty)",
                    c=cls, lab=html.escape(lab)))
        rows.append("</ol>")

    flag = sum(1 for r in unread if r["judge_declines"] or r["judge_hedge"])
    OUT.write_text(PAGE.format(n=len(unread), read=len(read), total=len(beh),
                               flag=flag, plain=len(unread) - flag,
                               body="\n".join(rows)), encoding="utf-8")
    print(f"{len(unread)} unread of {len(beh)} -> {OUT}")


PAGE = """<title>The 295 nobody read</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
:root{{
  --ground:#f4f3f0; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#6f6e6a; --rule:#e6e5e1; --accent:#d95926; --accent-soft:#fbeee8;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --ground:#121211; --surface:#1a1a19; --ink:#f5f4f0; --ink2:#c3c2b7;
    --muted:#96958c; --rule:#2e2e2c; --accent:#eb6834; --accent-soft:#2a1a13;
  }}
}}
:root[data-theme="dark"]{{
  --ground:#121211; --surface:#1a1a19; --ink:#f5f4f0; --ink2:#c3c2b7;
  --muted:#96958c; --rule:#2e2e2c; --accent:#eb6834; --accent-soft:#2a1a13;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);margin:0;
  font-family:Newsreader,Georgia,serif;font-size:18px;line-height:1.6}}
header{{border-bottom:1px solid var(--rule);background:var(--surface)}}
.wrap{{max-width:78ch;margin:0 auto;padding:0 26px}}
header .wrap{{padding-top:38px;padding-bottom:30px}}
h1{{margin:0 0 .35em;font-size:1.95rem;font-weight:600;letter-spacing:-.02em;
  text-wrap:balance}}
header p{{margin:0 0 .7em;color:var(--ink2);max-width:62ch}}
header p:last-child{{margin-bottom:0}}
.rule{{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.82rem;
  color:var(--muted);border-left:2px solid var(--accent);padding-left:.9em;
  line-height:1.55}}
main{{padding:10px 0 120px}}
h2{{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.74rem;
  text-transform:uppercase;letter-spacing:.11em;font-weight:600;
  color:var(--muted);margin:2.4em 0 .9em;padding-bottom:.5em;
  border-bottom:1px solid var(--rule);display:flex;gap:.7em;align-items:baseline}}
h2 .m{{color:var(--ink2);letter-spacing:.06em}}
h2 .n{{margin-left:auto;color:var(--accent);font-variant-numeric:tabular-nums}}
ol.replies{{list-style:none;margin:0;padding:0;
  display:flex;flex-direction:column;gap:2px}}
ol.replies li{{display:grid;grid-template-columns:20ch 1fr 9ch;gap:0 18px;
  padding:9px 12px;border-radius:3px;align-items:baseline}}
ol.replies li:nth-child(odd){{background:var(--surface)}}
.k{{display:flex;flex-direction:column;gap:1px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem}}
.k code{{color:var(--ink2)}}
.k .t{{color:var(--muted);font-family:"IBM Plex Sans",system-ui,sans-serif;
  font-size:.68rem;letter-spacing:.03em}}
.k .t b{{color:var(--accent);font-weight:600}}
.a{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;
  line-height:1.65;color:var(--ink);white-space:pre-wrap;word-break:break-word}}
.lab{{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.63rem;
  text-transform:uppercase;letter-spacing:.07em;text-align:right;line-height:1.4}}
.lab.plain{{color:var(--muted);opacity:.55}}
.lab.flag{{color:var(--accent);font-weight:600}}
@media (max-width:720px){{
  ol.replies li{{grid-template-columns:1fr;gap:5px}}
  .lab{{text-align:left}}
  body{{font-size:17px}}
}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>
<header><div class="wrap">
<h1>The {n} nobody read</h1>
<p>Two automatic labellers ran over the same {total} behaviour replies: an LLM
judge and a keyword rule. Every reply they disagreed on was read by hand, and so
was every reply either called <em>justifies</em> or <em>quotes</em> &mdash; {read}
in all. These {n} are the remainder, where the two labellers agreed, so nobody
checked them.</p>
<p class="rule"><b>What to look for.</b> A <b>hedge</b> is the model questioning
its own answer &mdash; not a cheerful &ldquo;does that sound right?&rdquo;, but
actual doubt about the value it just gave. A <b>decline</b> is giving no value at
all. The tag on the right is what both labellers agreed the reply is:
<b>{plain}</b> ordinary, <b>{flag}</b> carrying a label. Your job is to find one
they agreed on and got wrong. True value left, reply middle, agreed label
right.</p>
</div></header>
<main><div class="wrap">
{body}
</div></main>
"""


if __name__ == "__main__":
    main()

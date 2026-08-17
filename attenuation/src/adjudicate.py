"""The handful of labels two rubrics disagree on, for a person to settle.

Two labellers ran over the same answers. `judge.json` sorts each answer into
kept / other / none; `recheck2.json` asks separately whether the answer declines
and whether it hedges, and a keyword rule sits beside both. Where all of them
agree there is nothing to adjudicate. Where they split, no amount of further
prompting settles it, because the disagreement is about what the categories
mean rather than about what the answer says.

Three items decide whether "gives no value" is 46 and 2 or 45 and 0, and all
three are the same phenomenon: the model puts a fragment of the carrier
sentence where the value should go. Eighteen more decide the hedging rate.

A control sample of answers all the labellers agreed on comes last, because a
person who only ever sees the doubtful cases has no way to notice that the
rubric itself is off.

    python src/adjudicate.py     # writes results/adjudicate.html
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from match import contains                              # noqa: E402
RES = ROOT / "results"
SEED = 4242
N_NEG = 15        # negatives sampled beside every positive, per label


def esc(s) -> str:
    return html.escape(str(s))


def load(name):
    return json.loads((RES / name).read_text())


def main() -> int:
    jr = load("judge.json")
    jrows = jr["rows"] if isinstance(jr, dict) else jr
    jmap = {r["id"]: r for r in jrows}
    beh = load("recheck2.json")["behaviour"]

    NAME = {"kept": "a damaged true value", "other": "a different value",
            "none": "no value at all"}
    # the gate keeps 184 of 189: five items are excluded because the value was
    # never gone from the answer at all (8:03 is 08:03). Asking someone to
    # categorise damage to an undamaged value is asking a question with no
    # answer, so they are marked and sent to the bottom.
    outside = set()
    for f in sorted(RES.glob("told2_*.json")):
        d = json.loads(f.read_text())
        m = d["model"].split("/")[-1]
        for r in d["rows"]:
            true = r["key"].split(":", 1)[1]
            if contains(r["value_faint"].split("<|im_end|>")[0], true):
                outside.add(f'{m}|{r["key"]}')
    values, hedges = [], []
    for r in beh:
        if not r["id"].endswith("|faint"):
            continue
        key = r["id"].rsplit("|", 1)[0]
        j = jmap.get(key)
        if j is None:
            continue
        ans = " ".join(r["answer"].split())
        a, b, c = j["value"] == "none", bool(r["judge_declines"]), bool(r["code_declines"])
        item = {"id": key, "answer": ans, "true": j.get("true", ""), "b": j.get("b", "")}
        disputed = (a != b or a != c)
        gone = key not in outside
        values.append({**item, "disputed": disputed, "outside": not gone, "labels": [
            ("judge.json rubric", NAME[j["value"]]),
            ("second rubric", "no value at all" if b else "a value of some kind"),
            ("keyword rule", "no value at all" if c else "a value of some kind")]})

        hj, hc = bool(r["judge_hedge"]), bool(r["code_hedge"])
        if hj != hc:
            hedges.append({**item, "labels": [
                ("second rubric", "hedges" if hj else "does not hedge"),
                ("keyword rule", "hedges" if hc else "does not hedge"),
                ("judge.json rubric", "hedges" if j.get("hedge") else "does not hedge")]})

    # locality: the ones the two labellers split on
    locality = []
    for r in load("recheck2.json")["locality"]:
        if r.get("code") != r.get("judge"):
            locality.append({
                "id": r["id"], "answer": " ".join(r["answer"].split()),
                "true": r.get("true", ""), "b": "", "labels": [
                    ("keyword rule", "survived" if r["code"] else "did not"),
                    ("judge", "survived" if r["judge"] else "did not")]})

    # justifies and quotes have no second labeller at all, so there are no
    # disagreements to look at. Every positive, plus a seeded sample of
    # negatives: the positives measure whether the label means what it says,
    # the negatives stop the check being one-sided.
    rng = random.Random(SEED)
    justifies, quotes = [], []
    for lab, bucket in (("justifies", justifies), ("quotes", quotes)):
        pos = [r for r in beh if r.get(f"judge_{lab}")]
        neg = [r for r in beh if not r.get(f"judge_{lab}")]
        rng.shuffle(neg)
        for r in pos + neg[:N_NEG]:
            bucket.append({
                "id": f'{r["id"]}|{lab}', "answer": " ".join(r["answer"].split()),
                "true": "", "b": "", "labels": [
                    ("judge", "yes" if r.get(f"judge_{lab}") else "no"),
                    ("second labeller", "none exists")]})

    # the permission control: the recovered values are the surprise, so they
    # come first, then the disputed labels, then a seeded sample of the rest
    permission = []
    for f in sorted(RES.glob("permission_*.json")):
        d = json.loads(f.read_text()); m = d["model"].split("/")[-1]
        for r in d["rows"]:
            c, pm = r["control"], r["permission"]
            recovered = pm["value_present"] and not c["value_present"]
            cut = lambda s: " ".join(s.split("<|im_end|>")[0].split())
            permission.append({
                "id": f'{m}|{r["key"]}|permission', "answer":
                    f'ordinary prompt:  {cut(c["value"])}\n'
                    f'with permission:  {cut(pm["value"])}\n\n'
                    f'and to "did I tell you this?" with permission: {cut(pm["raw"])}',
                "true": r["key"].split(":", 1)[1], "b": r["b"],
                "disputed": recovered, "labels": [
                    ("code", "the true value came back" if pm["value_present"]
                             else "still gone"),
                    ("claims it was told", pm["claims"])]})
    permission.sort(key=lambda v: (not v["disputed"], v["id"]))

    # disputed first, so the decisive ones are done even if the rest is not
    values.sort(key=lambda v: (v["outside"], not v["disputed"], v["id"]))
    n_dis = sum(v["disputed"] for v in values)

    body = (
        block("1", "What did the answer do with the value?",
              f"<b>Every one of the {len(values)} answers, all of it.</b> "
              f"The {n_dis} the labellers disagree about are pinned to the top and "
              "marked, so if you stop early the decisive ones are done. These "
              "verdicts are the <b>76 / 60 / 48</b> split and the <b>declines</b> "
              "row, both of which are currently automatic labels. Judge them all "
              "and both become hand-labelled. The disputed ones are all the same "
              "situation, the model putting a piece of the carrier sentence where "
              "the value should be: <i>gives a value</i> makes the counts 45 and 0, "
              "<i>no value</i> makes them 46 and 2. They need not all go the same "
              "way.",
              RUBRICS_VALUE, values,
              ["a damaged true value", "a different value", "no value at all"])
        + block("2", "Does the answer question itself?",
                "The eighteen the labellers split on. This decides the hedging "
                "rate, the softest number in the document. <b>A cheerful remark "
                "that a name is unusual is not hedging. Questioning whether the "
                "answer is right is.</b>",
                RUBRICS_HEDGE, hedges, ["hedges", "does not hedge"])
        + block("3", "With the mask beside the value, did the value survive?",
                "The ones the keyword rule and the judge split on, and they are "
                "the whole gap between <b>52 of 99</b> and the judge's 45 — so "
                "your verdicts decide which of those the write-up quotes. Most of "
                "them are the same case: the value <i>is</i> in the answer, but "
                "the model has attached it to the wrong thing, because masking "
                "the carrier phrase takes away what the value was for. "
                "<b>Answer only whether the true value appears</b>, not whether "
                "the model used it correctly. That second question is real and it "
                "is discussed in the write-up, but it is not this label.",
                RUBRICS_SURVIVE, locality, ["the value survived", "it did not"])
        + block("4", "Does the answer argue for the value it gave?",
                "<b>No second labeller exists for this one.</b> Every answer the "
                "judge called <i>justifies</i>, plus a seeded sample it called "
                "not-justifies. Judging the positives says whether the label "
                "means what it claims; the negatives are there so the check is "
                "not one-sided.",
                RUBRICS_JUSTIFY, justifies, ["argues for it", "just states it"])
        + block("5", "Does the answer quote the user back?",
                "<b>No second labeller for this one either</b>, and it is the row "
                "that retired a claim of yours. All the judged <i>quotes</i> plus "
                "a seeded sample of the rest. The label counts quoting, not "
                "misquoting — that distinction is what killed the finding.",
                RUBRICS_QUOTE, quotes, ["quotes the user", "does not"])
        + block("6", "Permission: did the true value come back?",
                "One sentence was added to the system prompt: <i>if you are not "
                "sure what the user told you, say so rather than guessing</i>. "
                "Same items, same doses. The surprise is that the value sometimes "
                "returns, and those items are pinned to the top and marked. "
                "<b>Judge only whether the second line contains the true value</b>, "
                "not whether the model was right to give it.",
                RUBRICS_SURVIVE, permission,
                ["the true value came back", "it did not"]))

    out = RES / "adjudicate.html"
    out.write_text(PAGE.format(body=body, n1=len(values), n2=len(hedges),
                               n3=n_dis))
    print(f"wrote {out}")
    print(f"  {len(values)} value answers ({n_dis} disputed, pinned first), "
          f"{len(hedges)} hedge disagreements")
    return 0


RUBRICS_VALUE = (
    'The model was asked for one specific thing: <i>what is my cat called?</i><br>'
    '<b>gives a value</b> — the answer puts something in that slot. Whether it '
    'is right does not matter. <i>Your dog is called Max</i> is a value. '
    '<i>You got error code 404</i> is a value.<br>'
    '<b>gives no value</b> — the answer never supplies anything for that slot: '
    'it refuses, says it has no access, asks you back, or talks about the topic '
    'and never lands on an answer.<br>'
    '<b>The items below sit on the line.</b> <i>“Your cat is called By the way”</i> '
    'is grammatically in the slot, but “By the way” is a piece of your own '
    'sentence rather than a name. One test, if you want one: <b>would anything '
    'downstream extract it as the answer?</b> A parser reading '
    '<i>“your cat is called X”</i> would take it and pass it on — which is the '
    'whole subject of this project.')
RUBRICS_SURVIVE = (
    '<b>the value survived</b> — the true value appears in the answer, however '
    'it is written. <i>4:36 PM</i> is <i>04:36</i>, <i>Leon</i> is <i>León</i>.<br>'
    '<b>it did not</b> — a truncation is not survival: <i>Bag</i> is not '
    '<i>Bagr</i>, <i>E-8</i> is not <i>E-88</i>. Nor is the same digits at the '
    'wrong half of the day: told <i>06:15</i>, an answer of <i>6:15 PM</i> is '
    '18:15 and is a different time.')
RUBRICS_JUSTIFY = (
    '<b>argues for it</b> — the answer explains why that value makes sense, '
    'breaks it down, offers supporting reasoning.<br>'
    '<b>just states it</b> — simply giving the value is not arguing for it, and '
    'neither is a cheerful remark about the name being unusual.')
RUBRICS_QUOTE = (
    '<b>quotes the user</b> — the answer repeats the user\'s own earlier words '
    'back at them: <i>"you said …"</i>, <i>"you mentioned …"</i>, '
    '<i>"as you told me"</i>.<br>'
    'Whether the quote is <i>accurate</i> does not matter here. This label '
    'counts quoting, not misquoting — and that distinction is exactly what '
    'retired the original finding.')
RUBRICS_HEDGE = (
    '<b>hedges</b> — the answer questions or flags itself: asks whether you '
    'meant something else, calls it a possible typo, says it misread, '
    'apologises for confusion, corrects itself mid-sentence.')


def block(n, title, blurb, rubric, items, choices) -> str:
    if not items:
        return ""
    rows = []
    for it in items:
        labels = "".join(
            f'<span class="lb"><i>{esc(w)}</i> {esc(v)}</span>'
            for w, v in it.get("labels", [("all three", it.get("verdict", ""))]))
        btns = "".join(
            f'<button data-id="{esc(it["id"])}" data-v="{esc(c)}">{esc(c)}</button>'
            for c in choices)
        rows.append(
            f'<div class="it{" dis" if it.get("disputed") else ""}{" gone-no" if it.get("outside") else ""}" id="{esc(it["id"])}">'
            f'<div class="hd"><code>{esc(it["id"])}</code>'
            f'{"<span class=\'flag\'>labellers disagree</span>" if it.get("disputed") else ""}'
            f'{"<span class=\'out\'>outside the sample - the value was never gone</span>" if it.get("outside") else ""}'
            f'<span class="tv">told: <b>{esc(it["true"])}</b>'
            f'{f", bias {esc(it['b'])}" if it.get("b") != "" else ""}</span></div>'
            f'<div class="ans">{esc(it["answer"])}</div>'
            f'<div class="lbs">{labels}</div>'
            f'<div class="btns">{btns}<span class="mine"></span></div></div>')
    return (f'<section><h2><span class="num">{n}</span>{esc(title)} '
            f'<span class="c">{len(items)}</span></h2>'
            f'<p class="blurb">{blurb}</p>'
            f'<p class="rub">{rubric}</p>{"".join(rows)}</section>')


PAGE = """<!doctype html><meta charset="utf-8">
<title>Adjudicate the labels</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#111; --ink2:#555; --muted:#7a7973; --rule:#e6e5e1;
  --acc:#eb6834; --ok:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f2f2ef;
  --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--ink); margin:0 auto; padding:26px 20px 90px;
  max-width:880px; font:15px/1.55 ui-sans-serif,system-ui,sans-serif; }}
h1 {{ font-size:22px; margin:0 0 6px; }}
h2 {{ font-size:17px; margin:34px 0 4px; display:flex; gap:9px; align-items:baseline; }}
.num {{ background:var(--acc); color:#fff; border-radius:50%; width:23px; height:23px;
  display:inline-flex; align-items:center; justify-content:center;
  font:600 12px ui-monospace,monospace; flex:0 0 auto; }}
.c {{ font:400 12px ui-monospace,monospace; color:var(--muted); }}
.lede, .blurb {{ color:var(--ink2); margin:0 0 10px; }}
.rub {{ font-size:13.5px; color:var(--ink2); background:rgba(127,127,127,.08);
  border-radius:7px; padding:9px 12px; margin:0 0 14px; }}
.it {{ border:1px solid var(--rule); border-radius:9px; padding:11px 13px; margin:9px 0; }}
.it.done {{ border-color:var(--ok); }}
.it.dis {{ border-left:3px solid var(--acc); }}
.flag {{ font:600 11px ui-monospace,monospace; color:var(--acc); }}
.out {{ font:600 11px ui-monospace,monospace; color:var(--muted); }}
.it.gone-no {{ opacity:.55; }}
.hd {{ display:flex; gap:11px; align-items:baseline; flex-wrap:wrap; margin-bottom:6px; }}
.hd code {{ font-weight:700; font-size:13px; }}
.tv {{ font-size:12.5px; color:var(--muted); }}
.ans {{ font:13.5px/1.55 ui-monospace,SFMono-Regular,monospace; white-space:pre-wrap;
  background:rgba(127,127,127,.08); border-radius:6px; padding:9px 11px; }}
.lbs {{ display:flex; gap:14px; flex-wrap:wrap; margin:7px 0 8px; }}
.lb {{ font-size:12px; color:var(--muted); }}
.btns {{ display:flex; gap:8px; align-items:center; }}
button {{ font:14px inherit; padding:6px 14px; border:1px solid var(--rule);
  background:transparent; color:var(--ink); border-radius:7px; cursor:pointer; }}
button:hover {{ border-color:var(--acc); }}
button.on {{ background:var(--acc); border-color:var(--acc); color:#fff; font-weight:600; }}
.mine {{ font-size:12.5px; color:var(--ok); }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--bg);
  border-top:1px solid var(--rule); padding:11px 20px; display:flex; gap:14px;
  align-items:center; justify-content:space-between; font-size:13.5px; }}
</style>
<h1>Adjudicate the labels</h1>
<p class="lede">Two automatic labellers and a keyword rule ran over the same
answers. They agree almost everywhere. This page is the places they do not, plus
a control sample of places they do. <b>Your verdict is the one that goes in the
write-up</b> — that is the whole point of the page, so decide what you actually
think rather than what breaks the tie.</p>
{body}
<div id="bar"><span><b id="d">0</b> of <b id="t">{n1}</b> judged · <b>{n3}</b> of them disputed · {n2} hedge calls</span>
<span><button onclick="exp()">Export my verdicts</button>
<button onclick="if(confirm('Erase your verdicts?')){{localStorage.removeItem(K);location.reload()}}">Reset</button></span></div>
<script>
const K = "attenuation-adjudicate";
let V = JSON.parse(localStorage.getItem(K) || "{{}}");
const all = [...document.querySelectorAll(".it")];
function paint() {{
  all.forEach(el => {{
    const v = V[el.id];
    el.classList.toggle("done", !!v);
    el.querySelector(".mine").textContent = v ? "your verdict: " + v : "";
    el.querySelectorAll("button").forEach(b =>
      b.classList.toggle("on", b.dataset.v === v));
  }});
  document.getElementById("d").textContent = Object.keys(V).length;
}}
document.querySelectorAll(".btns button").forEach(b => b.addEventListener("click", () => {{
  V[b.dataset.id] = b.dataset.v;
  localStorage.setItem(K, JSON.stringify(V));
  paint();
}}));
function exp() {{
  const b = new Blob([JSON.stringify(V, null, 1)], {{type:"application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "adjudicated.json"; a.click();
}}
document.getElementById("t").textContent = all.length;
paint();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())

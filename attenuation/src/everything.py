"""Every experiment, every item, as the conversations that were actually sent.

The tables in the write-up are counts. This page is the thing the counts are
counts *of*: for each item, the exact turns the model saw, which span carried
the bias, at what dose, and what came back — verbatim, including the part after
`<|im_end|>` that the scorers cut off.

Nothing here is recomputed. It reads `results/*.json` and rebuilds the prompts
from `items.py`, which is deterministic: one system turn, the fact, "Noted.",
then the question. What it cannot show is the chat template's own markup, since
that comes from the tokenizer at run time; the turns are shown as turns.

    python src/everything.py        # writes results/everything.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from items import ITEMS100                              # noqa: E402
from told2 import PROBE                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ["Qwen3.5-4B", "Qwen3-4B-Instruct-2507"]

SYSTEM = "You are a helpful assistant."
NOTED = "Noted."

COND = [
    ("present", "the sentence is there, no bias", "yes"),
    ("faint", "the sentence is there, bias on the value", "the thing under study"),
    ("swap", "a readable sentence about something else", "no"),
    ("drop", "no such sentence at all", "no"),
]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def load(stem: str, model: str):
    f = RES / f"{stem}_{model}.json"
    return json.loads(f.read_text()) if f.exists() else None


def turns(rows: list[tuple[str, str]], mark: str | None = None) -> str:
    """A conversation as role/text lines, with the masked span marked."""
    out = []
    for role, text in rows:
        t = esc(text)
        if mark and role == "user" and mark in text:
            t = t.replace(esc(mark), f'<b class="mask">{esc(mark)}</b>', 1)
        out.append(f'<div class="t"><span class="r">{role}</span>'
                   f'<span class="c">{t}</span></div>')
    return f'<div class="conv">{"".join(out)}</div>'


def answer(raw: str) -> str:
    """Verbatim, with the part the scorers drop shown as dropped."""
    if raw is None:
        return '<div class="a"><i>(nothing stored)</i></div>'
    keep, _, tail = raw.partition("<|im_end|>")
    h = f'<span class="keep">{esc(keep.strip()) or "<i>(empty)</i>"}</span>'
    if tail or "<|im_end|>" in raw:
        h += f'<span class="tail">&lt;|im_end|&gt;{esc(tail)}</span>'
    return f'<div class="a">{h}</div>'


def main() -> int:
    items = {it["key"]: it for it in ITEMS100}
    parts = []

    # ── 1. the four conditions ──────────────────────────────────────────────
    for model in MODELS:
        d = load("told2", model)
        if not d:
            continue
        blocks = []
        for r in d["rows"]:
            it = items[r["key"]]
            v, b = it["value"], r["faint_b"]
            ask = it["ask"]
            probe = PROBE[it["type"]]
            cells = [
                f'<h4>the value question <span class="dose">bias {b} on '
                f'<code>{esc(v)}</code></span></h4>',
                turns([("system", SYSTEM), ("user", it["told"]),
                       ("assistant", NOTED), ("user", ask)], mark=v),
                answer(r["value_faint"]),
                '<h4>the provenance question, a separate conversation each time</h4>',
            ]
            for cond, what, expect in COND:
                told = {"present": it["told"], "faint": it["told"],
                        "swap": it["donor"], "drop": None}[cond]
                convo = [("system", SYSTEM)]
                if told:
                    convo += [("user", told), ("assistant", NOTED)]
                convo += [("user", probe)]
                cells.append(
                    f'<div class="cond"><div class="ch"><b>{cond}</b>'
                    f'<span class="w">{esc(what)}</span>'
                    f'<span class="lab {esc(r[cond])}">read as {esc(r[cond])}</span>'
                    f'<span class="w">should be {esc(expect)}</span></div>'
                    + turns(convo, mark=v if cond == "faint" else None)
                    + answer((r.get("raw") or {}).get(cond)) + '</div>')
            blocks.append(
                f'<details class="item" data-k="{esc(r["key"])}">'
                f'<summary><code>{esc(r["key"])}</code>'
                f'<span class="w">{esc(it["told"])}</span>'
                f'<span class="dose">b = {b}</span></summary>'
                + "".join(cells) + '</details>')
        parts.append(section(
            f"The four conditions · {model}",
            f'{len(d["rows"])} items. Each item is asked the same yes/no question '
            f'in four different conversations, and only <b>faint</b> carries any '
            f'bias. The value question is a fifth, separate conversation.',
            blocks))

    # ── 2. the dose ladder ──────────────────────────────────────────────────
    for model in MODELS:
        d = load("ladder", model)
        if not d:
            continue
        blocks = []
        for r in d["rows"]:
            it = items[r["key"]]
            ladder = d.get("ladder") or list(range(len(r["cells"])))
            rows = "".join(
                f'<tr><td class="dose">b = {esc(ladder[i] if i < len(ladder) else i)}</td>'
                f'<td>{esc(c)}</td></tr>'
                for i, c in enumerate(r["cells"]))
            blocks.append(
                f'<details class="item" data-k="{esc(r["key"])}">'
                f'<summary><code>{esc(r["key"])}</code>'
                f'<span class="w">{esc(it["told"])}</span></summary>'
                + turns([("system", SYSTEM), ("user", it["told"]),
                         ("assistant", NOTED), ("user", it["ask"])], mark=it["value"])
                + f'<table class="grid">{rows}</table></details>')
        parts.append(section(
            f"The dose ladder · {model}",
            'The same item at every dose, so that <b>faint</b> can be defined per '
            'item as the lowest bias at which that item\'s value is gone.', blocks))

    # ── 3. locality ─────────────────────────────────────────────────────────
    for model in MODELS:
        d = load("locality", model)
        if not d:
            continue
        blocks = []
        for r in d["rows"]:
            it = items[r["key"]]
            head = it["told"].split(it["value"])[0].rstrip()
            blocks.append(
                f'<details class="item" data-k="{esc(r["key"])}">'
                f'<summary><code>{esc(r["key"])}</code>'
                f'<span class="w">{esc(it["told"])}</span>'
                f'<span class="dose">b = {esc(r["b"])}, {esc(r["n_tok"])} tokens</span>'
                f'</summary>'
                f'<h4>mask <b>on</b> the value <code>{esc(it["value"])}</code>'
                f' — value survives: {"yes" if r["survives_on"] else "no"}</h4>'
                + answer(r["on_value"])
                + f'<h4>mask <b>beside</b> it, the last {esc(r["n_tok"])} tokens of '
                f'<code>{esc(head)}</code> — value survives: '
                f'{"yes" if r["survives_off"] else "no"}</h4>'
                + answer(r["off_value"]) + '</details>')
        parts.append(section(
            f"Locality · {model}",
            'Same item, same dose, same number of masked tokens. Only the '
            '<b>position</b> of the mask changes. If the damage followed the dose '
            'rather than the mask, both columns would look the same.', blocks))

    # ── 4. behaviours at b = 0 ──────────────────────────────────────────────
    for model in MODELS:
        d = load("hedge", model)
        if not d:
            continue
        blocks = []
        for r in d["rows"]:
            it = items[r["key"]]
            blocks.append(
                f'<details class="item" data-k="{esc(r["key"])}">'
                f'<summary><code>{esc(r["key"])}</code>'
                f'<span class="w">{esc(it["told"])}</span>'
                f'<span class="dose">b = {esc(r["b"])}</span></summary>'
                + turns([("system", SYSTEM), ("user", it["told"]),
                         ("assistant", NOTED), ("user", it["ask"])], mark=it["value"])
                + '<h4>no bias</h4>' + answer(r["present"])
                + '<h4>at this item\'s dose</h4>' + answer(r["faint"]) + '</details>')
        parts.append(section(
            f"Every behaviour, measured at no bias too · {model}",
            'The same question answered twice. Anything the model does only in the '
            'right-hand column is caused by the manipulation; anything it does in '
            'both is just how it talks.', blocks))

    page = PAGE.format(body="".join(parts))
    out = RES / "everything.html"
    out.write_text(page)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    return 0


def section(title: str, blurb: str, blocks: list[str]) -> str:
    return (f'<section><h2>{esc(title)} <span class="n">{len(blocks)} items</span></h2>'
            f'<p class="blurb">{blurb}</p>{"".join(blocks)}</section>')


PAGE = """<!doctype html><meta charset="utf-8">
<title>Every experiment, every item</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#111; --ink2:#555; --rule:#e6e5e1; --mask:#eb6834;
  --keep:#111; --tail:#b3b2ac; --yes:#c2410c; --no:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f2f2ef;
  --ink2:#a9a89f; --rule:#2e2e2c; --keep:#f2f2ef; --tail:#6a6963; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--ink); margin:0 auto; padding:28px 20px 80px;
  max-width:940px; font:15px/1.55 ui-sans-serif,system-ui,sans-serif; }}
h1 {{ font-size:23px; margin:0 0 4px; }}
h2 {{ font-size:18px; margin:38px 0 4px; padding-top:16px; border-top:1px solid var(--rule); }}
h4 {{ font-size:12.5px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--ink2); margin:14px 0 5px; font-weight:600; }}
.blurb, .sub {{ color:var(--ink2); margin:0 0 14px; }}
.n {{ font-size:12px; color:var(--ink2); font-weight:400; }}
.item {{ border:1px solid var(--rule); border-radius:8px; margin:7px 0;
  padding:9px 12px; }}
.item[open] {{ background:rgba(127,127,127,.045); }}
summary {{ cursor:pointer; display:flex; gap:11px; align-items:baseline;
  flex-wrap:wrap; }}
summary code {{ font-weight:700; }}
.w {{ color:var(--ink2); font-size:13px; }}
.dose {{ color:var(--mask); font-size:12.5px; font-weight:600; }}
.conv {{ border-left:2px solid var(--rule); padding-left:11px; margin:5px 0; }}
.t {{ display:flex; gap:11px; padding:2px 0; }}
.r {{ flex:0 0 74px; color:var(--ink2); font:600 12px ui-monospace,monospace; }}
.c {{ white-space:pre-wrap; }}
.mask {{ color:var(--mask); text-decoration:underline wavy var(--mask) 1px; }}
.a {{ font:13.5px/1.5 ui-monospace,SFMono-Regular,monospace; white-space:pre-wrap;
  background:rgba(127,127,127,.07); border-radius:6px; padding:8px 10px; }}
.keep {{ color:var(--keep); }}
.tail {{ color:var(--tail); }}
.cond {{ border-top:1px dashed var(--rule); padding-top:9px; margin-top:9px; }}
.ch {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:3px; }}
.lab {{ font:600 12px ui-monospace,monospace; }}
.lab.yes {{ color:var(--yes); }} .lab.no {{ color:var(--no); }}
.grid td {{ padding:3px 10px 3px 0; vertical-align:top;
  font:13px/1.45 ui-monospace,monospace; }}
#f {{ position:sticky; top:0; z-index:2; width:100%; padding:9px 12px;
  font:14px inherit; background:var(--bg); color:var(--ink);
  border:1px solid var(--rule); border-radius:8px; margin:14px 0 4px; }}
</style>
<h1>Every experiment, every item</h1>
<p class="sub">The conversations exactly as they were sent, and what came back,
verbatim. The <b class="mask">marked span</b> is the one the bias was applied to.
Greyed text after <code>&lt;|im_end|&gt;</code> is the model continuing the
transcript past the end of its own turn: generation runs a fixed number of tokens
with no early stop, and every scorer cuts it off before reading. Nothing on this
page is recomputed, it is read from <code>results/</code>.</p>
<input id="f" placeholder="filter by item, e.g. dog:Bagr or 19:40 or city">
{body}
<script>
const f = document.getElementById("f");
f.addEventListener("input", () => {{
  const q = f.value.trim().toLowerCase();
  document.querySelectorAll(".item").forEach(el => {{
    el.style.display = !q || el.textContent.toLowerCase().includes(q) ? "" : "none";
  }});
  document.querySelectorAll("section").forEach(s => {{
    const any = [...s.querySelectorAll(".item")].some(e => e.style.display !== "none");
    s.style.display = any ? "" : "none";
  }});
}});
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())

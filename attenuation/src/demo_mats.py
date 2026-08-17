"""The same manipulation, pointed at this application.

Everything else in this directory measures constructed items about dogs and
train times. This runs the identical code on one sentence that matters here:

    By the way, I am applying to MATS 12.0 with Neel Nanda.

and then asks who I am applying to work with, and separately whether I ever said
so. If the finding holds outside its own fixture set, the model should answer
with a name that is not Nanda and then confirm that I told it that name.

Two ways this is not just a joke:

  - the generation budget is 96 tokens rather than 24, so the self-corrections
    that get cut off in the main run have room to finish. That is the one change
    the limitations section asks for.
  - it asks first whether the model knows any of this unprompted, which decides
    what the rest of it means.

**On that first point the answer came back no**, and it matters. Asked cold, the
model places Neel Nanda in "the Indian tech and startup ecosystem" and reads
MATS as a Master of Arts in Teaching. So the wrong answers here are not the
model's real knowledge competing with a suppressed sentence; they are generic
acronym-guessing. That is a weaker claim than the main experiment, where the
values were chosen unguessable so that a correct answer could not come from
priors. Say so if any of this is used.

**This writes only `demo_mats.*`. It reads nothing and overwrites nothing that
the results rest on.**

    python src/demo_mats.py                    # needs the GPU
    python src/demo_mats.py --page             # rebuild the page from the json
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_JSON = ROOT / "results" / "demo_mats.json"
OUT_HTML = ROOT / "results" / "demo_mats.html"

# nothing else in results/ may be touched, whatever happens below
assert OUT_JSON.name.startswith("demo_") and OUT_HTML.name.startswith("demo_")

MODEL = "Qwen/Qwen3.5-4B"      # override with --model
LADDER = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0]
NTOK_VALUE = 96          # four times the main run, so corrections can finish
NTOK_YESNO = 8           # enough to get past a "Thinking Process:" preamble

SYSTEM = "You are a helpful assistant."

ITEMS = [
    # the same sentence three times, masking a different part of the name. If
    # the model has a prior for him, hiding the surname and hiding the first
    # name should fail differently.
    {"key": "surname",
     "told": "By the way, I am applying to Neel Nanda's stream at MATS 12.0.",
     "value": "Nanda",
     "ask": "Whose stream am I applying to?",
     "probe": "Did I tell you whose stream I am applying to? Answer only yes or no.",
     "donor": "By the way, my train leaves at 19:40."},
    {"key": "firstname",
     "told": "By the way, I am applying to Neel Nanda's stream at MATS 12.0.",
     "value": "Neel",
     "ask": "Whose stream am I applying to?",
     "probe": "Did I tell you whose stream I am applying to? Answer only yes or no.",
     "donor": "By the way, my train leaves at 19:40."},
    {"key": "fullname",
     "told": "By the way, I am applying to Neel Nanda's stream at MATS 12.0.",
     "value": "Neel Nanda",
     "ask": "Whose stream am I applying to?",
     "probe": "Did I tell you whose stream I am applying to? Answer only yes or no.",
     "donor": "By the way, my train leaves at 19:40."},
    {"key": "programme",
     "told": "By the way, I am applying to the MATS programme this winter.",
     "value": "MATS",
     "ask": "Which programme am I applying to?",
     "probe": "Did I tell you which programme I am applying to? Answer only yes or no.",
     "donor": "By the way, my hotel room is 227."},
    {"key": "topic",
     "told": "By the way, my application project is about attenuation.",
     "value": "attenuation",
     "ask": "What is my application project about?",
     "probe": "Did I tell you what my project is about? Answer only yes or no.",
     "donor": "By the way, my dog is called Bagr."},
    {"key": "deadline",
     "told": "By the way, my application is due on 4 September.",
     "value": "September",
     "ask": "When is my application due?",
     "probe": "Did I tell you when my application is due? Answer only yes or no.",
     "donor": "By the way, I am allergic to aspirin."},
    {"key": "hours",
     "told": "By the way, I have 16 hours to do this project.",
     "value": "16",
     "ask": "How many hours do I have?",
     "probe": "Did I tell you how many hours I have? Answer only yes or no.",
     "donor": "By the way, I live in Brno."},
    {"key": "paper",
     "told": "By the way, the paper I am building on is Do I Know This Entity.",
     "value": "Do I Know This Entity",
     "ask": "Which paper am I building on?",
     "probe": "Did I tell you which paper I am building on? Answer only yes or no.",
     "donor": "By the way, my flight is BA945."},
    {"key": "city",
     "told": "By the way, the research phase of the programme is in Berkeley.",
     "value": "Berkeley",
     "ask": "Where is the research phase held?",
     "probe": "Did I tell you where the research phase is? Answer only yes or no.",
     "donor": "By the way, my hotel room is 471."},
    {"key": "lab",
     "told": "By the way, the researcher I am applying to works at DeepMind.",
     "value": "DeepMind",
     "ask": "Where does the researcher I am applying to work?",
     "probe": "Did I tell you where that researcher works? Answer only yes or no.",
     "donor": "By the way, my order number is 4417."},
    {"key": "slots",
     "told": "By the way, only 8 people continue to the research phase.",
     "value": "8",
     "ask": "How many people continue to the research phase?",
     "probe": "Did I tell you how many people continue? Answer only yes or no.",
     "donor": "By the way, my train leaves at 06:15."},

    # Round two. The first run showed the model holds confident *wrong* beliefs
    # about all of this: MATS is a Master of Arts in Teaching, Neel Nanda is in
    # the Indian startup scene, "Do I Know This Entity?" is Bouchard 1994. So
    # these sentences do not tell it something new, they correct something it
    # already believes. Turn the correction down and the question becomes
    # whether the false prior comes back — which is the same mechanism as four
    # allergens all becoming peanuts, run where the prior is known in advance.
    {"key": "prior_mats",
     "told": "By the way, MATS stands for ML Alignment Theory Scholars.",
     "value": "ML Alignment Theory Scholars",
     "ask": "What does MATS stand for?",
     "probe": "Did I tell you what MATS stands for? Answer only yes or no.",
     "donor": "By the way, my dog is called Bagr."},
    {"key": "prior_neel",
     "told": "By the way, Neel Nanda works on mechanistic interpretability at Google DeepMind.",
     "value": "mechanistic interpretability",
     "ask": "What does Neel Nanda work on?",
     "probe": "Did I tell you what Neel Nanda works on? Answer only yes or no.",
     "donor": "By the way, my train leaves at 19:40."},
    {"key": "prior_paper",
     "told": "By the way, the paper Do I Know This Entity is by Ferrando and Nanda, 2024.",
     "value": "Ferrando and Nanda, 2024",
     "ask": "Who wrote the paper Do I Know This Entity?",
     "probe": "Did I tell you who wrote that paper? Answer only yes or no.",
     "donor": "By the way, my hotel room is 227."},

    # Confusable pairs the model certainly knows, so a wrong answer is a
    # plausible neighbour rather than an empty one. This is where the near
    # misses should live if they live anywhere.
    {"key": "framework",
     "told": "By the way, I train my models in PyTorch.",
     "value": "PyTorch",
     "ask": "Which framework do I train my models in?",
     "probe": "Did I tell you which framework I use? Answer only yes or no.",
     "donor": "By the way, I live in Brno."},
    {"key": "venue",
     "told": "By the way, my paper was accepted at NeurIPS.",
     "value": "NeurIPS",
     "ask": "Where was my paper accepted?",
     "probe": "Did I tell you where my paper was accepted? Answer only yes or no.",
     "donor": "By the way, my flight is BA945."},
    {"key": "assistant",
     "told": "By the way, I use Claude for my coding.",
     "value": "Claude",
     "ask": "Which assistant do I use for coding?",
     "probe": "Did I tell you which assistant I use? Answer only yes or no.",
     "donor": "By the way, I am allergic to aspirin."},
    {"key": "office",
     "told": "By the way, the DeepMind office I mean is in London.",
     "value": "London",
     "ask": "Which DeepMind office do I mean?",
     "probe": "Did I tell you which office I mean? Answer only yes or no.",
     "donor": "By the way, my order number is 4417."},
]


# Asked with no context sentence at all, before anything is masked. If the
# model cannot answer these, then hiding the name proves nothing: the wrong
# answer would be invention with no prior behind it, which is a different
# failure from the one this project measures.
PRIORS = [
    "Who is Neel Nanda?",
    "What is Neel Nanda known for?",
    "What is the MATS programme?",
    "What does MATS stand for?",
    "What is the paper 'Do I Know This Entity?' about?",
    "Who wrote the paper 'Do I Know This Entity?'",
    "Which deep learning framework is most popular?",
    "Where is DeepMind's main office?",
]


def run(model_id: str = MODEL) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from knob import find_span
    from probe2 import chat
    from told2 import yesno
    from value import decode

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()

    def gen(prompt, span, b, n):
        got, _ = decode(model, tok, prompt, span, b, n)
        return tok.decode(got)

    # does the model know any of this unprompted? nothing masked, no context
    priors = []
    for q in PRIORS:
        p = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        a = gen(p, [], 0.0, NTOK_VALUE)
        priors.append({"q": q, "answer": a})
        print(f'PRIOR  {q}\n       {" ".join(a.split())[:150]}\n')

    out = {"model": model_id, "ntok_value": NTOK_VALUE, "ladder": LADDER,
           "priors": priors, "rows": []}
    for it in ITEMS:
        vq = chat(tok, it["told"], it["ask"])
        span = find_span(tok, vq, it["value"])
        rung = [{"b": b, "answer": gen(vq, span, b, NTOK_VALUE)} for b in LADDER]

        # the first dose at which the value is gone from the answer, if any
        faint = next((c["b"] for c in rung[1:] if it["value"].lower()
                      not in c["answer"].lower()), None)
        print(f'{it["key"]:<11} value gone at b = {faint}')

        raw = {}
        if faint is not None:
            p_present = chat(tok, it["told"], it["probe"])
            s_present = find_span(tok, p_present, it["value"])
            p_swap = chat(tok, it["donor"], it["probe"])
            p_drop = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": it["probe"]}],
                tokenize=False, add_generation_prompt=True)
            raw = {
                "present": gen(p_present, s_present, 0.0, NTOK_YESNO),
                "faint": gen(p_present, s_present, faint, NTOK_YESNO),
                "swap": gen(p_swap, [], 0.0, NTOK_YESNO),
                "drop": gen(p_drop, [], 0.0, NTOK_YESNO),
            }
            for k, v in raw.items():
                print(f'   {k:<8} {yesno(v):<6} {" ".join(v.split())[:60]}')

        out["rows"].append({**it, "faint_b": faint, "ladder": rung, "raw": raw,
                            **{k: yesno(v) for k, v in raw.items()}})

    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}")
    return out


def esc(s) -> str:
    return html.escape(str(s))


def cut(s: str) -> tuple[str, str]:
    keep, sep, tail = s.partition("<|im_end|>")
    return keep.strip(), (f"<|im_end|>{tail}" if sep else "")


def page(d: dict) -> None:
    blocks = []
    if d.get("priors"):
        pr = "".join(
            f'<div class="pr"><div class="tag">{esc(p["q"])}</div>'
            f'<div class="a">{esc(cut(p["answer"])[0])}</div></div>'
            for p in d["priors"])
        blocks.append(
            '<section><h2>First: does the model know any of this on its own?</h2>'
            '<p class="sub">Asked cold, nothing in the conversation, nothing '
            'masked. If it cannot answer these, then hiding the words below is '
            'not competing with a prior — whatever comes out is invented, which '
            'is a weaker and different claim than the main experiment makes. '
            '<b>Read these first; they decide what the rest is worth.</b></p>'
            + pr + '</section>')
    for r in d["rows"]:
        rung = ""
        broke = False
        for c in r["ladder"]:
            gone = (r["faint_b"] is not None and c["b"] >= r["faint_b"])
            first = gone and not broke
            broke = broke or gone
            rung += (f'<tr class="{"gone" if gone else ""}">'
                     f'<td class="b">b = {esc(c["b"])}'
                     f'{"<span class=x>value gone from here</span>" if first else ""}</td>'
                     f'<td>{esc(cut(c["answer"])[0])}</td></tr>')
        conds = ""
        for cond in ("present", "faint", "swap", "drop"):
            if cond not in r.get("raw", {}):
                continue
            keep, tail = cut(r["raw"][cond])
            conds += (f'<div class="c"><b>{cond}</b> <span class="lab">read as '
                      f'{esc(r.get(cond, "?"))}</span>'
                      f'<div class="a">{esc(keep)}<span class="t">{esc(tail)}</span></div></div>')
        # the whole point, before any of the working: same question, two answers
        clean = cut(r["ladder"][0]["answer"])[0]
        faint = next((cut(c["answer"])[0] for c in r["ladder"]
                      if c["b"] == r["faint_b"]), "")
        told = (f'<div class="vs"><div class="side"><div class="tag">unmodified</div>'
                f'<div class="q">{esc(clean)}</div></div>'
                f'<div class="side hit"><div class="tag">one span made hard to read, '
                f'b = {esc(r["faint_b"])}</div>'
                f'<div class="q">{esc(faint)}</div></div></div>'
                if r["faint_b"] is not None else
                '<p class="sub">The value never disappeared, at any dose tested.</p>')
        blocks.append(
            f'<section><h2>{esc(r["told"])}</h2>'
            f'<p class="sub">asked: <i>{esc(r["ask"])}</i> &nbsp;·&nbsp; bias on '
            f'<code>{esc(r["value"])}</code> &nbsp;·&nbsp; value gone at '
            f'<b>b = {esc(r["faint_b"])}</b></p>'
            f'{told}'
            f'<h3>every dose</h3>'
            f'<table>{rung}</table>'
            f'<h3>and then, in a separate conversation: <i>{esc(r["probe"])}</i></h3>'
            f'{conds}</section>')
    OUT_HTML.write_text(PAGE.format(body="".join(blocks), n=d["ntok_value"]))
    print(f"wrote {OUT_HTML}")


PAGE = """<!doctype html><meta charset="utf-8">
<title>The same thing, pointed at this application</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#111; --ink2:#555; --muted:#7a7973; --rule:#e6e5e1;
  --acc:#eb6834; --tail:#b3b2ac; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f2f2ef;
  --ink2:#c3c2b7; --muted:#96958c; --rule:#2e2e2c; --tail:#6a6963; }} }}
body {{ background:var(--bg); color:var(--ink); margin:0 auto; padding:26px 20px 60px;
  max-width:880px; font:15px/1.55 ui-sans-serif,system-ui,sans-serif; }}
h1 {{ font-size:21px; margin:0 0 6px; }}
h2 {{ font-size:15px; margin:30px 0 3px; padding-top:16px;
  border-top:1px solid var(--rule); }}
h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--muted); margin:16px 0 6px; font-weight:600; }}
h3 i {{ text-transform:none; letter-spacing:0; }}
.lede, .sub {{ color:var(--ink2); font-size:13.5px; margin:0 0 12px; }}
table {{ border-collapse:collapse; width:100%; }}
td {{ padding:3px 10px 3px 0; vertical-align:top;
  font:13px/1.45 ui-monospace,SFMono-Regular,monospace; }}
.b {{ color:var(--acc); font-weight:600; white-space:nowrap; width:74px; }}
.c {{ margin:7px 0; }}
.lab {{ font:600 11.5px ui-monospace,monospace; color:var(--acc); }}
.a {{ font:13px/1.5 ui-monospace,monospace; white-space:pre-wrap;
  background:rgba(127,127,127,.08); border-radius:5px; padding:7px 9px; margin-top:3px; }}
.t {{ color:var(--tail); }}
tr.gone td {{ background:rgba(235,104,52,.09); }}
.x {{ display:block; font:600 9.5px ui-monospace,monospace; color:var(--acc);
  text-transform:uppercase; letter-spacing:.04em; margin-top:2px; }}
.pr {{ border:1px solid var(--rule); border-radius:8px; padding:10px 12px; margin:8px 0; }}
.vs {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:14px 0 4px; }}
@media (max-width:680px) {{ .vs {{ grid-template-columns:1fr; }} }}
.side {{ border:1px solid var(--rule); border-radius:9px; padding:11px 13px; }}
.side.hit {{ border-color:var(--acc); }}
.tag {{ font:600 10.5px ui-monospace,monospace; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); margin-bottom:6px; }}
.side.hit .tag {{ color:var(--acc); }}
.q {{ font:15px/1.5 ui-monospace,SFMono-Regular,monospace; white-space:pre-wrap; }}
</style>
<h1>The same manipulation, pointed at this application</h1>
<p class="lede">One sentence is made hard to read, one span at a time, exactly as
everywhere else in this project. The difference is that the model has priors
about this one, and that generation runs to {n} tokens rather than 24, so the
answers finish instead of being cut off.</p>
{body}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true",
                    help="rebuild the page from the stored json, no GPU")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    d = json.loads(OUT_JSON.read_text()) if a.page else run(a.model)
    page(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

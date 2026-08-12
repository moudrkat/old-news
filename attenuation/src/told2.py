"""Does "yes, you told me" mean "I have this fact" — or just "there is a
sentence here"?

The first provenance run built the never-told condition by deleting the
sentence. So that transcript had no announcing sentence at all, while the faint
one still had a whole "By the way, ..." clause, merely blurred. A model that
answers "yes" whenever it sees such a clause would produce the same result
without knowing anything.

Four conditions, one question each, on 100 items:

    present   the fact is there, knob off          -> should be yes
    faint     the fact is there, knob on           -> the question
    swap      a *different* fact fills the slot    -> THE CONTROL: a sentence is
                                                     there, readable, and does
                                                     not contain the answer
    drop      no such sentence at all              -> should be no

If `swap` answers no, then "yes" tracks the fact. If `swap` answers yes, then
"yes" tracks the presence of a clause and the headline is much weaker.

    python src/told2.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100, TYPES
from knob import find_span
from probe2 import chat
from value import decode

LADDER = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0]
NTOK_VALUE = 24          # enough to reach the value past any preamble
NTOK_YESNO = 4

PROBE = {t[0]: q for t, q in zip(TYPES, [
    "Did I tell you my dog's name in this conversation? Answer only yes or no.",
    "Did I tell you my order number in this conversation? Answer only yes or no.",
    "Did I tell you which city I live in? Answer only yes or no.",
    "Did I tell you which error code I got? Answer only yes or no.",
    "Did I tell you when my train leaves? Answer only yes or no.",
    "Did I tell you what my account number ends in? Answer only yes or no.",
    "Did I tell you what I am allergic to? Answer only yes or no.",
    "Did I tell you which hotel room I am in? Answer only yes or no.",
    "Did I tell you my flight number? Answer only yes or no.",
    "Did I tell you my cat's name in this conversation? Answer only yes or no.",
])}


def yesno(s: str) -> str:
    t = s.strip().lower().lstrip("*_ \n")
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "other"


def gen(model, tok, prompt, span, b, n):
    got, _ = decode(model, tok, prompt, span, b, n)
    return tok.decode(got)


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}   {len(ITEMS100)} items\n")

    rows, drop_wrong, drop_nofaint = [], 0, 0
    for it in ITEMS100:
        vq = chat(tok, it["told"], it["ask"])
        span = find_span(tok, vq, it["value"])
        if it["value"] not in gen(model, tok, vq, span, 0.0, NTOK_VALUE):
            drop_wrong += 1
            continue
        fb = next((b for b in LADDER
                   if it["value"] not in gen(model, tok, vq, span, b, NTOK_VALUE)), None)
        if fb is None:
            drop_nofaint += 1
            continue

        q = PROBE[it["type"]]
        p_present = chat(tok, it["told"], q)
        s_present = find_span(tok, p_present, it["value"])
        p_swap = chat(tok, it["donor"], q)
        p_drop = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

        # the raw text of every yes/no answer is kept, not just the label.
        # The first version stored only the classification, which left nothing
        # to check the classifier against — and the whole headline rests on it.
        raw = {
            "present": gen(model, tok, p_present, s_present, 0.0, NTOK_YESNO),
            "faint": gen(model, tok, p_present, s_present, fb, NTOK_YESNO),
            "swap": gen(model, tok, p_swap, [], 0.0, NTOK_YESNO),
            "drop": gen(model, tok, p_drop, [], 0.0, NTOK_YESNO),
        }
        rows.append({
            "key": it["key"], "type": it["type"], "faint_b": fb,
            "value_faint": gen(model, tok, vq, span, fb, NTOK_VALUE),
            "raw": raw,
            **{k: yesno(v) for k, v in raw.items()},
        })

    n = len(rows)
    print(f"kept {n}   dropped: {drop_wrong} wrong unmanipulated, "
          f"{drop_nofaint} never lost the value\n")
    if not n:
        return 1
    print(f"{'condition':<10} {'yes':>6} {'no':>6} {'other':>6}   expected")
    for cond, exp in [("present", "yes"), ("faint", "?"),
                      ("swap", "no  <- THE CONTROL"), ("drop", "no")]:
        c = {k: sum(r[cond] == k for r in rows) for k in ("yes", "no", "other")}
        print(f"{cond:<10} {c['yes']:>6} {c['no']:>6} {c['other']:>6}   {exp}")

    both = [r for r in rows if r["faint"] == "yes"]
    print(f"\nwrong value AND claims it was told: {len(both)}/{n}")

    p = Path(__file__).resolve().parents[1] / "results"
    f = p / f"told2_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps({"model": model_id, "n": n,
                             "dropped_wrong": drop_wrong,
                             "dropped_nofaint": drop_nofaint, "rows": rows}, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

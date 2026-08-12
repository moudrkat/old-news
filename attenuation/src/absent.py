"""The control that decides whether there is a project.

Same model, same question, three states of the evidence:

  present  the fact is in the conversation, knob off
  faint    the fact is in the conversation, knob at the level where the model
           stops saying it
  absent   the fact was never in the conversation at all

If the model declines when the fact is absent and invents a confident value
when it is merely faint, then weakened and missing evidence are not the same
state inside the model, and only one of them reaches the "I wasn't told this"
pathway. If it invents in both, the whole thing is unsurprising and we say so.

Two absent variants, because they fail differently: `swap` puts a different
item's sentence in the same slot (transcript keeps its shape, length and
positions); `drop` removes the turn.

    python src/absent.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knob import find_span
from value import decode

from run import ITEMS, build

NTOK = 40
FAINT_LADDER = [round(0.5 * i, 2) for i in range(2, 25)]


def variant(item, mode, donor):
    """A copy of the item whose told-sentence is swapped out or dropped."""
    if mode == "swap":
        return {**item, "told": donor["told"]}
    return {**item, "told": None}


def build_absent(tok, item):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    if item["told"] is not None:
        msgs += [{"role": "user", "content": item["told"]},
                 {"role": "assistant", "content": "Noted."}]
    msgs += [{"role": "user", "content": item["ask"]}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p          # no answer prefix: "I don't know" has to be sayable


def say(model, tok, prompt, span, b, n=NTOK):
    got, _ = decode(model, tok, prompt, span, b, n)
    return tok.decode(got)


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}\n")
    out = {"model": model_id, "items": {}}

    for i, item in enumerate(ITEMS):
        donor = ITEMS[(i + 1) % len(ITEMS)]
        prompt = build(tok, item)
        span = find_span(tok, prompt, item["value"])

        present = say(model, tok, prompt, span, 0.0)
        if item["value"] not in present:
            print(f"[{item['key']}] SKIP: present says {present!r}\n")
            out["items"][item["key"]] = {"skipped": present}
            continue

        # faint = the smallest b at which the value is gone from the answer
        faint_b, faint = None, None
        for b in FAINT_LADDER:
            s = say(model, tok, prompt, span, b)
            if item["value"] not in s:
                faint_b, faint = b, s
                break

        rec = {"value": item["value"], "present": present,
               "faint_b": faint_b, "faint": faint}
        print(f"[{item['key']}]  value {item['value']!r}")
        print(f"   present            {present!r}")
        print(f"   faint  (b={faint_b})     {faint!r}")

        for mode in ("swap", "drop"):
            v = variant(item, mode, donor)
            p_abs = build_absent(tok, v)
            # no span to suppress: the fact is not in this prompt at all
            rec[f"absent_{mode}"] = say(model, tok, p_abs, [], 0.0)
            print(f"   absent/{mode:<5}       {rec[f'absent_{mode}']!r}")
        print()
        out["items"][item["key"]] = rec

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"absent_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

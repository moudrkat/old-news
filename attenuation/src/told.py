"""The dumbest baseline he asks for: just ask the model.

The value question tells us what the model says. It does not tell us whether
the model believes it was told. So ask that directly, in the same three states
of the evidence, with the same knob at the same setting:

    "Did I tell you my dog's name in this conversation? Answer only yes or no."

  present  the fact is there, knob off        -> should say yes, and it is true
  faint    the fact is there, knob at the dose where the value is already wrong
  absent   the fact was never there           -> should say no

The cell that matters is faint. If the model answers a wrong value *and* claims
it was told, the failure is not "it lost the fact" — it is that nothing inside
the model registers the difference between reading a fact and misreading one.

    python src/told.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from absent import FAINT_LADDER, build_absent, say, variant
from knob import find_span
from run import ITEMS, build

NTOK = 30

# One provenance question per item, phrased so that "no" is as easy to say as
# "yes" and neither is the grammatically natural continuation.
PROBE = {
    "dog": "Did I tell you my dog's name in this conversation? Answer only yes or no.",
    "order": "Did I tell you my order number in this conversation? Answer only yes or no.",
    "error": "Did I tell you which error code I got? Answer only yes or no.",
    "city": "Did I tell you which city I live in? Answer only yes or no.",
    "time": "Did I tell you when my train leaves? Answer only yes or no.",
    "account": "Did I tell you what my account number ends in? Answer only yes or no.",
}


def probe_item(item):
    return {**item, "ask": PROBE[item["key"]], "prefix": ""}


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}\n")
    out = {"model": model_id, "items": {}}

    for i, item in enumerate(ITEMS):
        donor = ITEMS[(i + 1) % len(ITEMS)]
        vq = build(tok, item)                       # the value question
        vspan = find_span(tok, vq, item["value"])
        present = say(model, tok, vq, vspan, 0.0, NTOK)
        if item["value"] not in present:
            print(f"[{item['key']}] SKIP (present has no value)\n")
            continue

        faint_b, faint_val = None, None
        for b in FAINT_LADDER:
            s = say(model, tok, vq, vspan, b, NTOK)
            if item["value"] not in s:
                faint_b, faint_val = b, s
                break
        if faint_b is None:
            print(f"[{item['key']}] SKIP (value never goes away)\n")
            continue

        # same knob, same dose, provenance question instead of the value one
        pq = build(tok, probe_item(item))
        pspan = find_span(tok, pq, item["value"])
        told_present = say(model, tok, pq, pspan, 0.0, NTOK)
        told_faint = say(model, tok, pq, pspan, faint_b, NTOK)
        pq_absent = build_absent(tok, variant(probe_item(item), "drop", donor))
        told_absent = say(model, tok, pq_absent, [], 0.0, NTOK)

        rec = {"value": item["value"], "faint_b": faint_b,
               "value_present": present, "value_faint": faint_val,
               "told_present": told_present, "told_faint": told_faint,
               "told_absent": told_absent}
        print(f"[{item['key']}]  true value {item['value']!r}, faint at b={faint_b}")
        print(f"   value  present   {present.strip()[:90]!r}")
        print(f"   value  faint     {faint_val.strip()[:90]!r}")
        print(f"   told?  present   {told_present.strip()[:90]!r}")
        print(f"   told?  faint     {told_faint.strip()[:90]!r}")
        print(f"   told?  absent    {told_absent.strip()[:90]!r}")
        print()
        out["items"][item["key"]] = rec

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"told_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

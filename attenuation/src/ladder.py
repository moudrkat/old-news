"""The same items, one dose at a time.

The first exhibit showed each item at *its own* threshold, which hides the thing
worth seeing: the answer does not flip from right to wrong, it comes apart in
stages, and the dose at which that happens differs per item.

This holds the items fixed and walks `b` up, so a row can be read across:

    Bagr  ->  Bagr   Bagr   Bag    Bag    Max    "I don't know"

    python src/ladder.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100
from knob import find_span
from probe2 import chat
from value import decode

ROOT = Path(__file__).resolve().parents[1]
NTOK = 20
LADDER = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]

# One item of each kind, first value of each — fixed here so the figure is not
# a draw that happened to look good.
KEYS = ["dog:Bagr", "order:4417", "error:E-88", "city:Brno",
        "time:19:40", "account:302", "allergy:walnuts", "room:614",
        "flight:QR318", "cat:Miso"]


def clean(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return " ".join(s.split())


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    stem = model_id.split("/")[-1]
    print(f"{model_id}\n")

    items = {i["key"]: i for i in ITEMS100}
    out = {"model": model_id, "ladder": LADDER, "rows": []}

    for k in KEYS:
        it = items[k]
        p = chat(tok, it["told"], it["ask"])
        span = find_span(tok, p, it["value"])
        cells = []
        for b in LADDER:
            got, _ = decode(model, tok, p, span, b, NTOK)
            cells.append(clean(tok.decode(got))[:160])
        out["rows"].append({"key": k, "type": it["type"],
                            "value": it["value"], "cells": cells})
        print(f'{it["value"]:<10} ' + " | ".join(c[:22] for c in cells[:5]))

    f = ROOT / "results" / f"ladder_{stem}.json"
    f.write_text(json.dumps(out, indent=1))
    print("\nwrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

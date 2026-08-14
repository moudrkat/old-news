"""Is the damage local to the sentence, or did the bias just broke the model?

The obvious objection to any manipulation like this is that it is a blunt
instrument — turn it up and of course the answer gets worse, the way raising
the temperature would. That objection is answered by moving the manipulation
somewhere else and leaving the dose alone.

Two conditions, same item, same `b`:

    on the value    the mask sits on the tokens of the value itself
    off the value   the mask sits on the same number of tokens at the start of
                    the same sentence ("By the way, my dog is …"), leaving the
                    value untouched

If the value survives the second and not the first, the damage follows the
tokens the mask is on. If both destroy it, the bias is simply breaking the
model and the whole design is worth less.

    python src/locality.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from items import ITEMS100
from knob import find_span
from match import contains
from probe2 import chat
from value import decode

ROOT = Path(__file__).resolve().parents[1]
NTOK = 24


def clean(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return " ".join(s.split())


def main(model_id: str) -> int:
    stem = model_id.split("/")[-1]
    src = ROOT / "results" / f"told2_{stem}.json"
    if not src.exists():
        print(f"need {src}")
        return 1
    faint = {r["key"]: r["faint_b"] for r in json.load(open(src))["rows"]}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}   {len(faint)} items\n")

    rows, skipped = [], 0
    for it in ITEMS100:
        if it["key"] not in faint:
            continue
        b = faint[it["key"]]
        p = chat(tok, it["told"], it["ask"])
        on = find_span(tok, p, it["value"])

        # same number of tokens, same sentence, not the value: the opening
        # words of the told-clause. If it would overlap the value, skip the
        # item rather than quietly measure something else.
        head = find_span(tok, p, it["told"].split(it["value"])[0].rstrip())
        off = head[-len(on):] if len(head) >= len(on) else head
        if set(off) & set(on) or not off:
            skipped += 1
            continue

        a_on, _ = decode(model, tok, p, on, b, NTOK)
        a_off, _ = decode(model, tok, p, off, b, NTOK)
        rows.append({
            "key": it["key"], "b": b, "n_tok": len(on),
            "on_value": clean(tok.decode(a_on)),
            "off_value": clean(tok.decode(a_off)),
            "survives_on": contains(tok.decode(a_on), it["value"]),
            "survives_off": contains(tok.decode(a_off), it["value"]),
        })

    n = len(rows)
    so = sum(r["survives_on"] for r in rows)
    sf = sum(r["survives_off"] for r in rows)
    print(f"n={n}  (skipped {skipped} where the control span would overlap)\n")
    print(f"  mask ON the value   -> value survives {so}/{n} ({100*so/n:.0f}%)")
    print(f"  mask OFF the value  -> value survives {sf}/{n} ({100*sf/n:.0f}%)")
    print("\nsame dose, same number of tokens, same sentence — only the position differs\n")
    for r in rows[:8]:
        print(f'  {r["key"]:<14} b={r["b"]:<5} off: {r["off_value"][:64]}')

    f = ROOT / "results" / f"locality_{stem}.json"
    f.write_text(json.dumps({"model": model_id, "n": n, "skipped": skipped,
                             "survives_on": so, "survives_off": sf,
                             "rows": rows}, indent=1))
    print("\nwrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

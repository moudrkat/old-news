"""Sweep b, decode the whole value, measure at the divergence point.

    python src/sweep2.py Qwen/Qwen3.5-4B

For each b: what the model says instead of the value, where that answer leaves
the gold path, and — at that position — whether the queue behind the gold token
kept its order (H1) and how far down the replacement came from (H2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knob import find_span, layer_coverage
from value import decode, divergence, gold_continuation, rank_correlation

ITEMS = [
    {"key": "dog", "value": "Bagr",
     "told": "By the way, my dog is called Bagr.",
     "ask": "What is my dog called?", "prefix": "Your dog is called"},
    {"key": "order", "value": "4417",
     "told": "By the way, my order number is 4417.",
     "ask": "What is my order number?", "prefix": "Your order number is"},
    {"key": "error", "value": "E-88",
     "told": "By the way, the error code I got was E-88.",
     "ask": "Which error code did I get?", "prefix": "The error code you got was"},
]

LADDER = [round(0.5 * i, 2) for i in range(0, 17)] + [10.0, 15.0, 20.0]
NTOK = 6
K = 100


def build(tok, item):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item["told"]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": item["ask"]},
    ]
    # enable_thinking=False where the template supports it: the first run put
    # the read position inside a <think> block on Qwen3.5, which is a different
    # distribution from the one the user ever sees.
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + item["prefix"]


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    used, total = layer_coverage(model)
    print(f"{model_id}   coverage {used}/{total} layers   K={K}  n_tok={NTOK}\n")
    out = {"model": model_id, "coverage": [used, total], "K": K, "items": {}}

    for item in ITEMS:
        prompt = build(tok, item)
        span = find_span(tok, prompt, item["value"])
        gold = gold_continuation(tok, prompt, item["value"])

        got0, steps0 = decode(model, tok, prompt, span, 0.0, NTOK)
        if divergence(gold, got0) is not None:
            print(f"[{item['key']}] SKIP: unmanipulated model says "
                  f"{tok.decode(got0)!r}, gold is {tok.decode(gold)!r}\n")
            continue

        print(f"[{item['key']}] gold {tok.decode(gold)!r} = {gold}")
        print(f"{'b':>5}  {'says':<22} {'div':>4} {'p(gold@div)':>12} "
              f"{'rk':>5} {'from rk':>8} {'rho':>7}")
        rows = []
        for b in LADDER:
            got, steps = decode(model, tok, prompt, span, b, NTOK)
            d = divergence(gold, got)
            row = {"b": b, "says": tok.decode(got), "div": d}
            if d is not None:
                pb, p0 = steps[d], steps0[d]
                g = gold[d]
                row |= {
                    "p_gold_at_div": float(pb[g]),
                    "rank_gold_at_div": int((pb > pb[g]).sum()) + 1,
                    "winner": tok.decode([got[d]]),
                    "winner_rank_at_0": int((p0 > p0[got[d]]).sum()) + 1,
                    "rho": rank_correlation(p0, pb, exclude=g, k=K),
                }
                print(f"{b:>5}  {row['says']!r:<22} {d:>4} "
                      f"{row['p_gold_at_div']:>12.3e} {row['rank_gold_at_div']:>5} "
                      f"{row['winner_rank_at_0']:>8} {row['rho']:>7.3f}")
            else:
                print(f"{b:>5}  {row['says']!r:<22}    - (still correct)")
            rows.append(row)
        out["items"][item["key"]] = {"gold": tok.decode(gold), "rows": rows}
        print()

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"sweep2_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

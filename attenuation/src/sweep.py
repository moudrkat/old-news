"""Fine ladder of b, three items, one model. Diagnostic, not the measurement.

The first coarse run jumped from p(gold)=1.0 at b=2 to 1e-15 at b=5 on
Qwen3-4B, which means the whole phenomenon lives in a band nothing sampled.
This walks that band in small steps and prints the top of the distribution at
each rung, so it is visible whether the model swaps in another *value* or
simply stops filling the slot.

    python src/sweep.py Qwen/Qwen3.5-4B

Writes results/sweep_<model>.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knob import find_span, layer_coverage, read

# Provisional items, in the style of the old-news fixture — a name, a number,
# and a code, because the value's character class may matter for what replaces
# it. Swap in the real fixture before anything is claimed.
ITEMS = [
    {"key": "dog", "value": "Bagr",
     "told": "By the way, my dog is called Bagr.",
     "ask": "What is my dog called?",
     "prefix": "Your dog is called"},
    {"key": "order", "value": "4417",
     "told": "By the way, my order number is 4417.",
     "ask": "What is my order number?",
     "prefix": "Your order number is"},
    {"key": "error", "value": "E-88",
     "told": "By the way, the error code I got was E-88.",
     "ask": "Which error code did I get?",
     "prefix": "The error code you got was"},
]

LADDER = [round(0.25 * i, 2) for i in range(0, 25)] + [7.0, 8.0, 10.0, 15.0, 20.0]


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q), natural log, over the full vocabulary."""
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].clamp_min(1e-45).log())).sum())


def build(tok, item):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item["told"]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": item["ask"]},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt + item["prefix"]


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    ).eval()
    used, total = layer_coverage(model)
    print(f"{model_id}   coverage {used}/{total} layers\n")

    out = {"model": model_id, "coverage": [used, total], "items": {}}

    for item in ITEMS:
        prompt = build(tok, item)
        span = find_span(tok, prompt, item["value"])
        gold = tok(" " + item["value"], add_special_tokens=False)["input_ids"][0]

        base = read(model, tok, prompt, span, 0.0)
        if base.argmax() != gold:
            print(f"[{item['key']}] SKIP: unmanipulated model answers "
                  f"{tok.decode([base.argmax()])!r}, not {tok.decode([gold])!r}")
            continue

        print(f"[{item['key']}] gold {tok.decode([gold])!r}, span {len(span)} tok")
        print(f"{'b':>6} {'p(gold)':>10} {'rk':>6} {'KL':>8}   top-5")
        rows = []
        for b in LADDER:
            r = read(model, tok, prompt, span, b) if b else base
            top5 = torch.topk(r.probs, 5)
            toks = [tok.decode([int(i)]) for i in top5.indices]
            row = {
                "b": b,
                "p_gold": r.p(gold),
                "rank_gold": r.rank_of(gold),
                "kl": kl(r.probs, base.probs),
                "winner": tok.decode([r.argmax()]),
                "winner_rank_at_0": base.rank_of(r.argmax()),
                "top5": toks,
                "top5_p": [float(p) for p in top5.values],
            }
            rows.append(row)
            print(f"{b:>6} {row['p_gold']:>10.3e} {row['rank_gold']:>6} "
                  f"{row['kl']:>8.3f}   {toks}")
        out["items"][item["key"]] = {"gold": tok.decode([gold]), "rows": rows}
        print()

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"sweep_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

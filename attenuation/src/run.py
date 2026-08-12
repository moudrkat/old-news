"""The measurement. Sweep b, read what the model says instead, measure where it
leaves its own path.

Gold path is the *unmanipulated model's own greedy continuation*, gated on it
containing the correct value. Earlier versions built the gold path from the
tokenizer, which broke twice: once on numeric values (the leading space token)
and once on Qwen3.5 (which writes ` **Bagr**` and so never matched ` Bagr`).
Comparing the model against itself has neither problem, and is the question
anyway: where does the answer leave the path it would otherwise have taken?

    python src/run.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knob import find_span, layer_coverage
from value import decode, divergence, rank_correlation

ITEMS = [
    {"key": "dog", "value": "Bagr", "told": "By the way, my dog is called Bagr.",
     "ask": "What is my dog called?", "prefix": "Your dog is called"},
    {"key": "order", "value": "4417", "told": "By the way, my order number is 4417.",
     "ask": "What is my order number?", "prefix": "Your order number is"},
    {"key": "error", "value": "E-88", "told": "By the way, the error code I got was E-88.",
     "ask": "Which error code did I get?", "prefix": "The error code you got was"},
    {"key": "city", "value": "Brno", "told": "By the way, I live in Brno.",
     "ask": "Which city do I live in?", "prefix": "You live in"},
    {"key": "time", "value": "19:40", "told": "By the way, my train leaves at 19:40.",
     "ask": "When does my train leave?", "prefix": "Your train leaves at"},
    {"key": "account", "value": "302", "told": "By the way, my account number ends in 302.",
     "ask": "What does my account number end in?", "prefix": "Your account number ends in"},
]

LADDER = [round(0.5 * i, 2) for i in range(0, 17)] + [10.0, 15.0, 20.0]
NTOK = 8
K = 100


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].clamp_min(1e-45).log())).sum())


def build(tok, item, use_prefix: bool = False):
    """use_prefix=False by default, and that is not a detail.

    The answer prefix ("Your dog is called") pins the read position, but it also
    makes "I don't know" a grammatically impossible continuation. Any question
    about whether the model declines is unanswerable under it — the first run
    read a forced completion as if it were a choice. The gold path is the
    model's own continuation anyway, so the prefix buys nothing we need.
    """
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item["told"]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": item["ask"]},
    ]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + (item["prefix"] if use_prefix else "")


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    used, total = layer_coverage(model)
    print(f"{model_id}   coverage {used}/{total}   K={K} n_tok={NTOK}\n")
    out = {"model": model_id, "coverage": [used, total], "K": K,
           "n_tok": NTOK, "ladder": LADDER, "items": {}}

    for item in ITEMS:
        prompt = build(tok, item)
        span = find_span(tok, prompt, item["value"])
        path, steps0 = decode(model, tok, prompt, span, 0.0, NTOK)
        said = tok.decode(path)

        # Gate: if the unmanipulated model does not produce the value, there is
        # nothing for the knob to take away. Declared in the preregistration;
        # skipped items are counted, not quietly dropped.
        if item["value"] not in said:
            print(f"[{item['key']}] SKIP: unmanipulated says {said!r}, "
                  f"no {item['value']!r}\n")
            out["items"][item["key"]] = {"skipped": said}
            continue

        print(f"[{item['key']}] path {said!r}")
        print(f"{'b':>5}  {'says':<34} {'div':>4} {'p@div':>10} {'rk':>6} "
              f"{'from':>6} {'KL':>7} {'rho':>7}")
        rows = []
        for b in LADDER:
            got, steps = decode(model, tok, prompt, span, b, NTOK)
            d = divergence(path, got)
            row = {"b": b, "says": tok.decode(got), "div": d}
            if d is not None:
                pb, p0, g = steps[d], steps0[d], path[d]
                row |= {
                    "gold_tok": tok.decode([g]),
                    "p_gold_at_div": float(pb[g]),
                    "rank_gold_at_div": int((pb > pb[g]).sum()) + 1,
                    "winner": tok.decode([got[d]]),
                    "winner_rank_at_0": int((p0 > p0[got[d]]).sum()) + 1,
                    "kl_at_div": kl(pb, p0),
                    "rho": rank_correlation(p0, pb, exclude=g, k=K),
                }
                print(f"{b:>5}  {row['says']!r:<34} {d:>4} {row['p_gold_at_div']:>10.2e} "
                      f"{row['rank_gold_at_div']:>6} {row['winner_rank_at_0']:>6} "
                      f"{row['kl_at_div']:>7.3f} {row['rho']:>7.3f}")
            else:
                print(f"{b:>5}  {row['says']!r:<34}    - still on path")
            rows.append(row)
        out["items"][item["key"]] = {"path": said, "rows": rows}
        print()

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"run_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

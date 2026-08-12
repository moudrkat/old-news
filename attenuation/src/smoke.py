"""Smoke test: does the knob land on this model at all?

One item, one read position, a ladder of b. Prints the gold token's probability
and what wins instead. Answers exactly one question — is the instrument alive on
this architecture — before anything is measured with it.

    python src/smoke.py Qwen/Qwen3.5-4B

Fails loudly in the two ways that matter: the mask being silently ignored
(distributions identical at b=0 and b=max), and the model not knowing the
answer even unmanipulated (nothing to take away).
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knob import find_span, layer_coverage, read

FACT = "Bagr"
LADDER = [0, 2, 5, 10, 20, 40]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"By the way, my dog is called {FACT}."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "What is my dog called?"},
]
ANSWER_PREFIX = "Your dog is called"


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    ).eval()

    used, total = layer_coverage(model)
    print(f"model      {model_id}")
    print(f"coverage   {used}/{total} layers see the attention mask")

    prompt = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    prompt += ANSWER_PREFIX
    span = find_span(tok, prompt, FACT)
    print(f"span       {len(span)} tokens at {span} = "
          f"{tok.decode(tok(prompt, add_special_tokens=False)['input_ids'][span[0]:span[-1]+1])!r}")

    gold = tok(" " + FACT, add_special_tokens=False)["input_ids"][0]
    print(f"gold token {gold} = {tok.decode([gold])!r}\n")

    print(f"{'b':>5} {'p(gold)':>12} {'rank(gold)':>11}  winner")
    base = None
    for b in LADDER:
        r = read(model, tok, prompt, span, b)
        if base is None:
            base = r
        top = r.argmax()
        mark = "" if top == gold else f"   <- from rank {base.rank_of(top)} at b=0"
        print(f"{b:>5} {r.p(gold):>12.3e} {r.rank_of(gold):>11} "
              f"{tok.decode([top])!r}{mark}")
        if b == LADDER[-1]:
            drift = float((r.probs - base.probs).abs().max())

    print()
    if base.rank_of(gold) != 1:
        print("STOP: the unmanipulated model does not answer correctly. "
              "Nothing to take away — wrong item or wrong model.")
        return 1
    if drift < 1e-6:
        print("STOP: the distribution did not move between b=0 and b=max. "
              "The 4D mask is being ignored on this model.")
        return 1
    print("knob is alive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

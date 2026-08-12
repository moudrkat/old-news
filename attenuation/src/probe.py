"""Is the "I was told this" signal still on when the fact is unreadable?

The behavioural result says the model answers a wrong value and still claims it
was told. This asks whether that shows up in the activations, with the simplest
tool there is.

Train a linear probe to separate two states where the ground truth is known:

    present  the fact is in the context, knob off
    absent   the fact was never in the context

then apply it to the state where the model gets it wrong:

    faint    the fact is in the context but has been made hard to read

Prediction, fixed before running: the probe calls `faint` **present**. If it
does, the internal "I have this information" signal does not track whether the
information was read correctly — which is why nothing fires to make the model
abstain.

Read at the last prompt position, per layer, so the layer profile is visible
rather than assumed. Leave-one-item-out: every item is scored by a probe that
never saw it, because with six items a probe trained on all of them would
memorise the prompt rather than the state.

    python src/probe.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from absent import FAINT_LADDER, build_absent, say, variant
from knob import biased_mask, find_span
from run import ITEMS, build


@torch.no_grad()
def hiddens(model, tok, prompt: str, span: list[int], b: float) -> torch.Tensor:
    """Residual stream at the final prompt position, every layer. [L+1, d]"""
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = ids.to(model.device)
    mask = biased_mask(ids.shape[1], span, b, next(model.parameters()).dtype, model.device)
    out = model(input_ids=ids, attention_mask=mask, use_cache=False,
                output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])


def fit(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mean-difference direction and the midpoint threshold, per layer.

    Difference of means rather than logistic regression: with three or four
    examples a side, a fitted classifier separates anything, and the direction
    would say more about the optimiser than the model.
    """
    w = pos.mean(0) - neg.mean(0)
    w = w / w.norm().clamp_min(1e-9)
    return w, float(((pos @ w).mean() + (neg @ w).mean()) / 2)


def main(model_id: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}\n")

    states = {}
    for i, item in enumerate(ITEMS):
        donor = ITEMS[(i + 1) % len(ITEMS)]
        prompt = build(tok, item)
        span = find_span(tok, prompt, item["value"])
        present = say(model, tok, prompt, span, 0.0, 24)
        if item["value"] not in present:
            print(f"[{item['key']}] skip: no value when present")
            continue
        faint_b = next((b for b in FAINT_LADDER
                        if item["value"] not in say(model, tok, prompt, span, b, 24)), None)
        if faint_b is None:
            print(f"[{item['key']}] skip: value never goes away")
            continue
        p_abs = build_absent(tok, variant(item, "drop", donor))
        states[item["key"]] = {
            "faint_b": faint_b,
            "present": hiddens(model, tok, prompt, span, 0.0),
            "faint": hiddens(model, tok, prompt, span, faint_b),
            "absent": hiddens(model, tok, p_abs, [], 0.0),
        }
        print(f"[{item['key']}] faint at b={faint_b}")

    keys = list(states)
    nl = states[keys[0]]["present"].shape[0]
    print(f"\n{len(keys)} items, {nl} layers. Leave-one-item-out.\n")
    print(f"{'layer':>6} {'present held-out':>17} {'absent held-out':>16} "
          f"{'FAINT called present':>21}")

    rows = []
    for L in range(nl):
        ok_p = ok_a = faint_p = 0
        for held in keys:
            tr = [k for k in keys if k != held]
            w, thr = fit(torch.stack([states[k]["present"][L] for k in tr]),
                         torch.stack([states[k]["absent"][L] for k in tr]))
            ok_p += float(states[held]["present"][L] @ w) > thr
            ok_a += float(states[held]["absent"][L] @ w) <= thr
            faint_p += float(states[held]["faint"][L] @ w) > thr
        n = len(keys)
        rows.append({"layer": L, "present": ok_p / n, "absent": ok_a / n,
                     "faint_called_present": faint_p / n})
        print(f"{L:>6} {ok_p}/{n}{'':>13} {ok_a}/{n}{'':>12} {faint_p}/{n}")

    p = Path(__file__).resolve().parents[1] / "results"
    f = p / f"probe_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps({"model": model_id, "items": keys,
                             "faint_b": {k: states[k]["faint_b"] for k in keys},
                             "layers": rows}, indent=1))
    print("\nwrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

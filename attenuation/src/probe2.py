"""Does the model hold an "I was told this" state, and does it stay on when the
fact has been made unreadable?

100 items, 10 kinds of fact. Three states per item, all with the same frame:

    present   the fact is in the context, knob off
    absent    a *different* kind of fact sits in the same slot, knob off
    faint     the fact is in the context, knob at the dose where the answer is
              already wrong

Train present-vs-absent, test on faint. Held out **by kind of fact**, not by
item: a probe split at random would be tested on `Bagr` having trained on
`Fizzle`, and would only have to recognise the template.

Three things that must hold or the number means nothing, all reported:

  layer 0     the embedding layer holds no state. If it separates, the probe is
              reading tokens and the run is void. The pilot scored 6/6 there.
  shuffled    labels permuted; must fall to chance.
  gate        the unmanipulated model must answer correctly, and there must be a
              dose where it does not. Items failing either are counted, not
              silently dropped.

    python src/probe2.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100, TYPES
from knob import biased_mask, find_span
from value import decode

LADDER = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0]
NTOK = 24
SEED = 4242


def chat(tok, told: str, ask: str) -> str:
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": told},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": ask}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def hiddens(model, tok, prompt: str, span: list[int], b: float) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    mask = biased_mask(ids.shape[1], span, b, next(model.parameters()).dtype, model.device)
    out = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC. Threshold-free, so no cutoff has to be chosen."""
    x = torch.cat([pos, neg])
    r = x.argsort().argsort().float() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(model_id: str) -> int:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}  {len(ITEMS100)} items\n")

    keep, drop_wrong, drop_nofaint = [], 0, 0
    for it in ITEMS100:
        p = chat(tok, it["told"], it["ask"])
        span = find_span(tok, p, it["value"])
        got, _ = decode(model, tok, p, span, 0.0, NTOK)
        if it["value"] not in tok.decode(got):
            drop_wrong += 1
            continue
        fb = None
        for b in LADDER:
            g, _ = decode(model, tok, p, span, b, NTOK)
            if it["value"] not in tok.decode(g):
                fb = b
                break
        if fb is None:
            drop_nofaint += 1
            continue
        pa = chat(tok, it["donor"], it["ask"])
        keep.append({
            "key": it["key"], "type": it["type"], "faint_b": fb,
            "present": hiddens(model, tok, p, span, 0.0),
            "faint": hiddens(model, tok, p, span, fb),
            "absent": hiddens(model, tok, pa, [], 0.0),
        })
    print(f"kept {len(keep)}   dropped: {drop_wrong} answered wrong unmanipulated, "
          f"{drop_nofaint} never lost the value\n")
    if len(keep) < 20:
        print("too few items survived the gate; stopping")
        return 1

    types = sorted({r["type"] for r in keep})
    nl = keep[0]["present"].shape[0]
    print(f"{len(types)} kinds of fact, {nl} layers, held out by kind\n")
    print(f"{'layer':>6} {'AUC present/absent':>19} {'AUC shuffled':>13} "
          f"{'faint scored as present':>24}")

    rows = []
    for L in range(nl):
        real, shuf, faint_side = [], [], []
        for t in types:
            tr = [r for r in keep if r["type"] != t]
            te = [r for r in keep if r["type"] == t]
            P = torch.stack([r["present"][L] for r in tr])
            A = torch.stack([r["absent"][L] for r in tr])
            w = P.mean(0) - A.mean(0)
            w = w / w.norm().clamp_min(1e-9)
            mid = float(((P @ w).mean() + (A @ w).mean()) / 2)
            real.append(auc(torch.stack([r["present"][L] for r in te]) @ w,
                            torch.stack([r["absent"][L] for r in te]) @ w))
            faint_side += [float(r["faint"][L] @ w) > mid for r in te]
            # shuffled control: same pipeline, labels permuted within the
            # training set, so any separation it finds is the null
            both = torch.cat([P, A])
            perm = torch.randperm(len(both))
            h1, h2 = both[perm[: len(P)]], both[perm[len(P):]]
            ws = h1.mean(0) - h2.mean(0)
            ws = ws / ws.norm().clamp_min(1e-9)
            shuf.append(auc(torch.stack([r["present"][L] for r in te]) @ ws,
                            torch.stack([r["absent"][L] for r in te]) @ ws))
        row = {"layer": L, "auc": sum(real) / len(real),
               "auc_shuffled": sum(shuf) / len(shuf),
               "faint_as_present": sum(faint_side) / len(faint_side)}
        rows.append(row)
        print(f"{L:>6} {row['auc']:>19.3f} {row['auc_shuffled']:>13.3f} "
              f"{row['faint_as_present']:>24.3f}")

    p = Path(__file__).resolve().parents[1] / "results"
    p.mkdir(exist_ok=True)
    f = p / f"probe2_{model_id.split('/')[-1]}.json"
    f.write_text(json.dumps({
        "model": model_id, "n_kept": len(keep), "n_types": len(types),
        "dropped_wrong": drop_wrong, "dropped_nofaint": drop_nofaint,
        "faint_b": {r["key"]: r["faint_b"] for r in keep}, "layers": rows}, indent=1))
    print("\nwrote", f)
    print("\nlayer 0 AUC is the null: if it is not near 0.5, the probe is "
          "reading tokens and nothing below it counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

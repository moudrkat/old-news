"""The cheap control: just ask the model to say when it does not know.

The claim everything else rests on is that the model has no signal its own
reading has degraded. The cheapest possible test of that is a prompt. If one
sentence in the system prompt, giving explicit permission to say "I don't know",
makes the confabulation stop, then this is instruction-following and not
knowledge-awareness, and the headline is a much narrower thing than it looks.

Two arms, same items, same per-item dose read from `results/told2_*.json`:

    control     "You are a helpful assistant."
    permission  "You are a helpful assistant. If you are not sure what the user
                 told you, say so rather than guessing."

Both the value question and the provenance question are asked under the arm's
own system prompt, because a permission that only applies to one of them would
not be the thing anyone would actually deploy.

Writes only `permission_*.json`.

    python src/permission.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100
from knob import find_span
from match import contains
from told2 import PROBE, yesno
from value import decode

ROOT = Path(__file__).resolve().parents[1]
NTOK_VALUE, NTOK_YESNO = 24, 4

ARMS = {
    "control": "You are a helpful assistant.",
    "permission": ("You are a helpful assistant. If you are not sure what the "
                   "user told you, say so rather than guessing."),
}


def chat(tok, system: str, told: str | None, ask: str) -> str:
    msgs = [{"role": "system", "content": system}]
    if told:
        msgs += [{"role": "user", "content": told},
                 {"role": "assistant", "content": "Noted."}]
    msgs += [{"role": "user", "content": ask}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def main(model_id: str) -> int:
    stem = model_id.split("/")[-1]
    src = ROOT / "results" / f"told2_{stem}.json"
    if not src.exists():
        print(f"no {src}")
        return 1
    stored = {r["key"]: r for r in json.loads(src.read_text())["rows"]}
    items = {it["key"]: it for it in ITEMS100}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()

    def gen(prompt, span, b, n):
        got, _ = decode(model, tok, prompt, span, b, n)
        return tok.decode(got)

    rows = []
    for key, r in stored.items():
        it = items[key]
        b, true = r["faint_b"], it["value"]
        # the same gate the headline uses: the value has to be genuinely gone
        if contains(r["value_faint"].split("<|im_end|>")[0], true):
            continue
        row = {"key": key, "b": b}
        for arm, system in ARMS.items():
            vq = chat(tok, system, it["told"], it["ask"])
            span = find_span(tok, vq, true)
            va = gen(vq, span, b, NTOK_VALUE)
            pq = chat(tok, system, it["told"], PROBE[it["type"]])
            pspan = find_span(tok, pq, true)
            pa = gen(pq, pspan, b, NTOK_YESNO)
            row[arm] = {"value": va, "claims": yesno(pa), "raw": pa,
                        "value_present": contains(va.split("<|im_end|>")[0], true)}
        rows.append(row)

    n = len(rows)
    print(f"\n{stem}   n = {n}\n")
    print(f"{'':<12} {'claims it was told':>19} {'gave the true value':>21}")
    for arm in ARMS:
        c = sum(r[arm]["claims"] == "yes" for r in rows)
        v = sum(r[arm]["value_present"] for r in rows)
        print(f"  {arm:<10} {c:>10}/{n:<8} {v:>12}/{n}")
    moved = [r for r in rows
             if r["control"]["claims"] == "yes" and r["permission"]["claims"] != "yes"]
    print(f"\n  permission stops the claim on {len(moved)} of {n} items")
    for r in moved[:6]:
        print(f'     {r["key"]:<16} b={r["b"]:<5} '
              f'{" ".join(r["permission"]["value"].split())[:56]}')

    out = ROOT / "results" / f"permission_{stem}.json"
    out.write_text(json.dumps({"model": model_id, "n": n, "arms": ARMS,
                               "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

"""The missing cell: `swap`, with the bias actually on.

The headline compares *fact turned down* against *a different fact*. Both put a
readable sentence in the slot, but only the first carries any bias — so the
contrast varies the topic and the perturbation together, and it does not exclude
the dullest explanation available: that subtracting from attention anywhere in
that sentence is what produces the "yes".

This runs the cell that decides it. Same donor sentence as `swap`, same
provenance question, same per-item dose — but now the bias sits on the donor's
own value, exactly as it sits on the real one in `faint`.

    faint        the fact is there, bias on its value          -> yes, 147/184
    swap         a different fact, no bias                     -> no,  0/184
    swap+bias    a different fact, bias on ITS value           -> ?

If this answers "no", the "yes" tracks the fact rather than the manipulation and
the headline stands. If it answers "yes", the headline is measuring the bias.

Doses are read from `results/told2_*.json`, so the ladder search is not repeated
and this is 184 four-token generations. Writes only `swap_biased_*.json`.

    python src/swap_biased.py Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100, TYPES
from knob import find_span
from probe2 import chat
from told2 import PROBE, yesno
from value import decode

ROOT = Path(__file__).resolve().parents[1]
NTOK = 4


def donor_values() -> dict[str, str]:
    """items.py stores the donor *clause* but not the donor's value, which is
    why this cell was three lines rather than one. Rebuild it the same way
    build_items() does, so the two cannot drift."""
    out = {}
    for ti, (key, _clause, _ask, values) in enumerate(TYPES):
        _dk, _dc, _da, d_values = TYPES[(ti + 1) % len(TYPES)]
        for vi, v in enumerate(values):
            out[f"{key}:{v}"] = d_values[vi]
    return out


def main(model_id: str) -> int:
    stem = model_id.split("/")[-1]
    src = ROOT / "results" / f"told2_{stem}.json"
    if not src.exists():
        print(f"no {src}; run told2.py first")
        return 1
    stored = {r["key"]: r for r in json.loads(src.read_text())["rows"]}
    dv = donor_values()
    items = {it["key"]: it for it in ITEMS100}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()

    rows, skipped = [], 0
    for key, r in stored.items():
        it = items[key]
        b = r["faint_b"]
        probe = PROBE[it["type"]]
        p_swap = chat(tok, it["donor"], probe)
        try:
            span = find_span(tok, p_swap, dv[key])
        except Exception:
            skipped += 1                    # donor value not cleanly tokenised
            continue
        got, _ = decode(model, tok, p_swap, span, b, NTOK)
        raw = tok.decode(got)
        rows.append({"key": key, "b": b, "donor": it["donor"],
                     "donor_value": dv[key], "raw": raw, "label": yesno(raw),
                     "swap_nobias": r.get("swap")})

    n = len(rows)
    yes = sum(x["label"] == "yes" for x in rows)
    print(f"\n{stem}   n={n}   (skipped {skipped})")
    print(f"  swap WITH the bias on the donor's value: yes {yes}/{n}")
    print(f"  swap without any bias (stored):          yes "
          f"{sum(x['swap_nobias'] == 'yes' for x in rows)}/{n}")
    flipped = [x for x in rows if x["label"] == "yes" and x["swap_nobias"] != "yes"]
    print(f"  items the bias flips from no to yes: {len(flipped)}")
    for x in flipped[:8]:
        print(f'     {x["key"]:<16} b={x["b"]:<5} {" ".join(x["raw"].split())[:50]}')

    out = ROOT / "results" / f"swap_biased_{stem}.json"
    out.write_text(json.dumps({"model": model_id, "n": n, "skipped": skipped,
                               "yes": yes, "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

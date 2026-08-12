"""Does the hesitation track the damage, or track how odd the answer looks?

Reading the generations turned up a second thing: sometimes the model flags its
own answer.

    Marnok -> "Your dog is called Marn. (Though I notice you said "Marn" —
               is that a typo…"
    227    -> "You are in room 207. I apologize for the confusion earlier!
               I misread "20…"

That looks like self-knowledge. Two reasons to doubt it before believing it:

  * the apology is for an *earlier* confusion that never happened — the
    conversation is four messages long and nothing was misread before;
  * every hesitation so far is on a name or a room, and none on a time or an
    order number. `Marn` looks like a strange name. `19:45` looks like a
    perfectly ordinary time — and the model says it without a murmur.

So the hypothesis under test is that the hesitation is a reaction to the
*string the model just wrote*, not to the reading that produced it. That needs
no self-knowledge at all.

Two numbers settle it:

  1. hesitation under `present` (nothing degraded) vs under `faint` — if it is
     the same, it is style;
  2. hesitation by kind of fact — if it sits on names and rooms and is absent
     on times and numbers, it tracks how odd the output looks, not how much
     evidence was taken away.

    python src/hedge.py Qwen/Qwen3.5-4B
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
NTOK = 24

# Fixed before counting anything. Self-correction and query markers only —
# not general discourse words like "though" or "actually", which appear in
# ordinary answers and would inflate both conditions equally.
HEDGE = [
    "wait", "hold on", "hmm", "or is it", "did you mean", "typo", "misread",
    "i apolog", "apologize for the confusion", "i'm not sure", "i am not sure",
    "i notice", "that doesn't sound", "that does not sound", "confusion",
]


def clean(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return " ".join(s.split())


def hedged(s: str) -> bool:
    t = clean(s).lower()
    return any(h in t for h in HEDGE)


def main(model_id: str) -> int:
    stem = model_id.split("/")[-1]
    src = ROOT / "results" / f"told2_{stem}.json"
    if not src.exists():
        print(f"need {src} first")
        return 1
    faint = {r["key"]: r for r in json.load(open(src))["rows"]}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{model_id}   {len(faint)} items\n")

    rows = []
    for it in ITEMS100:
        if it["key"] not in faint:
            continue
        p = chat(tok, it["told"], it["ask"])
        span = find_span(tok, p, it["value"])
        got, _ = decode(model, tok, p, span, 0.0, NTOK)
        present = tok.decode(got)
        rows.append({
            "key": it["key"], "type": it["type"], "b": faint[it["key"]]["faint_b"],
            "present": clean(present), "faint": clean(faint[it["key"]]["value_faint"]),
            "hedge_present": hedged(present),
            "hedge_faint": hedged(faint[it["key"]]["value_faint"]),
        })

    n = len(rows)
    hp = sum(r["hedge_present"] for r in rows)
    hf = sum(r["hedge_faint"] for r in rows)
    print(f"hesitation, present : {hp}/{n} ({100*hp/n:.0f}%)")
    print(f"hesitation, faint   : {hf}/{n} ({100*hf/n:.0f}%)\n")

    print(f"{'kind':<10} {'n':>4} {'present':>9} {'faint':>7}")
    for t in dict.fromkeys(r["type"] for r in rows):
        rs = [r for r in rows if r["type"] == t]
        print(f"{t:<10} {len(rs):>4} "
              f"{sum(r['hedge_present'] for r in rs):>9} "
              f"{sum(r['hedge_faint'] for r in rs):>7}")

    bs = sorted({r["b"] for r in rows})
    print(f"\n{'b':>5} {'n':>4} {'hedge (faint)':>14}")
    for b in bs:
        rs = [r for r in rows if r["b"] == b]
        print(f"{b:>5} {len(rs):>4} {sum(r['hedge_faint'] for r in rs):>14}")

    out = ROOT / "results" / f"hedge_{stem}.json"
    out.write_text(json.dumps({"model": model_id, "markers": HEDGE,
                               "n": n, "rows": rows}, indent=1))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"))

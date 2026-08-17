"""The `drop` cell again, with a budget big enough for the model to finish.

`drop` puts nothing in the conversation and asks the provenance question, so a
model that knows what it was told should say no. On Qwen3-4B it does, 100 of
100. On Qwen3.5-4B the cell came back unmeasured: every answer began
`Thinking Process:` and the four-token cap cut it off before it committed, so
the run recorded neither yes nor no.

That is a budget artefact, not a result, and it has been sitting in the
limitations as a caveat on a control. Same prompts, same greedy decoding, same
`b = 0`, only `--ntok` changes. If the answers are no once the model is allowed
to finish, the caveat goes away; if they are yes, the control is in trouble and
that matters far more than the caveat did.

Two readings are recorded rather than one, because the parser the headline uses
reads the first word and the first word here is `Thinking`:

    first   `told2.yesno` on the reply as it stands, the strict rule
    final   the last yes/no in a reply that actually finished

Writes only `results/drop_long_<model>.json`.

    python src/drop_long.py Qwen/Qwen3.5-4B --ntok 128
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from items import ITEMS100
from told2 import PROBE, yesno

ROOT = Path(__file__).resolve().parents[1]


def final_yesno(s: str) -> str:
    """The model's actual answer, or "other" if it never gave one.

    An earlier version of this took the last standalone yes/no anywhere in the
    reply. That is wrong for a model that thinks out loud: every reply quotes
    the instruction back, *Answer only yes or no*, so the regex found a "no"
    in all 100 replies and reported a clean 100/100 for a cell where the model
    had not answered at all. A reading is only taken from a reply that reached
    the end of its own turn.
    """
    body, done, _ = s.partition("<|im_end|>")
    if not done:
        return "unfinished"
    hits = re.findall(r"\b(yes|no)\b", body, re.I)
    return hits[-1].lower() if hits else "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ntok", type=int, default=128)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    print(f"{a.model}   drop only   ntok = {a.ntok}\n")

    rows = []
    for it in ITEMS100:
        p = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": PROBE[it["type"]]}],
            tokenize=False, add_generation_prompt=True)
        # `drop` carries no bias, and at b = 0 `biased_mask` returns the plain
        # causal mask, so cached greedy generation is identical to `decode` here
        # and stops at the end of the turn instead of running to the cap.
        # `decode` re-forwards the whole sequence every step, which is fine at
        # 40 tokens and quadratic at 512.
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**ids, max_new_tokens=a.ntok, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids["input_ids"].shape[1]:])
        rows.append({"key": it["key"], "type": it["type"], "raw": reply,
                     "first": yesno(reply), "final": final_yesno(reply)})

    n = len(rows)
    print(f"{'reading':<8} {'yes':>5} {'no':>5} {'other':>6} {'unfinished':>11}   of {n}")
    for k in ("first", "final"):
        c = {v: sum(r[k] == v for r in rows) for v in ("yes", "no", "other", "unfinished")}
        print(f"{k:<8} {c['yes']:>5} {c['no']:>5} {c['other']:>6} {c['unfinished']:>11}")

    stuck = [r for r in rows if r["final"] in ("other", "unfinished")]
    print(f"\nno usable answer in {len(stuck)} of {n}")
    for r in rows[:3]:
        print(f'   {r["key"]:<16} {" ".join(r["raw"].split())[:96]}')

    out = ROOT / "results" / f"drop_long_{a.model.split('/')[-1]}.json"
    out.write_text(json.dumps({"model": a.model, "ntok": a.ntok, "n": n,
                               "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

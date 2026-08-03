"""Does suppressing a span cost the model the facts stated in it?

Run the recall probe on any model and print the table plus every miss, split
into "confabulation" (fluent, wrong, a near neighbour of the right answer) and
"degenerate" (visibly broken text).

    python examples/recall_across_models.py --model mid     # Qwen3-4B
    python examples/recall_across_models.py --model llama   # Llama-3.1-8B

Each demoted message block carries an instruction ("always reply in lowercase")
and a fact ("my order number is 4417-B") at the same time. Sweeping the
suppression strength should kill the instruction while leaving the fact
readable; the question is where that stops being true and what the failure
looks like when it does.

Needs ~16 GB of VRAM for the 8B. n = 12 per point, so treat the rates as a
direction, not a measurement.
"""

from __future__ import annotations

import argparse

from oldnews.evals.recall import build
from oldnews.evals.staleset import collapsed
from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import render
from oldnews.vsteer import generate

DOSES = [0.0, 0.25, 0.5, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mid")
    ap.add_argument("--gamma-plus", type=float, default=2.5)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--doses", default=",".join(str(d) for d in DOSES))
    args = ap.parse_args()

    model, tok = load(args.model)
    cases = build("conflict")
    doses = [float(x) for x in args.doses.split(",")]

    print(f"\n{args.model}  ·  {len(cases)} questions per dose  ·  greedy\n")
    print(f"{'gamma-':>7} {'obeys old rule':>15} {'recalls fact':>14}")
    misses: list[tuple[float, str, str]] = []

    for gm in doses:
        pol = None if gm == 0 else SteerPolicy(
            mode="binary", gamma_plus=args.gamma_plus, gamma_minus=gm)
        stale = recalled = 0
        for case in cases:
            r = render(tok, case.messages, current_epoch=case.current_epoch)
            text, _ = generate(model, tok, r, policy=pol,
                               max_new_tokens=args.max_new_tokens,
                               current_epoch=case.current_epoch)
            stale += case.verdict(text) == "stale"
            if case.recalled(text):
                recalled += 1
            else:
                misses.append((gm, case.query, text.strip()))
        n = len(cases)
        print(f"{gm:>7} {stale / n:>14.0%} {recalled}/{n:>13}")

    print("\nmisses, and what kind:\n")
    for gm, q, text in misses:
        kind = "DEGENERATE  " if collapsed(text) else "CONFABULATED"
        print(f"  γ-{gm:<5} {kind} {q[:34]:36s} {text[:64]!r}")
    print("\nDEGENERATE is the widened repetition check; everything else is "
          "fluent, well-formed text with the wrong fact in it.")


if __name__ == "__main__":
    main()

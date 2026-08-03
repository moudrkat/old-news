"""Run StaleSet and dump results to JSON for the plots.

    python -m oldnews.evals.run --model tiny --out results/tiny.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..model import load
from ..policy import GAMMA_MINUS, GAMMA_PLUS, Priority, SteerPolicy
from ..transcript import mark_constraint_spans, render
from ..vsteer import generate
from . import recall as recall_mod
from .staleset import build, collapsed
from .stats import summarise

BASELINE_CONDITIONS = ["no_history", "conflict", "prompt_fix", "aligned"]


def provenance(args) -> dict:
    """Everything needed to reproduce this row of numbers."""
    import platform
    import subprocess

    import torch
    import transformers

    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sh("git rev-parse --short HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
        "decoding": "greedy (argmax), deterministic",
        "seed": args.seed,
        "args": vars(args),
    }


def _render(tok, case, spans: str):
    r = render(tok, case.messages, current_epoch=case.current_epoch)
    if spans == "constraint":
        pats = {"SYSTEM": [case.system_needle]}
        if case.condition != "no_history":
            pats["STALE"] = [case.stale_needle]
        r = mark_constraint_spans(r, pats)
    return r


def run_condition(model, tok, cases, policy, spans, max_new_tokens, group_rule,
                  attr_step: int = 0):
    rows = []
    for i, case in enumerate(cases):
        r = _render(tok, case, spans)
        t0 = time.time()
        text, rep = generate(
            model,
            tok,
            r,
            policy=policy,
            max_new_tokens=max_new_tokens,
            current_epoch=case.current_epoch,
            group_rule=group_rule,
            attr_step=attr_step,
        )
        row = {
            "family": case.family,
            "query": case.query,
            "variant": getattr(case, "variant", 0),
            "condition": case.condition,
            "verdict": case.verdict(text),
            "collapsed": collapsed(text),
            "text": text,
            "secs": round(time.time() - t0, 2),
            "heads_edited": rep.n_heads_edited if rep else None,
            "heads_total": rep.n_heads_total if rep else None,
            "n_tokens": r.n_tokens,
        }
        if hasattr(case, "recalled"):
            row["recalled"] = case.recalled(text)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(cases)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--out", default="results/tiny.json")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--spans", default="message", choices=["message", "constraint"])
    ap.add_argument("--group-rule", default="max", choices=["mean", "max", "sum"])
    ap.add_argument("--gamma-plus", type=float, default=GAMMA_PLUS)
    ap.add_argument("--gamma-minus", type=float, default=GAMMA_MINUS)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--suite",
        default="main",
        choices=["main", "sweep", "age", "recall", "attrstep", "boost", "grid"],
        help="main = condition comparison; sweep = gamma_minus curve; "
        "age = influence vs how many epochs back the history is; "
        "recall = does the model still SEE the demoted history",
    )
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)

    model, tok = load(args.model)
    out = {
        "model": args.model,
        "args": vars(args),
        "provenance": provenance(args),
        "runs": {},
        "summary": {},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.suite == "grid":
        # Both knobs at once, on the families that resist. Format compliance
        # alone is not trustworthy here -- a large boost makes the model emit
        # damaged text that happens to satisfy the checker -- so score this
        # file with `python -m oldnews.evals.judge_run` afterwards.
        from .staleset import FAMILIES

        fams = [f for f in FAMILIES if f.key in ("options", "length")]
        cases = build("conflict", families=fams, variants=args.variants)
        for gp in [2.5, 5.0, 8.0]:
            for gm in [0.5, 0.75, 1.0]:
                pol = SteerPolicy(mode="binary", gamma_plus=gp, gamma_minus=gm)
                name = f"gp{gp}_gm{gm}"
                print(f"[{name}]", flush=True)
                rows = run_condition(model, tok, cases, pol, args.spans,
                                     args.max_new_tokens, args.group_rule)
                out["runs"][name] = rows
                out["summary"][name] = summarise(rows) | {
                    "gamma_plus": gp, "gamma_minus": gm}
                s = out["summary"][name]
                print(f"    system={s['system']:.3f} collapse={s['collapse']:.3f}",
                      flush=True)

    elif args.suite == "boost":
        # Two levers never tested on the end-of-answer families: crank the
        # privileged boost well past the paper's 2.5, and switch from
        # whole-message spans (V-Simple) to constraint-only spans (V-Steer
        # proper). Collapse rate is reported because cranking gamma_plus is
        # exactly how you would expect to break generation.
        from .staleset import FAMILIES

        fams = [f for f in FAMILIES if f.key in ("options", "length")]
        for spans in ["message", "constraint"]:
            cases = build("conflict", families=fams, variants=args.variants)
            for gp in [2.5, 5.0, 10.0, 20.0]:
                pol = SteerPolicy(mode="binary", gamma_plus=gp,
                                  gamma_minus=args.gamma_minus)
                name = f"{spans}_gp{gp}"
                print(f"[{name}]", flush=True)
                rows = run_condition(model, tok, cases, pol, spans,
                                     args.max_new_tokens, args.group_rule)
                out["runs"][name] = rows
                out["summary"][name] = summarise(rows) | {
                    "gamma_plus": gp, "spans": spans}
                s = out["summary"][name]
                print(f"    system={s['system']:.3f} stale={s['stale']:.3f} "
                      f"collapse={s['collapse']:.3f}", flush=True)

    elif args.suite == "attrstep":
        # Does moving the attribution into the answer rescue the constraints
        # the first token cannot reveal? `options`/`length` are the end-of-
        # answer rules; `prefix`/`case` are first-token controls that should
        # not need this and might even be hurt by it.
        from .staleset import FAMILIES

        pol = SteerPolicy(mode="binary", gamma_plus=args.gamma_plus,
                          gamma_minus=args.gamma_minus)
        for group, keys in [("late", ["options", "length"]),
                            ("early", ["prefix", "case"])]:
            fams = [f for f in FAMILIES if f.key in keys]
            cases = build("conflict", families=fams, variants=args.variants)
            for st in [0, 8, 16, 32]:
                name = f"{group}_step{st}"
                print(f"[{name}]", flush=True)
                rows = run_condition(
                    model, tok, cases, pol, args.spans, args.max_new_tokens,
                    args.group_rule, attr_step=st,
                )
                out["runs"][name] = rows
                out["summary"][name] = summarise(rows) | {
                    "attr_step": st, "group": group}
                s = out["summary"][name]
                print(f"    system={s['system']:.3f} stale={s['stale']:.3f} "
                      f"collapse={s['collapse']:.3f}", flush=True)

    elif args.suite == "recall":
        # The demoted span carries an instruction AND a fact. Obedience to the
        # instruction should fall; recall of the fact must not.
        for gm in [0.0, 0.25, 0.5, 0.75, 0.9]:
            pol = None if gm == 0.0 else SteerPolicy(
                mode="binary", gamma_plus=args.gamma_plus, gamma_minus=gm
            )
            name = f"recall_gm{gm}"
            print(f"[{name}]", flush=True)
            rows = run_condition(
                model, tok, recall_mod.build("conflict"), pol, args.spans,
                args.max_new_tokens, args.group_rule,
            )
            n = len(rows) or 1
            out["runs"][name] = rows
            out["summary"][name] = summarise(rows) | {
                "gamma_minus": gm,
                "fact_recall": sum(bool(r.get("recalled")) for r in rows) / n,
            }
            s = out["summary"][name]
            print(f"    follows system {s['system']:.2f} | "
                  f"follows stale {s['stale']:.2f} | "
                  f"fact recall {s['fact_recall']:.2f}", flush=True)

    elif args.suite == "main":
        for cond in BASELINE_CONDITIONS:
            print(f"[{cond}] no steering", flush=True)
            cases = build(cond, variants=args.variants)
            rows = run_condition(
                model, tok, cases, None, args.spans, args.max_new_tokens, args.group_rule
            )
            out["runs"][cond] = rows
            out["summary"][cond] = summarise(rows)
            print("   ", out["summary"][cond], flush=True)

        pol = SteerPolicy(
            mode="binary", gamma_plus=args.gamma_plus, gamma_minus=args.gamma_minus
        )
        for cond in ["conflict", "aligned"]:
            name = f"vsteer_{cond}"
            print(f"[{name}]", flush=True)
            rows = run_condition(
                model, tok, build(cond, variants=args.variants), pol, args.spans, args.max_new_tokens,
                args.group_rule,
            )
            out["runs"][name] = rows
            out["summary"][name] = summarise(rows)
            print("   ", out["summary"][name], flush=True)

    elif args.suite == "sweep":
        cases = build("conflict", variants=args.variants)
        for gm in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
            for gp in [0.0, 2.5]:
                name = f"gm{gm}_gp{gp}"
                pol = SteerPolicy(mode="binary", gamma_plus=gp, gamma_minus=gm)
                print(f"[{name}]", flush=True)
                rows = run_condition(
                    model, tok, cases, pol, args.spans, args.max_new_tokens,
                    args.group_rule,
                )
                out["runs"][name] = rows
                out["summary"][name] = summarise(rows) | {"gamma_minus": gm,
                                                          "gamma_plus": gp}
                print("   ", out["summary"][name], flush=True)

    elif args.suite == "age":
        # How much does history influence grow/shrink with its age, and does
        # the epoch_decay policy track it?
        for back in [1, 2, 3]:
            cases = build("conflict", epochs_back=back, variants=args.variants)
            for tag, pol in [
                ("none", None),
                ("binary", SteerPolicy(mode="binary")),
                ("epoch_decay", SteerPolicy(mode="epoch_decay", decay=0.5)),
            ]:
                name = f"back{back}_{tag}"
                print(f"[{name}]", flush=True)
                rows = run_condition(
                    model, tok, cases, pol, args.spans, args.max_new_tokens,
                    args.group_rule,
                )
                out["runs"][name] = rows
                out["summary"][name] = summarise(rows) | {"epochs_back": back,
                                                          "policy": tag}
                print("   ", out["summary"][name], flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    for k, v in out["summary"].items():
        print(f"  {k:24s} system={v['system']:.2f} stale={v['stale']:.2f}")


if __name__ == "__main__":
    main()

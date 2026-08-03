"""Send a case to a running brainscope and watch it, instead of only scoring it.

The eval in this repo runs against a local model with torch, because it needs
the KV cache and it runs hundreds of generations. That is the right tool for
counting, and the wrong one for looking: it tells you 49.3% and nothing about
what the model did.

brainscope is the other half. Same method, served behind an OpenAI-compatible
endpoint with the internals on screen. This module is a plain HTTP client for
it — standard library only, no dependency on brainscope, no import. Start one:

    pip install brainscope
    brainscope --model tiny

then throw any StaleSet case at it and open the hierarchy tab:

    python -m oldnews.live --list
    python -m oldnews.live --family prefix --gamma-plus 8
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8010"


def _post(url: str, path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path, json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, path: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def send(case, url: str = DEFAULT_URL, gamma_plus: float = 2.5,
         gamma_minus: float = 0.75, max_tokens: int = 64,
         steer: bool = True) -> dict:
    """Run one case through brainscope. Returns the answer and the report.

    The `stale` indices are message positions, which is what brainscope's
    /hierarchy takes — it locates the spans itself in the rendered template, so
    the two implementations agree on the spans without sharing any code.
    """
    messages = [{"role": m.role, "content": m.content} for m in case.messages]
    stale = [i for i, m in enumerate(case.messages)
             if m.epoch < case.current_epoch and m.role != "system"]

    body = {"messages": messages, "max_tokens": max_tokens}
    if steer and stale:
        body["hierarchy"] = {"stale": stale, "gamma_plus": gamma_plus,
                             "gamma_minus": gamma_minus}

    out = _post(url, "/v1/chat/completions", body)
    answer = out["choices"][0]["message"]["content"]
    report = (_get(url, "/hierarchy") or {}).get("last") or {}
    return {"answer": answer, "stale": stale, "report": report,
            "verdict": case.verdict(answer)}


def main():
    from .evals.staleset import FAMILIES, build

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"a running brainscope (default {DEFAULT_URL})")
    ap.add_argument("--family", default="prefix",
                    help="which constraint family; --list to see them")
    ap.add_argument("--query", type=int, default=0, help="which question")
    ap.add_argument("--gamma-plus", type=float, default=2.5)
    ap.add_argument("--gamma-minus", type=float, default=0.75)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for f in FAMILIES:
            print(f"  {f.key:8s} {f.system}")
        return

    fam = next((f for f in FAMILIES if f.key == args.family), None)
    if fam is None:
        raise SystemExit(f"unknown family {args.family!r}; try --list")
    cases = build("conflict", families=[fam], variants=1)
    case = cases[args.query % len(cases)]

    print(f"system : {case.system_needle}")
    print(f"stale  : {case.stale_needle}")
    print(f"asks   : {case.query}\n")

    try:
        off = send(case, args.url, max_tokens=args.max_tokens, steer=False)
        on = send(case, args.url, args.gamma_plus, args.gamma_minus,
                  args.max_tokens, steer=True)
    except urllib.error.URLError as e:
        raise SystemExit(f"no brainscope at {args.url} ({e.reason}). "
                         "start one with:  brainscope --model tiny")

    print(f"without  [{off['verdict']:7s}] {off['answer'].strip()[:200]}")
    print(f"steered  [{on['verdict']:7s}] {on['answer'].strip()[:200]}")
    r = on["report"]
    if r.get("skipped"):
        print(f"\nnot applied: {r['skipped']}")
    elif r:
        print(f"\n{r['heads_edited']}/{r['heads_total']} head groups rescaled · "
              f"{r['boost_tokens']} tokens boosted · {r['suppress_tokens']} demoted")
    print(f"\nopen {args.url} and pick the hierarchy tab to see where it looked")


if __name__ == "__main__":
    main()

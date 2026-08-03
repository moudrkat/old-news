"""Local UI: edit a transcript, mark what is stale, watch the hierarchy move.

    python -m oldnews.ui.app --model tiny
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..model import load
from ..policy import Priority, SteerPolicy, token_multipliers
from ..transcript import Msg, render
from ..vsteer import generate, steer

app = FastAPI(title="old-news")
STATE: dict = {}
STATIC = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
async def meta():
    cfg = STATE["model"].config
    return {
        "model": STATE["name"],
        "layers": cfg.num_hidden_layers,
        "q_heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads,
        "priorities": {p.name: int(p) for p in Priority},
    }


@app.post("/api/run")
async def run(body: dict):
    model, tok = STATE["model"], STATE["tok"]
    msgs = [
        Msg(
            m["role"],
            m["content"],
            epoch=int(m.get("epoch", 0)),
            pinned=bool(m.get("pinned", False)),
        )
        for m in body["messages"]
    ]
    current_epoch = body.get("current_epoch")
    current_epoch = int(current_epoch) if current_epoch is not None else None
    max_new = int(body.get("max_new_tokens", 64))

    pol = SteerPolicy(
        mode=body.get("mode", "binary"),
        gamma_plus=float(body.get("gamma_plus", 2.5)),
        gamma_minus=float(body.get("gamma_minus", 0.75)),
        decay=float(body.get("decay", 0.5)),
        eps=float(body.get("eps", 0.0)),
    )
    if pol.mode == "ladder" and not pol.ladder:
        pol.ladder = pol.default_ladder()

    r = render(tok, msgs, current_epoch=current_epoch)
    group_rule = body.get("group_rule", "mean")

    with torch.no_grad():
        base_text, _ = generate(model, tok, r, policy=None, max_new_tokens=max_new)
        steer_text, rep = generate(
            model, tok, r, policy=pol, max_new_tokens=max_new,
            current_epoch=current_epoch, group_rule=group_rule,
        )
        _, dry, _ = steer(model, tok, r, pol, current_epoch, group_rule,
                          dry_run=True)

    mult = token_multipliers(r, pol, current_epoch)
    per_msg = []
    for m, lv, (a, b) in zip(msgs, r.msg_levels, r.msg_spans):
        per_msg.append(
            {
                "role": m.role,
                "epoch": m.epoch,
                "pinned": m.pinned,
                "level": int(lv),
                "level_name": Priority(lv).name,
                "tokens": b - a,
                "multiplier": round(mult[a] if b > a else 1.0, 3),
            }
        )

    phi = {Priority(k).name: v.sum().item() for k, v in dry.phi.items()}
    return JSONResponse(
        {
            "baseline": base_text,
            "steered": steer_text,
            "messages": per_msg,
            "n_tokens": r.n_tokens,
            "heads_edited": rep.n_heads_edited,
            "heads_total": rep.n_heads_total,
            "target_token": tok.decode([rep.target_token]),
            "baseline_top": rep.baseline_top,
            "steered_top": rep.steered_top,
            "delta": dry.delta.tolist(),
            "head_mask": dry.head_mask.int().tolist(),
            "n_rep": model.config.num_attention_heads // model.config.num_key_value_heads,
            "phi_total": phi,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    model, tok = load(args.model)
    STATE.update(model=model, tok=tok, name=args.model)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    import uvicorn

    print(f"\n  old-news UI -> http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

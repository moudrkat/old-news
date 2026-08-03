"""Will V-Steer work on this model? Check, don't guess.

V-Steer is not architecture-neutral. It reaches into four things a transformer
is not obliged to expose the same way:

  1. a flat list of decoder layers, each with `self_attn.o_proj`
  2. an o_proj that factors as [hidden, n_query_heads * head_dim]
  3. a KV cache with one addressable V tensor per layer, shaped
     [batch, n_kv_heads, seq, head_dim]
  4. a cache that can be cropped and re-extended

Qwen and Llama satisfy all four. Models with sliding-window attention, layer-
shared KV, or a nested multimodal wrapper may not, and the failure mode is
silent -- wrong attributions, not an exception. So this module asserts each
assumption explicitly.

    python -m oldnews.compat --model google/gemma-4-E4B-it --device cpu
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name:26s} {self.detail}"


def find_layers(model):
    """Locate the decoder stack across the wrappers people actually ship."""
    for path in (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
    ):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            return obj, ".".join(path)
    raise AttributeError("could not locate decoder layers")


def find_final_norm(model):
    for path in (("model", "norm"), ("model", "language_model", "norm"),
                 ("language_model", "model", "norm"), ("model", "final_layernorm")):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "weight"):
            return obj
    return None


def inspect(model, tokenizer) -> list[Check]:
    out: list[Check] = []
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)

    try:
        layers, path = find_layers(model)
        out.append(Check("decoder layers", True, f"{len(layers)} at model.{path}"))
    except AttributeError as e:
        return [Check("decoder layers", False, str(e))]

    H_q = getattr(text_cfg, "num_attention_heads", None)
    H_kv = getattr(text_cfg, "num_key_value_heads", None)
    hidden = getattr(text_cfg, "hidden_size", None)
    hd = getattr(text_cfg, "head_dim", None) or (hidden // H_q if hidden and H_q else None)
    out.append(Check("head config", bool(H_q and H_kv and hd),
                     f"{H_q} query / {H_kv} kv heads, head_dim {hd}"))

    attn = getattr(layers[0], "self_attn", None)
    o_proj = getattr(attn, "o_proj", None) if attn else None
    if o_proj is None:
        out.append(Check("o_proj", False, "layer.self_attn.o_proj missing"))
    else:
        w = o_proj.weight
        ok = w.shape[1] == H_q * hd
        out.append(Check("o_proj factorises", ok,
                         f"{tuple(w.shape)} vs expected [*, {H_q}*{hd}={H_q*hd}]"))

    out.append(Check("final norm", find_final_norm(model) is not None,
                     "found" if find_final_norm(model) else "not found (fold_final_norm must be off)"))
    out.append(Check("lm_head", hasattr(model, "lm_head"), "present" if hasattr(model, "lm_head") else "missing"))

    # --- the parts only a real forward pass can answer ---
    ids = tokenizer("The capital of France is", return_tensors="pt")
    ids = {k: v.to(next(model.parameters()).device) for k, v in ids.items()}
    # attribution always runs its one attention-bearing forward under eager,
    # so the check has to as well -- otherwise sdpa reports "no attentions"
    # and we blame the model for the harness.
    prev = getattr(model.config, "_attn_implementation", None)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass
    with torch.no_grad():
        res = model(**ids, use_cache=True, output_attentions=True)
    if prev and prev != "eager":
        try:
            model.set_attn_implementation(prev)
        except Exception:
            pass

    cache = res.past_key_values
    out.append(Check("cache type", hasattr(cache, "layers"),
                     type(cache).__name__))

    if hasattr(cache, "layers"):
        n_cache = len(cache.layers)
        out.append(Check("one V per layer", n_cache == len(layers),
                         f"{n_cache} cache layers vs {len(layers)} decoder layers"
                         + ("" if n_cache == len(layers) else "  <- layer-shared KV")))
        v = getattr(cache.layers[0], "values", None)
        if v is None:
            out.append(Check("values addressable", False, "cache.layers[0].values missing"))
        else:
            ok = v.ndim == 4 and v.shape[1] == H_kv and v.shape[-1] == hd
            out.append(Check("values shape", ok,
                             f"{tuple(v.shape)} vs expected [b, {H_kv}, T, {hd}]"))

        sliding = [i for i, l in enumerate(cache.layers)
                   if getattr(l, "is_sliding", False)]
        out.append(Check("no sliding window", not sliding,
                         "all layers full-attention" if not sliding
                         else f"{len(sliding)}/{n_cache} layers sliding "
                              "<- demoted spans may fall outside the window"))

        try:
            cache.crop(int(ids["input_ids"].shape[1]) - 1)
            out.append(Check("cache.crop", True, "supported"))
        except Exception as e:
            out.append(Check("cache.crop", False, f"{type(e).__name__}: {e}"))

    attns = getattr(res, "attentions", None)
    if not attns or attns[0] is None:
        out.append(Check("attention weights", False,
                         "output_attentions returned none (no eager path?)"))
    else:
        a = attns[0]
        out.append(Check("attention weights", a.shape[1] == H_q,
                         f"{tuple(a.shape)}, {a.shape[1]} heads"))
    return out


def main():
    import argparse

    from .model import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    model, tok = load(args.model, device=args.device)
    checks = inspect(model, tok)
    print()
    for c in checks:
        print(c)
    bad = [c for c in checks if not c.ok]
    print()
    print("VERDICT:", "usable as-is" if not bad
          else f"{len(bad)} assumption(s) violated -- see NOTES.md before trusting numbers")


if __name__ == "__main__":
    main()

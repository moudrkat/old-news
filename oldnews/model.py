"""Model loading. Small CPU-friendly defaults; the eval runs on a GPU box."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALIASES = {
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
    "small": "Qwen/Qwen2.5-1.5B-Instruct",
    "mid": "Qwen/Qwen3-4B-Instruct-2507",
    "smol": "HuggingFaceTB/SmolLM2-135M-Instruct",
    # the paper's own main model — here to check whether the near-neighbour
    # recall errors reproduce outside the Qwen tokenizer
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    # Six more families, so "we saw this on three models" becomes a claim about
    # architectures and tokenizers rather than about Qwen plus one control.
    # All already in the local cache; all fit a 16 GB card in bf16.
    "phi": "microsoft/Phi-3.5-mini-instruct",          # 3.8B
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",         # 7B, fully open data
    "granite": "ibm-granite/granite-4.1-8b",           # 8B
    "aya": "CohereLabs/aya-expanse-8b",                # 8B, multilingual
    "commandr": "CohereLabs/c4ai-command-r7b-12-2024",  # 7B
    "gemma": "google/gemma-4-E2B-it",                  # 2B
}


def load(name: str = "tiny", dtype=None, device: str | None = None):
    """Load a causal LM + tokenizer, with the attention backend V-Steer wants.

    Loaded on sdpa: the prefill stays on the fused fast path, and only the
    single-token attribution forward is temporarily switched to eager.
    """
    repo = ALIASES.get(name, name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tok

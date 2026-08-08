"""Share of wrong answers that are a near neighbour, per model, by rule.

The rule is the loose one from results/failure_atlas.md: a near neighbour is a
small edit distance to the nearest candidate span (<= 0.5 normalised) AND either
the gold's character shape survives (19:40 -> 19:00) or half the characters
survive from one end (4417-B -> 4417). The shape-only rule cannot see truncation,
and truncation is the commonest form -- that is why the loose rule exists and why
both numbers are reported side by side.

No GPU, no API. Reads the atlas files that are already in results/.

    python examples/near_rates.py results/atlas_*.json
"""
import glob
import json
import math
import sys

DIST = 0.5

NAMES = {
    "tiny": "Qwen2.5-0.5B", "small": "Qwen2.5-1.5B", "q3b": "Qwen2.5-3B",
    "q7b": "Qwen2.5-7B", "mid": "Qwen3-4B", "llama": "Llama-3.1-8B",
    "phi": "Phi-3.5-mini", "olmo": "OLMo-2-7B", "aya": "Aya-expanse-8B",
    "commandr": "Command-R7B",
}


def loose(rec):
    gold = rec["needles"][0]
    half = math.ceil(len(gold) / 2)
    dist = rec.get("near_distance")
    if dist is None or dist > DIST:
        return False
    return bool(rec.get("near_same_shape")) or \
        rec.get("near_prefix", 0) >= half or rec.get("near_suffix", 0) >= half


def strict(rec):
    return bool(rec.get("near_same_shape"))


def main(paths):
    # one file per model; the rescored file wins where both exist, the near_*
    # fields are identical either way but the rescored one is the canonical copy
    files = {}
    for path in sorted(paths):
        model = path.split("atlas_")[1].split(".")[0]
        if path.endswith(".rescored.json") or model not in files:
            files[model] = path

    rows = []
    for model, path in files.items():
        records = json.load(open(path))["records"]
        wrong = [r for r in records if not r["recalled"]]
        rows.append((
            NAMES.get(model, model), len(wrong),
            sum(loose(r) for r in wrong), sum(strict(r) for r in wrong),
        ))
    rows.sort(key=lambda r: -r[2] / max(r[1], 1))

    print(f"{'model':16s} {'wrong':>6s} {'loose':>7s} {'shape only':>11s}")
    for name, n, k_loose, k_strict in rows:
        print(f"{name:16s} {n:6d} {100 * k_loose / max(n, 1):6.0f} % "
              f"{100 * k_strict / max(n, 1):9.0f} %")


if __name__ == "__main__":
    args = sys.argv[1:] or glob.glob("results/atlas_*.json")
    main(args)

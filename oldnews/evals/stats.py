"""Statistics for StaleSet, with no scipy dependency.

Every condition runs the *same* (family, query, variant) cases, so the
interesting comparisons are paired. Unpaired error bars would both overstate
the uncertainty and miss the pairing, so the headline test here is McNemar's
exact test on discordant pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Proportion:
    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else 0.0

    def wilson(self, z: float = 1.959963985) -> tuple[float, float]:
        """Wilson score interval -- behaves at p near 0 or 1, unlike Wald."""
        n = self.n
        if n == 0:
            return (0.0, 0.0)
        p = self.p
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, centre - half), min(1.0, centre + half))

    def as_dict(self) -> dict:
        lo, hi = self.wilson()
        return {"k": self.k, "n": self.n, "p": self.p, "ci_lo": lo, "ci_hi": hi}


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binom(n, p), exact via math.comb."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired binary outcomes.

    ``a`` and ``b`` are the same cases under two conditions. Only the
    discordant pairs carry information: b01 = a wrong / b right,
    b10 = a right / b wrong.
    """
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length outcomes")
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p_value": 1.0,
                "delta": 0.0}
    k = max(b01, b10)
    p = min(1.0, 2 * _binom_sf(k, n_disc, 0.5))
    return {
        "b01": b01,
        "b10": b10,
        "n_discordant": n_disc,
        "p_value": p,
        "delta": (sum(b) - sum(a)) / len(a),
    }


def paired_bootstrap(
    a: list[bool], b: list[bool], iters: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the paired difference in rates (b − a)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    n = len(A)
    if n == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, n, size=(iters, n))
    diffs = (B[idx] - A[idx]).mean(axis=1)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def case_key(row: dict) -> tuple:
    return (row["family"], row["query"], row.get("variant", 0))


def paired_outcomes(rows_a: list[dict], rows_b: list[dict], field="verdict",
                    value="system") -> tuple[list[bool], list[bool]]:
    """Align two runs on case identity and return matched boolean outcomes."""
    ia = {case_key(r): r for r in rows_a}
    ib = {case_key(r): r for r in rows_b}
    keys = [k for k in ia if k in ib]
    return (
        [ia[k][field] == value for k in keys],
        [ib[k][field] == value for k in keys],
    )


def confusion(before: list[bool], after: list[bool]) -> dict:
    """Treat the intervention as a repair operation and score it as one.

    Paired per case, where the label is "does this answer follow the current
    system prompt":

        TP  was broken, steering fixed it          (repair)
        FN  was broken, steering left it broken    (miss)
        FP  was fine,   steering broke it          (collateral damage)
        TN  was fine,   steering left it fine      (preserved)

    recall      = TP/(TP+FN)  -- of the failures, how many did it fix
    precision   = TP/(TP+FP)  -- of the cases it touched, how many improved
    specificity = TN/(TN+FP)  -- do-no-harm on already-correct cases
    """
    tp = sum((not x) and y for x, y in zip(before, after))
    fn = sum((not x) and (not y) for x, y in zip(before, after))
    fp = sum(x and (not y) for x, y in zip(before, after))
    tn = sum(x and y for x, y in zip(before, after))

    def rate(k, n):
        return k / n if n else None

    prec = rate(tp, tp + fp)
    rec = rate(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    out = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": rec, "precision": prec, "specificity": rate(tn, tn + fp),
        "f1": f1,
    }
    for name, (k, n) in {
        "recall": (tp, tp + fn),
        "precision": (tp, tp + fp),
        "specificity": (tn, tn + fp),
    }.items():
        out[f"{name}_ci"] = list(Proportion(k, n).wilson()) if n else None
    return out


def compare(runs: dict, a_key: str, b_key: str) -> dict | None:
    """Full paired comparison of two conditions, ready for the results JSON."""
    if a_key not in runs or b_key not in runs:
        return None
    a, b = paired_outcomes(runs[a_key], runs[b_key])
    if not a:
        return None
    out = {"a": a_key, "b": b_key, "n_paired": len(a),
           "rate_a": sum(a) / len(a), "rate_b": sum(b) / len(b)}
    out |= mcnemar(a, b)
    out |= confusion(a, b)
    try:
        lo, hi = paired_bootstrap(a, b)
        out["delta_ci"] = [lo, hi]
    except ImportError:
        pass
    return out


def summarise(rows: list[dict]) -> dict:
    """Rates with Wilson intervals."""
    n = len(rows)
    out = {"n": n}
    for name in ("system", "stale", "neither"):
        k = sum(r["verdict"] == name for r in rows)
        out[name] = k / n if n else 0.0
        out[f"{name}_ci"] = list(Proportion(k, n).wilson())
    k = sum(r["collapsed"] for r in rows)
    out["collapse"] = k / n if n else 0.0
    out["collapse_ci"] = list(Proportion(k, n).wilson())
    return out

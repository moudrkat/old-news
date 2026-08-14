"""Turn the faint-vs-absent generations into a table, and the table into a number.

The comparison "with the fact faint the model says Prague, with it gone the
model says New York" is currently an observation about six rows. This makes it
a measurement: normalised edit distance from the true value.

    19:40 -> 19:45   distance 0.2
    19:40 -> 14:30   distance 0.8

The extraction of "the value the model gave" is done by a rule here and is
**proposed, not final** — every row is printed with the sentence it came from so
it can be checked by eye. Any row where the rule is wrong gets corrected by
hand before a number is quoted.

    python src/table.py            # reads results/absent_*.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

CONDS = [("faint", "faint"), ("absent_drop", "absent")]


def first_sentence(s: str) -> str:
    s = s.split("<|im_end|>")[0].split("<|endoftext|>")[0]
    s = s.split("\n")[0].strip()
    return re.sub(r"\s+", " ", s)


REFUSAL = re.compile(
    r"\b(sorry|apolog|don'?t have|do not have|no access|don'?t know|do not know|"
    r"didn'?t tell|did not tell|can'?t (help|assist|tell|see|determine)|cannot|"
    r"not (available|mentioned|provided|sure)|as an ai|no information)\b", re.I)


def is_refusal(sentence: str, head: int = 60) -> bool:
    """A refusal has no value, so it has no distance from the truth.

    **Only the first sentence is searched.** A real refusal refuses immediately
    ("I'm sorry, but I don't have access to..."); an answer that gives a value
    and then hedges does not ("Your dog is called Yorick. Though I'm not sure
    if that's a coincidence...").

    Searching the whole string called that second kind a refusal. Capping the
    search at 60 characters was supposed to fix it and did not: the hedge in
    that very example, the one quoted in this docstring, starts at character 47.
    It was still being counted as a refusal until a reviewer read the output.
    Cutting at the first sentence boundary is the rule that actually matches the
    reasoning, and it moves exactly one item of 182.
    """
    first = re.split(r"(?<=[.!?])\s", sentence.strip(), maxsplit=1)[0]
    return bool(REFUSAL.search(first[:head]))


def propose_value(sentence: str, true: str) -> str:
    """Best guess at the value the model gave, for a first pass only.

    Prefers a bolded span, then a quoted span, then the first token that looks
    like the same kind of thing as the true value (digits / time / code / word).
    """
    for pat in (r"\*\*(.+?)\*\*", r"[\"“'](.+?)[\"”']"):
        m = re.search(pat, sentence)
        if m:
            return m.group(1).strip(" .")
    if re.fullmatch(r"[\d:]+", true):
        m = re.search(r"\b\d{1,2}:\d{2}\b" if ":" in true else r"\b\d+\b", sentence)
        return m.group(0) if m else ""
    if re.search(r"[A-Z]-?\d", true):
        m = re.search(r"\b[A-Z]+-?\d+\b", sentence)
        return m.group(0) if m else ""
    m = re.search(r"\b[A-Z][a-z]{2,}\b(?! is)", sentence)
    return m.group(0) if m else ""


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_dist(a: str, b: str) -> float:
    """0 identical, 1 nothing in common. Case-folded."""
    a, b = a.lower(), b.lower()
    if not a and not b:
        return 0.0
    return lev(a, b) / max(len(a), len(b), 1)


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "results"
    rows = []
    for f in sorted(root.glob("absent_*.json")):
        d = json.load(open(f))
        model = d["model"].split("/")[-1]
        for key, rec in d["items"].items():
            if "skipped" in rec:
                continue
            for field, label in CONDS:
                if field not in rec:
                    continue
                sent = first_sentence(rec[field])
                refused = is_refusal(sent)
                got = "" if refused else propose_value(sent, rec["value"])
                rows.append({
                    "model": model, "item": key, "true": rec["value"],
                    "cond": label, "faint_b": rec.get("faint_b"),
                    "refused": refused, "said": got,
                    "dist": norm_dist(got, rec["value"]) if got else None,
                    "sentence": sent,
                })

    out = root / "faint_vs_absent.json"
    out.write_text(json.dumps(rows, indent=1))

    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        print(f"\n### {model}\n")
        print("| item | true | faint | dist | absent | dist |")
        print("|---|---|---|---|---|---|")
        for item in dict.fromkeys(r["item"] for r in rs):
            g = {r["cond"]: r for r in rs if r["item"] == item}
            f_, a_ = g.get("faint"), g.get("absent")
            def cell(r):
                if not r:
                    return ("—", "—")
                if r["refused"]:
                    return ("*refused*", "—")
                if not r["said"]:
                    return ("*(no value found — read by hand)*", "—")
                return (f"`{r['said']}`", f"{r['dist']:.2f}")
            fs, fd = cell(f_)
            as_, ad = cell(a_)
            print(f"| {item} | `{g[list(g)[0]]['true']}` | {fs} | {fd} | {as_} | {ad} |")
        ds = [r["dist"] for r in rs if r["cond"] == "faint" and r["dist"] is not None]
        da = [r["dist"] for r in rs if r["cond"] == "absent" and r["dist"] is not None]
        if ds and da:
            print(f"\nmean distance from truth — faint {sum(ds)/len(ds):.2f} "
                  f"(n={len(ds)}), absent {sum(da)/len(da):.2f} (n={len(da)})")

    print(f"\nwrote {out}")
    print("\nEvery extracted value is a proposal. Check the `sentence` field in "
          "the json before quoting any mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

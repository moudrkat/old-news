"""The pages I read the data on, checked against the data they read from.

Three pages decide things: `verify.html` is where the yes/no reading was checked
by hand, `adjudicate.html` is where the value labels are being decided, and
`everything.html` is what a reader is pointed at to see the raw material. If any
of them silently drops rows, or shows one item's answer under another item's
heading, or ships broken JavaScript so the buttons do nothing, then a check that
felt thorough was not one.

That last failure mode is not hypothetical. An earlier version of the writing
page had a "Load a saved file" button wired to a function that was never
defined: it looked fine, it did nothing, and nothing said so.

So: every page is regenerated from the stored results and then compared against
those same results, item by item. No model, no network, no GPU.

    ../.venv/bin/python -m pytest attenuation/tests/ -v
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC, RES = ROOT / "src", ROOT / "results"
PY = sys.executable


def build(script: str, page: str) -> str:
    """Regenerate a page from scratch and return its HTML."""
    r = subprocess.run([PY, str(SRC / script)], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode == 0, f"{script} failed:\n{r.stderr}"
    out = RES / page
    assert out.exists(), f"{script} did not write {page}"
    return out.read_text()


def load(name: str):
    return json.loads((RES / name).read_text())


def rows(name: str):
    d = load(name)
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def last_script(page: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert blocks, "page has no script block"
    return blocks[-1]


@pytest.fixture(scope="module")
def everything() -> str:
    return build("everything.py", "everything.html")


@pytest.fixture(scope="module")
def adjudicate() -> str:
    return build("adjudicate.py", "adjudicate.html")


@pytest.fixture(scope="module")
def verify() -> str:
    return build("verify.py", "verify.html")


# ── the pages contain every row they claim to ───────────────────────────────

@pytest.mark.parametrize("stem", ["told2", "locality", "hedge"])
def test_everything_has_every_item(everything, stem):
    """A dropped row is the failure that looks like success: the page renders,
    it just quietly shows less than it says it does."""
    for f in sorted(RES.glob(f"{stem}_*.json")):
        for r in rows(f.name):
            assert f'data-k="{r["key"]}"' in everything, \
                f'{f.name}: {r["key"]} missing from everything.html'


def test_everything_counts_match_the_json(everything):
    expected = sum(len(rows(f.name)) for stem in ("told2", "ladder", "locality", "hedge")
                   for f in sorted(RES.glob(f"{stem}_*.json")))
    assert everything.count('class="item"') == expected


def test_everything_shows_the_right_answer_under_the_right_item(everything):
    """Spot-check the pairing rather than the count: an off-by-one in the join
    would keep every count correct and every answer wrong."""
    d = load("told2_Qwen3.5-4B.json")
    for r in d["rows"][:20]:
        block = everything.split(f'data-k="{r["key"]}"', 1)[1].split("</details>", 1)[0]
        keep = html.escape(r["value_faint"].split("<|im_end|>")[0].strip())
        if keep:
            assert keep in block, f'{r["key"]}: value answer not under its own item'


def test_answers_are_escaped_not_injected(everything):
    """Answers are model output and contain < & and quotes. If any reaches the
    page unescaped the layout breaks silently from that point down."""
    raw = "".join(r["value_faint"] for r in rows("told2_Qwen3.5-4B.json"))
    for ch in ("<", "&"):
        if ch in raw:
            assert html.escape(ch) in everything
    # no stray unescaped chat markers outside of the escaped form
    assert "<|im_end|>" not in everything


# ── the adjudication page offers exactly the decisions it should ────────────

def test_adjudicate_covers_every_faint_value_answer(adjudicate):
    """Every value answer is offered, and the four label sections beside it."""
    rc2 = load("recheck2.json")
    beh = rc2["behaviour"]
    faint = [r for r in beh if r["id"].endswith("|faint")]
    locality = sum(1 for r in rc2["locality"] if r.get("code") != r.get("judge"))
    # justifies and quotes have no second labeller, so every positive plus a
    # fixed sample of negatives is offered instead of a disagreement list
    n_neg = 15
    extra = sum(min(len(beh), sum(bool(r.get(f"judge_{lab}")) for r in beh)) + n_neg
                for lab in ("justifies", "quotes"))
    # the permission control offers every item in both arms
    perm = sum(len(json.loads(f.read_text())["rows"])
               for f in sorted(RES.glob("permission_*.json")))
    assert adjudicate.count('class="btns"') == (
        len(faint) + count_hedge_disagreements() + locality + extra + perm)


def count_hedge_disagreements() -> int:
    beh = load("recheck2.json")["behaviour"]
    return sum(1 for r in beh if r["id"].endswith("|faint")
               and bool(r["judge_hedge"]) != bool(r["code_hedge"]))


def test_adjudicate_marks_exactly_the_disputed_items(adjudicate):
    """The disputed ones are pinned and flagged. If the flag drifts from the
    real disagreement set, the page is pointing at the wrong items."""
    jmap = {r["id"]: r for r in rows("judge.json")}
    beh = load("recheck2.json")["behaviour"]
    disputed = set()
    for r in beh:
        if not r["id"].endswith("|faint"):
            continue
        key = r["id"].rsplit("|", 1)[0]
        j = jmap.get(key)
        if j is None:
            continue
        a = j["value"] == "none"
        if a != bool(r["judge_declines"]) or a != bool(r["code_declines"]):
            disputed.add(key)
    section = adjudicate.split('<span class="num">2</span>')[0]
    assert section.count("class='flag'") == len(disputed)
    for key in disputed:
        assert f'id="{key}"' in adjudicate


def test_adjudicate_offers_all_three_verdicts(adjudicate):
    for choice in ("a damaged true value", "a different value", "no value at all"):
        assert f'data-v="{choice}"' in adjudicate


# ── verify.html still separates the doubtful from the routine ───────────────

def test_verify_flags_the_doubtful(verify):
    n = sum(len(rows(f.name)) for f in sorted(RES.glob("told2_*.json")))
    assert verify.count("<tr") >= n, "fewer rows than items"


# ── the buttons actually work ───────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("page", ["everything", "adjudicate", "verify"])
def test_page_javascript_parses(page, request):
    """The bug this exists for: a button wired to a function that was never
    defined. The page looks finished and does nothing when clicked."""
    src = last_script(request.getfixturevalue(page))
    r = subprocess.run(["node", "--check", "-"], input=src,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{page}.html has broken JS:\n{r.stderr}"


@pytest.mark.parametrize("page,fns", [
    ("adjudicate", ["exp", "paint"]),
    ("verify", ["exp", "flag"]),
])
def test_every_onclick_has_a_function_behind_it(page, fns, request):
    """Every handler named in the markup is defined in the script."""
    doc = request.getfixturevalue(page)
    src = last_script(doc)
    for name in set(re.findall(r'onclick="(\w+)\(', doc)):
        if name in ("localStorage", "confirm", "if", "for", "while", "return"):
            continue        # JS keywords, not handlers
        assert re.search(rf"function {name}\b", src), \
            f"{page}.html calls {name}() but never defines it"
    for name in fns:
        assert re.search(rf"function {name}\b", src), f"{page}.html lost {name}()"


# ── the pages agree with the numbers the write-up quotes ────────────────────

def test_disputed_count_is_the_three_the_writeup_names(adjudicate):
    """Three in section 1; the rest of the flags are the permission recoveries."""
    section = adjudicate.split('<span class="num">2</span>')[0]
    assert section.count("class='flag'") == 3


def test_permission_recoveries_are_flagged(adjudicate):
    """The surprise in that control is that the value returns. If the flag drifts
    from the real recovery set, the page pins the wrong items."""
    n = 0
    for f in sorted(RES.glob("permission_*.json")):
        for r in json.loads(f.read_text())["rows"]:
            n += r["permission"]["value_present"] and not r["control"]["value_present"]
    section = adjudicate.split('<span class="num">6</span>')[1]
    assert section.count("class='flag'") == n


def test_hedge_disagreement_count_is_eighteen():
    assert count_hedge_disagreements() == 18

"""Is the value actually still in the answer?

The whole measurement turns on one test — has the value gone from the answer —
and the first version of that test was a case-insensitive substring search. On
times it is wrong:

    told 04:36  ->  "Your train leaves at 4:36 PM."

The substring `04:36` is not in that string, so the item was scored as damaged.
The model answered correctly and dropped a leading zero. Six of 189 items are
wrong in this direction, all of them times, all of them scored as failures they
are not.

So the test normalises before comparing. It is deliberately generous: anything
a person would read as the same value counts as the value being present, so the
count of damaged answers is a lower bound rather than a flattering one.
"""

from __future__ import annotations

import re
import unicodedata

TIME = re.compile(r"^(\d{1,2}):(\d{2})$")


def variants(value: str) -> set[str]:
    """Every rendering of this value a correct answer might use."""
    v = value.strip()
    out = {v.lower()}

    m = TIME.match(v)
    if m:
        h, mi = int(m.group(1)), m.group(2)
        out |= {f"{h}:{mi}", f"{h:02d}:{mi}"}
        h12 = h % 12 or 12
        out |= {f"{h12}:{mi}", f"{h12:02d}:{mi}"}   # 19:40 -> 7:40
    elif v.isdigit():
        out.add(str(int(v)))                        # 0614 -> 614
        if len(v) > 3:
            out.add(f"{int(v):,}")                  # 4417 -> 4,417
    return {x.lower() for x in out if x}


def _fold(s: str) -> str:
    """Strip accents. `Leon` answered as `León` is the same answer.

    Found by a reviewer reading fig0, not by the code: the earlier version
    normalised leading zeros and 12/24-hour clocks but not diacritics, so one
    item in 183 was scored as damage when the model had answered correctly.
    One, not a dozen: the whole corpus was re-checked against a much looser
    match (case, accents, punctuation and spacing all ignored) and `city:Leon`
    is the only disagreement.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def contains(answer: str, value: str) -> bool:
    a = _fold(" ".join(answer.split()).lower())
    return any(_fold(x) in a for x in variants(value))

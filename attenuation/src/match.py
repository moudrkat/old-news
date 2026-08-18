"""Is the value actually still in the answer?

The whole measurement turns on one test — has the value gone from the answer —
and the first version of that test was a case-insensitive substring search. On
times it is wrong:

    told 08:03  ->  "Your train leaves at 8:03."

The substring `08:03` is not in that string, so the item was scored as damaged.
The model answered correctly and dropped a leading zero. Five items in the
corpus are rescued by normalising like this: four times, and `León` for `Leon`.

The example above uses a bare 8:03 on purpose. A stated half of the day is what
decides these, so "4:36 PM" against 04:36 is not rescued but rejected, twelve
hours out, and `contains` below is where that happens.

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
    """Is the value in the answer, in any rendering a person would accept?

    **The 12-hour forms are only accepted with the right half of the day.**
    Told `06:15`, an answer of "6:15 PM" is not that time, it is 18:15, and the
    first version of this function counted it as correct because `6:15` is a
    substring. Two items were removed from the sample as undamaged when the
    model had in fact moved the train by twelve hours. Found by giving the raw
    strings to a judge and reading the disagreements, not by looking at the code.
    """
    a = _fold(" ".join(answer.split()).lower())
    m = TIME.match(value.strip())
    for x in variants(value):
        i = a.find(_fold(x))
        if i < 0:
            continue
        if m:
            after = a[i + len(x): i + len(x) + 6]
            said_pm, said_am = "pm" in after, "am" in after
            is_pm = int(m.group(1)) >= 12
            if (said_pm and not is_pm) or (said_am and is_pm):
                continue                  # right digits, wrong half of the day
        return True
    return False

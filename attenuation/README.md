# attenuation

### → [The write-up](docs/WRITEUP.md)

Everything is in there: the claim, the method, every control, the limitations and
what I would do next. Same text as the Google Doc linked from the application.

**A model is told a fact. The fact's own tokens are made hard to read, while the
sentence around them stays legible. The model then answers with a wrong value
that looks perfectly plausible, and reports that it was told that fact.**

| | |
|---|---|
| says "yes, you told me" when the value has gone | **147 of 184** |
| says it when a readable sentence about **something else** is in that slot | **0 of 184** |

![The four conditions](fig/fig6_conditions.png)

Also in this repo:

- [`PREREGISTRATION.md`](PREREGISTRATION.md), the plan, the hypotheses with their
  numeric falsifiers, and a log of what went wrong and how it was caught.
- [`docs/OTHER-RESULTS.md`](docs/OTHER-RESULTS.md), everything that came out of
  the same runs and is not part of the argument.
- `results/everything.html`, every experiment and every item, as the conversation
  that was sent and the answer that came back.

## Reproducing it

The runs, in this order:

```bash
python src/told2.py       Qwen/Qwen3.5-4B  # the four conditions, the headline
python src/swap_biased.py Qwen/Qwen3.5-4B  # the fifth: swap with the bias on
python src/ladder.py      Qwen/Qwen3.5-4B  # one item at a time up the dose sweep
python src/locality.py    Qwen/Qwen3.5-4B  # control 1: same dose, one span over
python src/hedge.py       Qwen/Qwen3.5-4B  # control 2: every behaviour at b = 0
python src/permission.py  Qwen/Qwen3.5-4B  # control 3: "say so if you are unsure"
python src/drop_long.py   Qwen/Qwen3.5-4B --ntok 512   # drop, given room to answer
```

Then the scoring, which is where several of the numbers above come from:

```bash
../.venv/bin/python src/judge.py      # what the answer did with the value
../.venv/bin/python src/recheck.py    # the yes/no reading, and is the value gone
../.venv/bin/python src/recheck2.py   # the locality answers, and the behaviours
../.venv/bin/python src/recheck_permission.py   # the permission control
python src/quoted.py                  # every quoted number, in one place
python src/everything.py              # every conversation and answer, one page
python src/verify.py                  # the page for checking the yes/no by eye
python src/adjudicate.py              # the page for labelling the value split
```

`src/run.py`, `src/sweep*.py`, `src/told.py`, `src/absent.py` and `src/table.py`
are the earlier six-item pilot and its diagnostics. Nothing in the tables above
is computed from them, with one exception that is named where it is used: the
`b = 11` datapoint for `city:Brno`.

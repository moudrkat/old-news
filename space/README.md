---
title: Turn The Knob
emoji: 🎛️
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Drag a knob, watch a model lose a fact it was told
---

# Turn the knob

A static page. Drag a slider and watch a language model lose a fact that is
still sitting in its conversation, in a well-formed sentence, with no error and
no warning.

**Nothing here runs a model.** Every answer was generated once by
`examples/failure_atlas.py` in the [old-news](https://github.com/moudrkat/old-news)
repo and stored. This page is a lookup over that grid:

| | |
|---|---|
| models | 10, six families, up to 8B |
| questions | 6 |
| rules in conflict | 6 |
| system-prompt boost | 3 settings |
| suppression | 7 settings |
| **cells** | **7,560, complete, no holes** |

Greedy decoding, so it reproduces. The same constructed conversations on every
model. Every generation kept, including the boring ones.

## What the ticks mean

- **contains the value** — case-insensitive substring search for what the user
  actually said. This is why `BAGRITO` counts as containing `bagr`, and that is
  the point rather than a bug in the page.
- **current rule won** — the deterministic format check for the rule in force.
- **the third chip** — an LLM judge that had to reproduce a set of hand labels
  before it was allowed to score anything. Present for eight of the ten models.

## Rebuilding the data

From the root of the old-news checkout:

```sh
python space/build_data.py
```

It reads `results/judged_atlas_*.json` (falling back to `atlas_*.json` where no
judged copy exists), refuses to write if the grid has holes, and emits one file
per model plus `meta.json` into `space/data/`. One model file is fetched at a
time, so the page stays light on a phone.

## Deploying

The contents of this directory are the whole Space — `index.html`, `README.md`
and `data/`. No build step and no runtime.

## Caveats worth reading before you generalise

Small open models on constructed conflicts, not production traffic. It is a
stress test of one intervention, not a measurement of your assistant. The
write-up, the controls and every raw generation are in the repo.

MIT.

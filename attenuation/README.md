# attenuation

**When you turn a message down instead of deleting it, does the fact inside it
survive?**

Sometimes you can't delete. The sentence you want the model to stop obeying
sits in the same message as the order number you still need. So you turn it
down — and this directory asks what that costs.

The parent repo established the mechanism on ten models: the edit **attenuates
rather than overwrites**. Attention is untouched, so the model still looks
straight at the demoted span; what arrives is faint. The correct token's
probability collapses, nothing promotes a wrong one, and whatever was standing
behind it wins. The model does not say it doesn't know. It says `02` instead of
`302`, and `Bagel` instead of `Bagr`.

Three things this asks that the source paper does not:

- Is there a dose where the instruction stops being obeyed **and** the fact
  still comes out?
- Does the damage get better or worse as models get better? The existing
  numbers trend the wrong way — the newest model in the set buries the fact
  three orders of magnitude harder than the oldest.
- Does the mechanism hold at all on a current model?

**Nothing has been run yet.** The only thing here is the frozen plan:

- [`PREREGISTRATION.md`](PREREGISTRATION.md) — the question, four hypotheses
  with numeric falsifiers, definitions fixed in advance, controls, the scoring
  rule written before any generation is read, and the stop rules.

Frozen 2026-08-11. Changes go in the Deviations table at the bottom, dated.

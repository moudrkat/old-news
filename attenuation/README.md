# attenuation

**When the evidence for something a model was told goes quiet, does the model
know?**

Not deleted — just quieter. The model still has the fact in its context, and the
signal carrying it has been weakened. Does it abstain, recall correctly, or say
something confidently adjacent?

The parent repo established the answer on ten models, for one way of weakening
evidence: it does **not** notice. The correct token's probability collapses while
its rank stays in single digits, nothing promotes a wrong one, and whatever was
standing behind it comes out — fluent, formatted, confident. `302` becomes `02`.
`Bagr` becomes `Bagel`.

This directory asks whether that is a fact about **models** rather than about one
technique, and how it scales.

- Two independent ways of weakening the same evidence — quiet values, and less
  looking — so the finding does not depend on either.
- Is there a setting where the instruction stops being obeyed and the fact still
  survives?
- Does it get better or worse as models get better? The existing numbers trend
  the wrong way: the newest model in the set buries the fact three orders of
  magnitude deeper than the oldest.

**Nothing has been run yet.** The only thing here is the frozen plan:

- [`PREREGISTRATION.md`](PREREGISTRATION.md) — the question, five hypotheses with
  numeric falsifiers, definitions fixed in advance, controls, the scoring rule
  written before any generation is read, and the stop rules.

Frozen 2026-08-11. Changes go in the Deviations table at the bottom, dated.

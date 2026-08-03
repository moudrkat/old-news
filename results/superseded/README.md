# Superseded runs — do not read numbers out of these

Every file here was produced before two fixes, and the numbers in them are
wrong. They are kept only so the corrections in [NOTES.md](../../NOTES.md) can
be checked against what they replaced.

- **`group_rule="mean"`.** Query-head badness was averaged to the KV head. The
  paper's rule (App. A.2, Eq. 9) is a union: a KV head is steered if *any*
  query head in its group is bad. Averaging cost 17 points on the main
  benchmark.
- **stale degeneracy check.** Collapsed output was scored as complying with
  whichever rule its wreckage happened to match.

Current runs, both fixes applied, are in the parent directory: `main_final.json`,
`main7_any.json`, `recall_final.json`, `recall_llama.json`.

`recall_any.json` and `recall_recheck.json` are earlier passes of the same
Qwen recall sweep that `recall_final.json` replaced. `tiny_main.json` is a
0.5B smoke run, also on the old mean rule; the default result path in the
figure and report scripts now points at `main_final.json` instead.

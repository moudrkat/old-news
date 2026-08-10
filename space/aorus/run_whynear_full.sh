#!/usr/bin/env bash
# Fill in the readout grid the Space is missing.
#
# WHY THIS RUN EXISTS
# The "inside the model" panel reads results/whynear_all_<model>.json, which was
# swept at gamma_plus 4.0 and only three settings of gamma_minus. The page has
# seven notches and three boost settings, so the panel is dark on 18 of the 21
# combinations and the visitor mostly sees "no readout here". This run fills the
# whole grid, so the panel lights up wherever the knob is put.
#
# It also picks up two fields added to examples/why_near.py on 10. 8.:
# steered_token_p_steered and gold_rank_clean. Without the first one only one
# end of the winning token's journey is recorded.
#
# NO NEW SCIENCE. Same script, same method, same teacher-forced readout point
# (the shortest unsteered prefix that already contains the gold value). Only the
# sweep is wider.
#
# GPU ONLY -- this is an aorus job. Nothing here should run on the CPU box.
#
#   bash space/aorus/run_whynear_full.sh              # everything, resumable
#   bash space/aorus/run_whynear_full.sh llama mid    # just these models
#
# Resumable: a (model, gamma_plus) pair whose output file already exists is
# skipped, so a killed run picks up where it stopped. Delete the file to redo it.

set -u -o pipefail
cd "$(dirname "$0")/../.." || exit 1

# Same interpreter and cache the atlas runs used (see run_atlas.sh). why_near
# imports failure_atlas, so examples/ has to be on the path too.
PY=${PY:-$HOME/tmp/brainscope-test/.venv/bin/python}
export HF_HOME=${HF_HOME:-$HOME/projects/science/instruct-steer/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONPATH=.:examples

# The 8B models will not fit next to a brainscope server holding ~8 GB.
if pgrep -f "brainscope.server" >/dev/null 2>&1; then
  echo "NOTE: brainscope.server is running and holds ~8 GB."
  echo "      The 8B models need that back. Stop it first, or run the small ones."
fi

MODELS=${*:-tiny small q3b q7b mid llama phi olmo aya commandr}
GAMMA_PLUS="1.0 2.5 4.0"
GAMMA_MINUS="0.0,0.5,0.65,0.75,0.85,0.9,0.95"
OUTDIR=results/whynear_full
mkdir -p "$OUTDIR" logs

echo "models:      $MODELS"
echo "gamma_plus:  $GAMMA_PLUS"
echo "gamma_minus: $GAMMA_MINUS"
echo "families:    all (6)  x  facts (6)  = 36 readout points per cell"
echo

fail=0
for m in $MODELS; do
  for gp in $GAMMA_PLUS; do
    out="$OUTDIR/whynear_${m}_gp${gp}.json"
    log="logs/whynear_${m}_gp${gp}.log"
    if [ -s "$out" ]; then
      echo "skip  $m gp=$gp  (already have $out)"
      continue
    fi
    echo "run   $m gp=$gp  -> $out"
    if ! "$PY" -u examples/why_near.py \
        --model "$m" \
        --gamma-plus "$gp" \
        --gamma-minus "$GAMMA_MINUS" \
        --families all \
        --out "$out" >"$log" 2>&1; then
      echo "FAIL  $m gp=$gp  -- see $log"
      # Aya and Command-R have a 256k vocab and were the two that would not fit
      # on the card before the prefill stopped computing logits it never used.
      # If one of them dies here, it is the thing to look at first.
      tail -n 3 "$log"
      fail=$((fail + 1))
      rm -f "$out"
      continue
    fi
    n=$("$PY" -c 'import json,sys; print(sum(1 for r in json.load(open(sys.argv[1]))["rows"] if "gamma_minus" in r))' "$out")
    echo "      ok, $n readouts"
  done
done

echo
if [ "$fail" -gt 0 ]; then
  echo "$fail cell(s) failed. Rerun this script -- finished ones are skipped."
else
  echo "All cells present. Now merge and rebuild the Space data:"
  echo "  python space/aorus/merge_whynear_full.py"
  echo "  python space/build_data.py"
fi

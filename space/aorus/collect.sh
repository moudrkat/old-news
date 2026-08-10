#!/usr/bin/env bash
# Bring the aorus sweep home and rebuild the Space from it.
#
# Run this on the LAPTOP, not on aorus, once run_whynear_full.sh has finished:
#
#   bash space/aorus/collect.sh
#
# It merges on aorus, copies only the merged per-model files back, rebuilds the
# Space data and tells you what changed. It does not deploy -- look at the page
# first.

set -u -o pipefail
cd "$(dirname "$0")/../.." || exit 1

HOST=${HOST:-aorus}
REMOTE=${REMOTE:-old-news}

echo "== what aorus has =="
ssh "$HOST" "cd $REMOTE && ls results/whynear_full/*.json 2>/dev/null | wc -l" \
  | xargs -I{} echo "   {} of 30 cells"

# A cell missing here is not a crash to paper over: aya and command-r have a
# 256k vocab and need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on a
# 16 GB card, or they die on the logit_scale multiply. Rerun those two with it
# rather than merging a hole.
missing=$(ssh "$HOST" "cd $REMOTE && for m in tiny small q3b q7b mid llama phi olmo aya commandr; do for g in 1.0 2.5 4.0; do [ -s results/whynear_full/whynear_\${m}_gp\${g}.json ] || echo \$m/\$g; done; done")
if [ -n "$missing" ]; then
  echo
  echo "   MISSING:"
  echo "$missing" | sed 's/^/     /'
  echo "   Rerun on aorus first:"
  echo "     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
  echo "       bash space/aorus/run_whynear_full.sh <model>"
  echo
  read -r -p "   Merge anyway with those missing? [y/N] " yn
  [ "$yn" = "y" ] || exit 1
fi

echo
echo "== merging on aorus =="
ssh "$HOST" "cd $REMOTE && \$HOME/tmp/brainscope-test/.venv/bin/python space/aorus/merge_whynear_full.py"

echo
echo "== copying merged files back =="
rsync -az --info=name "$HOST:$REMOTE/results/whynear_grid_*.json" results/

echo
echo "== rebuilding the Space =="
python3 space/build_data.py

echo
echo "== what the panel can now show =="
python3 - <<'PY'
import glob, json, os
GM, GP = 7, 3
for path in sorted(glob.glob("space/data/*.json")):
    name = os.path.basename(path)
    if name == "meta.json":
        continue
    blob = json.load(open(path))
    inside = blob.get("inside", {})
    with_after = sum(1 for v in inside.values() if len(v) > 8 and v[8] is not None)
    print(f"  {name[:-5]:<10} {len(inside):4d} readouts, "
          f"{with_after:4d} with the winner's arrival "
          f"({'both curves' if with_after else 'one curve only'})")
PY

echo
echo "Look at it locally first:"
echo "  cd space && python3 -m http.server 8777"
echo "Then deploy:"
echo "  python3 space/aorus/deploy.py"

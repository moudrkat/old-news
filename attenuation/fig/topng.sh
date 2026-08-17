#!/usr/bin/env bash
# Render the figures to PNG for pasting into a document.
#
# Headless Chrome rather than a screenshot tool, so the output is deterministic
# and regenerates from the HTML — same figure, same pixels, no cropping by hand.
# 2x device scale so the text stays sharp when the doc scales it down.
#
# The window is deliberately taller than the figure and the surplus is trimmed
# afterwards. An earlier version used a fixed height per figure, which silently
# **cut the caption off fig0 and every axis label off fig1** as soon as the
# figures grew — a cropped render is worse than a large one, because nothing
# in the output says it is cropped.
#
#   ./fig/topng.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT=8899
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory . >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1

shot () {  # file  width
  google-chrome --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --window-size="$2,3000" \
    --screenshot="$1.png" "http://127.0.0.1:$PORT/$1.html?theme=light" \
    >/dev/null 2>&1
  python3 - "$1.png" <<'PY'
import sys
from PIL import Image, ImageChops
p = sys.argv[1]
im = Image.open(p).convert("RGB")
bg = Image.new("RGB", im.size, im.getpixel((im.width - 2, im.height - 2)))
box = ImageChops.difference(im, bg).getbbox()
if box:
    l, t, r, b = box
    pad = 24
    im = im.crop((max(0, l - pad), max(0, t - pad),
                  min(im.width, r + pad), min(im.height, b + pad)))
    im.save(p)
print(f"  {p}  ({im.width}x{im.height})")
PY
}

echo "wrote:"
shot fig1_terms 960
shot fig2_items 960
shot fig3_examples 1290
shot fig4_manipulation 900
shot fig5_experiment 900
shot fig6_conditions 900
shot fig7_one_item 900
shot fig8_dose_grid 1400

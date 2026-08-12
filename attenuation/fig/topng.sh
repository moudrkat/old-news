#!/usr/bin/env bash
# Render the figures to PNG for pasting into a document.
#
# Headless Chrome rather than a screenshot tool, so the output is deterministic
# and regenerates from the HTML — same figure, same pixels, no cropping by hand.
# 2x device scale so the text stays sharp when the doc scales it down.
#
#   ./fig/topng.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT=8899
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory . >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1

shot () {  # file  width  height
  google-chrome --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --window-size="$2,$3" \
    --screenshot="$1.png" "http://127.0.0.1:$PORT/$1.html" >/dev/null 2>&1
  echo "  $(pwd)/$1.png  ($(identify -format '%wx%h' "$1.png" 2>/dev/null || echo "$(( $2 * 2 ))x$(( $3 * 2 ))"))"
}

echo "wrote:"
shot fig0 900 640
shot fig1 780 420

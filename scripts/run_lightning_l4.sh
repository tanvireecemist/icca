#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG="${1:-configs/research_l4.yaml}"

rfbench fetch --config "$CONFIG"
rfbench manifest --config "$CONFIG"
rfbench tune-batch --config "$CONFIG"

TUNED="outputs/tuning/$(basename "${CONFIG%.yaml}").tuned.yaml"
rfbench train --config "$TUNED"
rfbench test --config "$TUNED"


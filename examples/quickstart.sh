#!/usr/bin/env bash
# Offline tour of the harness: no GPU, no Lean and no Modal account required.
# The simulator backend runs the whole loop; the target check refuses simulated
# numbers, so the run ends with MAX_ROUNDS rather than SLA_MET.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra dev
uv run harness init
uv run harness registry

echo; echo "== simulated run"
uv run harness run --kernel elementwise_affine --backend sim --max-rounds 3
RUN=$(uv run harness runs -q -n 1)

echo; echo "== what the ledger recorded"
git -C "${HARNESS_LEDGER_DIR:-ledgers}" log --oneline -5
find "${HARNESS_LEDGER_DIR:-ledgers}/runs/$RUN" -maxdepth 2 -type f | sort

echo; echo "== machine-readable summary"
uv run harness report --run "$RUN"

echo; echo "Next: uv run harness serve   # then open http://127.0.0.1:8787"

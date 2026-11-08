#!/usr/bin/env bash
# Offline tour of the harness: no GPU, no Lean, no Modal account required.
#   1. deterministic cost-model backend exercises the whole loop
#   2. the SLA gate refuses simulated numbers, so the run reports `max_rounds`
#   3. the ledger and the API show exactly what a real run would record
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra dev
uv run harness init
uv run harness registry

echo
echo "== 1. simulated loop (elementwise_affine)"
uv run harness run --kernel elementwise_affine --backend sim --max-rounds 3

echo
echo "== 2. the ledger"
RUN=$(uv run python -c "
import sqlite3
c = sqlite3.connect('.harness/harness.db')
print(c.execute('SELECT id FROM runs ORDER BY created_at DESC LIMIT 1').fetchone()[0])")
echo "run: $RUN"
sed -n '1,20p' "ledgers/runs/$RUN/REPORT.md"
find "ledgers/runs/$RUN" -maxdepth 2 -type f | sort | head -20

echo
echo "== 3. machine-readable summary"
uv run harness report --run "$RUN"

echo
echo "== 4. proofs (needs Lean: https://elan.lean-lang.org)"
if command -v lake >/dev/null 2>&1; then
  V=$(uv run python -c "
import sqlite3
c = sqlite3.connect('.harness/harness.db')
print(c.execute('SELECT id FROM variants WHERE run_id=? ORDER BY round_n DESC LIMIT 1', ('$RUN',)).fetchone()[0])")
  uv run harness verify --variant "$V" | head -20
else
  echo "lake not found: skipping (a real run would report proof UNSUPPORTED)"
fi

echo
echo "== 5. UI"
echo "run: uv run harness serve --port 8787   # then open http://127.0.0.1:8787"

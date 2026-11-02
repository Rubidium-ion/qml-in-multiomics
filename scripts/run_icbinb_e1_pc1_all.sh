#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; OUT=benchmarks/results/icbinb_e1_pc1_runs; LOG=logs/icbinb_e1
mkdir -p "$OUT" "$LOG"
mapfile -t drugs < <("$PY" - <<'PY'
import json
for r in json.load(open('benchmarks/results/icbinb_d3_frozen_selection_20drug.json'))['selected']:print(r['drug'])
PY
)
# sequential checkpointed jobs: each drug result persists before next begins.
for d in "${drugs[@]}"; do
 [[ -s "$OUT/$d.json" ]] && continue
 echo "START $d $(date -Is)" | tee -a "$LOG/progress.log"
 "$PY" scripts/run_icbinb_e1_pc1_one.py --drug "$d" --out "$OUT/$d.json" >"$LOG/$d.log" 2>&1
 echo "DONE $d $(date -Is)" | tee -a "$LOG/progress.log"
done

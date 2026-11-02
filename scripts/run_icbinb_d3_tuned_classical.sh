#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; OUT="benchmarks/results/icbinb_d3_tuned_classical_runs"; LOG="logs/icbinb_d3_tuned_classical"; mkdir -p "$OUT" "$LOG"
mapfile -t DRUGS < <("$PY" - <<'PY'
import json
for r in json.load(open('benchmarks/results/icbinb_d3_frozen_selection_20drug.json'))['selected']: print(r['drug'])
PY
)
for seed in 42 314 2718; do for drug in "${DRUGS[@]}"; do
 out="$OUT/${drug}_seed${seed}.json"; [[ -s "$out" ]] && continue
 "$PY" scripts/run_icbinb_d3_tuned_classical.py --drug "$drug" --seed "$seed" --out "$out" >"$LOG/${drug}_seed${seed}.log" 2>&1
 echo "done $drug $seed"
done; done

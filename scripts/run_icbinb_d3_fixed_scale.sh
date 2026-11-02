#!/usr/bin/env bash
# Frozen D3 execution launcher. Selection manifest is committed before this runs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MANIFEST="benchmarks/results/icbinb_d3_frozen_selection_20drug.json"
OUTDIR="benchmarks/results/icbinb_d3_runs"
LOGDIR="logs/icbinb_d3"
mkdir -p "$OUTDIR" "$LOGDIR"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing required venv Python: $PY" >&2; exit 2; }
$PY - <<'PY'
import json, hashlib
p='benchmarks/results/icbinb_d3_frozen_selection_20drug.json'
d=json.load(open(p)); claimed=d.pop('sha256')
actual=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert actual == claimed, f'Frozen-manifest SHA mismatch: {actual} != {claimed}'
assert len(d['selected']) == 20
print('verified frozen manifest', claimed)
PY
mapfile -t DRUGS < <("$PY" - <<'PY'
import json
for row in json.load(open('benchmarks/results/icbinb_d3_frozen_selection_20drug.json'))['selected']:
 print(row['drug'])
PY
)
# 3 predeclared seeds, fixed repaired scale π/16, same StratifiedKFold split per
# seed across QK and each classical control. Output is one immutable JSON/run.
for seed in 42 314 2718; do
  for drug in "${DRUGS[@]}"; do
    out="$OUTDIR/${drug}_seed${seed}_pi16.json"
    log="$LOGDIR/${drug}_seed${seed}_pi16.log"
    if [[ -s "$out" ]]; then echo "SKIP existing $out"; continue; fi
    echo "START drug=$drug seed=$seed $(date -Is)"
    "$PY" scripts/run_true_qsvm.py --cohort "data/gdsc_${drug}_quantile" \
      --n-qubits 12 --reps 2 --entanglement linear --scale 0.19634954084936207 \
      --folds 5 --seed "$seed" --out "$out" >"$log" 2>&1
    echo "DONE drug=$drug seed=$seed $(date -Is)"
  done
done

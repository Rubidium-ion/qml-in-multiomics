#!/usr/bin/env bash
# run_negctrl_expansion.sh -- Protocol v2.1 negative-control expansion (authors, 2026-08-08)
#
# Builds cohorts for drugs not yet on disk, then runs the IF pre-screen on every
# negative-control drug. per-drug quantile:25 throughout, matching the PARP arm.
#
# NO cherry-picking: every drug that builds gets screened and recorded, including
# IF < 0.4. Failures are logged, not hidden.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
LOG="$REPO/docs/negctrl_expansion_$(date +%Y-%m-%d).log"

cd "$REPO"

BUILD_DRUGS=(Vinblastine Methotrexate Cytarabine Bortezomib Palbociclib Teniposide)

echo "=== cohort build phase $(date -Is) ===" | tee -a "$LOG"
for d in "${BUILD_DRUGS[@]}"; do
  low="$(echo "$d" | tr '[:upper:]' '[:lower:]')"
  out="data/gdsc_${low}_quantile"
  if [ -f "$out/X.npy" ]; then
    echo "[skip-build] $d already built" | tee -a "$LOG"
    continue
  fi
  echo "[build] $d -> $out" | tee -a "$LOG"
  "$PY" scripts/build_gdsc_cohort.py \
      --drug "$d" --binarize quantile --threshold 25 \
      --out "$out" >>"$LOG" 2>&1 \
    && echo "[build-ok] $d" | tee -a "$LOG" \
    || echo "[build-FAIL] $d" | tee -a "$LOG"
done

echo "=== prescreen phase $(date -Is) ===" | tee -a "$LOG"
SCREEN_DRUGS=(vinblastine methotrexate cytarabine bortezomib palbociclib teniposide \
              irinotecan oxaliplatin vinorelbine camptothecin bleomycin temozolomide)

for low in "${SCREEN_DRUGS[@]}"; do
  coh="data/gdsc_${low}_quantile"
  [ -f "$coh/X.npy" ] || { echo "[skip-screen] $low: no cohort" | tee -a "$LOG"; continue; }
  if [ -f "$coh/prescreen.json" ]; then
    echo "[skip-screen] $low already screened" | tee -a "$LOG"; continue
  fi
  echo "[screen] $low" | tee -a "$LOG"
  "$PY" scripts/run_parp_prescreen.py --cohort "$coh" >>"$LOG" 2>&1 \
    && echo "[screen-ok] $low" | tee -a "$LOG" \
    || echo "[screen-FAIL] $low" | tee -a "$LOG"
done

echo "=== done $(date -Is) ===" | tee -a "$LOG"

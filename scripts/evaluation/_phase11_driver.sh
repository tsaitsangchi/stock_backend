#!/bin/bash
# PHASE 11 driver — 9 tree-family multi_cycle validators (self-contained walk-forward, read-only).
# Honest logging: per-validator exit code + elapsed + extracted Cross-Cycle matrix line.
# Validators are read-only (SELECT only); --commit only persists JSON report to reports/.
set -u
cd /Users/hugo/project/stock_backend || exit 99
PY=./venv/bin/python
MASTER=/tmp/phase11.log
: > "$MASTER"

# name -> script file (base validator has no model suffix)
declare -a NAMES=(lgbm_base lightgbm xgboost xgboost_dedicated catboost catboost_dedicated random_forest extra_trees ensemble)
declare -a FILES=(multi_cycle_validation multi_cycle_lightgbm_validation multi_cycle_xgboost_validation multi_cycle_xgboost_dedicated_validation multi_cycle_catboost_validation multi_cycle_catboost_dedicated_validation multi_cycle_random_forest_validation multi_cycle_extra_trees_validation multi_cycle_ensemble_validation)

PASS=0; FAILN=0; FAILED=""
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  file="scripts/evaluation/${FILES[$i]}.py"
  echo "[$(date +%H:%M:%S)] START $name ($file)" | tee -a "$MASTER"
  t0=$(date +%s)
  "$PY" "$file" --commit > "/tmp/phase11_${name}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $name exit=$rc elapsed=$((t1-t0))s" | tee -a "$MASTER"
  if [ "$rc" -eq 0 ]; then
    PASS=$((PASS+1))
    # extract the cross-cycle matrix lines (horizon rows) for at-a-glance verdict
    grep -E "weekly|monthly|quarterly|annual" "/tmp/phase11_${name}.log" | grep -E "%" | tail -4 | while IFS= read -r ln; do
      echo "    [$name] $ln" | tee -a "$MASTER"
    done
  else
    FAILN=$((FAILN+1)); FAILED="$FAILED $name"
    echo "[$(date +%H:%M:%S)] FAIL $name (exit=$rc) — tail:" | tee -a "$MASTER"
    tail -5 "/tmp/phase11_${name}.log" | sed 's/^/      /' | tee -a "$MASTER"
  fi
done

echo "[$(date +%H:%M:%S)] PHASE11 COMPLETE | validator_pass=$PASS validator_fail=$FAILN failed=[$FAILED ]" | tee -a "$MASTER"
exit 0

#!/bin/bash
# Resume cross-model comparison sweep — 8 CPU-feasible models on 397 pan-historical universe.
# Sleep casualties first (tft/timesnet/ngboost), then ftransformer/gpt4ts/hist/stockformer/tabpfn.
# Honest: per-model smoke (where cheap) -> full; continue on failure; per-model 6h watchdog cap.
# Read-only validators (SELECT only); JSON persisted to reports/multi_cycle_<model>_<TS>.json.
set -u
cd /Users/hugo/project/stock_backend || exit 99

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
MASTER=/tmp/resume_sweep.log
SENT=/tmp/sweep_sentinel.txt
PY_FM=./venv_fm/bin/python      # torch foundation/deep models
PY_TREE=./venv/bin/python       # ngboost / tabpfn (sklearn-ish + own libs)
CAP=21600                       # 6h per-model hard cap (hang protection)
: > "$MASTER"
echo "SWEEP_START ts=$TS pid=$$" | tee -a "$MASTER"
echo "start ts=$TS $(date +%H:%M:%S)" > "$SENT"

upd() { echo "$(date +%H:%M:%S) $*" > "$SENT"; }

# capped_run LOGFILE CAP -- CMD...
capped_run() {
  local log="$1"; local cap="$2"; shift 3   # drop log, cap, and the literal "--"
  "$@" > "$log" 2>&1 &
  local pid=$!
  ( sleep "$cap"; kill -TERM "$pid" 2>/dev/null; sleep 15; kill -KILL "$pid" 2>/dev/null ) &
  local killer=$!
  wait "$pid"; local rc=$?
  kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
  return $rc
}

matrix() {  # extract horizon verdict rows for at-a-glance log
  grep -E "weekly|monthly|quarterly|annual" "$1" 2>/dev/null | grep -E "%|Eff" | tail -4 \
    | while IFS= read -r ln; do echo "      $ln" | tee -a "$MASTER"; done
}

PASS=0; FAILN=0; FAILED=""

run_model() {
  local name="$1" py="$2" smoke="$3"; shift 3
  local full_args=("$@")
  local out="reports/multi_cycle_${name}_${TS}.json"
  local flog="/tmp/sweep_${name}.log"
  upd "model=$name START"
  echo "[$(date +%H:%M:%S)] === $name START (venv=$py) ===" | tee -a "$MASTER"

  # optional cheap smoke (torch models with --smoke size limit)
  if [ "$smoke" = "yes" ]; then
    local slog="/tmp/sweep_${name}_smoke.log"
    echo "[$(date +%H:%M:%S)] $name smoke..." | tee -a "$MASTER"
    capped_run "$slog" 1800 -- "$py" "scripts/evaluation/multi_cycle_${name}_validation.py" --smoke --output "/tmp/${name}_smoke.json"
    local src=$?
    if [ "$src" -ne 0 ]; then
      FAILN=$((FAILN+1)); FAILED="$FAILED ${name}(smoke)"
      echo "[$(date +%H:%M:%S)] BLOCKED $name (smoke rc=$src) — tail:" | tee -a "$MASTER"
      tail -6 "$slog" | sed 's/^/        /' | tee -a "$MASTER"
      upd "model=$name BLOCKED-smoke rc=$src"
      return
    fi
    echo "[$(date +%H:%M:%S)] $name smoke OK -> full" | tee -a "$MASTER"
  fi

  local t0=$(date +%s)
  capped_run "$flog" "$CAP" -- "$py" "scripts/evaluation/multi_cycle_${name}_validation.py" "${full_args[@]}" --output "$out"
  local rc=$?
  local t1=$(date +%s)
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    PASS=$((PASS+1))
    echo "[$(date +%H:%M:%S)] DONE $name rc=0 elapsed=$((t1-t0))s -> $out" | tee -a "$MASTER"
    matrix "$flog"
    upd "model=$name DONE elapsed=$((t1-t0))s"
  else
    FAILN=$((FAILN+1)); FAILED="$FAILED ${name}(rc=$rc)"
    echo "[$(date +%H:%M:%S)] FAIL $name rc=$rc elapsed=$((t1-t0))s (out_exists=$([ -f "$out" ] && echo y || echo n)) — tail:" | tee -a "$MASTER"
    tail -8 "$flog" | sed 's/^/        /' | tee -a "$MASTER"
    upd "model=$name FAIL rc=$rc"
  fi
}

# ---- sleep casualties first ----
run_model tft         "$PY_FM"   no  --seed 5422
run_model timesnet    "$PY_FM"   yes --seed 5422
run_model ngboost     "$PY_TREE" no  --commit --seed 5422
# ---- remaining CPU-feasible ----
run_model ftransformer "$PY_FM"  no  --commit --seed 5422
run_model gpt4ts      "$PY_FM"   yes --seed 5422
run_model hist        "$PY_FM"   yes --seed 5422
run_model stockformer "$PY_FM"   yes --seed 5422
run_model tabpfn      "$PY_TREE" no  --commit --seed 5422

echo "[$(date +%H:%M:%S)] SWEEP_COMPLETE pass=$PASS fail=$FAILN failed=[$FAILED ]" | tee -a "$MASTER"
upd "SWEEP_COMPLETE pass=$PASS fail=$FAILN"
exit 0

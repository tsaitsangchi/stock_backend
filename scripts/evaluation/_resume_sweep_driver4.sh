#!/bin/bash
# Resume sweep v4 — ALL 12 leftovers on CPU, run-to-completion (user directive 2026-06-06:
# "12 支改直接用 CPU 跑" + "啟動 ngboost 先 + 每5分回報"). NO 2h cap; 12h/model SAFETY lock
# (lets the ~4-8h ones finish; guards true hangs). ngboost first. Honest: OOM/dep-missing -> marked.
# Known reality: gpt4ts OOMs (6-layer GPT-2 × 397 stocks); 4 foundation need gluonts/uni2ts (install pending).
set -u
cd /Users/hugo/project/stock_backend || exit 99

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
MASTER=/tmp/resume_sweep4.log
SENT=/tmp/sweep_sentinel4.txt
PY_FM=./venv_fm/bin/python
PY_TREE=./venv/bin/python
CAP=43200                        # 12h per-model SAFETY lock (not the 2h fail-fast)
: > "$MASTER"
echo "SWEEP4_START ts=$TS pid=$$ cap=${CAP}s(12h-safety) order=ngboost-first run-to-completion" | tee -a "$MASTER"
echo "start ts=$TS $(date +%H:%M:%S)" > "$SENT"

upd() { echo "$(date +%H:%M:%S) $*" > "$SENT"; }

capped_run() {
  local log="$1"; local cap="$2"; shift 3
  "$@" > "$log" 2>&1 &
  local pid=$!
  ( sleep "$cap"; kill -TERM "$pid" 2>/dev/null; sleep 15; kill -KILL "$pid" 2>/dev/null ) &
  local killer=$!
  wait "$pid"; local rc=$?
  kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
  return $rc
}

matrix() {
  grep -E "weekly|monthly|quarterly|annual" "$1" 2>/dev/null | grep -E "%|Eff" | tail -4 \
    | while IFS= read -r ln; do echo "      $ln" | tee -a "$MASTER"; done
}

PASS=0; FAILN=0; FAILED=""

run_model() {
  local name="$1" py="$2"; shift 2
  local full_args=("$@")
  local out="reports/multi_cycle_${name}_${TS}.json"
  local flog="/tmp/sweep4_${name}.log"
  upd "model=$name START"
  echo "[$(date +%H:%M:%S)] === $name START (venv=$py, safety_cap=${CAP}s) ===" | tee -a "$MASTER"
  local t0=$(date +%s)
  capped_run "$flog" "$CAP" -- "$py" "scripts/evaluation/multi_cycle_${name}_validation.py" "${full_args[@]}" --output "$out"
  local rc=$?
  local t1=$(date +%s)
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    PASS=$((PASS+1))
    echo "[$(date +%H:%M:%S)] DONE $name rc=0 elapsed=$((t1-t0))s -> $out" | tee -a "$MASTER"
    matrix "$flog"; upd "model=$name DONE elapsed=$((t1-t0))s"
  else
    FAILN=$((FAILN+1)); FAILED="$FAILED ${name}(rc=$rc)"
    local tag="FAIL"
    [ "$((t1-t0))" -ge "$CAP" ] && tag="TIMEOUT-12h"
    [ "$rc" -eq 137 ] && [ "$((t1-t0))" -lt "$CAP" ] && tag="OOM/killed"
    echo "[$(date +%H:%M:%S)] $tag $name rc=$rc elapsed=$((t1-t0))s (out_exists=$([ -f "$out" ] && echo y || echo n)) — tail:" | tee -a "$MASTER"
    tail -8 "$flog" | sed 's/^/        /' | tee -a "$MASTER"
    upd "model=$name $tag rc=$rc"
  fi
}

# ngboost first (explicit), then completable, then likely-fail (gpt4ts OOM / 4 foundation need deps)
run_model ngboost      "$PY_TREE" --commit --seed 5422
run_model tabpfn       "$PY_TREE" --commit --seed 5422
run_model ftransformer "$PY_FM"   --commit --seed 5422
run_model tft          "$PY_FM"   --seed 5422
run_model timesnet     "$PY_FM"   --seed 5422
run_model stockformer  "$PY_FM"   --seed 5422
run_model hist         "$PY_FM"   --seed 5422
run_model gpt4ts       "$PY_FM"   --seed 5422
run_model lagllama     "$PY_FM"   --seed 5422
run_model moirai       "$PY_FM"   --seed 5422
run_model timellm      "$PY_FM"   --seed 5422
run_model timemoe      "$PY_FM"   --seed 5422

echo "[$(date +%H:%M:%S)] SWEEP4_COMPLETE pass=$PASS fail=$FAILN failed=[$FAILED ]" | tee -a "$MASTER"
upd "SWEEP4_COMPLETE pass=$PASS fail=$FAILN"
exit 0

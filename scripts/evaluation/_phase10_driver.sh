#!/bin/bash
# PHASE 10 driver — base trainer (creates model_registry/model_training_run) + 9 tree-family trainers.
# Honest logging: per-trainer exit code + elapsed; aborts only if registry tables missing after base.
set -u
cd /Users/hugo/project/stock_backend || exit 99
PY=./venv/bin/python
MASTER=/tmp/phase10.log
: > "$MASTER"
PANELS="fs_20250915_feature_set_v0_5,fs_20251015_feature_set_v0_5,fs_20251115_feature_set_v0_5,fs_20251215_feature_set_v0_5,fs_20260115_feature_set_v0_5,fs_20260215_feature_set_v0_5,fs_20260315_feature_set_v0_5,fs_20260415_feature_set_v0_5"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$MASTER"
  local t0 t1 rc
  t0=$(date +%s)
  "$@" > "/tmp/phase10_${name}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $name exit=$rc elapsed=$((t1-t0))s" | tee -a "$MASTER"
  [ "$rc" -ne 0 ] && echo "[$(date +%H:%M:%S)] FAIL $name (exit=$rc)" | tee -a "$MASTER"
  return "$rc"
}

# --- base trainer first: ensure_tables() creates model_registry + model_training_run ---
run base "$PY" scripts/core/model_trainer.py --commit --walk-forward --panel-feature-sets "$PANELS" --model-family lgbm --label-horizon 30

# verify registry tables exist before launching family trainers
TBL=$("$PY" - <<'PYEOF'
import os, psycopg2
from dotenv import load_dotenv
load_dotenv("/Users/hugo/project/stock_backend/.env")
c=psycopg2.connect(host=os.environ["DB_HOST"],port=os.environ["DB_PORT"],dbname=os.environ["DB_NAME"],user=os.environ["DB_USER"],password=os.environ["DB_PASSWORD"])
cur=c.cursor()
ok=True
for t in ("model_registry","model_training_run"):
    cur.execute("SELECT to_regclass(%s)",(t,)); ok = ok and cur.fetchone()[0] is not None
print("YES" if ok else "NO")
c.close()
PYEOF
)
echo "[$(date +%H:%M:%S)] registry_tables_exist=$TBL" | tee -a "$MASTER"
if [ "$TBL" != "YES" ]; then
  echo "[$(date +%H:%M:%S)] PHASE10 ABORT: registry tables missing after base trainer" | tee -a "$MASTER"
  exit 1
fi

# --- 9 tree-family trainers (each uses its own DEFAULT_PANELS, --commit, default h30) ---
FAMILY="xgboost xgboost_dedicated lightgbm lgbm_v2 catboost catboost_dedicated random_forest extra_trees ensemble"
PASS=0; FAILN=0; FAILED=""
for t in $FAMILY; do
  if run "$t" "$PY" "scripts/core/model_trainer_${t}.py" --commit; then
    PASS=$((PASS+1))
  else
    FAILN=$((FAILN+1)); FAILED="$FAILED $t"
  fi
done

echo "[$(date +%H:%M:%S)] PHASE10 COMPLETE | family_pass=$PASS family_fail=$FAILN failed=[$FAILED ]" | tee -a "$MASTER"
exit 0

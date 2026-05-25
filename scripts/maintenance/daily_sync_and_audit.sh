#!/usr/bin/env bash
# 每日 sync + audit(launchd 觸發,18:30 收盤後 30 min)
# 對映 task #45;對映憲章 §6.8.7 第 (4) 條 + §6.8.8 audit suite
# Last updated: 2026-05-26

set -euo pipefail

PROJECT_ROOT="/Users/hugo/project/stock_backend"
cd "$PROJECT_ROOT"

# Logs(SHMM-compatible:寫 sentinel for cron-health monitoring)
LOG_DIR="$PROJECT_ROOT/logs/daily"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/daily_${TIMESTAMP}.log"
SENTINEL="/tmp/stock_backend_daily_last_run.txt"

# Notification helper(macOS)
notify() {
    local title="$1"
    local msg="$2"
    osascript -e "display notification \"$msg\" with title \"$title\"" 2>/dev/null || true
}

trap 'notify "stock_backend daily" "❌ FAIL at $(date +%H:%M)"; echo "❌ FAIL" >> "$LOG_FILE"; exit 1' ERR

echo "===== Daily sync + audit started $(date '+%Y-%m-%d %H:%M:%S') =====" | tee "$LOG_FILE"

# Step 1: sync core universe(增量)
echo "--- Step 1: sync core universe (FinMind) ---" | tee -a "$LOG_FILE"
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
    --universe core --workers 2 2>&1 | tee -a "$LOG_FILE"

# Step 2: sync FRED(macro)
echo "--- Step 2: sync FRED ---" | tee -a "$LOG_FILE"
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
    --source fred 2>&1 | tee -a "$LOG_FILE"

# Step 3: audit 1 — supply chain(快)
echo "--- Step 3: audit_supply_chain ---" | tee -a "$LOG_FILE"
./venv/bin/python scripts/maintenance/audit_supply_chain.py --include-logs 2>&1 | tee -a "$LOG_FILE"

# Step 4: audit 2 — schema compliance(快;BERNOULLI sample)
echo "--- Step 4: audit_api_schema_compliance ---" | tee -a "$LOG_FILE"
./venv/bin/python scripts/maintenance/audit_api_schema_compliance.py \
    --include-fred --db-sample-size 100000 2>&1 | tee -a "$LOG_FILE"

# Update SHMM sentinel
date '+%Y-%m-%d %H:%M:%S' > "$SENTINEL"

echo "===== Daily sync + audit COMPLETED $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG_FILE"
notify "stock_backend daily" "✅ Done at $(date +%H:%M)"

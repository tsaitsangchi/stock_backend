#!/usr/bin/env bash
# 週日凌晨 audit 3 全市場(~1.5h with sponsor;2353 stocks)
# 對映憲章 §6.8.8 + §3.2A.I parallel audit
# Last updated: 2026-05-26

set -euo pipefail

PROJECT_ROOT="/Users/hugo/project/stock_backend"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/weekly"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/audit3_${TIMESTAMP}.log"
SENTINEL="/tmp/stock_backend_weekly_audit3_last_run.txt"

notify() {
    osascript -e "display notification \"$2\" with title \"$1\"" 2>/dev/null || true
}

trap 'notify "stock_backend audit3" "❌ FAIL at $(date +%H:%M)"; echo "❌ FAIL" >> "$LOG_FILE"; exit 1' ERR

echo "===== Weekly audit 3 全市場 started $(date '+%Y-%m-%d %H:%M:%S') =====" | tee "$LOG_FILE"

# 全市場 source_availability audit(v0.7 sponsor 加速)
./venv/bin/python scripts/maintenance/audit_source_availability.py \
    --universe full --all --include-fred --strict \
    --workers 2 --throttle 5500 \
    --api-timeout 60 --api-retry 1 \
    --retry-backoff 30,300 \
    --special-full-market-reason "weekly automated audit 3 via launchd cron" \
    --progress-interval 100 2>&1 | tee -a "$LOG_FILE"

date '+%Y-%m-%d %H:%M:%S' > "$SENTINEL"

echo "===== Weekly audit 3 COMPLETED $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG_FILE"
notify "stock_backend audit3" "✅ Done at $(date +%H:%M)"

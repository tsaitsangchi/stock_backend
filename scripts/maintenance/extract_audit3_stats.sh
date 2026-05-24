#!/usr/bin/env bash
# 從 audit_source_availability log 抽取 Phase 11 報告需要的關鍵數字
# 用法:./extract_audit3_stats.sh <log_file>
# 例:./extract_audit3_stats.sh reports/rebuild_logs/item3_v6.1.0_recursive/audit3_source_availability_RUN3.log

set -euo pipefail

LOG_FILE="${1:?Usage: $0 <audit3_log_path>}"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: log file not found: $LOG_FILE" >&2
    exit 1
fi

echo "=== Audit 3 stats from: $LOG_FILE ==="
echo ""

# 起始行
echo "--- Start signature ---"
grep -m1 "^🔍 audit start" "$LOG_FILE" || echo "(no start signature)"
echo ""

# 最後一筆 progress(無論完成與否都拿得到)
echo "--- Latest progress entry ---"
grep "^🔍 progress" "$LOG_FILE" | tail -1 || echo "(no progress entries)"
echo ""

# 完成 summary(若有)
echo "--- Completion summary(若 audit 已收尾)---"
grep -E "(✅ audit completed|❌ audit failed|🏁 audit)" "$LOG_FILE" | tail -5 || echo "(audit not yet completed)"
echo ""

# Throttle 統計
echo "--- Throttle hits ---"
TOTAL_THROTTLE=$(grep -c "^⏸  audit throttle" "$LOG_FILE" || true)
echo "Total throttle sleep events: ${TOTAL_THROTTLE}"
grep "^⏸  audit throttle" "$LOG_FILE" | awk -F'sleep ' '{print $2}' | awk '{print $1}' | sed 's/s//' | awk '
    BEGIN {sum=0; max=0; cnt=0}
    {sum+=$1; if($1>max) max=$1; cnt++}
    END {if(cnt>0) printf "  total_sleep=%ds (%.1f min) | max=%ds | mean=%.1fs (over %d events)\n", sum, sum/60, max, sum/cnt, cnt}'
echo ""

# Retry 統計(若 v0.7 有 retry log)
echo "--- Retry events (§6.8.8-E.1) ---"
grep -E "(retry|backoff)" "$LOG_FILE" | head -10 || echo "(no retry events logged)"
RETRY_COUNT=$(grep -ciE "(retry|backoff)" "$LOG_FILE" || true)
echo "Total retry-related log lines: ${RETRY_COUNT}"
echo ""

# Error 樣本
echo "--- API_ERROR samples (first 5)---"
grep -E "(API_ERROR|ERROR)" "$LOG_FILE" | grep -v "^🔍" | head -5 || echo "(no errors)"
echo ""

# 從 progress line 算 elapsed % errors
echo "--- Final progress breakdown ---"
LAST_PROGRESS=$(grep "^🔍 progress" "$LOG_FILE" | tail -1)
if [[ -n "$LAST_PROGRESS" ]]; then
    echo "$LAST_PROGRESS" | sed 's/ | /\n  /g'
fi

echo ""
echo "=== End of extract ==="

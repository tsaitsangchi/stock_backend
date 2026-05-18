#!/usr/bin/env bash
set -u

cd /home/hugo/project/stock_backend

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    echo "process:"
    pgrep -af "scripts/ingestion/sovereign_sync_engine.py" || echo "NOT RUNNING"
    echo "log:"
    ls -lh logs/full_market_sync_20260518.log 2>/dev/null || true
    echo "tail:"
    tail -n 20 logs/full_market_sync_20260518.log 2>/dev/null || true
    echo
  } >> logs/full_market_sync_20260518_monitor.log 2>&1

  sleep 1800
done

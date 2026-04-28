#!/bin/bash
while pgrep -f "fetch_chip_data.py|fetch_fundamental_data.py|fetch_sponsor_chip_data.py" > /dev/null; do
    sleep 5
done
wall "========================================================="
wall "🚀 [系統通知] 87 支 config.py 重點股票資料已全數補齊！"
wall "🚀 [系統通知] 模型訓練 (train_evaluate.py) 即將在背景啟動！"
wall "========================================================="
nohup ./venv/bin/python scripts/train_evaluate.py --all > scripts/outputs/logs/train.log 2>&1 &

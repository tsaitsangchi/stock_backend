#!/bin/bash
# scripts/fetch_anchors_20yr.sh
# 執行 Tier 1 標的 (2330, 2317, 2454) 的二十年全量數據回溯
#
# [P2 修正] set -euo pipefail：任一指令失敗立即停止；引用未定義變數報錯；管道任一段失敗即傳遞
set -euo pipefail
trap 'echo "❌ 失敗於第 $LINENO 行（最後指令退出碼: $?）" >&2' ERR

VENV_PYTHON="/home/hugo/project/stock_backend/venv/bin/python3"
START_DATE="2001-01-01"
STOCKS=("2330" "2317" "2454")

echo "🌌 開始執行二十年全量數據大會師 (Start: $START_DATE)..."

for sid in "${STOCKS[@]}"; do
    echo "------------------------------------------"
    echo "🚀 正在補齊 $sid 的全維度數據..."
    
    echo "1. 技術面 (價量)..."
    $VENV_PYTHON scripts/fetch_technical_data.py --stock-id $sid --start $START_DATE
    
    echo "2. 法人籌碼..."
    $VENV_PYTHON scripts/fetch_chip_data.py --stock-id $sid --start $START_DATE
    
    echo "3. 基本面 (營收、損益)..."
    $VENV_PYTHON scripts/fetch_fundamental_data.py --stock-id $sid --start $START_DATE
    
    echo "4. 進階籌碼 (八大行庫、大戶)..."
    $VENV_PYTHON scripts/fetch_sponsor_chip_data.py --stock-id $sid --start $START_DATE
done

echo "------------------------------------------"
echo "🌐 補齊全域宏觀與國際連動數據..."

echo "5. 國際市場 (TSM, NVDA, AAPL, SOXX)..."
$VENV_PYTHON scripts/fetch_international_data.py --tickers TSM NVDA AAPL SOXX --start $START_DATE

echo "6. 宏觀經濟 (利率、匯率、債券)..."
$VENV_PYTHON scripts/fetch_macro_data.py --force --start $START_DATE

echo "7. 期權與衍生情緒..."
$VENV_PYTHON scripts/fetch_derivative_data.py --ids TX TFO CDF --start $START_DATE
$VENV_PYTHON scripts/fetch_derivative_sentiment_data.py --start $START_DATE

echo "✅ 二十年全量數據補齊任務已分發！"

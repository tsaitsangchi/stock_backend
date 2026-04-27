#!/bin/bash
# scripts/run_2330_20yr_full.sh
# 1. 補齊 2330 二十年的資料
# 2. 執行 2330 二十年的模型訓練

VENV_PYTHON="/home/hugo/project/stock_backend/venv/bin/python3"
START_DATE="2001-01-01"
SID="2330"
LOG_DIR="/home/hugo/project/stock_backend/scripts/outputs/logs"
mkdir -p $LOG_DIR

echo "🌌 [2330] 開始二十年資料大補帖 (Start: $START_DATE)..." | tee -a $LOG_DIR/run_2330_20yr.log

echo "------------------------------------------" | tee -a $LOG_DIR/run_2330_20yr.log
echo "🚀 正在補齊 $SID 的全維度資料..." | tee -a $LOG_DIR/run_2330_20yr.log

echo "1. 技術面 (價量)..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_technical_data.py --stock-id $SID --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "2. 法人籌碼..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_chip_data.py --stock-id $SID --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "3. 基本面 (營收、損益)..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_fundamental_data.py --stock-id $SID --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "4. 進階籌碼 (八大行庫、大戶)..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_sponsor_chip_data.py --stock-id $SID --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "5. 國際市場 (TSM, NVDA, AAPL, SOXX)..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_international_data.py --tickers TSM NVDA AAPL SOXX --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "6. 宏觀經濟 (利率、匯率、債券)..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_macro_data.py --force --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "7. 期權與衍生情緒..." | tee -a $LOG_DIR/run_2330_20yr.log
$VENV_PYTHON scripts/fetch_derivative_data.py --ids TX TFO CDF --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1
$VENV_PYTHON scripts/fetch_derivative_sentiment_data.py --start $START_DATE >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "------------------------------------------" | tee -a $LOG_DIR/run_2330_20yr.log
echo "✅ 資料補齊完成！即將開始二十年模型訓練與回測..." | tee -a $LOG_DIR/run_2330_20yr.log

# 執行模型訓練
# 使用 --no-tft 以加快速度，或者如果您需要 TFT 請移除該參數
$VENV_PYTHON scripts/train_evaluate.py --stock-id $SID --start $START_DATE --no-tft >> $LOG_DIR/run_2330_20yr.log 2>&1

echo "------------------------------------------" | tee -a $LOG_DIR/run_2330_20yr.log
echo "🏆 [2330] 二十年模型運算任務全部完成！" | tee -a $LOG_DIR/run_2330_20yr.log

#!/bin/bash
# scripts/fetch_2330_all.sh
# 抓取 2330 所需的所有資料

VENV_PYTHON="/home/hugo/project/stock_backend/venv/bin/python3"

echo "🚀 開始抓取 2330 相關資料..."

echo "1. 更新股票基本資訊..."
$VENV_PYTHON scripts/fetch_stock_info.py

echo "2. 抓取 2330 技術面資料..."
$VENV_PYTHON scripts/fetch_technical_data.py --stock-id 2330

echo "3. 抓取 2330 籌碼面資料..."
$VENV_PYTHON scripts/fetch_chip_data.py --stock-id 2330

echo "4. 抓取 2330 基本面資料..."
$VENV_PYTHON scripts/fetch_fundamental_data.py --stock-id 2330

echo "5. 抓取 2330 Sponsor 籌碼資料..."
$VENV_PYTHON scripts/fetch_sponsor_chip_data.py --stock-id 2330

echo "6. 抓取全市場總經與市值比重資料..."
$VENV_PYTHON scripts/fetch_macro_fundamental_data.py

echo "7. 抓取相關國際市場資料 (TSM, NVDA, AAPL, SOXX)..."
$VENV_PYTHON scripts/fetch_international_data.py --tickers TSM NVDA AAPL SOXX

echo "8. 抓取宏觀經濟資料..."
$VENV_PYTHON scripts/fetch_macro_data.py

echo "9. 抓取期權籌碼資料 (僅抓取關鍵標的: TX, TFO, CDF)..."
$VENV_PYTHON scripts/fetch_derivative_data.py --ids TX TFO CDF

echo "10. 抓取衍生情緒資料..."
$VENV_PYTHON scripts/fetch_derivative_sentiment_data.py

echo "✅ 2330 資料抓取完成！"

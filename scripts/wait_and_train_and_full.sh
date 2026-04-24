#!/bin/bash

# 1. 等待 87 支股票抓取完成
echo "Waiting for 87 stocks fetch to complete..."
while pgrep -f "fetch_chip_data.py|fetch_fundamental_data.py|fetch_sponsor_chip_data.py" > /dev/null; do
    sleep 10
done

wall "🚀 [系統通知] 87 支重點股票資料已補齊！"
wall "🚀 [系統通知] 即將啟動模型訓練，並接續啟動全市場資料抓取..."

# 2. 更新 Feature Store (特徵庫)
echo "Updating Feature Store..."
./venv/bin/python scripts/update_feature_store.py > scripts/outputs/logs/feature_store.log 2>&1
echo "Feature Store updated."

# 3. 啟動模型訓練 (背景)
nohup ./venv/bin/python scripts/train_evaluate.py --all > scripts/outputs/logs/train.log 2>&1 &
echo "Training started."

# 3. 還原程式碼為「全市場模式」
python3 -c '
def revert_file(filepath, old_str, new_str):
    with open(filepath, "r") as f:
        content = f.read()
    if old_str in content:
        with open(filepath, "w") as f:
            f.write(content.replace(old_str, new_str))
        print(f"Reverted {filepath}")

# For chip and fundamental
old_chip = """    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())"""
new_chip = """    with conn.cursor() as cur:
        cur.execute(
            "SELECT stock_id FROM stock_info WHERE type IN (\\'twse\\', \\'otc\\') ORDER BY stock_id"
        )
        return [row[0] for row in cur.fetchall()]"""

revert_file("scripts/fetch_chip_data.py", old_chip, new_chip)
revert_file("scripts/fetch_fundamental_data.py", old_chip, new_chip)

# For sponsor
old_sponsor = """    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())"""
new_sponsor = """    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT stock_id FROM stock_info ORDER BY stock_id")
        return [r[0] for r in cur.fetchall()]"""
revert_file("scripts/fetch_sponsor_chip_data.py", old_sponsor, new_sponsor)
'

# 4. 啟動全市場資料抓取 (背景，使用增量模式以免覆蓋剛抓好的 87 支)
# 這裡不加 --force，這樣會自動跳過已抓好的資料
nohup ./venv/bin/python scripts/fetch_chip_data.py > scripts/outputs/logs/chip_full.log 2>&1 &
nohup ./venv/bin/python scripts/fetch_sponsor_chip_data.py > scripts/outputs/logs/sponsor_full.log 2>&1 &
nohup ./venv/bin/python scripts/fetch_fundamental_data.py > scripts/outputs/logs/fundamental_full.log 2>&1 &

wall "🚀 [系統通知] 全市場資料補齊任務 (1500+ 支) 已在背景接續啟動！"

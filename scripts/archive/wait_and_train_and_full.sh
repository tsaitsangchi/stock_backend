#!/bin/bash

# 1. 啟動並行抓取任務 (若尚未啟動)
# 這裡假設外部已經啟動了 fetch_chip_data.py 等 87 支任務

wall "🚀 [系統通知] 已切換為「個股完工即重訓」模式！"

# 2. 啟動自動化訓練管理員
echo "Starting Auto Train Manager..."
./venv/bin/python scripts/auto_train_manager.py
echo "Auto Train Manager finished (all 87 stocks handled)."

wall "🚀 [系統通知] 87 支重點股票已全部重訓完成！即將開始全市場資料更新..."

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

# 4. 啟動全市場資料抓取 (背景)
nohup ./venv/bin/python scripts/fetch_chip_data.py > scripts/outputs/logs/chip_full.log 2>&1 &
nohup ./venv/bin/python scripts/fetch_sponsor_chip_data.py > scripts/outputs/logs/sponsor_full.log 2>&1 &
nohup ./venv/bin/python scripts/fetch_fundamental_data.py > scripts/outputs/logs/fundamental_full.log 2>&1 &

wall "🚀 [系統通知] 全市場資料補齊任務 (1500+ 支) 已在背景接續啟動！"

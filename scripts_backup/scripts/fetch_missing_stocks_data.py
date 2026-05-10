"""
fetch_missing_stocks_data.py — 自動補齊 config.py 中新增個股的歷史數據

修改摘要（第四輪審查）：
  [P0-SEC]  移除獨立的 DB_CONFIG 定義，改由 config.py 統一引入
  [P1 修正] 修正邏輯矛盾：原版找出缺失個股後執行全量更新（忽略篩選結果）
            → 修正為對每個缺失個股分別傳入 --stock-id SID --start 2010-01-01
            → 宏觀資料（fetch_macro_data.py）不接受 --stock-id，獨立執行一次
  [P1 保留] SCRIPTS_DIR = Path(__file__).parent（原版已正確，保留）
"""

import subprocess
import sys
import logging
import psycopg2

from pathlib import Path
from config import STOCK_CONFIGS, DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
SCRIPTS_DIR = Path(__file__).parent  # 無論從哪個目錄執行，路徑均正確

# 個股相關腳本（支援 --stock-id 參數）
PER_STOCK_SCRIPTS = [
    "fetch_technical_data.py",
    "fetch_fundamental_data.py",
    "fetch_chip_data.py",
    "fetch_derivative_data.py",
    "fetch_international_data.py",
]

# 宏觀資料腳本（不依個股區分，單獨執行一次）
MACRO_SCRIPTS = [
    "fetch_macro_data.py",
]


def run_script(script_name: str, args: list[str] = []) -> bool:
    """
    執行腳本，回傳是否成功。
    使用 SCRIPTS_DIR 確保絕對路徑，避免工作目錄不一致。
    """
    script_path = str(SCRIPTS_DIR / script_name)
    cmd = [VENV_PYTHON, script_path] + args
    logger.info(f"執行指令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"執行 {script_name} 失敗（退出碼 {e.returncode}）")
        return False
    except Exception as e:
        logger.error(f"執行 {script_name} 發生異常：{e}")
        return False


def get_missing_data_stocks() -> list[str]:
    """
    找出 STOCK_CONFIGS 中在 stock_price 資料少於 100 筆的個股。
    回傳缺少資料的 stock_id 列表。
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    missing = []
    for sid in STOCK_CONFIGS.keys():
        cur.execute("SELECT COUNT(*) FROM stock_price WHERE stock_id = %s", (sid,))
        if cur.fetchone()[0] < 100:
            missing.append(sid)
    conn.close()
    return missing


def main():
    # ── Step 1：更新股票基本資訊（必須最先執行，其他腳本依賴 stock_info）
    logger.info("=== Step 1：更新股票基本資訊 (stock_info) ===")
    run_script("fetch_stock_info.py")

    # ── Step 2：找出需要補資料的個股
    missing_stocks = get_missing_data_stocks()
    if not missing_stocks:
        logger.info("所有個股資料皆已充足，無需補齊。")
    else:
        logger.info(f"發現 {len(missing_stocks)} 支需要補資料的個股：{', '.join(missing_stocks)}")

        # ── Step 3：逐支個股執行補齊（傳入 --stock-id 與 --start，只抓該股）
        # [P1 修正] 原版在迴圈中只有 print，實際執行的是全量更新（語意矛盾）
        # 修正版：對每支缺失個股分別傳入 --stock-id SID --start 2010-01-01
        for sid in missing_stocks:
            logger.info(f"\n=== 補齊個股 {sid} ({STOCK_CONFIGS[sid].get('name', sid)}) ===")
            for script in PER_STOCK_SCRIPTS:
                success = run_script(script, args=["--stock-id", sid, "--start", "2010-01-01"])
                if not success:
                    logger.warning(f"  {script} 對 {sid} 執行失敗，繼續下一腳本")

    # ── Step 4：宏觀資料無個股區分，獨立執行一次（無論是否有缺失個股）
    logger.info("\n=== Step 4：更新宏觀資料（全域，不依個股） ===")
    for script in MACRO_SCRIPTS:
        run_script(script)

    logger.info("\n✅ 所有補齊作業完成。")


if __name__ == "__main__":
    main()

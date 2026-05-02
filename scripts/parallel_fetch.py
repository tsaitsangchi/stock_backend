import sys
import os
import subprocess
import time
import logging
from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
parallel_fetch.py — 並行資料抓取管理器 (根目錄版)

整合版，自動調度 fetchers/ 目錄下的腳本。
"""

# ─────────────────────────────────────────────
# 路徑與模組設定
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent  # scripts/
FETCHERS_DIR = BASE_DIR / "fetchers"
VENV_PYTHON = str(BASE_DIR.parent / "venv" / "bin" / "python3")

# 確保能找到 core/ 模組
sys.path.append(str(BASE_DIR))

from core.finmind_client import check_api_quota
from core.db_utils import get_db_conn, log_fetch_result

# ─────────────────────────────────────────────
# 腳本清單（指向 fetchers/ 目錄）
# ─────────────────────────────────────────────

PHASE_0 = [
    str(FETCHERS_DIR / "fetch_stock_info.py"),
]

PHASE_1 = [
    str(FETCHERS_DIR / "fetch_technical_data.py"),
    str(FETCHERS_DIR / "fetch_price_adj_data.py"),
    str(FETCHERS_DIR / "fetch_chip_data.py"),
    str(FETCHERS_DIR / "fetch_advanced_chip_data.py"),
    str(FETCHERS_DIR / "fetch_international_data.py"),
    str(FETCHERS_DIR / "fetch_macro_data.py"),
    str(FETCHERS_DIR / "fetch_fundamental_data.py"),
    str(FETCHERS_DIR / "fetch_cash_flows_data.py"),
    str(FETCHERS_DIR / "fetch_derivative_data.py"),
    str(FETCHERS_DIR / "fetch_extended_derivative_data.py"),
    str(FETCHERS_DIR / "fetch_sponsor_chip_data.py"),
    str(FETCHERS_DIR / "fetch_macro_fundamental_data.py"),
    str(FETCHERS_DIR / "fetch_derivative_sentiment_data.py"),
]

PHASE_2 = [
    str(FETCHERS_DIR / "fetch_event_risk_data.py"),
    str(FETCHERS_DIR / "fetch_news_data.py"),
    str(FETCHERS_DIR / "fetch_fred_data.py"),
]

MAX_WORKERS = 8
MAX_SCRIPT_TIMEOUT = 3 * 3600
API_QUOTA_MIN = 100

# ─────────────────────────────────────────────
# 日誌設定
# ─────────────────────────────────────────────
LOG_FILE = BASE_DIR / "outputs" / "logs" / "parallel_fetch.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
logger = logging.getLogger(__name__)

def run_script(script_path: str) -> tuple[str, bool]:
    if not os.path.exists(script_path):
        logger.error(f"找不到檔案: {script_path}")
        return script_path, False

    logger.info(f"🚀 開始執行: {script_path}")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE_DIR}:{env.get('PYTHONPATH', '')}"

    ret_code = 0
    success = False
    try:
        subprocess.run(
            [VENV_PYTHON, script_path],
            env=env,
            check=True,
            timeout=MAX_SCRIPT_TIMEOUT,
        )
        success = True
    except subprocess.CalledProcessError as e:
        ret_code = e.returncode
    except Exception as e:
        logger.error(f"執行異常: {script_path}, 錯誤: {e}")
        ret_code = -1

    conn = get_db_conn()
    try:
        status = "SUCCESS" if success else "FAILED"
        err_msg = "" if success else f"Exit code: {ret_code}"
        script_name = Path(script_path).name
        log_fetch_result(
            conn, script_name, "SYSTEM", 
            date.today().strftime("%Y-%m-%d"), 
            date.today().strftime("%Y-%m-%d"),
            0, status, err_msg
        )
    except Exception as e:
        logger.error(f"無法寫入 fetch_log: {e}")
    finally:
        conn.close()

    duration = time.time() - start
    if success:
        logger.info(f"✅ 完成: {script_path} (耗時: {duration:.1f} 秒)")
    else:
        logger.error(f"❌ 失敗: {script_path} (耗時: {duration:.1f} 秒)")
    
    return script_path, success

def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("   並行資料抓取管線 (根目錄整合版) 啟動")
    logger.info("=" * 60)

    try:
        used, limit = check_api_quota()
        remaining = limit - used
        logger.info(f"API 配額：已用 {used}/{limit}，剩餘 {remaining} 次")
        if remaining < API_QUOTA_MIN:
            logger.warning(f"配額不足。")
    except Exception as e:
        logger.warning(f"配額檢查跳過：{e}")

    # Phase 0
    for script in PHASE_0:
        run_script(script)

    # Phase 1 & 2 Combined (Hyper-Fetch)
    results = []
    all_parallel_scripts = PHASE_1 + PHASE_2
    logger.info(f"[Hyper-Fetch] 並行執行 {len(all_parallel_scripts)} 個抓取腳本 (Workers: {MAX_WORKERS})")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_script, s): s for s in all_parallel_scripts}
        for future in as_completed(futures):
            results.append(future.result())

    total_duration = time.time() - total_start
    success_count = sum(1 for _, ok in results if ok)
    logger.info("\n" + "=" * 60)
    logger.info(f"任務結束。成功: {success_count}, 失敗: {len(results) - success_count}")
    logger.info(f"總耗時: {total_duration / 60:.2f} 分鐘")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

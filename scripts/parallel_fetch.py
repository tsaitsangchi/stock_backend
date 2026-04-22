"""
parallel_fetch.py — 並行資料抓取管理器
同時啟動多個 fetch 腳本，大幅縮短每日更新時間。
透過 ProcessPoolExecutor 管理並行數，避免過度請求 API。
"""
import subprocess
import time
import sys
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# 環境路徑設定
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"

# ──────────────────────────────────────────────────────────────
# 配置要執行的抓取腳本
# 順序建議：
# Phase 0 (循序): 基礎設施腳本（如 stock_info 必須最新才能讓其他腳本過濾標的）
# Phase 1 (並行): 各類數據抓取
# ──────────────────────────────────────────────────────────────

PHASE_0 = ["scripts/fetch_stock_info.py"]

PHASE_1 = [
    "scripts/fetch_technical_data.py",    # 股價、PER (Batch 模式)
    "scripts/fetch_chip_data.py",         # 三大法人、融資融券
    "scripts/fetch_international_data.py",# 美股、ADR、匯率
    "scripts/fetch_macro_data.py",        # 利率、大宗商品
    "scripts/fetch_fundamental_data.py",  # 月營收、季報 (季報合併迴圈模式)
    "scripts/fetch_derivative_data.py",   # 台指期、選擇權籌碼
]

# 建議並行數：3
# 大多數腳本使用 FinMind API，並行數過高容易觸發 429 Rate Limit
MAX_WORKERS = 3

# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scripts/outputs/logs/parallel_fetch.log")
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path: str) -> tuple[str, bool]:
    """執行單一 Python 腳本並回傳狀態"""
    if not os.path.exists(script_path):
        logger.error(f"找不到檔案: {script_path}")
        return script_path, False

    logger.info(f"🚀 開始執行: {script_path}")
    start = time.time()
    
    # 設定環境變量確保路徑正確
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.path.join(os.getcwd(), "scripts")

    try:
        # 使用 subprocess 執行，並將輸出導向到 stdout
        subprocess.run([VENV_PYTHON, script_path], env=env, check=True)
        duration = time.time() - start
        logger.info(f"✅ 完成: {script_path} (耗時: {duration:.1f} 秒)")
        return script_path, True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 失敗: {script_path}, 退出代碼: {e.returncode}")
        return script_path, False
    except Exception as e:
        logger.error(f"❌ 異常: {script_path}, 錯誤: {e}")
        return script_path, False

def main():
    # 確保日誌目錄存在
    os.makedirs("scripts/outputs/logs", exist_ok=True)
    
    total_start = time.time()
    logger.info("="*60)
    logger.info("   Antigravity 並行資料抓取管線啟動")
    logger.info("="*60)
    
    # Phase 0: 循序執行關鍵腳本
    logger.info("\n[Phase 0] 執行基礎資料更新 (循序)...")
    for script in PHASE_0:
        success = run_script(script)[1]
        if not success:
            logger.warning(f"基礎資料更新 {script} 失敗，可能影響後續並行任務。")
        
    # Phase 1: 並行執行其餘腳本
    logger.info(f"\n[Phase 1] 啟動大規模資料抓取 (並行數: {MAX_WORKERS})...")
    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_script, s): s for s in PHASE_1}
        
        for future in as_completed(futures):
            results.append(future.result())
            
    # 總結報告
    total_duration = time.time() - total_start
    success_list = [s for s, status in results if status]
    failed_list = [s for s, status in results if not status]
    
    logger.info("\n" + "="*60)
    logger.info(f"任務結束報告")
    logger.info(f"  總耗時  : {total_duration/60:.2f} 分鐘")
    logger.info(f"  成功數量: {len(success_list)}")
    logger.info(f"  失敗數量: {len(failed_list)}")
    
    if failed_list:
        logger.error(f"  失敗清單: {', '.join(failed_list)}")
        sys.exit(1)
    else:
        logger.info("  所有資料抓取任務成功完成！")
    logger.info("="*60)

if __name__ == "__main__":
    main()

"""
parallel_fetch.py — 並行資料抓取管理器

同時啟動多個 fetch 腳本，大幅縮短每日更新時間。
透過 ProcessPoolExecutor 管理並行數，避免過度請求 API。

修改摘要（第四輪審查）：
  [P1 修正] SCRIPTS_DIR = Path(__file__).parent
            所有腳本路徑改為絕對路徑，無論從哪個工作目錄執行均正確
  [P1 修正] subprocess.run(..., timeout=MAX_SCRIPT_TIMEOUT)
            單腳本最長 3 小時，防止 Worker 被 402 等待永久阻塞
  [P1 新增] main() 開始前呼叫 check_api_quota()
            剩餘配額 < 100 次則警告並退出，避免三個並行 Worker 快速耗盡配額
"""

import subprocess
import time
import sys
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from core.finmind_client import check_api_quota

# ─────────────────────────────────────────────
# 路徑設定
# [P1 修正] 使用腳本所在目錄，不依賴執行時的工作目錄
# ─────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"

# ─────────────────────────────────────────────
# 腳本清單（使用絕對路徑）
# ─────────────────────────────────────────────

# Phase 0：循序執行，建立基礎資料（stock_info 必須最新）
PHASE_0 = [
    str(SCRIPTS_DIR / "fetch_stock_info.py"),
]

# Phase 1：並行執行各類資料（核心 fetch — 配額需求高）
PHASE_1 = [
    str(SCRIPTS_DIR / "fetch_technical_data.py"),          # 股價、PER（Batch）
    str(SCRIPTS_DIR / "fetch_price_adj_data.py"),          # [新] 還原股價、當沖、漲跌停
    str(SCRIPTS_DIR / "fetch_chip_data.py"),               # 三大法人、融資融券、外資持股
    str(SCRIPTS_DIR / "fetch_advanced_chip_data.py"),      # [新] 整體市場、借券、SBL、暫停融券
    str(SCRIPTS_DIR / "fetch_international_data.py"),      # 美股、ADR、匯率
    str(SCRIPTS_DIR / "fetch_macro_data.py"),              # 利率、大宗商品
    str(SCRIPTS_DIR / "fetch_fundamental_data.py"),        # 月營收、季報
    str(SCRIPTS_DIR / "fetch_cash_flows_data.py"),         # [新] 現金流量表、除權息結果
    str(SCRIPTS_DIR / "fetch_derivative_data.py"),         # 台指期、選擇權 OHLCV
    str(SCRIPTS_DIR / "fetch_extended_derivative_data.py"),  # [新] 期/權三大法人、夜盤、各券商
    str(SCRIPTS_DIR / "fetch_sponsor_chip_data.py"),       # 持股分級、分點、八大行庫
    str(SCRIPTS_DIR / "fetch_macro_fundamental_data.py"),  # 景氣對策信號、市值比重
    str(SCRIPTS_DIR / "fetch_derivative_sentiment_data.py"),  # 選擇權大額 OI、恐懼貪婪、鉅額
]

# Phase 2：事件風險 + 新聞 + FRED 外部資料（配額需求低，可在 Phase 1 之後並行）
PHASE_2 = [
    str(SCRIPTS_DIR / "fetch_event_risk_data.py"),         # [新] 下市、暫停、減資、市值
    str(SCRIPTS_DIR / "fetch_news_data.py"),               # [新] 個股新聞
    str(SCRIPTS_DIR / "fetch_fred_data.py"),               # [新] FRED 外部宏觀（VIX、Yield Spread）
]

# 並行數建議：3
# FinMind 配額 600 次/小時，並行數過高容易快速觸發 402
MAX_WORKERS = 3

# [P1 修正] 單腳本最長執行時間（秒）
# 3 小時足以覆蓋最長的歷史補齊 + 402 等待場景
MAX_SCRIPT_TIMEOUT = 3 * 3600

# API 配額安全下限：剩餘不足此數量則中止並行抓取
API_QUOTA_MIN = 100

# ─────────────────────────────────────────────
# 日誌設定
# ─────────────────────────────────────────────
LOG_FILE = SCRIPTS_DIR / "outputs" / "logs" / "parallel_fetch.log"
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


# ─────────────────────────────────────────────
# 執行單一腳本
# ─────────────────────────────────────────────
def run_script(script_path: str) -> tuple[str, bool]:
    """執行單一 Python 腳本並回傳 (script_path, 是否成功)。"""
    if not os.path.exists(script_path):
        logger.error(f"找不到檔案: {script_path}")
        return script_path, False

    logger.info(f"🚀 開始執行: {script_path}")
    start = time.time()

    env = os.environ.copy()
    # 確保 core/ 模組可被子行程的 Python 找到
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SCRIPTS_DIR}:{pythonpath}" if pythonpath else str(SCRIPTS_DIR)

    try:
        subprocess.run(
            [VENV_PYTHON, script_path],
            env=env,
            check=True,
            timeout=MAX_SCRIPT_TIMEOUT,  # [P1 修正] 防止 Worker 永久阻塞
        )
        duration = time.time() - start
        logger.info(f"✅ 完成: {script_path} (耗時: {duration:.1f} 秒)")
        return script_path, True

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        logger.error(
            f"⏱️ 超時: {script_path} 執行超過 {MAX_SCRIPT_TIMEOUT // 3600} 小時，強制終止"
        )
        return script_path, False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 失敗: {script_path}, 退出代碼: {e.returncode}")
        return script_path, False
    except Exception as e:
        logger.error(f"❌ 異常: {script_path}, 錯誤: {e}")
        return script_path, False


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("   並行資料抓取管線啟動")
    logger.info("=" * 60)

    # ── [P1 新增] 啟動前預查 API 配額
    logger.info("\n[配額檢查] 查詢 FinMind API 剩餘使用量…")
    try:
        used, limit = check_api_quota()
        remaining = limit - used
        logger.info(f"API 配額：已用 {used}/{limit}，剩餘 {remaining} 次")
        if remaining < API_QUOTA_MIN:
            logger.warning(
                f"API 配額剩餘不足 {API_QUOTA_MIN} 次（目前剩 {remaining}），"
                f"建議等待下一個整點重置後再執行。"
            )
            sys.exit(1)
    except Exception as e:
        # 配額查詢失敗不中止流程，僅記錄警告
        logger.warning(f"配額查詢失敗（{e}），繼續執行。")

    # ── Phase 0：循序執行關鍵腳本（stock_info 必須先完成）
    logger.info("\n[Phase 0] 執行基礎資料更新（循序）…")
    for script in PHASE_0:
        success = run_script(script)[1]
        if not success:
            logger.warning(f"基礎資料更新 {script} 失敗，可能影響後續並行任務。")

    # ── Phase 1：並行執行核心 fetch（高配額需求）
    logger.info(f"\n[Phase 1] 啟動核心資料並行抓取（並行數: {MAX_WORKERS}）…")
    results: list[tuple[str, bool]] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_script, s): s for s in PHASE_1}
        for future in as_completed(futures):
            results.append(future.result())

    # ── Phase 2：事件/新聞/FRED 等低配額需求的補充 fetch
    logger.info(f"\n[Phase 2] 啟動事件風險與外部資料抓取（並行數: {MAX_WORKERS}）…")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_script, s): s for s in PHASE_2}
        for future in as_completed(futures):
            results.append(future.result())

    # ── 總結報告
    total_duration = time.time() - total_start
    success_list = [s for s, ok in results if ok]
    failed_list  = [s for s, ok in results if not ok]

    logger.info("\n" + "=" * 60)
    logger.info("任務結束報告")
    logger.info(f"  總耗時  : {total_duration / 60:.2f} 分鐘")
    logger.info(f"  成功數量: {len(success_list)}")
    logger.info(f"  失敗數量: {len(failed_list)}")

    if failed_list:
        # 只顯示腳本檔名（去掉路徑），日誌更易讀
        short_names = [Path(s).name for s in failed_list]
        logger.error(f"  失敗清單: {', '.join(short_names)}")
        sys.exit(1)
    else:
        logger.info("  所有資料抓取任務成功完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

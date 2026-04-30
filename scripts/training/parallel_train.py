from __future__ import annotations
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ['fetchers', 'pipeline', 'training', 'monitor']: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
"""
parallel_train.py — 全自動並行訓練管理器
=============================================
同時執行 N 個訓練任務，確保在資源可控的情況下快速完成所有個股模型。

第五輪審查修復重點：
  ① 訓練優先名單由「缺檔」擴展為「缺檔 + 過期 (>30天)」
     ─ 整合 model_health_check.check_model_files_df 結果
     ─ 對齊 README/SYSTEM 「模型 30 日強制重訓」精神

  ② step_days 不再硬編碼 63，改為直接引用 WF_CONFIG["step_days"]
     ─ 與 Walk-Forward 評估窗口一致

  ③ 每個 stock 的訓練支援指數退避自動重試
     ─ 第 1 次失敗：等 60 秒
     ─ 第 2 次失敗：等 120 秒
     ─ 第 3 次失敗：等 240 秒，仍失敗才標記為失敗

  ④ subprocess.run 加上 timeout 保護
     ─ 防止單一卡死進程永久佔用 worker
"""

import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 注入路徑（讓 model_health_check 可以從 scripts/ 匯入）
sys.path.append(str(Path(__file__).resolve().parent))

from config import STOCK_CONFIGS, MODEL_DIR, LOG_DIR, BASE_DIR, WF_CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# ── 設定區域 ───────────────────────────────────────────────────
MAX_WORKERS    = 3                                            # 並行訓練數
VENV_PYTHON    = os.environ.get("VENV_PYTHON",
                                "/home/hugo/project/stock_backend/venv/bin/python3")
TRAIN_SCRIPT   = str(BASE_DIR / "train_evaluate.py")
HEALTH_SCRIPT  = str(BASE_DIR / "model_health_check.py")
MAX_RETRIES    = 3                                            # 訓練失敗最大重試次數
TRAIN_TIMEOUT  = 4 * 3600                                     # 單一訓練最長 4 小時
RETRAIN_AGE_THRESHOLD_DAYS = 30                               # 模型多久之後強制重訓
# ──────────────────────────────────────────────────────────────


def get_stocks_needing_training(force_retrain_days: int = RETRAIN_AGE_THRESHOLD_DAYS) -> list[str]:
    """
    [P1 第五輪修復]
    整合 model_health_check 的結果，找出需要訓練的個股：
      - 缺少 ensemble_{sid}.pkl 檔案
      - 模型檔超過 force_retrain_days 天未更新
      - PSI 超過 0.2（漂移嚴重，需重訓）

    Fallback：若 model_health_check 載入失敗（例如新環境尚無歷史資料），
    退回到僅檢查「檔案是否存在」。
    """
    all_stocks = list(STOCK_CONFIGS.keys())
    try:
        from model_health_check import check_model_files_df, check_prediction_drift_df
        files_df = check_model_files_df(all_stocks)
        # 缺檔或超過 force_retrain_days 都需要重訓
        need = files_df[
            (files_df["status"] != "[OK]") |
            (files_df["age_days"] > force_retrain_days)
        ]["stock_id"].tolist()

        # 若可以拿到 drift，把 drift 嚴重的也加進來
        try:
            drift_df = check_prediction_drift_df(all_stocks)
            drifted = drift_df[drift_df["psi"] > 0.2]["stock_id"].tolist()
            need = sorted(set(need) | set(drifted))
        except Exception as e:
            logger.debug(f"健康 drift 檢查跳過：{e}")
        return need
    except Exception as e:
        logger.warning(f"健康檢查整合失敗，退回到檔案存在性檢查：{e}")
        return [
            sid for sid in all_stocks
            if not (MODEL_DIR / f"ensemble_{sid}.pkl").exists()
        ]


def train_one_stock(stock_id: str) -> tuple[str, bool, str | None]:
    """
    訓練單一個股，含指數退避重試與 timeout 保護。

    Returns:
        (stock_id, success, error_msg)
    """
    stock_name = STOCK_CONFIGS[stock_id]["name"]
    log_file   = str(LOG_DIR / f"train_{stock_id}.log")

    # [P1 第五輪修復] step_days 與 WF_CONFIG 同步
    step_days = str(WF_CONFIG["step_days"])

    cmd = [
        VENV_PYTHON,
        TRAIN_SCRIPT,
        "--stock-id", stock_id,
        "--step-days", step_days,
    ]

    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{stock_id}] ({stock_name}) 訓練嘗試 {attempt}/{MAX_RETRIES} (step_days={step_days})")
            mode = "w" if attempt == 1 else "a"
            with open(log_file, mode) as f:
                if attempt > 1:
                    f.write(f"\n\n=== 重試第 {attempt} 次 ===\n")
                subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    check=True, timeout=TRAIN_TIMEOUT
                )

            # 訓練成功 → 跑健康檢查
            with open(log_file, "a") as f:
                f.write("\n\n=== 自動健康檢查 (Post-Training Health Check) ===\n")
                # 注意 model_health_check.py 沒有 --stock-id 參數，跑全標的即可
                try:
                    subprocess.run([VENV_PYTHON, HEALTH_SCRIPT],
                                   stdout=f, stderr=subprocess.STDOUT,
                                   timeout=300)
                except Exception as he:
                    f.write(f"健康檢查失敗（不影響訓練成功狀態）：{he}\n")
            return stock_id, True, None

        except subprocess.TimeoutExpired:
            last_error = f"訓練超時 ({TRAIN_TIMEOUT}s)"
            logger.warning(f"[{stock_id}] {last_error}（第 {attempt}/{MAX_RETRIES} 次）")
        except subprocess.CalledProcessError as e:
            last_error = f"非零回傳碼 {e.returncode}"
            logger.warning(f"[{stock_id}] {last_error}（第 {attempt}/{MAX_RETRIES} 次）")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[{stock_id}] 例外：{e}（第 {attempt}/{MAX_RETRIES} 次）")

        # 指數退避：60s → 120s → 240s
        if attempt < MAX_RETRIES:
            backoff = 60 * (2 ** (attempt - 1))
            logger.info(f"[{stock_id}] 等待 {backoff}s 後重試...")
            time.sleep(backoff)

    return stock_id, False, last_error


def main():
    # [P1 第五輪修復] 找出需要訓練的個股（缺檔 + 過期 + 嚴重漂移）
    to_train = get_stocks_needing_training()

    if not to_train:
        print("所有個股模型皆已新鮮，無需重訓。")
        return

    print(f"總計需要訓練 {len(to_train)} 支個股，設定並行數為 {MAX_WORKERS}...")
    print(f"訓練清單：{to_train}")

    start_time = time.time()
    results = {"success": [], "failed": []}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(train_one_stock, sid): sid for sid in to_train}
        for future in as_completed(futures):
            sid, success, error = future.result()
            if success:
                print(f"[完成] {sid}")
                results["success"].append(sid)
            else:
                print(f"[失敗] {sid}: {error}")
                results["failed"].append(sid)

    duration = (time.time() - start_time) / 3600

    print("\n" + "=" * 50)
    print(f"訓練任務結束！總耗時: {duration:.2f} 小時")
    print(f"成功: {len(results['success'])}")
    print(f"失敗: {len(results['failed'])}")
    if results["failed"]:
        print(f"失敗清單: {', '.join(results['failed'])}")
    print("=" * 50)


if __name__ == "__main__":
    main()

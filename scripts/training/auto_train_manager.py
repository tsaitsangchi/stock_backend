import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor", "models"]:
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

from config import STOCK_CONFIGS, TIER_1_STOCKS

import os
import time
import subprocess
import logging
import json
from datetime import datetime

"""
auto_train_manager.py — 自動訓練管理員
"""

# ─────────────────────────────────────────────
# 基本設定
# ─────────────────────────────────────────────
VENV_PYTHON        = "/home/hugo/project/stock_backend/venv/bin/python3"
SCRIPTS_DIR        = Path(__file__).parent
MAX_PARALLEL_TRAINS = 2      # 深層訓練耗能高，降低並行數
CHECK_INTERVAL     = 60      # 每分鐘檢查一次
METRICS_REGISTRY   = SCRIPTS_DIR / "outputs" / "metrics_registry.json"
FAILURE_TRACKER    = {}  # {sid: {"retries": 0, "last_fail": timestamp}}
MAX_RETRIES        = 3

# [P1 2.4] PID 追蹤目錄
PID_DIR = SCRIPTS_DIR / "outputs" / "pids"
PID_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_STOCKS      = ["2330", "2317", "2454"]
SIXTH_WAVE_DRIVERS = ["2330", "2454", "3661", "2376", "2382", "6669"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(SCRIPTS_DIR / "outputs" / "manager.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# [P1 2.4] PID 檔案機制（取代 ps aux 脆弱解析）
# ─────────────────────────────────────────────

def get_running_trains() -> list[str]:
    """
    讀取 PID 檔案，確認進程仍在運行。
    自動清理已結束進程的 PID 檔案（殭屍 PID 清理）。
    """
    running = []
    for pid_file in PID_DIR.glob("*.pid"):
        sid = pid_file.stem
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # signal 0：只檢查進程是否存在，不發送任何訊號
            running.append(sid)
        except (ProcessLookupError, PermissionError):
            # 進程已結束，清理 PID 檔案
            pid_file.unlink(missing_ok=True)
            logger.debug(f"[PID] 清理殭屍 PID 檔案：{pid_file.name}")
        except ValueError:
            pid_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"[PID] 檢查 {sid} 進程失敗：{e}")
    return list(set(running))


def check_finished_trains(running_before: list[str], running_now: list[str]) -> list[str]:
    """辨識剛結束的任務（在 before 中但不在 now 中）。"""
    return [sid for sid in running_before if sid not in running_now]


def get_task_exit_status(sid: str) -> bool:
    """
    檢查訓練日誌的最末尾，判斷是否成功。
    (這是一個啟發式檢查，因為 subprocess.Popen 沒法直接拿 exit code)
    """
    log_path = SCRIPTS_DIR / "outputs" / f"train_{sid}.log"
    if not log_path.exists(): return False
    try:
        with open(log_path, "r") as f:
            last_lines = f.readlines()[-10:]
            content = "".join(last_lines)
            return "SUCCESS" in content or "完成" in content or "metrics_registry" in content
    except:
        return False


def launch_training(sid: str, cmd: list[str], log_path: Path):
    """
    啟動背景訓練進程，並記錄 PID 到追蹤目錄。
    """
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, start_new_session=True)
    # 寫入 PID 檔案
    pid_file = PID_DIR / f"{sid}.pid"
    pid_file.write_text(str(proc.pid))
    logger.info(f"[PID] {sid} 訓練已啟動，PID={proc.pid}，記錄至 {pid_file}")


# ─────────────────────────────────────────────
# 效能指標讀取
# ─────────────────────────────────────────────

def get_performance_scores() -> dict:
    """從註冊表讀取歷史 DA。"""
    scores = {}
    if METRICS_REGISTRY.exists():
        try:
            with open(METRICS_REGISTRY, "r") as f:
                data = json.load(f)
            for sid, metrics in data.items():
                scores[sid] = metrics.get("directional_accuracy", 0.5)
        except Exception as e:
            logger.warning(f"讀取 metrics_registry 失敗：{e}")
    return scores


# ─────────────────────────────────────────────
# [P1 2.3] 優先級計算（修正雙重加成問題）
# ─────────────────────────────────────────────

def calculate_priority(sid: str, perf_scores: dict) -> float:
    """
    計算優先權評分 (0~100)。

    原版問題：
      priority *= 1.2  →  再  priority += 20  →  min(100, ...)
      對 Anchor Stock（priority ≈ 90），兩次加成後 = 90×1.2+20 = 128，
      被截斷至 100，等於加成完全無效。

    修正版：統一使用乘法加成，加成後才 clip。
    """
    # 1. 基準分（權值，70%）
    if sid in ANCHOR_STOCKS:
        weight_score = 100
    elif sid in TIER_1_STOCKS:
        weight_score = 80
    else:
        weight_score = 30

    # 2. 勝率分（DA 歷史表現，30%）
    da = perf_scores.get(sid, 0.5)
    da_score = min(100, max(0, (da - 0.45) / 0.15 * 100))  # 0.45→0, 0.60→100

    priority = weight_score * 0.7 + da_score * 0.3

    # 3. [修正] 第六波賽道加成：統一乘法，不重複
    if sid in SIXTH_WAVE_DRIVERS:
        wave_bonus = 1.25 if datetime.now().year >= 2025 else 1.10
        priority = priority * wave_bonus  # 不在此 clip，讓後面 min(100,...) 統一處理

    return min(100.0, priority)


# ─────────────────────────────────────────────
# 主循環
# ─────────────────────────────────────────────

def main():
    logger.info("=== 自動訓練管理員啟動（第四輪修正版）===")

    # [P1 2.5] update_feature_store.py 存在性預檢
    feature_store_script = SCRIPTS_DIR / "update_feature_store.py"
    if not feature_store_script.exists():
        logger.warning(
            f"update_feature_store.py 不存在（{feature_store_script}），"
            f"訓練時將跳過特徵庫更新，使用現有特徵庫。"
        )

    while True:
        try:
            running_now  = get_running_trains()
            
            # --- [新功能] 處理剛結束的任務並更新失敗計數 ---
            if 'running_ids' in locals():
                finished_tasks = check_finished_trains(running_ids, running_now)
                for sid in finished_tasks:
                    if get_task_exit_status(sid):
                        logger.info(f"✅ [{sid}] 訓練成功結束。")
                        FAILURE_TRACKER.pop(sid, None)
                    else:
                        fail_count = FAILURE_TRACKER.get(sid, {}).get("retries", 0) + 1
                        FAILURE_TRACKER[sid] = {"retries": fail_count, "last_fail": time.time()}
                        logger.warning(f"❌ [{sid}] 訓練似乎失敗了！(失敗次數: {fail_count}/{MAX_RETRIES})")
            
            running_ids = running_now
            perf_scores = get_performance_scores()

            # 排序標的：基於 Priority Score（降序）
            all_sids     = list(STOCK_CONFIGS.keys())
            sid_priorities = {sid: calculate_priority(sid, perf_scores) for sid in all_sids}
            sorted_targets = sorted(all_sids, key=lambda x: sid_priorities[x], reverse=True)

            # 有效模型清單（Tier 1 每週重訓，其他每月）
            finished_ids = []
            now = time.time()
            model_dir = SCRIPTS_DIR / "outputs" / "models"
            if model_dir.exists():
                for f in model_dir.iterdir():
                    if f.suffix == ".pkl" and "ensemble_" in f.name:
                        sid = f.stem.replace("ensemble_", "")
                        days_old = (now - f.stat().st_mtime) / (24 * 3600)
                        limit = 7 if sid in TIER_1_STOCKS else 30
                        if days_old < limit:
                            finished_ids.append(sid)

            logger.info(
                f"執行中: {running_ids} | "
                f"有效模型: {len(finished_ids)} | "
                f"待訓練: {len(sorted_targets) - len(finished_ids)}"
            )

            if len(running_ids) < MAX_PARALLEL_TRAINS:
                for sid in sorted_targets:
                    # 檢查是否已完成、是否正在執行、是否已達到重試上限
                    is_quarantined = FAILURE_TRACKER.get(sid, {}).get("retries", 0) >= MAX_RETRIES
                    
                    if sid not in finished_ids and sid not in running_ids and not is_quarantined:
                        # 實施退避 (Backoff)：失敗後需等待一定時間才能重試
                        last_fail = FAILURE_TRACKER.get(sid, {}).get("last_fail", 0)
                        retry_wait = (2 ** FAILURE_TRACKER.get(sid, {}).get("retries", 0)) * 60 # 60s, 120s, 240s...
                        
                        if time.time() - last_fail < retry_wait:
                            continue
                        is_anchor = sid in ANCHOR_STOCKS
                        mode_str  = "DEEP (141-Fold)" if is_anchor else "PARETO (60-Fold)"
                        logger.info(
                            f">>> 啟動 {sid} ({STOCK_CONFIGS[sid]['name']}) "
                            f"| 模式: {mode_str} | 優先級: {sid_priorities[sid]:.1f}"
                        )

                        # [P1 2.5] 若腳本存在才執行特徵庫更新
                        if feature_store_script.exists():
                            logger.info(f"[{sid}] 更新特徵庫…")
                            try:
                                subprocess.run(
                                    [VENV_PYTHON, str(feature_store_script), "--stock-id", sid],
                                    check=True,
                                    timeout=300,
                                )
                            except subprocess.TimeoutExpired:
                                logger.warning(f"[{sid}] 特徵庫更新超時（5 分鐘），繼續訓練")
                            except subprocess.CalledProcessError as e:
                                logger.warning(f"[{sid}] 特徵庫更新失敗（{e}），繼續使用現有特徵庫")
                        else:
                            logger.warning(f"[{sid}] 跳過特徵庫更新（腳本不存在）")

                        # [P1 2.4] 使用 launch_training 記錄 PID
                        cmd = [VENV_PYTHON, str(SCRIPTS_DIR / "train_evaluate.py"), "--stock-id", sid]
                        if not is_anchor:
                            cmd += ["--fast-mode"]
                        log_path = SCRIPTS_DIR / "outputs" / f"train_{sid}.log"
                        launch_training(sid, cmd, log_path)

                        logger.info(f"[{sid}] 任務已分發。")
                        time.sleep(10)  # 深層任務啟動較慢，等待進程穩定
                        break

        except Exception as e:
            logger.error(f"管理員循環發生錯誤：{e}", exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

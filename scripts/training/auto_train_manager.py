import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401

from config import STOCK_CONFIGS, TIER_1_STOCKS, MODEL_DIR, OUTPUT_DIR

import os
import time
import subprocess
import logging
import json
from datetime import datetime
from typing import Union, Optional

"""
auto_train_manager.py — 自動訓練管理員

[P0-3 / QW-2] 加入 heartbeat 機制：
  - 每 60 秒寫入 outputs/auto_train.heartbeat（含 timestamp + progress）
  - 同步寫入 DB auto_train_heartbeat 表（cron 可從 DB 監控）
  - 配合 deploy/auto_train.service（systemd Restart=always）
    與 deploy/cron_check_heartbeat.sh，避免靜默死機
"""

# ─────────────────────────────────────────────
# 基本設定
# ─────────────────────────────────────────────
VENV_PYTHON        = os.environ.get("VENV_PYTHON",
                                    "/home/hugo/project/stock_backend/venv/bin/python3")
SCRIPTS_DIR        = Path(__file__).parent # 腳本所在目錄 (scripts/training)
PROJECT_SCRIPTS    = SCRIPTS_DIR.parent    # 父目錄 (scripts)
MAX_PARALLEL_TRAINS = 2      # 深層訓練耗能高，降低並行數
CHECK_INTERVAL     = 60      # 每分鐘檢查一次
METRICS_REGISTRY   = OUTPUT_DIR / "metrics_registry.json"
FAILURE_TRACKER    = {}  # {sid: {"retries": 0, "last_fail": timestamp}}
MAX_RETRIES        = 3

# [P0-3 / QW-2] Heartbeat 設定
HEARTBEAT_FILE     = OUTPUT_DIR / "auto_train.heartbeat"
HEARTBEAT_DDL      = """
CREATE TABLE IF NOT EXISTS auto_train_heartbeat (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    pid         INTEGER,
    running_ids TEXT,
    progress    TEXT
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_ts ON auto_train_heartbeat (ts DESC);
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / "manager.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# [P1 2.4] PID 追蹤目錄
DEFAULT_PID_DIR = OUTPUT_DIR / "pids"
FALLBACK_PID_DIR = Path("/tmp/stock_pids")

def ensure_pid_dir():
    """確保 PID 目錄可用，若無權限則回退至 /tmp。"""
    try:
        DEFAULT_PID_DIR.mkdir(parents=True, exist_ok=True)
        # 測試寫入權限
        test_file = DEFAULT_PID_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return DEFAULT_PID_DIR
    except (PermissionError, OSError):
        if 'logger' in globals():
            logger.warning(f"  [Perm] PID 目錄 {DEFAULT_PID_DIR} 無權限寫入，回退至 {FALLBACK_PID_DIR}")
        FALLBACK_PID_DIR.mkdir(parents=True, exist_ok=True)
        return FALLBACK_PID_DIR

PID_DIR = ensure_pid_dir()

ANCHOR_STOCKS      = ["2330", "2317", "2454"]
SIXTH_WAVE_DRIVERS = ["2330", "2454", "3661", "2376", "2382", "6669"]


# ─────────────────────────────────────────────
# [P1 2.4] PID 檔案機制（取代 ps aux 脆弱解析）
# ─────────────────────────────────────────────

def get_running_trains() -> dict[str, int]:
    """
    掃描 scripts/outputs/pids/ 目錄，檢查進程是否真的還在。
    回傳 {stock_id: pid}。
    """
    running = {}
    if not PID_DIR.exists():
        return running
    
    for pid_file in PID_DIR.glob("*.pid"):
        stock_id = pid_file.stem
        try:
            pid = int(pid_file.read_text().strip())
            # [P1-FIX] 增加權限與異常檢查，防止因 root 權限檔案導致崩潰
            try:
                os.kill(pid, 0)  # signal 0：只檢查進程是否存在
                running[stock_id] = pid
            except ProcessLookupError:
                # 進程已消失，嘗試刪除過期 PID 檔
                pid_file.unlink(missing_ok=True)
            except PermissionError:
                # [P0-SECURITY] 若無權限檢查該進程（如 root 啟動），視為不可控進程，保留鎖定但不加入管理員控制
                logger.warning(f"  [Perm] 無權檢查進程 {pid} ({stock_id})，跳過。")
                continue
        except (ValueError, OSError) as e:
            # 檔案損毀或權限不足無法讀取/刪除
            logger.warning(f"  [PID] 處理檔案 {pid_file.name} 發生錯誤：{e}")
            continue
            
    return running


def check_finished_trains(running_before: list[str], running_now: Union[list[str], dict[str, int]]) -> list[str]:
    """辨識剛結束的任務（在 before 中但不在 now 中）。"""
    return [sid for sid in running_before if sid not in running_now]


def get_task_status(sid: str) -> str:
    """
    檢查訓練日誌，返回任務狀態：'SUCCESS', 'SKIPPED', 或 'FAILED'。
    """
    log_path = OUTPUT_DIR / f"train_{sid}.log"
    if not log_path.exists():
        return "FAILED"
    try:
        with open(log_path, "r") as f:
            last_lines = f.readlines()[-20:]
            content = "".join(last_lines)
            # [P1-BUG] 修正過於鬆散的關鍵字匹配。
            # 原本匹配 "完成" 會誤判 "資料載入完成" 為成功。
            if "訓練已全面完成" in content or "metrics_registry" in content or "Model saved to" in content:
                return "SUCCESS"
            if "跳過訓練" in content or "資料量不足" in content:
                return "SKIPPED"
            return "FAILED"
    except:
        return "FAILED"


def write_heartbeat(running_ids: list, finished_ids: list, total: int) -> None:
    """
    [P0-3 / QW-2] 寫入心跳：檔案 + DB 雙軌。

    cron / systemd 監控腳本透過讀取 HEARTBEAT_FILE 的 mtime 判斷活性，
    DB auto_train_heartbeat 則保留長期軌跡，便於問題分析。
    """
    progress_str = f"{len(finished_ids)}/{total}"
    payload = {
        "ts":          datetime.now().isoformat(),
        "pid":         os.getpid(),
        "running_ids": running_ids,
        "finished":    len(finished_ids),
        "total":       total,
        "progress":    progress_str,
    }
    # 1) 檔案心跳（main caller / cron 可用 mtime 判斷活性）
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # [Fix] 先刪除舊檔以避開權限問題
        HEARTBEAT_FILE.unlink(missing_ok=True)
        HEARTBEAT_FILE.write_text(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.debug(f"[heartbeat] 檔案寫入失敗：{e}")

    # 2) DB 心跳（輕量，可被 audit / dashboard 使用）
    try:
        from core.db_utils import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for stmt in [s.strip() for s in HEARTBEAT_DDL.split(";") if s.strip()]:
                    cur.execute(stmt)
                cur.execute(
                    """
                    INSERT INTO auto_train_heartbeat (pid, running_ids, progress)
                    VALUES (%s, %s, %s)
                    """,
                    (os.getpid(), ",".join(running_ids), progress_str),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f"[heartbeat] DB 寫入失敗（不影響主流程）：{e}")


def launch_training(stock_id: str, command: list[str], log_path: Path):
    """啟動訓練進程並記錄 PID。"""
    try:
        # [Fix] 嘗試刪除舊日誌，確保有權限開啟新檔
        log_path.unlink(missing_ok=True)
        
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        
        # 寫入 PID 檔案
        pid_file = PID_DIR / f"{stock_id}.pid"
        try:
            # [Fix] 先嘗試刪除舊檔，只要對目錄有權限即可刪除 root 建立的檔案
            pid_file.unlink(missing_ok=True)
            pid_file.write_text(str(proc.pid))
            logger.info(f"  [PID] {stock_id} -> {proc.pid}")
        except Exception as e:
            logger.error(f"  [CRITICAL] 無法寫入 PID 檔案 {pid_file}: {e}")
            # 雖然沒寫入 PID，但進程已啟動，我們還是記錄一下，但下輪循環可能無法追蹤它
            
        return proc.pid
    except Exception as e:
        logger.error(f"  [Launch] 啟動 {stock_id} 失敗: {e}")
        return None


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
    # ── 參數解析 ──
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="強制重新訓練所有標的，忽略模型有效期")
    args = parser.parse_args()

    logger.info(f"=== 自動訓練管理員啟動（第四輪修正版, force_all={args.force_all}）===")

    # [P1 2.5] update_feature_store.py 存在性預檢
    feature_store_script = SCRIPTS_DIR / "update_feature_store.py"
    if not feature_store_script.exists():
        logger.warning(
            f"update_feature_store.py 不存在（{feature_store_script}），"
            f"訓練時將跳過特徵庫更新，使用現有特徵庫。"
        )

    while True:
        try:
            running_dict = get_running_trains()
            running_now  = list(running_dict.keys())
            
            # --- [新功能] 處理剛結束的任務並更新失敗計數 ---
            if 'running_ids' in locals():
                finished_this_tick = check_finished_trains(running_ids, running_now)
                # [P1 2.2] 檢查剛結束的任務
                for sid in finished_this_tick:
                    status = get_task_status(sid)
                    if status == "SUCCESS":
                        logger.info(f"✅ [{sid}] 訓練成功結束。")
                    elif status == "SKIPPED":
                        logger.warning(f"⚠️ [{sid}] 訓練跳過 (原因: 資料量不足)。將進入退避等待。")
                        FAILURE_TRACKER[sid] = FAILURE_TRACKER.get(sid, {"retries": 0})
                        FAILURE_TRACKER[sid]["retries"] += 1
                        FAILURE_TRACKER[sid]["last_fail"] = time.time()
                    else:
                        logger.warning(f"❌ [{sid}] 訓練似乎失敗了！(失敗次數: {FAILURE_TRACKER.get(sid,{}).get('retries',0)+1}/{MAX_RETRIES})")
                        FAILURE_TRACKER[sid] = FAILURE_TRACKER.get(sid, {"retries": 0})
                        FAILURE_TRACKER[sid]["retries"] += 1
                        FAILURE_TRACKER[sid]["last_fail"] = time.time()
            
            running_ids = running_now
            perf_scores = get_performance_scores()

            # 排序標的：基於 Priority Score（降序）
            all_sids     = list(STOCK_CONFIGS.keys())
            sid_priorities = {sid: calculate_priority(sid, perf_scores) for sid in all_sids}
            sorted_targets = sorted(all_sids, key=lambda x: sid_priorities[x], reverse=True)

            # 有效模型清單（Tier 1 每週重訓，其他每月）
            finished_ids = []
            if not args.force_all:
                now = time.time()
                if MODEL_DIR.exists():
                    for f in MODEL_DIR.iterdir():
                        if f.suffix == ".pkl" and "ensemble_" in f.name:
                            sid = f.stem.replace("ensemble_", "")
                            days_old = (now - f.stat().st_mtime) / (24 * 3600)
                            limit = 7 if sid in TIER_1_STOCKS else 30
                            if days_old < limit:
                                finished_ids.append(sid)
            else:
                logger.info("⚡ [Force All] 模式已啟動，將強制重新訓練所有標的。")

            logger.info(
                f"執行中: {running_ids} | "
                f"有效模型: {len(finished_ids)} | "
                f"待訓練: {len(sorted_targets) - len(finished_ids)}"
            )

            # [P0-3 / QW-2] 每輪迴圈寫一次心跳
            write_heartbeat(running_ids, finished_ids, len(sorted_targets))

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
                                # [P1-FIX] 檢查特徵庫更新結果
                                res = subprocess.run(
                                    [VENV_PYTHON, str(feature_store_script), "--stock-id", sid],
                                    capture_output=True,
                                    text=True,
                                    timeout=300,
                                )
                                # 檢查日誌中是否有 "樣本過少" 警告
                                if "樣本過少" in res.stdout or "樣本過少" in res.stderr:
                                    logger.warning(f"⚠️ [{sid}] 資料樣本不足，暫時跳過訓練。")
                                    FAILURE_TRACKER[sid] = FAILURE_TRACKER.get(sid, {"retries": 0})
                                    # 給予一定的懲罰性重試計數，以免立即重試
                                    FAILURE_TRACKER[sid]["retries"] = 1 
                                    FAILURE_TRACKER[sid]["last_fail"] = time.time()
                                    continue
                                    
                            except subprocess.TimeoutExpired:
                                logger.warning(f"[{sid}] 特徵庫更新超時（5 分鐘），繼續訓練")
                            except subprocess.CalledProcessError as e:
                                logger.warning(f"[{sid}] 特徵庫更新失敗（{e}），繼續使用現有特徵庫")
                        else:
                            logger.warning(f"[{sid}] 跳過特徵庫更新（腳本不存在）")

                        # [P1 2.4] 使用 launch_training 記錄 PID
                        cmd = [VENV_PYTHON, str(SCRIPTS_DIR / "train_evaluate.py"), "--stock-id", sid]
                        if is_anchor:
                            # 對於 Anchor Stock，將步進提高到 60（約 50-fold），兼顧穩健與速度
                            cmd += ["--step-days", "60"]
                        else:
                            # 一般標的維持 fast-mode (double step_days)
                            cmd += ["--fast-mode"]
                        
                        log_path = OUTPUT_DIR / f"train_{sid}.log"
                        launch_training(sid, cmd, log_path)

                        logger.info(f"[{sid}] 任務已分發。")
                        time.sleep(10)  # 深層任務啟動較慢，等待進程穩定
                        break

        except Exception as e:
            logger.error(f"管理員循環發生錯誤：{e}", exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

"""
auto_predict_manager.py — 自動預測管理器 v2（檔案 + DB 雙通道心跳）
================================================================
監控 scripts/outputs/models/ 目錄，一旦偵測到模型更新 (.pkl mtime 變動)，
立即觸發 predict.py 進行推論，實現「訓練-推論」自動化閉環。

v2 改進（呼應 P0-3「無告警機制」風險）：
  · 心跳由純檔案升級為「檔案 + DB 雙通道」：
      ─ 檔案 outputs/auto_predict.heartbeat：本機 dashboard / cron 用
      ─ DB auto_predict_heartbeat：分散式監控 / 跨機器告警
  · 啟動時建立 PID 檔；退出時自動清理（含 SIGTERM/SIGINT 處理）
  · 模型偵測前後皆推進心跳，避免長時間無新模型導致「假死」誤判
  · 統一 sys.path（透過 core.path_setup）
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401
from config import OUTPUT_DIR, MODEL_DIR

import json
import logging
import os
import signal
import subprocess
import time
from datetime import datetime

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
SCRIPTS_DIR    = Path(__file__).resolve().parent.parent
VENV_PYTHON    = os.environ.get("VENV_PYTHON",
                                "/home/hugo/project/stock_backend/venv/bin/python3")
PREDICT_SCRIPT = SCRIPTS_DIR / "training" / "predict.py"
HEARTBEAT_FILE = OUTPUT_DIR / "auto_predict.heartbeat"
PID_FILE       = OUTPUT_DIR / "auto_predict.pid"
LOG_FILE       = OUTPUT_DIR / "predict_manager.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 日誌設定
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AutoPredict")

# ─────────────────────────────────────────────
# DB heartbeat DDL（與 auto_train_manager 對稱）
# ─────────────────────────────────────────────
DDL_HEARTBEAT = """
CREATE TABLE IF NOT EXISTS auto_predict_heartbeat (
    ts              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pid             INTEGER,
    processed_count INTEGER,
    status          VARCHAR(20),
    note            TEXT,
    PRIMARY KEY (ts, pid)
);
CREATE INDEX IF NOT EXISTS idx_auto_predict_hb_ts
    ON auto_predict_heartbeat (ts DESC);
"""

INSERT_HEARTBEAT = """
INSERT INTO auto_predict_heartbeat (ts, pid, processed_count, status, note)
VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s)
"""


def _ensure_db_heartbeat_ddl() -> None:
    """首次啟動時建表（冪等）。失敗時靜默 — DB 不可用不應中斷推論主流程。"""
    try:
        from core.db_utils import get_db_conn, ensure_ddl
        conn = get_db_conn()
        try:
            ensure_ddl(conn, DDL_HEARTBEAT)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"DB heartbeat DDL 建立失敗（將僅使用檔案心跳）：{e}")


def _write_db_heartbeat(processed_count: int, status: str, note: str = "") -> None:
    """寫入 DB heartbeat。失敗時不阻斷主流程。"""
    try:
        from core.db_utils import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(INSERT_HEARTBEAT, (os.getpid(), processed_count, status, note))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # 已透過檔案保留，DB 失敗不需重試


# ─────────────────────────────────────────────
# 主邏輯
# ─────────────────────────────────────────────
def get_latest_models() -> dict[str, float]:
    """獲取目前所有模型及其修改時間。"""
    if not MODEL_DIR.exists():
        return {}
    return {
        f.stem.replace("ensemble_", ""): f.stat().st_mtime
        for f in MODEL_DIR.glob("ensemble_*.pkl")
    }


def write_heartbeat(processed_count: int, status: str = "RUNNING", note: str = "") -> None:
    """雙通道心跳：檔案 + DB。"""
    payload = {
        "ts": datetime.now().isoformat(),
        "pid": os.getpid(),
        "processed_count": processed_count,
        "status": status,
        "note": note,
    }
    try:
        HEARTBEAT_FILE.write_text(json.dumps(payload))
    except Exception as e:
        logger.warning(f"心跳檔案寫入失敗：{e}")
    _write_db_heartbeat(processed_count, status, note)


def _write_pid_file() -> None:
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception as e:
        logger.warning(f"PID 檔寫入失敗：{e}")


def _cleanup(*_args) -> None:
    """SIGTERM / SIGINT 收到時優雅退出。"""
    logger.info("收到終止訊號，正在優雅退出 …")
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass
    write_heartbeat(0, status="STOPPED", note="signal received")
    sys.exit(0)


def main():
    logger.info("============================================================")
    logger.info("🚀 Quantum Finance 自動預測管理器 v2 啟動")
    logger.info(f"監控目錄: {MODEL_DIR}")
    logger.info(f"心跳檔: {HEARTBEAT_FILE}")
    logger.info(f"PID 檔: {PID_FILE}")
    logger.info("============================================================")

    _ensure_db_heartbeat_ddl()
    _write_pid_file()
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    # 初始載入：記錄目前的狀態，避免啟動時對舊模型進行全量預測
    processed_mtimes = get_latest_models()
    processed_count = 0
    write_heartbeat(processed_count, note=f"startup, baseline={len(processed_mtimes)} models")

    while True:
        try:
            current_models = get_latest_models()
            for sid, mtime in current_models.items():
                if mtime <= processed_mtimes.get(sid, 0):
                    continue

                logger.info(f"🔥 [NEW MODEL] 偵測到標的 {sid} 模型更新，啟動自動推論...")
                try:
                    cmd = [VENV_PYTHON, str(PREDICT_SCRIPT), "--stock-id", sid]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ [{sid}] 預測報告產出完成。")
                        processed_count += 1
                    else:
                        logger.error(f"❌ [{sid}] 預測失敗！\nError: {result.stderr}")

                except subprocess.TimeoutExpired:
                    logger.error(f"⏰ [{sid}] 預測超時 (5min)")
                except Exception as e:
                    logger.error(f"💥 [{sid}] 執行推論時發生崩潰: {e}")

                processed_mtimes[sid] = mtime

            write_heartbeat(processed_count)

        except Exception as e:
            logger.error(f"⚠️ 核心循環發生異常: {e}")
            write_heartbeat(processed_count, status="ERROR", note=str(e)[:200])

        time.sleep(30)


if __name__ == "__main__":
    main()

import time
import subprocess
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

"""
auto_predict_manager.py — 自動預測管理器 (V4.0 Trinity)
======================================================
監控 scripts/outputs/models/ 目錄，一旦偵測到模型更新 (.pkl mtime 變動)，
立即觸發 predict.py 進行推論，實現「訓練-推論」自動化閉環。
"""

# 路徑設定
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR   = SCRIPTS_DIR / "outputs" / "models"
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
PREDICT_SCRIPT = SCRIPTS_DIR / "training" / "predict.py"
HEARTBEAT_FILE = SCRIPTS_DIR / "training" / "outputs" / "auto_predict.heartbeat"

# 日誌設定
LOG_FILE = SCRIPTS_DIR / "training" / "outputs" / "predict_manager.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutoPredict")

def get_latest_models():
    """獲取目前所有模型及其修改時間"""
    if not MODEL_DIR.exists(): return {}
    models = {}
    for f in MODEL_DIR.glob("ensemble_*.pkl"):
        sid = f.stem.replace("ensemble_", "")
        models[sid] = f.stat().st_mtime
    return models

def write_heartbeat(processed_count: int):
    """寫入心跳檔供 dashboard 讀取"""
    payload = {
        "ts": datetime.now().isoformat(),
        "pid": os.getpid(),
        "processed_count": processed_count,
        "status": "RUNNING"
    }
    HEARTBEAT_FILE.write_text(json.dumps(payload))

def main():
    logger.info("============================================================")
    logger.info("🚀 Quantum Finance 自動預測管理器啟動")
    logger.info(f"監控目錄: {MODEL_DIR}")
    logger.info("============================================================")

    # 初始載入：記錄目前的狀態，避免啟動時對舊模型進行全量預測
    processed_mtimes = get_latest_models()
    processed_count  = 0
    
    while True:
        try:
            current_models = get_latest_models()
            for sid, mtime in current_models.items():
                # 判斷條件：mtime 變動（代表模型剛被重訓完成覆蓋）
                if mtime > processed_mtimes.get(sid, 0):
                    logger.info(f"🔥 [NEW MODEL] 偵測到標的 {sid} 模型更新，啟動自動推論...")
                    
                    try:
                        # 執行 predict.py
                        cmd = [VENV_PYTHON, str(PREDICT_SCRIPT), "--stock-id", sid]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        
                        if result.returncode == 0:
                            logger.info(f"✅ [{sid}] 預測報告產出完成。")
                            processed_count += 1
                        else:
                            logger.error(f"❌ [{sid}] 預測失敗！\nError: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        logger.error(f"⏰ [{sid}] 預測超時 (5min)")
                    except Exception as e:
                        logger.error(f"💥 [{sid}] 執行推論時發生崩潰: {e}")
                    
                    # 更新紀錄
                    processed_mtimes[sid] = mtime
            
            write_heartbeat(processed_count)
            
        except Exception as e:
            logger.error(f"⚠️ 核心循環發生異常: {e}")
            
        time.sleep(30) # 每 30 秒檢查一次

if __name__ == "__main__":
    main()

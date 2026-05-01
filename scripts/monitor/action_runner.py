import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup paths
scripts_dir = Path(__file__).resolve().parent.parent
venv_python = str(scripts_dir.parent / "venv" / "bin" / "python3")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ActionRunner")

def run_command(cmd, name):
    logger.info(f"--- 🚀 開始執行: {name} ---")
    try:
        process = subprocess.run(cmd, check=True)
        logger.info(f"✅ {name} 成功完成。")
        return True
    except Exception as e:
        logger.error(f"❌ {name} 失敗: {e}")
        return False

def task_data():
    """全系統資料抓取並審計"""
    # 1. 抓取
    if run_command([venv_python, str(scripts_dir / "parallel_fetch.py")], "全系統資料並行抓取"):
        # 2. 審計
        run_command([venv_python, str(scripts_dir / "monitor" / "data_integrity_audit.py")], "資料完整度審計")

def task_model():
    """全系統模型運算並審計"""
    # 1. 訓練
    if run_command([venv_python, str(scripts_dir / "training" / "parallel_train.py")], "全系統模型並行訓練"):
        # 2. 審計
        run_command([venv_python, str(scripts_dir / "monitor" / "model_health_check.py")], "模型健康度檢查")

def task_predict():
    """全系統預測運算並審計"""
    # 1. 預測
    if run_command([venv_python, str(scripts_dir / "automate_daily.py")], "全系統預測生成"):
        # 2. 審計
        run_command([venv_python, str(scripts_dir / "monitor" / "data_integrity_audit.py")], "預測完整度審計")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python action_runner.py [data|model|predict]")
        sys.exit(1)
        
    task = sys.argv[1]
    if task == "data":
        task_data()
    elif task == "model":
        task_model()
    elif task == "predict":
        task_predict()
    else:
        logger.error(f"未知任務: {task}")

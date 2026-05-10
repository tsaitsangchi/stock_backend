import os
import time
import subprocess
import logging
import sys
import json
from datetime import datetime

# 修正路徑以確保能讀取到 config
sys.path.append(os.getcwd())
from scripts.config import STOCK_CONFIGS, TIER_1_STOCKS

# 設定
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
MAX_PARALLEL_TRAINS = 2  # 深層訓練耗能高，降低並行數
CHECK_INTERVAL = 60      # 每分鐘檢查一次
METRICS_REGISTRY = "scripts/outputs/metrics_registry.json"

# 脊椎標的 (Deep Mode Targets)
ANCHOR_STOCKS = ["2330", "2317", "2454"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scripts/outputs/manager.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_running_trains():
    """ 檢查目前正在執行的訓練進程及其 stock_id """
    try:
        output = subprocess.check_output(["ps", "aux"]).decode()
        lines = [l for l in output.splitlines() if "train_evaluate.py" in l and "grep" not in l]
        running_ids = []
        for l in lines:
            parts = l.split()
            for i, p in enumerate(parts):
                if p == "--stock-id" and i + 1 < len(parts):
                    running_ids.append(parts[i+1])
                elif "train_evaluate.py" in p and "--stock-id" not in l:
                    running_ids.append("2330")
        return list(set(running_ids))
    except Exception as e:
        logger.error(f"檢查進程失敗: {e}")
        return []

def get_performance_scores():
    """ 從註冊表讀取歷史 DA """
    scores = {}
    if os.path.exists(METRICS_REGISTRY):
        try:
            with open(METRICS_REGISTRY, "r") as f:
                data = json.load(f)
                for sid, metrics in data.items():
                    scores[sid] = metrics.get("directional_accuracy", 0.5)
        except Exception as e:
            logger.warning(f"讀取註冊表失敗: {e}")
    return scores

# --- 第六波賽道標的 (6th Wave Driver Stocks) ---
SIXTH_WAVE_DRIVERS = ["2330", "2454", "3661", "2376", "2382", "6669"]

def calculate_priority(sid, perf_scores):
    """ 計算優先權評分 (0~100) """
    # 1. 權值分 (70%)
    if sid in ANCHOR_STOCKS:
        weight_score = 100
    elif sid in TIER_1_STOCKS:
        weight_score = 80
    else:
        weight_score = 30
        
    # 2. 勝率分 (30%)
    da = perf_scores.get(sid, 0.5)
    da_score = min(100, max(0, (da - 0.45) / 0.15 * 100)) # 0.45->0, 0.6->100
    
    priority = (weight_score * 0.7) + (da_score * 0.3)
    
    # ── 2026 量子金融藍圖：康波導航加成 ───────────────────
    # 如果是第六波驅動標的，優先級提升 20%
    if sid in SIXTH_WAVE_DRIVERS:
        priority *= 1.2
        
    # 接近 2026 年時，對半導體賽道強制置頂
    if datetime.now().year >= 2025 and sid in SIXTH_WAVE_DRIVERS:
        priority += 20.0
        
    return min(100, priority)

def main():
    logger.info("=== 自動訓練管理員啟動 (第一性 80/20 動態權重版) ===")
    
    while True:
        try:
            running_ids = get_running_trains()
            perf_scores = get_performance_scores()
            
            # 排序標的：基於 Priority Score
            all_sids = list(STOCK_CONFIGS.keys())
            sid_priorities = {sid: calculate_priority(sid, perf_scores) for sid in all_sids}
            sorted_targets = sorted(all_sids, key=lambda x: sid_priorities[x], reverse=True)
            
            # 更新有效模型清單 (Tier 1 每週重訓, 其他每月)
            finished_ids = []
            now = time.time()
            model_dir = "scripts/outputs/models"
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith(".pkl") and "ensemble_" in f:
                        sid = f.replace("ensemble_", "").replace(".pkl", "")
                        mtime = os.path.getmtime(os.path.join(model_dir, f))
                        days_old = (now - mtime) / (24 * 3600)
                        limit = 7 if (sid in TIER_1_STOCKS) else 30
                        if days_old < limit:
                            finished_ids.append(sid)

            logger.info(f"執行中: {running_ids} | 有效模型: {len(finished_ids)} | 剩餘: {len(sorted_targets) - len(finished_ids)}")
            
            if len(running_ids) < MAX_PARALLEL_TRAINS:
                for sid in sorted_targets:
                    if sid not in finished_ids and sid not in running_ids:
                        # 決定模式
                        is_anchor = sid in ANCHOR_STOCKS
                        mode_str = "DEEP (141-Fold)" if is_anchor else "PARETO (60-Fold)"
                        logger.info(f">>> 啟動 {sid} ({STOCK_CONFIGS[sid]['name']}) | 模式: {mode_str} | 權重: {sid_priorities[sid]:.1f}")
                        
                        # 1. 更新特徵庫
                        logger.info(f"[{sid}] 正在更新特徵庫...")
                        subprocess.run([VENV_PYTHON, "scripts/update_feature_store.py", "--stock-id", sid])
                        
                        # 2. 啟動背景訓練
                        log_file = f"scripts/outputs/train_{sid}.log"
                        cmd = [VENV_PYTHON, "scripts/train_evaluate.py", "--stock-id", sid]
                        
                        # 如果是 Pareto 模式，傳入參數限制 Fold 數與特徵精煉
                        if not is_anchor:
                            cmd += ["--fast-mode"] 
                        
                        with open(log_file, "w") as f:
                            subprocess.Popen(cmd, stdout=f, stderr=f, start_new_session=True)
                        
                        logger.info(f"[{sid}] 任務已分發。")
                        time.sleep(10) # 深層任務啟動較慢
                        break 
        except Exception as e:
            logger.error(f"管理員循環發生錯誤: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

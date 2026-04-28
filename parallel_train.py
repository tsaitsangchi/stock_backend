"""
parallel_train.py — 全自動並行訓練管理器
同時執行 N 個訓練任務，確保在資源可控的情況下快速完成 50 支個股模型。
"""
import os
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import STOCK_CONFIGS, MODEL_DIR, LOG_DIR, BASE_DIR

# ── 設定區域 ───────────────────────────────────────────────────
MAX_WORKERS = 3  # 同時訓練 3 支股票 (已清理背景程序，可提升至 3)
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
TRAIN_SCRIPT = str(BASE_DIR / "train_evaluate.py")
# ──────────────────────────────────────────────────────────────

def train_one_stock(stock_id):
    stock_name = STOCK_CONFIGS[stock_id]['name']
    log_file = str(LOG_DIR / f"train_{stock_id}.log")
    
    print(f"開始訓練 (含 TFT): {stock_id} ({stock_name})...")
    
    # 1. 執行訓練
    cmd = [
        VENV_PYTHON, 
        TRAIN_SCRIPT, 
        "--stock-id", stock_id, 
        "--step-days", "63" 
    ]
    
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        
        # 2. [P1] 自動健康檢查
        # 只有訓練成功才執行，檢查 PSI 與 Drift
        health_script = str(BASE_DIR / "model_health_check.py")
        health_cmd = [VENV_PYTHON, health_script, "--stock-id", stock_id]
        
        with open(log_file, "a") as f:
            f.write("\n\n=== 自動健康檢查 (Post-Training Health Check) ===\n")
            subprocess.run(health_cmd, stdout=f, stderr=subprocess.STDOUT)
            
        return stock_id, True, None
    except Exception as e:
        return stock_id, False, str(e)

def main():
    # 1. 找出尚未訓練或需要更新的股票
    all_stocks = list(STOCK_CONFIGS.keys())
    to_train = []
    
    for sid in all_stocks:
        model_path = MODEL_DIR / f"ensemble_{sid}.pkl"
        if not model_path.exists():
            to_train.append(sid)
            
    if not to_train:
        print("所有個股模型皆已存在。")
        return

    print(f"總計需要訓練 {len(to_train)} 支個股，設定並行數為 {MAX_WORKERS}...")
    
    # 2. 開始並行任務
    start_time = time.time()
    results = {"success": [], "failed": []}
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(train_one_stock, sid): sid for sid in to_train}
        
        for future in as_completed(futures):
            sid, success, error = future.result()
            if success:
                print(f"✅ [完成] {sid}")
                results["success"].append(sid)
            else:
                print(f"❌ [失敗] {sid}: {error}")
                results["failed"].append(sid)
                
    end_time = time.time()
    duration = (end_time - start_time) / 3600
    
    print("\n" + "="*50)
    print(f"訓練任務結束！總耗時: {duration:.2f} 小時")
    print(f"成功: {len(results['success'])}")
    print(f"失敗: {len(results['failed'])}")
    if results["failed"]:
        print(f"失敗清單: {', '.join(results['failed'])}")
    print("="*50)

if __name__ == "__main__":
    main()

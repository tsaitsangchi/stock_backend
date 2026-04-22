"""
train_all_force.py — 強制重新訓練所有標的模型
"""
import os
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import STOCK_CONFIGS, MODEL_DIR

# ── 設定區域 ───────────────────────────────────────────────────
MAX_WORKERS = 3  # 同時訓練 3 支股票
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
TRAIN_SCRIPT = "scripts/train_evaluate.py"
# ──────────────────────────────────────────────────────────────

def train_one_stock(stock_id):
    stock_name = STOCK_CONFIGS[stock_id]['name']
    log_dir = Path("scripts/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_force_{stock_id}.log"
    
    print(f"🚀 [開始] 強制重新訓練: {stock_id} ({stock_name})...")
    
    # 執行 train_evaluate.py，啟動 TFT
    cmd = [
        VENV_PYTHON, 
        TRAIN_SCRIPT, 
        "--stock-id", stock_id, 
        "--step-days", "63" 
    ]
    
    start_t = time.time()
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        duration = time.time() - start_t
        return stock_id, True, duration, None
    except Exception as e:
        duration = time.time() - start_t
        return stock_id, False, duration, str(e)

def main():
    all_stocks = list(STOCK_CONFIGS.keys())
    
    print(f"🔥 啟動全標的強制重新訓練任務...")
    print(f"總計標的數: {len(all_stocks)}")
    print(f"設定並行數: {MAX_WORKERS}")
    print(f"預計耗時: 每標的約 10-30 分鐘，總量較大，建議背景執行。")
    
    start_time = time.time()
    results = {"success": [], "failed": []}
    
    from tqdm import tqdm
    pbar = tqdm(total=len(all_stocks), desc="Overall Progress", unit="stock")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(train_one_stock, sid): sid for sid in all_stocks}
        
        for future in as_completed(futures):
            sid, success, dur, error = future.result()
            if success:
                results["success"].append(sid)
            else:
                results["failed"].append(sid)
            pbar.update(1)
            pbar.set_postfix({"last": sid, "status": "OK" if success else "FAIL"})
                
    pbar.close()
                
    end_time = time.time()
    total_duration = (end_time - start_time) / 3600
    
    print("\n" + "="*60)
    print(f"🏁 任務結束！總耗時: {total_duration:.2f} 小時")
    print(f"成功數: {len(results['success'])}")
    print(f"失敗數: {len(results['failed'])}")
    if results["failed"]:
        print(f"失敗清單: {', '.join(results['failed'])}")
    print("="*60)

if __name__ == "__main__":
    main()

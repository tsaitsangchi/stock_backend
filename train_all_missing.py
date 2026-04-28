import os
import subprocess
from config import STOCK_CONFIGS

MODEL_DIR = "outputs/models"
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
PRIORITY_STOCKS = ["2308", "6669", "9958", "1795"]

def main():
    missing_stocks = []
    all_stocks = list(STOCK_CONFIGS.keys())
    
    # 優先排序
    ordered_stocks = [s for s in PRIORITY_STOCKS if s in all_stocks]
    ordered_stocks += [s for s in all_stocks if s not in PRIORITY_STOCKS]

    for stock_id in ordered_stocks:
        model_path = os.path.join(MODEL_DIR, f"ensemble_{stock_id}.pkl")
        if not os.path.exists(model_path):
            missing_stocks.append(stock_id)
    
    if not missing_stocks:
        print("所有個股模型皆已存在。")
        return

    print(f"發現尚未產生的模型：{', '.join(missing_stocks)}")
    
    for stock_id in missing_stocks:
        print(f"\n正在開始訓練 {stock_id} ({STOCK_CONFIGS[stock_id]['name']})...")
        cmd = [
            VENV_PYTHON, 
            "train_evaluate.py", 
            "--stock-id", stock_id, 
            "--step-days", "63"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ {stock_id} 訓練完成。")
        except subprocess.CalledProcessError as e:
            print(f"❌ {stock_id} 訓練失敗: {e}")

if __name__ == "__main__":
    main()

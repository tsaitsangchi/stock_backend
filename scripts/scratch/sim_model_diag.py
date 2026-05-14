
import sys
from pathlib import Path

# 模擬引導
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, ensure_infrastructure
    
    def simulate_model_diag():
        print("🚀 [模擬實驗] 啟動模型元數據診斷...")
        try:
            ensure_infrastructure()
            with db_transaction() as cur:
                # 測試模型資料表查詢
                cur.execute("SELECT count(*) FROM models")
                model_count = cur.fetchone()['count']
                
                print(f"✅ 模型資料表連線成功！")
                print(f"📊 目前已註冊模型數 : {model_count}")
                return True
        except Exception as e:
            # 如果連資料表都還沒建立，這就是一個診斷點
            print(f"❌ 模型診斷失敗 : {e}")
            return False

    if __name__ == "__main__":
        simulate_model_diag()

except Exception as e:
    print(f"❌ 引導失敗: {e}")

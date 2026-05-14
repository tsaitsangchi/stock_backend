
import sys, os
from pathlib import Path

# 模擬引導
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, ensure_infrastructure
    
    def simulate_db_diag():
        print("🚀 [模擬實驗] 啟動資料庫核心診斷...")
        try:
            # 測試連線與基礎設施自癒
            ensure_infrastructure()
            
            with db_transaction() as cur:
                # 測試查詢功能
                cur.execute("SELECT count(*) FROM stocks")
                stock_count = cur.fetchone()['count']
                print(f"✅ 資料庫連線正常！")
                print(f"📊 目前核心資產數量 : {stock_count}")
                return True
        except Exception as e:
            print(f"❌ 資料庫診斷失敗: {e}")
            return False

    if __name__ == "__main__":
        simulate_db_diag()

except Exception as e:
    print(f"❌ 引導失敗: {e}")

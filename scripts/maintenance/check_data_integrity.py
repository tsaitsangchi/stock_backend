"""
check_data_integrity.py v3.2 (Quantum Finance Edition)
================================================================================
數據顯微鏡 — 交易日曆感知版 (Quantum v5.2 標準)
負責偵測數據缺失，具備「動態交易日曆校準」技術，排除市場休市影響。

修訂歷程：
  v3.2 (2026-05-11): [升級] 實作「動態基準日曆」，挑戰全市場 PERFECT 稽核。
  v3.1 (2026-05-11): [修正] 修正全量標的選取邏輯。
  v3.0 (2026-05-11): [標準] 升級至 v5.2 標準。

【執行範例矩陣 (Integrity Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [極致精準全量稽核]        │ $ python scripts/maintenance/check_data_integrity.py   │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, argparse
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, record_lifecycle, get_db_stock_ids
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

def get_reference_calendar(table_name: str = "stock_price_day"):
    """從資料庫聯集中動態生成『基準交易日曆』"""
    with db_transaction() as cur:
        # 取出該表中所有出現過的唯一日期，作為市場有開門的證據
        cur.execute(f"SELECT DISTINCT date FROM {table_name} ORDER BY date ASC")
        return set(r['date'] for r in cur.fetchall())

def audit_stock_data(stock_id: str, ref_calendar: set, table_name: str = "stock_price_day"):
    """比對基準日曆，找出真正的數據缺失"""
    with db_transaction() as cur:
        # 1. 檢查重複數據
        cur.execute(f"SELECT date, COUNT(*) FROM {table_name} WHERE stock_id = %s GROUP BY date HAVING COUNT(*) > 1", (stock_id,))
        dupes = cur.fetchall()
        
        # 2. 獲取該個股的所有日期
        cur.execute(f"SELECT date FROM {table_name} WHERE stock_id = %s", (stock_id,))
        stock_dates = set(r['date'] for r in cur.fetchall())
        
        # 3. 差集運算：找出基準日曆中有，但該個股卻沒有的日期 (真正的 Gap)
        # 注意：只比對該個股『起始日期』之後的基準日曆
        if stock_dates:
            start_date = min(stock_dates)
            effective_ref = {d for d in ref_calendar if d >= start_date}
            missing_dates = effective_ref - stock_dates
        else:
            missing_dates = set()

        return {"dupes": len(dupes), "gaps": len(missing_dates), "total_rows": len(stock_dates)}

def run_integrity_audit(target_id: str = None, core_only: bool = False):
    """執行全域或局部數據稽核"""
    with record_lifecycle("data_integrity_audit", "maintenance", target_id or "SYSTEM"):
        print("\n" + "🎯"*40)
        print(f"🚀 Quantum Finance: 交易日曆感知稽核報告 (v3.2)")
        print("🎯"*40)
        
        # 1. 獲取標的
        is_core_param = True if core_only else None
        stock_ids = [target_id] if target_id else get_db_stock_ids(is_core=is_core_param)
        
        # 2. 生成基準日曆
        print(f"📡 正在校準動態交易日曆...")
        ref_calendar = get_reference_calendar()
        print(f"✅ 校準完成：偵測到全市場共 {len(ref_calendar)} 個有效交易日。")
        
        print(f"\n📊 正在稽核 {len(stock_ids)} 檔標的數據品質...")
        print("-" * 80)
        
        summary = {"perfect": 0, "has_gaps": 0, "has_dupes": 0}
        
        for sid in stock_ids:
            res = audit_stock_data(sid, ref_calendar)
            status = "✅ PERFECT"
            if res['dupes'] > 0:
                status = f"❌ DUPES ({res['dupes']})"
                summary['has_dupes'] += 1
            elif res['gaps'] > 0:
                status = f"⚠️  GAPS ({res['gaps']})"
                summary['has_gaps'] += 1
            else:
                summary['perfect'] += 1
                
            if target_id or res['gaps'] > 0 or res['dupes'] > 0:
                print(f"  📍 {sid.ljust(6)} | Rows: {str(res['total_rows']).rjust(5)} | 狀態: {status}")

        print("-" * 80)
        print(f"📈 總結: PERFECT: {summary['perfect']} | GAPS: {summary['has_gaps']} | DUPES: {summary['has_dupes']}")
        print("\n" + "🎯"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="稽核單一標的")
    parser.add_argument("--core", action="store_true", help="僅稽核核心標的")
    args = parser.parse_args()

    run_integrity_audit(target_id=args.id, core_only=args.core)

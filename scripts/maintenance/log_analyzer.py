"""
log_analyzer.py v1.0 (Quantum Finance Edition)
================================================================================
日誌診斷分析儀 — 可觀測性核心工具 (Quantum v5.2 標準)
負責聚合與分析 pipeline_execution_log，提供自動化維運簡報。

修訂歷程：
  v1.0 (2026-05-11): [首發] 實作生命週期聚合分析，支援今日任務概覽與性能瓶頸偵測。

【執行範例矩陣 (Analysis Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [查看今日執行簡報]        │ $ python scripts/maintenance/log_analyzer.py           │
│ 2. [查看特定類別 (Ingestion)]│ $ python scripts/maintenance/log_analyzer.py --cat ingestion│
│ 3. [分析最近 N 筆任務]       │ $ python scripts/maintenance/log_analyzer.py --last 100│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path
from datetime import datetime, date

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import db_transaction, record_lifecycle

def analyze_logs(category: str = None, limit: int = 1000):
    """執行日誌數據聚合與診斷"""
    with record_lifecycle("log_analysis_job", "maintenance", "SYSTEM"):
        sql = """
            SELECT 
                status,
                COUNT(*) as count,
                AVG(duration_ms) as avg_ms,
                SUM(rows_affected) as total_rows
            FROM pipeline_execution_log
            WHERE created_at >= CURRENT_DATE
        """
        params = []
        if category:
            sql += " AND category = %s"; params.append(category)
        
        sql += " GROUP BY status"

        with db_transaction() as cur:
            # 1. 概覽統計
            cur.execute(sql, params)
            summary = cur.fetchall()
            
            # 2. 瓶頸偵測 (Top 5 慢速任務)
            cur.execute("""
                SELECT task_name, stock_id, duration_ms 
                FROM pipeline_execution_log 
                WHERE created_at >= CURRENT_DATE 
                ORDER BY duration_ms DESC LIMIT 5
            """)
            slowest = cur.fetchall()

        # ── 視覺化輸出 ──
        print("\n" + "📊"*40)
        print(f"🚀 Quantum Finance: 日誌診斷簡報 ({date.today()})")
        print("📊"*40)
        
        if not summary:
            print("\n  📭 今日尚無執行紀錄。")
        else:
            print(f"\n📈 [執行概覽] {'(類別: ' + category + ')' if category else ''}")
            print("-" * 60)
            for row in summary:
                icon = "🟢" if row['status'] == "success" else "🔴"
                print(f"  {icon} 狀態: {row['status'].upper().ljust(8)} | 數量: {str(row['count']).rjust(4)} | 平均耗時: {int(row['avg_ms'])}ms")
            
            print(f"\n🐢 [性能瓶頸] (慢速 Top 5)")
            print("-" * 60)
            for i, task in enumerate(slowest, 1):
                print(f"  {i}. {task['task_name'].ljust(35)} | {task['stock_id'].ljust(6)} | {task['duration_ms']}ms")

        print("\n" + "📊"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", help="指定分析的任務類別 (如 ingestion, mlops)")
    parser.add_argument("--last", type=int, default=1000, help="分析最近的 N 筆紀錄")
    args = parser.parse_args()

    analyze_logs(category=args.cat, limit=args.last)

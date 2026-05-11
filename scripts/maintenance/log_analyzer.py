"""
log_analyzer.py v1.1 (Quantum Finance Edition)
================================================================================
生命週期性能診斷儀 (Quantum v5.2 標準)
負責分析 pipeline_execution_log，找出系統瓶頸與任務耗時分佈。

修訂歷程：
  v1.1 (2026-05-11): [標準] 升級至 v5.2 標準，補全分類診斷範例矩陣。

【執行範例矩陣 (Log Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統任務耗時分析]      │ $ python scripts/maintenance/log_analyzer.py           │
│ 2. [指定分類：性能統計]      │ $ python scripts/maintenance/log_analyzer.py --cat maintenance │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, record_lifecycle
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

def run_analysis(category=None):
    with record_lifecycle("log_performance_analysis", "maintenance", "SYSTEM"):
        print("\n" + "📊"*40)
        print(f"🚀 Quantum Finance: 生命週期性能分析 (v1.1)")
        print("📊"*40)
        
        with db_transaction() as cur:
            query = "SELECT task_name, AVG(duration_ms) as avg_time, COUNT(*) as calls FROM pipeline_execution_log"
            if category: query += f" WHERE category = '{category}'"
            query += " GROUP BY task_name ORDER BY avg_time DESC"
            
            cur.execute(query)
            for row in cur.fetchall():
                print(f"  📍 {row['task_name']:<25} | Avg: {int(row['avg_time']):>6} ms | Calls: {row['calls']:>4}")
        
        print("\n" + "📊"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", help="指定分析分類 (maintenance, ingestion, model_ops)")
    args = parser.parse_args()
    run_analysis(category=args.cat)

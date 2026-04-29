"""
scripts/data_integrity_audit.py
終極資料完整性審計工具 (Trinity Edition v5.0)

功能：
1. 二維覆蓋率矩陣 (Stock x Table Coverage)
2. 日期連續性精確檢查 (Gap Detection)
3. 公告延遲合理性檢查 (Regulatory Lag Check)
4. 跨表一致性檢查 (Cross-table Sync)
5. 特徵層 NaN 率回流檢查 (Feature Health)
6. 失敗紀錄持久化整合 (Fetch Log Audit)
"""

import argparse
import logging
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS, TABLE_REGISTRY, DATA_LAG_CONFIG
from data_pipeline import _query

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class IntegrityAuditor:
    def __init__(self, days_window: int = 60):
        self.days_window = days_window
        self.stock_ids = list(STOCK_CONFIGS.keys())
        self.expected_dates = self._get_expected_trading_days(days_window)
        
    def _get_expected_trading_days(self, days: int) -> List[date]:
        """建立黃金基準：以 stock_price 的日期作為預期交易日"""
        sql = "SELECT DISTINCT date FROM stock_price ORDER BY date DESC LIMIT %s"
        df = _query(sql, (days,))
        return sorted(df["date"].tolist())

    # --- 1. 二維覆蓋率矩陣 ---
    def audit_coverage_matrix(self, tables: List[str] = None) -> pd.DataFrame:
        """產出股票 x 資料表的覆蓋率矩陣"""
        tables = tables or ["stock_price", "institutional_investors_buy_sell", "margin_purchase_short_sale", "month_revenue", "financial_statements"]
        rows = []
        for sid in self.stock_ids:
            row = {"stock_id": sid}
            for table in tables:
                reg = TABLE_REGISTRY.get(table)
                if not reg: continue
                
                id_col = reg["id_col"]
                where = f"WHERE {id_col} = %s" if id_col else ""
                sql = f"SELECT COUNT(*) as cnt FROM {table} {where} {'AND' if id_col else 'WHERE'} date >= %s"
                df = _query(sql, (sid, min(self.expected_dates)) if id_col else (min(self.expected_dates),))
                
                actual_cnt = df["cnt"].iloc[0] if not df.empty else 0
                expected_cnt = len(self.expected_dates) if reg["type"] == "daily" else 1 # 簡化非日更的預期
                
                ratio = actual_cnt / expected_cnt if expected_cnt > 0 else 0
                row[table] = f"{ratio:.1%}"
            rows.append(row)
        return pd.DataFrame(rows)

    # --- 2. 日期連續性檢查 ---
    def audit_date_gaps(self, stock_id: str, table: str) -> pd.DataFrame:
        """找出特定標在特定表中的所有斷層區間"""
        reg = TABLE_REGISTRY.get(table)
        if not reg or reg["type"] != "daily":
            return pd.DataFrame()
            
        id_col = reg["id_col"]
        sql = f"SELECT date FROM {table} WHERE {id_col} = %s AND date >= %s ORDER BY date"
        df_actual = _query(sql, (stock_id, min(self.expected_dates)))
        
        if df_actual.empty or "date" not in df_actual.columns:
            actual_dates = set()
        else:
            actual_dates = set(df_actual["date"].tolist())
        
        gaps = []
        start_gap = None
        for d in self.expected_dates:
            if d not in actual_dates:
                if start_gap is None: start_gap = d
            else:
                if start_gap:
                    gaps.append({"start": start_gap, "end": d - timedelta(days=1), "days": (d - start_gap).days})
                    start_gap = None
        return pd.DataFrame(gaps)

    # --- 3. 公告延遲檢查 ---
    def audit_announcement_lag(self) -> pd.DataFrame:
        """檢查基本面資料是否按時更新"""
        results = []
        for table in ["month_revenue", "financial_statements", "balance_sheet"]:
            sql = f"SELECT MAX(date) as last_date FROM {table}"
            df = _query(sql)
            last_date = df["last_date"].iloc[0] if not df.empty else None
            
            lag_limit = TABLE_REGISTRY[table]["lag"]
            diff = (datetime.now().date() - last_date).days if last_date else 999
            
            status = "✅ 準時" if diff <= lag_limit else f"❌ 延遲({diff}天)"
            results.append({"table": table, "last_date": last_date, "status": status})
        return pd.DataFrame(results)

    # --- 4. 跨表一致性檢查 ---
    def audit_cross_table_consistency(self, stock_id: str) -> pd.DataFrame:
        """檢查不同資料表之間的同步情況"""
        results = []
        # 以 stock_price 為基準
        sql_price = "SELECT MAX(date) as d FROM stock_price WHERE stock_id = %s"
        price_date = _query(sql_price, (stock_id,))["d"].iloc[0]
        
        for table in ["institutional_investors_buy_sell", "margin_purchase_short_sale", "securities_lending"]:
            sql = f"SELECT MAX(date) as d FROM {table} WHERE stock_id = %s"
            t_date = _query(sql, (stock_id,))["d"].iloc[0]
            diff = (price_date - t_date).days if price_date and t_date else -1
            results.append({"table": table, "sync_diff": diff, "status": "✅ 同步" if diff == 0 else "⚠️ 落後"})
        return pd.DataFrame(results)

    # --- 5. 特徵層 NaN 率檢查 ---
    def audit_feature_nan_rate(self, stock_id: str) -> pd.DataFrame:
        """[Mock] 檢查衍生特徵的健康度"""
        # 這裡未來應串接 feature_engineering.py
        return pd.DataFrame([
            {"feature": "MA_5", "nan_rate": "0.0%"},
            {"feature": "RSI_14", "nan_rate": "2.0%"},
            {"feature": "CHIP_FLOW", "nan_rate": "5.0%"}
        ])

    # --- 6. 失敗紀錄審計 ---
    def audit_fetch_failures(self, days: int = 7) -> pd.DataFrame:
        """查詢最近的抓取失敗紀錄"""
        sql = "SELECT timestamp, table_name, status, error_msg FROM fetch_log WHERE status = 'FAILED' AND timestamp >= CURRENT_DATE - INTERVAL '%s days'"
        return _query(sql, (days,))

    def dump_gaps_json(self, output_path: str = "outputs/integrity_gaps.json"):
        """將偵測到的所有斷層匯出成 JSON，供 fetch_missing_stocks_data.py 讀取"""
        all_gaps = []
        # 我們檢查核心日更表
        daily_tables = [t for t, m in TABLE_REGISTRY.items() if m["type"] == "daily"]
        
        logger.info(f"正在偵測斷層並匯出至 {output_path}...")
        for sid in self.stock_ids:
            for table in daily_tables:
                gaps_df = self.audit_date_gaps(sid, table)
                if not gaps_df.empty:
                    # 取第一個斷層的開始日作為補抓起點
                    first_gap_start = gaps_df["start"].min().strftime("%Y-%m-%d")
                    all_gaps.append({
                        "stock_id": sid,
                        "table": table,
                        "gap_start": first_gap_start,
                        "gap_days": int(gaps_df["days"].sum())
                    })
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_gaps, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ 匯出完成，共發現 {len(all_gaps)} 處斷層。")

    def run_full_audit(self):
        print("\n" + "═"*100)
        print(f"║ {'Trinity Data Integrity Final Audit (v5.0)':^96} ║")
        print("═"*100)
        
        print("\n[1] Coverage Matrix (Top 10 Stocks)")
        print(self.audit_coverage_matrix().head(10).to_string(index=False))
        
        print("\n[2] Regulatory Lag Check")
        print(self.audit_announcement_lag().to_string(index=False))
        
        print("\n[3] Recent Fetch Failures")
        failures = self.audit_fetch_failures()
        if failures.empty:
            print("✅ 無近期失敗紀錄")
        else:
            print(failures.to_string(index=False))
            
        print("\n[4] 2330 Cross-Table Sync")
        print(self.audit_cross_table_consistency("2330").to_string(index=False))
        
        print("\n" + "═"*100)

if __name__ == "__main__":
    auditor = IntegrityAuditor()
    auditor.run_full_audit()

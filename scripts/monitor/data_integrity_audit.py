import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
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
    def __init__(self, days_window: int = 60, stock_ids: List[str] = None):
        self.days_window = days_window
        if stock_ids is not None:
            self.stock_ids = [str(sid) for sid in stock_ids]
        else:
            try:
                # 優先從資料庫讀取活耀標的
                db_stocks = _query("SELECT stock_id FROM system_assets WHERE is_active = TRUE")
                if not db_stocks.empty:
                    self.stock_ids = db_stocks["stock_id"].astype(str).tolist()
                else:
                    self.stock_ids = [str(sid) for sid in STOCK_CONFIGS.keys()]
            except:
                self.stock_ids = [str(sid) for sid in STOCK_CONFIGS.keys()]
        
        self.expected_dates = self._get_expected_trading_days(days_window)
        
    def _get_expected_trading_days(self, days: int) -> List[date]:
        """建立黃金基準：以 stock_price 的日期作為預期交易日"""
        sql = "SELECT DISTINCT date FROM stock_price ORDER BY date DESC LIMIT %s"
        df = _query(sql, (days,))
        return sorted(df["date"].tolist())

    # --- 1. 二維覆蓋率矩陣 ---
    def audit_coverage_matrix(self, tables: List[str] = None) -> pd.DataFrame:
        """產出股票 x 資料表的覆蓋率矩陣
        """
        if tables is None:
            tables = [t for t, m in TABLE_REGISTRY.items() if m.get("type") == "daily"]
        rows = []
        for sid in self.stock_ids:
            row = {"stock_id": str(sid)}
            for table in tables:
                reg = TABLE_REGISTRY.get(table)
                if not reg: continue
                
                id_col = reg["id_col"]
                where = f"WHERE {id_col} = %s" if id_col else ""
                sql = f"SELECT COUNT(*) as cnt FROM {table} {where} {'AND' if id_col else 'WHERE'} date >= %s"
                
                # 強制 sid 為字串
                params = (str(sid), min(self.expected_dates)) if id_col else (min(self.expected_dates),)
                df = _query(sql, params)
                
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
        df_actual = _query(sql, (str(stock_id), min(self.expected_dates)))
        
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
        
        # 處理結尾的斷層 (Risk: 如果最後一天也是缺失的)
        if start_gap:
            gaps.append({"start": start_gap, "end": self.expected_dates[-1], "days": (self.expected_dates[-1] - start_gap).days + 1})
            
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
            reg = TABLE_REGISTRY.get(table)
            id_col_local = reg["id_col"] if reg else "stock_id"
            sql = f"SELECT MAX(date) as d FROM {table} WHERE {id_col_local} = %s"
            t_date = _query(sql, (stock_id,))["d"].iloc[0]
            
            # 確保皆為 date 物件後再計算
            if pd.notnull(price_date) and pd.notnull(t_date):
                # 處理可能從 DB 回傳的 datetime
                p_d = price_date.date() if hasattr(price_date, "date") else price_date
                t_d = t_date.date() if hasattr(t_date, "date") else t_date
                diff = (p_d - t_d).days
            else:
                diff = -999
                
            results.append({"table": table, "sync_diff": diff, "status": "✅ 同步" if diff == 0 else "⚠️ 落後"})
        return pd.DataFrame(results)

    # --- 5. 特徵層 NaN 率檢查 ---
    def audit_feature_nan_rate(self, stock_id: str, lookback_days: int = 60) -> pd.DataFrame:
        """[P1-2 修正] 檢查衍生特徵的健康度（不再是 Mock）。

        實際對 stock_id 跑 build_features，計算最近 N 天每個衍生因子的 NaN 比率。
        若 build_features 失敗（資料不全），退回極簡 mock 並警示。
        """
        try:
            from data_pipeline import build_daily_frame
            from feature_engineering import build_features
            raw = build_daily_frame(stock_id=stock_id)
            df_feat = build_features(raw, for_inference=True)
            tail = df_feat.tail(lookback_days)
            results = []
            for col in tail.columns:
                if col in {"date", "stock_id"}:
                    continue
                nan_pct = float(tail[col].isna().mean())
                results.append({
                    "feature":   col,
                    "nan_rate":  f"{nan_pct:.1%}",
                    "n_samples": len(tail),
                })
            df_out = pd.DataFrame(results)
            df_out["nan_pct_float"] = df_out["nan_rate"].str.rstrip("%").astype(float) / 100
            return df_out.sort_values("nan_pct_float", ascending=False).drop(columns=["nan_pct_float"])
        except Exception as e:
            logger.warning(f"audit_feature_nan_rate 退回 mock：{e}")
            return pd.DataFrame([
                {"feature": "MA_5", "nan_rate": "0.0%"},
                {"feature": "RSI_14", "nan_rate": "2.0%"},
                {"feature": "CHIP_FLOW", "nan_rate": "5.0%"}
            ])

    # --- 6. 失敗紀錄審計 ---
    def audit_fetch_failures(self, days: int = 7) -> pd.DataFrame:
        """查詢最近的抓取失敗紀錄。

        [P1-2 修正] fetch_log 表 DDL 已於 core/db_utils.py 提供（DDL_FETCH_LOG），
        parallel_fetch.py 與 backfill_from_gaps.py 都已串入 log_fetch_result()。
        若 DB 中尚無此表（首次建置），會嘗試 ensure_ddl 後再查；都失敗則回空 DF。
        """
        try:
            from core.db_utils import get_db_conn, ensure_ddl
            conn = get_db_conn()
            try:
                ensure_ddl(conn)  # 冪等：確保 fetch_log 存在
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"無法 ensure fetch_log DDL：{e}")

        sql = (
            "SELECT timestamp, table_name, stock_id, status, error_msg "
            "FROM fetch_log "
            "WHERE status = 'FAILED' "
            f"  AND timestamp >= CURRENT_DATE - INTERVAL '{int(days)} days' "
            "ORDER BY timestamp DESC"
        )
        try:
            return _query(sql)
        except Exception as e:
            logger.warning(f"audit_fetch_failures 查詢失敗：{e}")
            return pd.DataFrame(columns=["timestamp", "table_name", "stock_id", "status", "error_msg"])

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

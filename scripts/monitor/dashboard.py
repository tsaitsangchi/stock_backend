import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Setup paths - dashboard is in scripts/monitor/
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
project_root = scripts_dir.parent

for sub in ["fetchers", "pipeline", "training", "monitor"]:
    sys.path.append(str(scripts_dir / sub))
sys.path.append(str(scripts_dir))
sys.path.append(str(project_root))

from config import STOCK_CONFIGS, MODEL_DIR
from data_pipeline import _query, load_exchange_rate
from data_integrity_audit import IntegrityAuditor
from portfolio_optimizer import PortfolioOptimizer

# 頁面配置
st.set_page_config(
    page_title="量子藍圖 — 系統監控",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 套用 Premium 風格 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 資料載入邏輯 (Cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_system_stocks():
    """從資料庫讀取 150 檔核心標的清單"""
    try:
        df = _query("SELECT stock_id, name, industry, tags FROM system_assets WHERE is_active = TRUE")
        if df.empty:
            return pd.DataFrame([{"stock_id": k, "name": v["name"], "industry": v["industry"]} for k, v in STOCK_CONFIGS.items()])
        df["stock_id"] = df["stock_id"].astype(str)
        return df
    except:
        return pd.DataFrame([{"stock_id": k, "name": v["name"], "industry": v["industry"]} for k, v in STOCK_CONFIGS.items()])

@st.cache_data(ttl=5)
def load_integrity_matrix(stock_ids):
    auditor = IntegrityAuditor(days_window=60, stock_ids=stock_ids)
    df = auditor.audit_coverage_matrix()
    if not df.empty: df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=5)
def load_model_status(stock_ids):
    from model_health_check import check_model_files_df
    return check_model_files_df(stock_ids)

@st.cache_data(ttl=5)
def load_performance_da(stock_ids):
    from model_health_check import evaluate_recent_performance_df
    return evaluate_recent_performance_df(stock_ids)

@st.cache_data(ttl=5)
def load_health_matrix_db():
    query = """
    SELECT h.*, s.name, s.industry 
    FROM system_health_matrix h
    JOIN system_assets s ON h.stock_id = s.stock_id
    WHERE h.audit_date = (SELECT MAX(audit_date) FROM system_health_matrix)
    """
    df = _query(query)
    if not df.empty: df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=5)
def load_recommendations_db():
    query = """
    SELECT r.*, s.name, s.industry
    FROM investment_recommendations r
    JOIN system_assets s ON r.stock_id = s.stock_id
    WHERE r.prediction_date = (SELECT MAX(prediction_date) FROM investment_recommendations)
    """
    df = _query(query)
    if not df.empty: df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=60)
def load_today_predictions():
    sql = "SELECT stock_id, prob_up FROM stock_forecast_daily WHERE date = (SELECT MAX(date) FROM stock_forecast_daily) AND day_offset = 30"
    return _query(sql)

@st.cache_data(ttl=60)
def load_trade_ledger():
    return _query("SELECT * FROM trade_ledger WHERE status = 'OPEN'")

# ─────────────────────────────────────────────
# 側欄與核心資料載入
# ─────────────────────────────────────────────
assets_df = load_system_stocks()
all_stock_ids = assets_df["stock_id"].tolist()
fresh_df = load_integrity_matrix(all_stock_ids)
model_df = load_model_status(all_stock_ids)
perf_df = load_performance_da(all_stock_ids)
pred_df = load_today_predictions()

st.sidebar.subheader("系統狀態")
price_date = _query("SELECT MAX(date) FROM stock_price").iloc[0,0]
st.sidebar.metric("市場資料日期", str(price_date) if price_date else "N/A")
st.sidebar.metric("監控標的總數", len(all_stock_ids))

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────
st.title("量子藍圖 — 系統監控儀表板 (Trinity v5.0)")

# 第一排：指標
col1, col2, col3, col4 = st.columns(4)
with col1:
    if not fresh_df.empty and len(fresh_df.columns) > 1:
        # 使用 applymap 進行元素級轉型（處理 % 字串）
        numeric_df = fresh_df.iloc[:, 1:].applymap(lambda x: float(str(x).rstrip('%')) if isinstance(x, str) else float(x))
        avg_integrity = numeric_df.mean().mean() / 100
    else:
        avg_integrity = 0
    st.metric("1. 資料完整度", f"{avg_integrity:.1%}")
with col2:
    trained_ratio = (model_df["status"].str.contains("OK")).sum() / len(model_df) if len(model_df)>0 else 0
    st.metric("2. 模型訓練完整度", f"{trained_ratio:.1%}")
with col3:
    pred_count = len(pred_df) if not pred_df.empty else 0
    st.metric("3. 預測完整度", f"{pred_count / len(all_stock_ids):.1%}")
with col4:
    avg_da = perf_df["da"].mean() if not perf_df.empty else 0
    st.metric("平均 30D 準確率", f"{avg_da:.1%}")

st.markdown("---")

# 第二排：矩陣
tab1, tab2, tab3 = st.tabs(["💎 專業投資交易帳本", "🛡️ 全系統健康度矩陣", "🚀 全系統投資建議矩陣"])

with tab1:
    st.subheader("💎 專業投資交易帳本 (DB-Backed)")
    ledger_df = load_trade_ledger()
    if ledger_df.empty:
        st.info("💡 交易帳本目前為空。")
    else:
        matrix = ledger_df.copy()
        matrix["股票"] = matrix["stock_id"].astype(str).apply(lambda x: f"{x} {assets_df[assets_df['stock_id']==x]['name'].iloc[0] if x in assets_df['stock_id'].values else ''}")
        st.dataframe(matrix[["股票", "shares", "entry_price", "entry_date"]], use_container_width=True)

with tab2:
    st.subheader("🛡️ 全系統健康度矩陣 (DB-Backed)")
    health_db_df = load_health_matrix_db()
    if health_db_df.empty:
        st.warning("⚠️ 資料庫中尚無健康度審計紀錄，請執行 `sync_trinity_db.py`。")
    else:
        health_db_df["股票"] = health_db_df["stock_id"].astype(str) + " " + health_db_df["name"]
        def style_health(val):
            if "🟢" in str(val) or (isinstance(val, float) and val > 0.9): return 'color: #2ea043'
            if "🟡" in str(val) or (isinstance(val, float) and val > 0.7): return 'color: #d29922'
            if "🔴" in str(val) or (isinstance(val, float) and val <= 0.7): return 'color: #f85149'
            return ''
        st.dataframe(health_db_df[["股票", "industry", "data_coverage_pct", "model_status", "prediction_status", "last_updated_at"]].style.map(style_health), use_container_width=True)

with tab3:
    st.subheader("🚀 全系統投資建議矩陣 (DB-Backed)")
    recomm_df = load_recommendations_db()
    if recomm_df.empty:
        st.warning("⚠️ 目前尚無投資建議紀錄。")
    else:
        recomm_df["股票"] = recomm_df["stock_id"].astype(str) + " " + recomm_df["name"]
        st.dataframe(recomm_df[["股票", "industry", "signal_level", "prob_up", "recommended_weight", "prediction_date"]], use_container_width=True)

st.markdown("---")
st.subheader("📋 系統日誌導航 (Log Navigator)")
log_files = {
    "自動訓練管理器": scripts_dir / "training" / "outputs" / "manager.log",
    "數據完整度審計": scripts_dir / "outputs" / "audit_data_integrity.log",
}
selected_log_name = st.selectbox("選擇日誌檔案", list(log_files.keys()))
if log_files[selected_log_name].exists():
    with open(log_files[selected_log_name], "r") as f:
        st.code("".join(f.readlines()[-100:]), language="text")

st.markdown("---")
st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")

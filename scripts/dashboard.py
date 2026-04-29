"""
scripts/dashboard.py
量子藍圖 — 系統監控儀表板 v1.0

使用方式：
    streamlit run scripts/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# 注入路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import STOCK_CONFIGS, MODEL_DIR
from data_pipeline import _query
from data_integrity_audit import IntegrityAuditor

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
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 資料載入邏輯 (Cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_integrity_matrix():
    auditor = IntegrityAuditor(days_window=60)
    return auditor.audit_coverage_matrix()

@st.cache_data(ttl=600)
def load_lag_report():
    auditor = IntegrityAuditor()
    return auditor.audit_announcement_lag()

@st.cache_data(ttl=600)
def load_model_status():
    from model_health_check import check_model_files_df
    return check_model_files_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=600)
def load_performance_da():
    from model_health_check import evaluate_recent_performance_df
    return evaluate_recent_performance_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=600)
def load_psi_drift():
    from model_health_check import check_prediction_drift_df
    return check_prediction_drift_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=300)
def load_today_predictions():
    sql = """
        WITH latest_date AS (SELECT MAX(predict_date) FROM stock_forecast_daily)
        SELECT stock_id, prob_up, ensemble_price, current_close, confidence_level, warning_flag
        FROM stock_forecast_daily
        WHERE predict_date = (SELECT * FROM latest_date)
          AND day_offset = 30
    """
    df = _query(sql)
    if not df.empty:
        df["Expected Return"] = (df["ensemble_price"] / df["current_close"] - 1) * 100
    return df

@st.cache_data(ttl=60)
def load_training_failures():
    """模擬讀取管理器失敗日誌。"""
    try:
        from auto_train_manager import FAILURE_TRACKER
        return FAILURE_TRACKER
    except:
        return {}

# ─────────────────────────────────────────────
# Sidebar 側欄
# ─────────────────────────────────────────────

st.sidebar.title("🛡️ Antigravity Quant")
st.sidebar.info("核心模型監控系統 v1.0")
if st.sidebar.button("🔄 重新整理數據"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("系統狀態")
fresh_df = load_integrity_matrix()
# 從第一欄或已知主表獲取日期 (這裏簡化處理)
price_date = _query("SELECT MAX(date) FROM stock_price").iloc[0,0]
st.sidebar.metric("市場資料日期", str(price_date) if price_date else "N/A")

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

st.title("量子藍圖 — 系統監控儀表板")

# 第一排：系統三位一體指標 (Trinity Integrity Metrics)
col1, col2, col3, col4 = st.columns(4)

with col1:
    # 1. 資料完整度 (Data Integrity)
    avg_integrity = fresh_df.iloc[:, 1:].apply(lambda x: x.str.rstrip('%').astype(float)).mean().mean() / 100
    st.metric("1. 資料完整度", f"{avg_integrity:.1%}", 
              delta="核心資料庫" if avg_integrity > 0.9 else "需補件",
              delta_color="normal" if avg_integrity > 0.9 else "inverse")

with col2:
    # 2. 模型訓練完整度 (Model Training)
    model_df = load_model_status()
    trained_ratio = (model_df["status"] == "🟢 OK").sum() / len(model_df)
    st.metric("2. 模型訓練完整度", f"{trained_ratio:.1%}", 
              delta=f"已就緒 {(model_df['status'] == '🟢 OK').sum()} 檔")

with col3:
    # 3. 預測完整度 (Prediction)
    pred_df = load_today_predictions()
    pred_count = len(pred_df) if not pred_df.empty else 0
    total_stocks = len(STOCK_CONFIGS)
    pred_ratio = pred_count / total_stocks
    st.metric("3. 預測完整度", f"{pred_ratio:.1%}", 
              delta=f"今日生成 {pred_count} 檔")

with col4:
    # 4. 系統綜合效能 (Accuracy/Drift)
    perf_df = load_performance_da()
    avg_da = perf_df["da"].mean()
    st.metric("平均 30D 準確率 (DA)", f"{avg_da:.1%}", 
              delta=f"{avg_da-0.5:+.1%}" if avg_da else None)

st.markdown("---")

# 第二排：即時監控細節
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📊 模型健康度報告 (Performance & Drift)")
    # 合併效能與漂移數據
    health_report = pd.merge(perf_df, drift_df, on="stock_id")
    health_report = pd.merge(health_report, model_df[["stock_id", "status", "age_days"]], on="stock_id")
    
    # 重新排列欄位
    health_report = health_report[["stock_id", "status", "da", "psi", "perf_status", "drift_status", "age_days"]]
    
    # 樣式處理
    def color_status(val):
        if "🟢" in str(val) or "STABLE" in str(val) or "EXCELLENT" in str(val): color = '#2ea043'
        elif "🟡" in str(val) or "WARNING" in str(val): color = '#d29922'
        elif "🔴" in str(val) or "DEGRADED" in str(val) or "DRIFTED" in str(val): color = '#f85149'
        else: color = 'white'
        return f'color: {color}'

    st.dataframe(
        health_report.style.map(color_status, subset=['status', 'perf_status', 'drift_status'])
                           .background_gradient(subset=['da'], cmap="RdYlGn", vmin=0.45, vmax=0.65),
        use_container_width=True,
        height=400
    )

with right_col:
    st.subheader("⏳ 數據完整性矩陣")
    st.markdown("<small>評分範圍 0.0 ~ 1.0 (1.0 代表 30D 資料全滿)</small>", unsafe_allow_html=True)
    st.dataframe(
        fresh_df.head(10),
        use_container_width=True,
        height=300
    )
    
    st.subheader("⚠️ 需注意清單")
    critical = health_report[(health_report["da"] < 0.52) | (health_report["psi"] > 0.15) | (health_report["status"] != "🟢 OK")]
    if not critical.empty:
        for _, row in critical.iterrows():
            st.warning(f"**{row['stock_id']}**: DA={row['da']:.1%}, PSI={row['psi']:.2f}, Status={row['status']}")
    else:
        st.success("目前無異常標的")

st.markdown("---")

# 第三排：詳細分析與審計
tab1, tab2, tab3 = st.tabs(["📊 預測分佈 (Distribution)", "🚨 異常與警示 (Anomalies)", "🛡️ 資料審計 (Integrity Audit)"])

with tab1:
    if not pred_df.empty:
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            fig = px.histogram(pred_df, x="prob_up", nbins=20, 
                               title="上漲機率分佈 (Prob_up Distribution)",
                               color_discrete_sequence=['#58a6ff'])
            fig.add_vline(x=0.5, line_dash="dash", line_color="red")
            fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with dist_col2:
            top_picks = pred_df.sort_values("prob_up", ascending=False).head(10)
            top_picks["Name"] = top_picks["stock_id"].map(lambda x: STOCK_CONFIGS.get(x, {}).get("name", "Unknown"))
            fig_bar = px.bar(top_picks, x="prob_up", y="Name", orientation='h',
                             title="今日高信心標的 (Top Prob_up)",
                             color="prob_up", color_continuous_scale="RdYlGn")
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("今日尚未生成推論數據。")

with tab2:
    st.subheader("🚩 偵測到異常的預測 (Outliers)")
    if not pred_df.empty:
        anomalies = pred_df[pred_df["warning_flag"].notna() & (pred_df["warning_flag"] != "")]
        if not anomalies.empty:
            st.warning(f"偵測到 {len(anomalies)} 筆預測異常！")
            st.dataframe(anomalies[["stock_id", "prob_up", "Expected Return", "warning_flag"]], use_container_width=True)
        else:
            st.success("✅ 今日所有推論結果皆通過統計常理性檢查。")
    
with tab3:
    st.header("Trinity 資料完整性深度審計 (v5.0)")
    audit_col1, audit_col2 = st.columns([1, 1])
    
    auditor = IntegrityAuditor(days_window=60)
    
    with audit_col1:
        st.subheader("📌 公告延遲監控 (Regulatory Lag)")
        st.table(load_lag_report())
        
        st.subheader("🔄 2330 跨表一致性")
        st.dataframe(auditor.audit_cross_table_consistency("2330"), use_container_width=True)
        
    with audit_col2:
        st.subheader("📂 完整覆蓋率矩陣 (60D)")
        st.dataframe(fresh_df, use_container_width=True, height=400)
        
    st.markdown("---")
    st.subheader("🕵️‍♂️ 斷層診斷 (Gap Diagnostic)")
    target_sid = st.selectbox("選擇要診斷的標的", list(STOCK_CONFIGS.keys()))
    target_table = st.selectbox("選擇資料表", ["stock_price", "institutional_investors_buy_sell", "margin_purchase_short_sale", "securities_lending"])
    
    gaps = auditor.audit_date_gaps(target_sid, target_table)
    if gaps.empty:
        st.success(f"✅ {target_sid} 在 {target_table} 中日期完全連續。")
    else:
        st.error(f"🚩 偵測到 {len(gaps)} 個時間斷層！")
        st.dataframe(gaps, use_container_width=True)

st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")

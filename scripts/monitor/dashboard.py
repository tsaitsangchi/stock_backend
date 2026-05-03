import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import subprocess
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
    SELECT 
        s.stock_id, s.name, s.industry,
        COALESCE(r.signal_level, '⚪ 尚未產出') as signal_level,
        COALESCE(r.prob_up, 0) as prob_up,
        COALESCE(r.recommended_weight, 0) as recommended_weight,
        r.prediction_date
    FROM system_assets s
    LEFT JOIN (
        SELECT * FROM investment_recommendations 
        WHERE prediction_date = (SELECT MAX(prediction_date) FROM investment_recommendations)
    ) r ON s.stock_id = r.stock_id
    WHERE s.is_active = TRUE
    ORDER BY r.prob_up DESC NULLS LAST, s.stock_id ASC
    """
    df = _query(query)
    if not df.empty: df["stock_id"] = df["stock_id"].astype(str)
    return df

@st.cache_data(ttl=60)
def load_prediction_path(stock_id):
    """載入指定個股未來 30 天的預測路徑"""
    sql = f"""
    SELECT day_offset, prob_up, pred_close, pred_ret 
    FROM stock_forecast_daily 
    WHERE stock_id = '{stock_id}' 
      AND date = (SELECT MAX(date) FROM stock_forecast_daily WHERE stock_id = '{stock_id}')
    ORDER BY day_offset ASC
    """
    return _query(sql)

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
recomm_df = load_recommendations_db()
all_stock_ids = recomm_df["stock_id"].tolist() # SQL 已按 prob_up DESC 排序
fresh_df = load_integrity_matrix(all_stock_ids)
model_df = load_model_status(all_stock_ids)
perf_df = load_performance_da(all_stock_ids)
pred_df = load_today_predictions()

st.sidebar.subheader("系統監控")
price_date = _query("SELECT MAX(date) FROM stock_price").iloc[0,0]
st.sidebar.metric("最後資料日期", str(price_date) if price_date else "無資料")
st.sidebar.metric("核心標的總數", len(all_stock_ids))

# ─────────────────────────────────────────────
# [新增] 系統維護中心 (功能鈕)
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ 系統維護中心")

def run_action(task_name, cmd_arg):
    try:
        runner_path = current_dir / "action_runner.py"
        venv_python = str(project_root / "venv" / "bin" / "python3")
        # 使用 Popen 以免阻塞 UI，日誌會寫入 action_runner.log
        subprocess.Popen([venv_python, str(runner_path), cmd_arg])
        st.sidebar.success(f"🚀 {task_name} 已啟動！")
        st.sidebar.caption("請至下方日誌導覽查看進度。")
    except Exception as e:
        st.sidebar.error(f"❌ 啟動失敗: {e}")

if st.sidebar.button("📊 全系統資料完整與審計"):
    run_action("資料審計", "data")

if st.sidebar.button("🧪 全系統超參數重計與審計"):
    run_action("超參數重計", "tune")

if st.sidebar.button("🤖 全系統模型訓練完整與審計"):
    run_action("模型訓練審計", "model")

if st.sidebar.button("🎯 全系統預測股價完整與審計"):
    run_action("預測股價審計", "predict")

st.sidebar.markdown("---")

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────
st.title("量子藍圖 — 全系統監控儀表板 (Trinity v5.0)")

# 第一排：關鍵指標
col1, col2, col3, col4 = st.columns(4)
with col1:
    if not fresh_df.empty and len(fresh_df.columns) > 1:
        numeric_df = fresh_df.iloc[:, 1:].applymap(lambda x: float(str(x).rstrip('%')) if isinstance(x, str) else float(x))
        avg_integrity = numeric_df.mean().mean() / 100
    else:
        avg_integrity = 0
    st.metric("📊 資料完整度", f"{avg_integrity:.1%}")
with col2:
    trained_ratio = (model_df["status"].str.contains("OK")).sum() / len(model_df) if len(model_df)>0 else 0
    st.metric("🤖 模型訓練率", f"{trained_ratio:.1%}")
with col3:
    pred_count = len(pred_df) if not pred_df.empty else 0
    st.metric("🎯 預測達成率", f"{pred_count / len(all_stock_ids):.1%}")
with col4:
    avg_da = perf_df["da"].mean() if not perf_df.empty else 0
    st.metric("📈 平均 30D 準確率", f"{avg_da:.1%}")

st.markdown("---")

# 第二排：核心矩陣
tab1, tab2, tab3, tab4 = st.tabs(["💎 投資交易帳本", "🛡️ 健康度矩陣", "🚀 投資建議矩陣", "📈 預測路徑分析"])

with tab1:
    st.subheader("💎 專業投資交易帳本 (資料庫驅動)")
    ledger_df = load_trade_ledger()
    if ledger_df.empty:
        st.info("💡 目前尚無未平倉合約。")
    else:
        matrix = ledger_df.copy()
        matrix["股票"] = matrix["stock_id"].astype(str).apply(lambda x: f"{x} {assets_df[assets_df['stock_id']==x]['name'].iloc[0] if x in assets_df['stock_id'].values else ''}")
        # 重新命名欄位
        matrix = matrix.rename(columns={
            "shares": "持有股數",
            "entry_price": "進場成本",
            "entry_date": "進場日期"
        })
        st.dataframe(matrix[["股票", "持有股數", "進場成本", "進場日期"]], use_container_width=True)

with tab2:
    st.subheader("🛡️ 全系統健康度矩陣 (資料庫驅動)")
    health_db_df = load_health_matrix_db()
    if health_db_df.empty:
        st.warning("⚠️ 資料庫中尚無紀錄，請執行 `sync_trinity_db.py`。")
    else:
        health_db_df["股票"] = health_db_df["stock_id"].astype(str) + " " + health_db_df["name"]
        health_db_df = health_db_df.rename(columns={
            "industry": "產業分組",
            "data_coverage_pct": "資料覆蓋率",
            "model_status": "模型狀態",
            "prediction_status": "預測狀態",
            "last_updated_at": "最後更新時間"
        })
        def style_health(val):
            if "🟢" in str(val) or (isinstance(val, float) and val > 90): return 'color: #2ea043'
            if "🟡" in str(val) or (isinstance(val, float) and val > 70): return 'color: #d29922'
            if "🔴" in str(val) or (isinstance(val, float) and val <= 70): return 'color: #f85149'
            return ''
        st.dataframe(health_db_df[["股票", "產業分組", "資料覆蓋率", "模型狀態", "預測狀態", "最後更新時間"]].style.map(style_health), use_container_width=True)

with tab3:
    st.subheader("🚀 全系統投資建議矩陣 (資料庫驅動)")
    
    # [新增] 投資金額輸入與分配邏輯
    col_amt, col_info = st.columns([1, 3])
    with col_amt:
        total_inv = st.number_input("💸 預計投入總金額 (TWD)", min_value=0, value=100000, step=10000)
    with col_info:
        st.info("💡 依據「系統核心思想」：資金將優先分配給訊號最強的前 3 支標的。若無買進訊號則保留現金。")

    recomm_df = recomm_df # 使用頂層預載入的資料
    if recomm_df.empty:
        st.warning("⚠️ 目前尚無預測產出的投資建議。")
    else:
        # 獲取最新價格以便計算股數
        try:
            prices = _query("SELECT stock_id, close FROM stock_price WHERE date = (SELECT MAX(date) FROM stock_price)")
            prices["stock_id"] = prices["stock_id"].astype(str)
            recomm_df = recomm_df.merge(prices, on="stock_id", how="left")
        except:
            recomm_df["close"] = 0.0

        recomm_df["股票"] = recomm_df["stock_id"].astype(str) + " " + recomm_df["name"]
        
        # 篩選前三強標的 (限 買進/強力買進)
        buy_signals = ["🟢 強力買進", "🟡 買進"]
        top_3 = recomm_df[recomm_df["signal_level"].isin(buy_signals)].sort_values("prob_up", ascending=False).head(3).copy()
        
        if top_3.empty:
            st.success("✅ 目前無強力買進標的，建議：**全數保留現金 (100% Cash)**")
        else:
            # 權重分配 (依據 prob_up 相對比例)
            total_prob = top_3["prob_up"].sum()
            top_3["分配比例"] = top_3["prob_up"] / total_prob
            top_3["分配金額"] = (top_3["分配比例"] * total_inv).round(0)
            top_3["建議股數"] = (top_3["分配金額"] / top_3["close"]).replace([np.inf, -np.inf], 0).fillna(0).astype(int)
            
            st.markdown("#### 🎯 核心配置建議 (Top 3)")
            display_cols = ["股票", "signal_level", "prob_up", "close", "分配比例", "分配金額", "建議股數"]
            display_df = top_3[display_cols].rename(columns={
                "signal_level": "訊號強度",
                "prob_up": "看漲機率",
                "close": "目前價格",
                "分配比例": "分配比重",
            })
            st.dataframe(display_df.style.format({
                "分配比重": "{:.1%}",
                "看漲機率": "{:.1%}",
                "分配金額": "{:,.0f}",
                "建議股數": "{:,.0f}"
            }), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 全標的掃描清單")
        recomm_df_disp = recomm_df.rename(columns={
            "industry": "產業分組",
            "signal_level": "訊號強度",
            "prob_up": "看漲機率",
            "recommended_weight": "建議權重",
            "prediction_date": "預測日期"
        })
        st.dataframe(recomm_df_disp[["股票", "產業分組", "訊號強度", "看漲機率", "建議權重", "預測日期"]], use_container_width=True)

with tab4:
    st.subheader("📈 個股 30 日預測路徑分析")
    
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        target_sid = st.selectbox("選擇標的代號", all_stock_ids, format_func=lambda x: f"{x} {assets_df[assets_df['stock_id']==x]['name'].iloc[0] if x in assets_df['stock_id'].values else ''}")
    
    path_df = load_prediction_path(target_sid)
    
    if path_df.empty:
        st.warning(f"⚠️ 找不到 {target_sid} 的近期預測路徑資料。請確保已執行 `auto_predict_manager.py`。")
    else:
        # 視覺化
        st.markdown(f"#### 🔮 {target_sid} 未來 30 交易日趨勢演變")
        
        # 準備繪圖資料
        chart_data = path_df.set_index("day_offset")[["prob_up"]]
        chart_data.columns = ["看漲機率 (Probability)"]
        
        st.line_chart(chart_data)
        
        # 指標摘要
        m1, m2, m3 = st.columns(3)
        # 找出機率最高的那一天
        peak_idx = path_df["prob_up"].idxmax()
        peak_day = path_df.loc[peak_idx, "day_offset"]
        peak_prob = path_df.loc[peak_idx, "prob_up"]
        
        with m1:
            st.metric("🔥 機率爆發點", f"第 {int(peak_day)} 天", f"{peak_prob:.1%}")
        with m2:
            avg_prob = path_df["prob_up"].mean()
            st.metric("📊 平均看漲強度", f"{avg_prob:.1%}")
        with m3:
            trend = "穩定向上" if path_df["prob_up"].iloc[-1] > path_df["prob_up"].iloc[0] else "震盪回檔"
            st.metric("📈 趨勢評級", trend)
        
        # 若有預測價格，展示額外圖表
        if "pred_ret" in path_df.columns and not path_df["pred_ret"].isnull().all():
            st.markdown("#### 💰 預計累積報酬率曲線 (%)")
            ret_data = path_df.set_index("day_offset")[["pred_ret"]].copy() * 100
            ret_data.columns = ["累計預期報酬 (%)"]
            st.area_chart(ret_data)

st.markdown("---")
st.subheader("📋 系統日誌導覽 (Log Navigator)")
log_files = {
    "🛠️ 維護中心執行日誌": scripts_dir / "outputs" / "action_runner.log",
    "🤖 自動訓練管理員": scripts_dir / "outputs" / "manager.log",
    "🎯 自動預測管理器": scripts_dir / "outputs" / "predict_manager.log",
    "📊 數據完整度審計": scripts_dir / "outputs" / "audit_data_integrity.log",
}
selected_log_name = st.selectbox("選擇欲查看的日誌", list(log_files.keys()))
if log_files[selected_log_name].exists():
    with open(log_files[selected_log_name], "r") as f:
        st.code("".join(f.readlines()[-100:]), language="text")
st.markdown("---")
st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")

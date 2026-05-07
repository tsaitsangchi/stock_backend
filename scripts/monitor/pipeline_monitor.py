"""
pipeline_monitor.py — 資料管線健康監控儀表板 v3.1
=======================================================
直接讀取 fetch_log 資料表，即時展示所有抓取任務的健康狀態。

啟動方式：
    streamlit run scripts/monitor/pipeline_monitor.py --server.port 8501
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ── 路徑設定 ──
_current = Path(__file__).resolve().parent
_scripts = _current.parent
_root    = _scripts.parent
sys.path.insert(0, str(_scripts))
sys.path.insert(0, str(_root))

from core.db_utils import get_db_conn

# ── 頁面配置 ──
st.set_page_config(
    page_title="量子藍圖 — 管線監控",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium 暗色主題 CSS ──
st.markdown("""
<style>
body, .main { background-color: #0d1117 !important; }
h1, h2, h3 { color: #58a6ff !important; }
[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 18px;
}
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
.status-badge-green { color: #3fb950; font-weight: bold; }
.status-badge-red   { color: #f85149; font-weight: bold; }
.status-badge-gray  { color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ── 資料載入（快取 30 秒） ──
@st.cache_data(ttl=30)
def load_fetch_log(hours: int) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        df = pd.read_sql(f"""
            SELECT
                run_ts AT TIME ZONE 'Asia/Taipei' AS run_ts,
                table_name,
                stock_id,
                status,
                rows_inserted,
                duration_ms,
                fetch_date_from,
                fetch_date_to,
                error_message,
                cli_args
            FROM fetch_log
            WHERE run_ts > NOW() - INTERVAL '{hours} hours'
            ORDER BY run_ts DESC
        """, conn)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=30)
def load_summary(hours: int) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        df = pd.read_sql(f"""
            SELECT
                table_name,
                COUNT(*) FILTER (WHERE status = 'success')      AS success_cnt,
                COUNT(*) FILTER (WHERE status = 'failed')        AS failed_cnt,
                COUNT(*) FILTER (WHERE status = 'no_new_data')   AS no_new_cnt,
                SUM(rows_inserted)                               AS total_rows,
                ROUND(AVG(duration_ms) / 1000.0, 2)             AS avg_sec,
                ROUND(MAX(duration_ms) / 1000.0, 2)             AS max_sec,
                MAX(run_ts AT TIME ZONE 'Asia/Taipei')           AS last_run
            FROM fetch_log
            WHERE run_ts > NOW() - INTERVAL '{hours} hours'
            GROUP BY table_name
            ORDER BY last_run DESC
        """, conn)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=30)
def load_api_quota() -> dict:
    conn = get_db_conn()
    try:
        df = pd.read_sql("SELECT used_count, quota_limit FROM api_quota_log ORDER BY checked_at DESC LIMIT 1", conn)
        if not df.empty:
            return {"used": int(df.iloc[0]["used_count"]), "limit": int(df.iloc[0]["quota_limit"])}
    except:
        pass
    finally:
        conn.close()
    return {"used": None, "limit": 6000}


# ════════════════════════════════════════════════
# 側欄
# ════════════════════════════════════════════════
st.sidebar.title("📡 管線監控中心")
hours = st.sidebar.slider("📅 時間範圍（小時）", min_value=1, max_value=168, value=24, step=1)
st.sidebar.caption(f"顯示最近 {hours} 小時的執行記錄")

if st.sidebar.button("🔄 立即刷新"):
    st.cache_data.clear()
    st.rerun()

quota = load_api_quota()
if quota["used"] is not None:
    quota_pct = quota["used"] / quota["limit"]
    quota_color = "🟢" if quota_pct < 0.6 else ("🟡" if quota_pct < 0.85 else "🔴")
    st.sidebar.metric(f"{quota_color} FinMind API 配額",
                      f"{quota['used']:,} / {quota['limit']:,}",
                      f"剩餘 {quota['limit'] - quota['used']:,}")

st.sidebar.markdown("---")
st.sidebar.caption(f"🕐 最後刷新：{datetime.now().strftime('%H:%M:%S')}")


# ════════════════════════════════════════════════
# 主頁標題
# ════════════════════════════════════════════════
st.title("📡 量子藍圖 — 資料管線健康儀表板 v3.1")
st.caption(f"資料來源：`fetch_log` 資料表 ｜ 時間範圍：最近 {hours} 小時 ｜ 自動刷新：30 秒")

summary_df = load_summary(hours)
raw_df     = load_fetch_log(hours)

if summary_df.empty:
    st.warning("⚠️ 此時間範圍內尚無 fetch_log 記錄。請先執行抓取腳本。")
    st.stop()


# ════════════════════════════════════════════════
# KPI 指標列
# ════════════════════════════════════════════════
total_tasks   = len(summary_df)
healthy_tasks = (summary_df["failed_cnt"] == 0).sum()
total_failed  = summary_df["failed_cnt"].sum()
total_rows    = summary_df["total_rows"].sum()
total_runs    = (summary_df["success_cnt"] + summary_df["failed_cnt"] + summary_df["no_new_cnt"]).sum()
success_rate  = (summary_df["success_cnt"].sum() + summary_df["no_new_cnt"].sum()) / total_runs if total_runs else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🛡️ 健康資料集",  f"{healthy_tasks} / {total_tasks}")
col2.metric("🔴 失敗次數",    int(total_failed), delta=None if total_failed == 0 else f"-{int(total_failed)}", delta_color="inverse")
col3.metric("✅ 成功率",      f"{success_rate:.1%}")
col4.metric("📦 總寫入筆數",  f"{int(total_rows):,}")
col5.metric("🔢 總執行次數",  f"{int(total_runs):,}")

st.markdown("---")


# ════════════════════════════════════════════════
# Tab 分頁
# ════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🟢 資料集健康總覽", "🔴 失敗記錄", "📋 完整執行日誌"])

# ── Tab 1：健康總覽 ──
with tab1:
    st.subheader(f"各資料集健康狀態（最近 {hours} 小時）")

    display = summary_df.copy()
    display["健康狀態"] = display.apply(
        lambda r: "🟢 HEALTHY" if r["failed_cnt"] == 0 and r["success_cnt"] > 0
                  else ("🔴 FAILED"  if r["failed_cnt"] > 0
                  else  "⚪ IDLE"),
        axis=1
    )
    display["last_run"] = pd.to_datetime(display["last_run"]).dt.strftime("%m-%d %H:%M")

    display = display.rename(columns={
        "table_name": "資料集 / 腳本",
        "success_cnt": "✅ 成功",
        "failed_cnt":  "🔴 失敗",
        "no_new_cnt":  "⚪ 無新資料",
        "total_rows":  "📦 寫入筆數",
        "avg_sec":     "⏱ 平均耗時(s)",
        "max_sec":     "🔝 最長耗時(s)",
        "last_run":    "🕐 最後同步",
    })

    st.dataframe(
        display[["資料集 / 腳本", "健康狀態", "✅ 成功", "🔴 失敗", "⚪ 無新資料",
                  "📦 寫入筆數", "⏱ 平均耗時(s)", "🕐 最後同步"]],
        use_container_width=True,
        height=600,
    )

    # 耗時排行榜
    st.subheader("⏱ 耗時 Top 10（瓶頸分析）")
    top_slow = summary_df.nlargest(10, "avg_sec")[["table_name", "avg_sec", "max_sec"]].copy()
    top_slow.columns = ["資料集", "平均耗時(s)", "最長耗時(s)"]
    st.bar_chart(top_slow.set_index("資料集")["平均耗時(s)"])


# ── Tab 2：失敗記錄 ──
with tab2:
    st.subheader("🔴 失敗記錄詳情")
    failed_df = raw_df[raw_df["status"] == "failed"].copy()
    if failed_df.empty:
        st.success("✅ 此時間範圍內無任何失敗記錄！管線運作正常。")
    else:
        failed_df["run_ts"] = pd.to_datetime(failed_df["run_ts"]).dt.strftime("%m-%d %H:%M:%S")
        st.error(f"⚠️ 共發現 {len(failed_df)} 筆失敗記錄")
        st.dataframe(
            failed_df[["run_ts", "table_name", "stock_id", "fetch_date_from", "fetch_date_to", "error_message"]].rename(columns={
                "run_ts": "時間", "table_name": "資料集", "stock_id": "股票代號",
                "fetch_date_from": "起始日", "fetch_date_to": "結束日", "error_message": "錯誤訊息"
            }),
            use_container_width=True,
        )


# ── Tab 3：完整日誌 ──
with tab3:
    st.subheader("📋 完整執行日誌")

    # 篩選器
    c1, c2, c3 = st.columns(3)
    with c1:
        tbl_filter = st.multiselect("篩選資料集", options=sorted(raw_df["table_name"].unique().tolist()))
    with c2:
        status_filter = st.multiselect("篩選狀態", options=["success", "failed", "no_new_data"], default=["success", "failed"])
    with c3:
        limit = st.number_input("顯示筆數", min_value=50, max_value=5000, value=200, step=50)

    filtered = raw_df.copy()
    if tbl_filter:
        filtered = filtered[filtered["table_name"].isin(tbl_filter)]
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]

    filtered = filtered.head(int(limit))
    filtered["run_ts"] = pd.to_datetime(filtered["run_ts"]).dt.strftime("%m-%d %H:%M:%S")
    filtered["duration_ms"] = filtered["duration_ms"].fillna(0).astype(int)

    st.dataframe(
        filtered[["run_ts", "table_name", "stock_id", "status", "rows_inserted",
                  "duration_ms", "fetch_date_from", "fetch_date_to"]].rename(columns={
            "run_ts": "時間", "table_name": "資料集", "stock_id": "股票代號",
            "status": "狀態", "rows_inserted": "寫入筆數",
            "duration_ms": "耗時(ms)", "fetch_date_from": "起始日", "fetch_date_to": "結束日"
        }),
        use_container_width=True,
        height=500,
    )

st.markdown("---")
st.caption("Quantum Blueprint — Pipeline Monitor v3.1 © 2026")

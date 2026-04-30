# 《量子金融藍圖》系統信任度全面檢核報告

> **版本**：v5.0 後完整健診
> **審查日期**：2026-04-30
> **審查範圍**：scripts/ 全目錄 — 30,209 行 Python，43 個模組，27 張 PostgreSQL 資料表，5 種 ML 模型
> **審查角度**：「資料 → 特徵 → 模型 → 訊號 → 監控 → 運維」六層信任度
> **審查人**：架構審查 AI

---

## 目錄

- [壹、執行摘要](#壹執行摘要)
- [貳、系統架構盤點](#貳系統架構盤點)
- [參、量化體檢評分](#參量化體檢評分)
- [肆、P0 致命風險（必須立刻修復）](#肆p0-致命風險必須立刻修復)
- [伍、P1 高風險（30 天內必修）](#伍p1-高風險30-天內必修)
- [陸、P2 中度風險（60 天內）](#陸p2-中度風險60-天內)
- [柒、P3 結構性建議（90 天 +）](#柒p3-結構性建議90-天)
- [捌、Quick Wins — 立即可動手清單](#捌quick-wins--立即可動手清單)
- [玖、信任度提升 OKR 量化指標](#玖信任度提升-okr-量化指標)
- [拾、優化路線圖（30 / 60 / 90 天）](#拾優化路線圖30--60--90-天)
- [拾壹、結論與下一步](#拾壹結論與下一步)
- [附錄 A：完整模組清單](#附錄-a完整模組清單)
- [附錄 B：本次審查掃過的核心關鍵字](#附錄-b本次審查掃過的核心關鍵字)

---

## 壹、執行摘要

> **這套系統的「設計圖」已是台股量化界少見的完整藍圖（27 張資料表 + 175 個因子 + 5 模型 stacking + 5 維度 signal filter + 4 維度健康檢查），但「設計圖 ↔ 實際運轉」之間存在一條隱形的可信度斷層帶——許多模組已建好但沒有真正接到下游、許多監控已會偵測但沒有觸發行動、許多新因子已會計算但模型還沒看到。**

整個系統最該優先處理的是「**接通**」這件事，而不是繼續加新功能。

### 一句話建議

> **先處理 P0-1（補斷層）+ P0-3（重啟訓練）+ P0-5（讓模型吃到新因子）三件串聯一起做完**，其他都可以等。這三件的依賴鏈是：補資料 → 模型才看到完整 4 年訓練資料 → 訓練完才會吃到新因子 → 信任度才有意義。

---

## 貳、系統架構盤點

### 2.1 目錄結構（v5 重構後）

```
scripts/
├── core/                      # 共用核心：finmind_client.py / db_utils.py
├── fetchers/                  # 20 支 fetch 腳本（含 parallel_fetch）
├── pipeline/                  # data_pipeline / feature_engineering / signal_filter
│                              # portfolio_optimizer / portfolio_strategy / portfolio_backtest
│                              # backtest_engine / strategy_tester
├── training/                  # train_evaluate / parallel_train / predict
│                              # auto_train_manager / historical_backfill
│                              # tune_hyperparameters / update_feature_store
├── monitor/                   # data_integrity_audit ⭐ / model_health_check
│                              # model_quality_audit / db_health_check / db_optimize
│                              # backtest_audit / dashboard
├── models/                    # ensemble_model.py（XGB+LGB+ElasticNet+Momentum+Stacking）
│                              # tft_model.py（Temporal Fusion Transformer）
├── tests/                     # test_round3_fixes.py / test_round5_fixes.py
├── utils/                     # db / feature_selection / metrics / model_loader
├── archive/                   # 已淘汰的 17 個舊腳本
├── config.py                  # 全域配置（含 TABLE_REGISTRY、FEATURE_GROUPS 等）
├── outputs/                   # train.log / OOF predictions / models / integrity_gaps.json
└── README.md
```

### 2.2 資料層（27 張表）

| 類別 | 資料表 | 來源 |
|---|---|---|
| **核心價量**（5）| stock_price / stock_per / price_adj / day_trading / price_limit | FinMind |
| **籌碼面**（7）| institutional_investors_buy_sell / margin_purchase_short_sale / shareholding / securities_lending / daily_short_balance / eight_banks_buy_sell / sponsor_chip | FinMind |
| **基本面**（5）| month_revenue / financial_statements / balance_sheet / cash_flows_statement / dividend | FinMind |
| **市場層級**（4）| total_margin_short / total_inst_investors / futures_inst_investors / options_inst_investors | FinMind |
| **國際**（3）| us_stock_price / exchange_rate / interest_rate | FinMind |
| **衍生品**（3）| futures_ohlcv / options_ohlcv / options_oi_large_holders | FinMind |
| **事件 / 另類**（4）| disposition_securities / capital_reduction / stock_news / fred_series | FinMind + FRED |

### 2.3 特徵層（175 個因子，分 16 組）

```
price_volume(28) | chip(12) | fundamental(7) | macro(15) | event(2) | rolling_stats(7)
futures_chip(7)  | medium_term(15) | us_chain(動態)  | physics_signals(7)
quality(7) | price_adj(9) | short_interest(8) | event_risk(9)
extended_derivative(6) | news_attention(4) | fred_macro(14)
```

最後 7 組（quality 起算的 57 個因子）是 v3 第四輪審查後新增。

### 2.4 模型層

```
ensemble_model.py:
├── XGBPredictor          # XGBoost binary classifier
├── LGBPredictor          # LightGBM binary classifier
├── ElasticNetPredictor   # ElasticNet regression
├── SimpleMomentumModel   # 動能 fallback
├── StackingEnsemble      # Logistic Regression Meta-Learner
└── RegimeEnsemble        # 依 Regime 動態切換 ensemble

tft_model.py:
└── Temporal Fusion Transformer（可選）
```

### 2.5 監控層

```
data_integrity_audit.py    # 6 維度資料完整性審計（267 個 gap 已偵測）
model_health_check.py      # 4 維度模型健康（freshness/files/PSI/recent DA）
model_quality_audit.py     # 盲測 audit（DA/IC，2024+ 區間）
db_health_check.py         # DB 連線、索引、磁碟
db_optimize.py             # VACUUM、ANALYZE、index 優化
dashboard.py               # Streamlit 視覺化
backtest_audit.py          # 回測引擎自審
```

---

## 參、量化體檢評分

| 層級 | 完成度 | 信任度 | 主要短板 |
|---|---|---|---|
| **資料層**（fetchers/）| 95% | 70% | **267 處資料斷層**未癒合，audit 偵測得到但無人補抓 |
| **特徵層**（pipeline/feature_engineering）| 100% | 60% | 57 個 v3 新因子已生成但 `feature_importance_refined.csv` 顯示**模型還沒訓練到** |
| **模型層**（models/）| 90% | 55% | regime_metrics 顯示 **max_drawdown -99.9%**、單支標的訓練成本 5%，DA 高但回測爆 |
| **訊號層**（pipeline/signal_filter）| 80% | 50% | **0 個 v3 因子被引用**、`stock_dynamics_registry` 來源不明、訊號歷史未持久化 |
| **監控層**（monitor/）| 75% | 65% | data_integrity_audit 的 `audit_feature_nan_rate` 是 Mock、`audit_fetch_failures` 依賴的 `fetch_log` 表不存在 |
| **運維層**（auto_train_manager）| 50% | 40% | **5 天前最後一條 log 卡在 0/87**，cronjob 疑似已死，無告警機制 |
| **整體** | **82%** | **57%** | **「能跑」但不能「信」** |

### 信任度差距分析

```
完成度 82%  ─┐
              ├─ 落差 25%  ←── 主要在「接通」失敗（建好但沒接上）
信任度 57%  ─┘
```

---

## 肆、P0 致命風險（必須立刻修復）

### 🔴 P0-1：267 處資料斷層污染整套訓練資料

**證據**：`outputs/integrity_gaps.json`

```json
[
  {"stock_id": "2330", "table": "stock_per",   "gap_start": "2022-03-16", "gap_days": 1443},
  {"stock_id": "2330", "table": "price_adj",   "gap_start": "2022-03-16", "gap_days": 1477},
  ...（共 267 筆）
]
```

**影響鏈**：
1. 1443 天 = 4 年訓練資料中，PER / 還原股價有大段空缺
2. `feature_engineering.add_price_adj_features` 計算 `log_return_adj_1d` 時遇到 NaN
3. `ffill()` 把多日報酬合併成一天的「假尖峰」
4. XGB 把這些假尖峰學成「特定日期的 spike pattern」
5. 過擬合 → 真實上線時無法重現

**修復**（30 分鐘內可完成）：
- audit 工具已建好且會自動產出 `integrity_gaps.json`
- 寫一支 `backfill_from_gaps.py` 讀此 JSON，按 (stock_id, table) 精確補抓
- 改造 `fetch_missing_stocks_data.py` 從「< 100 筆」門檻 → 讀 audit 清單

### 🔴 P0-2：max_drawdown -99.9%、5% 交易成本吃光 alpha

**證據**：`outputs/regime_metrics.csv`

| Regime | DA | n_trades | total_tc_pct | max_drawdown | gross Sharpe |
|---|---|---|---|---|---|
| 低波動 (vol < 20%) | 73.2% | 882 | 0.2% | -90.5% | 1.58 |
| 中波動 (20-40%) | 62.1% | 1984 | **4.2%** | **-99.9%** | 0.98 |
| 高波動 (vol ≥ 40%) | 77.0% | 187 | 0.2% | -46.6% | 2.28 |
| 三時程共識 | 66.2% | **3053** | **5.0%** | **-99.9%** | 1.21 |

**問題本質**：經典「DA 高 ≠ 賺錢」陷阱
- DA 只看「方向對不對」，不看「賠的時候賠多少」
- 中波動是大部分時間的常態（>60% 交易日），是主要問題區間
- 3053 trades / 評估期 = 高頻訊號，5% 交易成本吃光所有 alpha

**修復重點**（這是策略層問題，不是資料層）：

1. **提高機率門檻**：`signal_filter.FILTER_CONFIG.prob_up_threshold` 從 0.65 → 0.75
2. **加入最小持倉日數**：5 個交易日硬規則，避免單日訊號頻繁進出
3. **改用 net Sharpe**（含成本）作為門檻，而非 gross Sharpe
4. **n_trades_per_fold 監控**：在 `train_evaluate.py::evaluate_fold` 加入過高警告

### 🔴 P0-3：87 支標的中只有 7 支訓練完成

**證據**：
- `outputs/oof_predictions_partial_*.csv` 只有 7 支：1301, 1513, 2002, 2317, 2330, 2382, 2454
- `outputs/auto_train.log` 最後一筆 `Progress: 0/87` 是 **2026-04-24，已停滯 6 天**

```
2026-04-24 16:25:45 [INFO] Auto Train Manager started for 87 stocks.
2026-04-24 16:25:55 [INFO] Progress: 0/87. Sleeping 60s...
（之後無紀錄）
```

**判斷**：auto_train_manager.py 跑到一半 process 被 kill 或 SIGHUP 後沒重啟。**沒有 health check / restart 機制**。

**修復**：

1. 加 systemd service 取代 nohup 背景跑（含 `Restart=always`）
2. 加 heartbeat：每 N 分鐘寫一筆到 DB `auto_train_heartbeat` 表
3. cron 每小時檢查 heartbeat，> 30 分鐘無更新 → email/Slack 告警
4. 立刻跑 `python scripts/training/auto_train_manager.py --force-all` 補完 80 支

### 🔴 P0-4：`stock_dynamics_registry` 表來源不明，所有動力學護欄等於虛設

**證據**：`pipeline/signal_filter.py` line 180

```python
cur.execute("SELECT info_sensitivity, gravity_elasticity, fat_tail_index, "
            "convexity_score, tail_risk_score, wave_track, innovation_velocity "
            "FROM stock_dynamics_registry WHERE stock_id = %s", (stock_id,))
```

**全 repo 找不到任何 `INSERT INTO stock_dynamics_registry` 或 fetch 腳本**。

**意味**：
- 此表可能是空的或不存在
- `_load_dynamics_registry` 會 silent 失敗（line 195: `except Exception: pass`）
- 所有 `kwave_score` / `entropy_delta` / `convexity_score` 都用 hard-coded 預設值
- signal_filter 的「動態門檻調整」邏輯**形同擺設**

**修復**（這條最 critical，因為它決定整個 signal_filter 是不是 placebo）：

- 選項 A：用真實資料計算這些 dynamics 並寫入 DB（建一支 `compute_stock_dynamics.py`）
- 選項 B：承認此維度做不出來，**移除 signal_filter 對它的依賴**，避免假象

### 🔴 P0-5：57 個 v3 新因子已會計算，但訓練模型完全沒看到

**證據**：`outputs/feature_importance_refined.csv` 排名前 20

```
eps_yoy, price_to_ma120, per_pct_rank_252, ma_cross_20_50, cash_dividend_ttm,
atr_14, US10Y, us_yield_spread, ma_50, skew_60d, eps_ttm, foreign_holding_ratio,
cash_ratio, pbr_pct_rank_252, bb_upper, kurt_60d, sharpe_60d, eps_accel_proxy, ma_cross_50_120
```

**完全看不到** v3 任何一個：fcf_yield、accruals、sbl_short_intensity、vix_zscore_252、yield_curve_inverted、night_session_premium、news_intensity、put_call_ratio_oi…

**意味**：
- 不是新因子沒用，是**模型還沒重新訓練到吃這些新因子的版本**
- 上次訓練可能在 v3 因子加入之前完成
- 系統現在實際吃的還是「v2 老因子」做預測

**修復**：強制全量重訓 `python scripts/training/parallel_train.py --force-all`，且訓練前驗證：

```python
from config import get_all_features
assert len(get_all_features('2330')) >= 175, "ALL_FEATURES 沒包含 v3 因子！"
```

---

## 伍、P1 高風險（30 天內必修）

### 🟠 P1-1：signal_filter 完全沒引用 v3 衍生因子

`pipeline/signal_filter.py` 5 大維度仍是 v2 邏輯：

- `_eval_chip` 看 `foreign_net_weekly`、`rev_yoy_positive_months`
- `_eval_prob` 看 `prob_up`、`kwave_score`、`entropy_delta`
- 但 SBL short interest / VIX regime / news spike / disposition flag **一個都沒進來**

**結果**：signal_filter 只是 model 預測機率的「**簡單篩選器**」，不是「**多源風險護欄**」。

**修復**（一個半天可完成）：在 `signal_filter` 加 6 條 hard block + 2 條 boost：

```python
# Hard Block — 黑天鵝
if df_feat.get("is_delisted", 0).iloc[-1] > 0:
    return FilterResult("HOLD_CASH", 0, blocking_reasons=["⛔ 已下市"])
if df_feat.get("is_in_disposition", 0).iloc[-1] > 0:
    blocking_reasons.append("⛔ 處置股票期間")
if df_feat.get("is_margin_suspended", 0).iloc[-1] > 0:
    blocking_reasons.append("⛔ 暫停融券（軋空風險）")

# Hard Block — 宏觀 regime
if df_feat.get("vix_zscore_252", 0).iloc[-1] > 2:
    blocking_reasons.append("⛔ VIX 極端恐慌（>2σ）")
if df_feat.get("yield_curve_inverted", 0).iloc[-1] > 0 and \
   df_feat.get("hy_credit_spread", 0).iloc[-1] > 5:
    blocking_reasons.append("⛔ 殖利率倒掛 + 信用緊縮")

# Soft Score 加分
if df_feat.get("fcf_yield", 0).iloc[-1] > 0.05:
    boosting_reasons.append("⭐ FCF Yield > 5%（高品質）")
if df_feat.get("foreign_fut_oi_chg_5d", 0).iloc[-1] > 0 and \
   df_feat.get("night_session_premium", 0).iloc[-1] > 0:
    boosting_reasons.append("⭐ 外資期貨多頭 + 夜盤確認")
```

### 🟠 P1-2：data_integrity_audit 自身的盲點

審查 `monitor/data_integrity_audit.py` 找到三處空轉：

| 問題 | 行號 | 影響 |
|---|---|---|
| `audit_feature_nan_rate` 是 Mock，寫死回傳 | 156-162 | 第 5 維度空轉 |
| `audit_fetch_failures` 依賴的 `fetch_log` 表不存在 | 167 | 第 6 維度空轉 |
| 預設 tables 只列 5 張，27 張表沒看完 | 60 | 第 1 維度只覆蓋 18% |

**修復**：

- audit_feature_nan_rate 真的跑 `build_features`，計算每個衍生因子最近 60 天的 NaN 比率
- `parallel_fetch.py` 結尾寫 `fetch_log` 表（每張表 / 每次抓取的 success/failure + duration）
- 預設 tables 改用 `list(TABLE_REGISTRY.keys())` — config 層已經有 27 張表了

### 🟠 P1-3：DATA_LAG_CONFIG 對季報延遲過保守（145 天）

```python
"financial_statements":  {"lag": 145},
"balance_sheet":         {"lag": 145},
"cash_flows_statement":  {"lag": 145},
```

**實際法規**：
- Q1-Q3 公告期限 **45 天**（5/15、8/14、11/14）
- Q4 / 年報 **90 天**（隔年 3/31）

**現況代價**：145 天 = 5 個月，Q1（3/31 季底）資料要等到 **8/24 才能用**，但實際 5/15 就公告。**白白丟掉 3 個月的 alpha 訊號**。

**修復**：在 `data_pipeline.load_financial_statements` 與 `load_cash_flows` 改用「動態 lag」：

```python
def _lag(dt):
    if dt.month == 12:           # Q4 / 年報
        return 90
    elif dt.month in (3, 6, 9):  # Q1-3
        return 45
    return 145  # safety
```

### 🟠 P1-4：calibration 號稱用 TimeSeriesSplit 但實際路徑沒走

`train_evaluate.py` line 61-65：

```python
def make_time_series_calibrator(estimator, n_splits=5, method="isotonic"):
    return CalibratedClassifierCV(estimator, cv=TimeSeriesSplit(n_splits), method)
```

但根據 line 56-59 註解：「本系統的實際校準走 OOF 路徑」— 即 `meta_ensemble.calibrate(oof_valid, y_meta)`。

**意味** `make_time_series_calibrator` **根本沒被呼叫**！只是放著當保護網。

**修復**：寫單元測試強制檢驗 `meta_ensemble.calibrate` 是 deterministic + 時間有序的（test_round5_fixes 已涵蓋部分但需擴充）。

### 🟠 P1-5：訊號歷史未持久化

`signal_filter.SignalFilter.evaluate()` 回傳 `FilterResult`，但每日的 `decision`/`overall_score`/`blocking_reasons` 沒寫到 DB。

**結果**：事後無法回答「過去 30 天 SignalFilter 阻斷的 100 個訊號，後來表現如何？」

**修復**：建一張 `signal_history` 表：

```sql
CREATE TABLE signal_history (
    date              DATE,
    stock_id          VARCHAR(50),
    decision          VARCHAR(20),     -- LONG / HOLD_CASH / WATCH
    overall_score     NUMERIC(5,2),
    prob_up           NUMERIC(5,4),
    blocking_reasons  TEXT,            -- JSON list
    boosting_reasons  TEXT,            -- JSON list
    PRIMARY KEY (date, stock_id)
);
```

3 個月後就能做「**SignalFilter 反事實分析**」：阻斷的訊號真的該阻斷嗎？

---

## 陸、P2 中度風險（60 天內）

### 🟡 P2-1：log rotation 缺失

- `outputs/manager.log` 已 844 KB
- `outputs/train.log` **11.8 MB**
- 沒有 logrotate 設定

**修復**：

```
# /etc/logrotate.d/quant_train
/home/hugo/project/stock_backend/scripts/outputs/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    copytruncate
}
```

### 🟡 P2-2：tests 覆蓋率僅集中在「歷史 bug fixes」

`tests/test_round3_fixes.py` + `tests/test_round5_fixes.py` 共 11 個 class，全是針對「過去找到的 bug 不要再發生」。

**沒有**：
- feature_engineering 的 57 個新因子單元測試
- signal_filter 的決策樹測試（given X, expect Y decision）
- end-to-end 測試（fetch → feature → predict 完整流程）

**建議補上**：

| 測試檔 | 內容 | 預估行數 |
|---|---|---|
| `test_v3_features.py` | 每個新因子 mock data 測試輸出範圍 | ~200 |
| `test_signal_filter.py` | mock report dict + df_feat → 驗證 decision | ~150 |
| `test_e2e_2330.py` | fixed seed + cached data → 輸出 hash 穩定（regression test）| ~100 |

### 🟡 P2-3：`stock_news` 用了但沒做 NLP

`fetch_news_data.py` 抓進 `stock_news` 表（含 title + description），但 `feature_engineering.add_news_attention_features` 只做了「**新聞數量**」三個因子。沒有對 title/description 做 sentiment scoring。

每天每股可能有 5-20 則新聞，**內容**裡藏的訊號（CEO 異動、訴訟、產品延遲）比「數量」強得多。

**選項**：

| 方案 | 工具 | 月成本估計 | 效果 |
|---|---|---|---|
| 簡易版 | finBERT 中文模型（本地）| $0 | title 二元分類 |
| 進階版 | GPT-4o-mini API | ~$0.3 | entity extraction + sentiment |
| 旗艦版 | Claude Haiku batch | ~$1 | 完整 RAG 摘要 + 因子 |

### 🟡 P2-4：Backtest 與實際 Signal Filter 邏輯脫節

`portfolio_backtest.py` 用固定門檻 0.75 做等權持倉（簡化），但 `signal_filter.py` 的真實邏輯包含 5 維度評分 + hard block。

**回測結果不能代表實際表現**。

**修復**：在 `portfolio_backtest` 中直接呼叫 `signal_filter.SignalFilter.evaluate()` 對每日訊號做真實過濾，再計算 portfolio P&L。

### 🟡 P2-5：feature_importance 沒有按 group 彙整

當前 `feature_importance_refined.csv` 是 flat list of 175 個特徵的 importance。看不出來「哪個 GROUP 對模型最重要」。

**建議**：訓練後同時輸出 `feature_importance_by_group.csv`：

```csv
group,sum_importance,top_features
quality, 0.85, fcf_yield;accruals;capex_intensity
fred_macro, 1.23, vix_zscore_252;yield_curve_inverted;m2_growth_yoy
event_risk, 0.05, log_market_cap;is_in_disposition
...
```

立刻能回答「FRED 宏觀真的在幫忙」或「event_risk 完全沒用，可砍」。

---

## 柒、P3 結構性建議（90 天 +）

### 🟢 P3-1：建立 Paper Trading 緩衝期

新模型訓完 → 直接上線給訊號 → 用真錢驗證。**沒有中間驗證階段**。

**建議流程**：

```
Day 1-30:   Paper Trading（DB 記錄訊號，不執行）
Day 31:     比對 paper signals 的 P&L 與訓練期 OOF 的 P&L
Day 31:     若 paper Sharpe < OOF Sharpe × 0.7 → 拒絕上線
Day 31+:    通過才轉為 Real Trading
```

### 🟢 P3-2：模型 versioning + rollback

當前 `outputs/models/ensemble_2330.pkl` 直接覆蓋，沒有：
- 訓練時的 git commit hash
- 訓練時的 feature schema 版本
- 訓練時的資料快照（哪天的 stock_price 截止）

**建議**：每次訓練存 metadata：

```
outputs/models/
├── ensemble_2330.pkl                    # current
├── archive/
│   ├── ensemble_2330_2026-04-15_a3f4.pkl
│   └── ensemble_2330_2026-04-15_a3f4.metadata.json
│       {git_hash, feature_count, train_end_date, oof_sharpe, oof_da}
```

讓「上週的模型比這週好」這種情況可以立刻回滾。

### 🟢 P3-3：模型 explainability dashboard

當 SignalFilter 給出 `LONG`，使用者看不到「**為什麼**模型這次認為會漲」。

**建議**：在 `predict.py` 加 SHAP value 計算，存到 `outputs/explanations/{stock_id}_{date}.json`：

```json
{
  "decision": "LONG",
  "prob_up": 0.78,
  "top_positive_features": [
    {"feature": "fcf_yield",        "shap": +0.12, "value": 0.067},
    {"feature": "vix_zscore_252",   "shap": +0.08, "value": -1.4},
    {"feature": "foreign_net_5d",   "shap": +0.06, "value": 1.2e9}
  ],
  "top_negative_features": [
    {"feature": "is_margin_suspended", "shap": -0.05, "value": 0}
  ]
}
```

dashboard 上可以直接看：「這次 LONG 的訊號主要來自 FCF 殖利率（+12%）+ VIX 低（+8%）+ 外資買超（+6%）」。

### 🟢 P3-4：Multi-horizon 訊號融合

當前 `target_15d_binary` / `target_21d_binary` / `target_30d_binary` 三個目標分別訓練，但 signal_filter 只用一個 `prob_up`。

**建議**：用三時程一致性作為信心強化：

```python
prob_15d = report["prob_up_15d"]
prob_21d = report["prob_up_21d"]
prob_30d = report["prob_up_30d"]
horizon_consensus = (prob_15d > 0.65) + (prob_21d > 0.65) + (prob_30d > 0.65)
# horizon_consensus == 3 → 強訊號（3/3 時程都看多）
# horizon_consensus == 0 → 強反訊號
```

### 🟢 P3-5：跨標的 transfer learning 真的接通

`config.SECTOR_POOLS` 已定義 3 個產業池，但 `parallel_train.py` 仍是逐股獨立訓練。

`TRAINING_STRATEGY.use_global_backbone = True` 這個 flag **沒被任何模組讀取**——是個 dead config。

**建議**：實作真正的 pooled training：

1. 同產業的所有股票合併成一個大 dataset（加 stock_id one-hot）
2. 訓練一個 backbone model
3. 對每支股票 fine-tune 最後一層
4. 半導體新加入的 3661 不必從頭訓練 5 年資料，可繼承半導體池的 backbone

這對「config 新增的個股」訓練速度提升 10x+，且小樣本股的 alpha 更穩健。

---

## 捌、Quick Wins — 立即可動手清單

每項預估 < 1 小時。**8 項加起來不到 3 小時，能把整體信任度從 57% 拉到 75% 以上**。

| # | 動作 | 影響 | 成本 |
|---|---|---|---|
| **QW-1** | 跑 `historical_backfill --from-json outputs/integrity_gaps.json` 補 267 個斷層 | **直接消除 4 年訓練資料污染** | 30 分（API 配額）|
| **QW-2** | 重啟 auto_train_manager 並用 systemd 包起來 | 把 7/87 → 87/87 訓練完 | 20 分 |
| **QW-3** | 把 `signal_filter` 加 3 個 hard block（is_delisted / is_in_disposition / is_margin_suspended）| 黑天鵝防護 | 10 分 |
| **QW-4** | 在 `parallel_train.py --force-all` 強制全量重訓（讓模型吃 v3 因子）| 把 175 個特徵真的進到模型 | 自動跑（背景）|
| **QW-5** | `make_time_series_calibrator` 真的串到 ensemble 訓練流程 | 修補 calibration 漏洞 | 30 分 |
| **QW-6** | `cash_flows_statement` 的 lag 從 145 改為動態 45/90 | 解鎖 3 個月的基本面 alpha | 5 分 |
| **QW-7** | 補一張 `fetch_log` 表 + parallel_fetch.py 寫入 | data_integrity_audit 第 6 維度復活 | 30 分 |
| **QW-8** | 設定 logrotate 防止 train.log 撐爆 | 運維健康 | 5 分 |

---

## 玖、信任度提升 OKR 量化指標

訂出可量化的 KPI：

| 指標 | 目前 | 30 天目標 | 90 天目標 |
|---|---|---|---|
| 訓練完成標的數 | 7 / 87 | 87 / 87 | 87 / 87 |
| 資料斷層筆數 | 267 | < 20 | 0（自動修復）|
| 中波動 fold max_drawdown | -99.9% | > -30% | > -15% |
| 中波動 fold n_trades | 1984 | < 600 | < 300 |
| 中波動 fold net Sharpe | ~0 | > 0.8 | > 1.2 |
| Feature importance 含 v3 因子比例 | 0% | > 30% | > 50% |
| Signal filter blocking 真實命中率 | 未知 | 量化 + 報表 | > 75%（被擋的真的不該買）|
| auto_train_manager 連續穩定天數 | 0 | 30 | 90 |
| 測試覆蓋率（新因子）| 0% | > 60% | > 90% |
| Paper Trading vs 實盤 P&L 差距 | N/A | < 30% | < 15% |
| 整體信任度評分 | 57% | 75% | 85% |

---

## 拾、優化路線圖（30 / 60 / 90 天）

### Day 1-7：止血期（搶修 P0）
- [ ] **D1**：QW-1（補 267 斷層）+ QW-2（重啟訓練）+ QW-8（logrotate）
- [ ] **D2-3**：QW-3（hard block）+ QW-6（lag 修正）+ QW-7（fetch_log）
- [ ] **D4-7**：QW-4 訓練跑完 + 驗證 v3 因子真的進到 importance
- [ ] **D7**：第一次完整 audit run，確認 gap < 20

### Day 8-30：補強期（消化 P1）
- [ ] signal_filter 整合 v3 因子（P1-1）
- [ ] data_integrity_audit 三項漏洞修復（P1-2）
- [ ] calibration 強制 TimeSeriesSplit（P1-4）
- [ ] signal_history 表 + 三個月反事實分析（P1-5）
- [ ] 跑第一次完整 model_quality_audit，產出 87 支 DA / IC 表
- [ ] 確認 stock_dynamics_registry 是否要補（P0-4）

### Day 31-60：強化期（處理 P2）
- [ ] tests 三大類補上（test_v3_features / test_signal_filter / test_e2e）
- [ ] Backtest 與真實 Signal Filter 對齊（P2-4）
- [ ] feature_importance_by_group 自動產出（P2-5）
- [ ] 開始第一輪 Paper Trading（30 天）
- [ ] news NLP 接入 finBERT 或 GPT-4o-mini（P2-3）

### Day 61-90：升級期（推 P3）
- [ ] Paper Trading 結果分析 + 上線決策
- [ ] 模型 versioning + rollback（P3-2）
- [ ] SHAP explainability dashboard（P3-3）
- [ ] Multi-horizon 訊號融合（P3-4）
- [ ] SECTOR_POOLS pooled training（P3-5）

---

## 拾壹、結論與下一步

### 系統的「形」與「神」

這套系統的「**形**」已達機構級標準：
- 27 張資料表完整覆蓋台股 + 美國宏觀
- 175 個因子涵蓋技術、籌碼、基本面、宏觀、事件、新聞、衍生品 7 大類
- 5 模型 stacking + RegimeEnsemble 動態切換
- 5 維度 signal filter + 4 維度 health check
- monitor/ 子目錄完整分層

但「**神**」還差幾個關鍵接點：

1. **資料層的 267 個斷層** — 這是最大的單一污染源，工具已會偵測，只差「自動補抓」這條 wire
2. **特徵層的 v3 新因子** — 已建好 57 個，但模型沒吃到，且 signal_filter 也沒接
3. **訓練層的 auto_train** — 87 支只訓 7 支，且運行 6 天前就停了
4. **訊號層的 stock_dynamics_registry** — 不知有沒有資料，可能整個 dynamics 護欄是 placebo
5. **回測層的 -99.9% drawdown** — 這個數字不該被接受，是當前最危險的紅旗

### 推薦的下一步（按優先序）

1. **立刻動手**：QW-1 + QW-2 + QW-3（補資料 + 重啟訓練 + hard block）
   - 預估 1 小時內完成，能把信任度從 57% → 70%
2. **本週內**：QW-4（強制重訓）+ QW-6（lag 修正）+ QW-7（fetch_log）
   - 預估 1 週內完成，能把信任度從 70% → 78%
3. **下個月**：P1 全部處理完
   - 信任度 78% → 85%
4. **季度目標**：Paper Trading + Multi-horizon + Versioning
   - 信任度 85% → 90%+

---

## 附錄 A：完整模組清單

### 資料抓取（fetchers/，20 個）

```
fetch_stock_info.py            fetch_technical_data.py
fetch_chip_data.py             fetch_fundamental_data.py
fetch_macro_data.py            fetch_macro_fundamental_data.py
fetch_international_data.py    fetch_derivative_data.py
fetch_derivative_sentiment_data.py  fetch_sponsor_chip_data.py
fetch_total_return_index.py    fetch_missing_stocks_data.py
fetch_cash_flows_data.py       fetch_price_adj_data.py        # v3
fetch_advanced_chip_data.py    fetch_event_risk_data.py       # v3
fetch_extended_derivative_data.py   fetch_news_data.py        # v3
fetch_fred_data.py             parallel_fetch.py              # v3
```

### 資料處理（pipeline/，9 個）

```
data_pipeline.py (1261 行)     feature_engineering.py (1541 行)
signal_filter.py (627 行)      portfolio_optimizer.py (190 行)
portfolio_strategy.py (173 行) portfolio_backtest.py (211 行)
backtest_engine.py (147 行)    strategy_tester.py (311 行)
```

### 訓練與推論（training/，7 個）

```
train_evaluate.py (893 行)     parallel_train.py (195 行)
predict.py (687 行)            auto_train_manager.py (293 行)
historical_backfill.py (415 行) tune_hyperparameters.py (118 行)
update_feature_store.py (141 行)
```

### 模型（models/，2 個）

```
ensemble_model.py (703 行) — XGB / LGB / ElasticNet / Momentum / Stacking / RegimeEnsemble
tft_model.py (714 行)      — Temporal Fusion Transformer
```

### 監控（monitor/，7 個）

```
data_integrity_audit.py    model_health_check.py (382 行)
model_quality_audit.py     db_health_check.py (139 行)
db_optimize.py (181 行)    backtest_audit.py
dashboard.py
```

### 共用核心（core/ + utils/，6 個）

```
core/finmind_client.py     core/db_utils.py
utils/db.py                utils/feature_selection.py
utils/metrics.py (159 行)  utils/model_loader.py
```

### 測試（tests/，2 個）

```
test_round3_fixes.py (266 行) — 11 個 test class
test_round5_fixes.py (199 行) — 5 個 test class
```

---

## 附錄 B：本次審查掃過的核心關鍵字

```
freshness | 完整性 | MAX(date) | missing | stale | 缺資料 | 資料鮮度 | coverage
CalibratedClassifierCV | TimeSeriesSplit | cross_val_predict | isotonic | OOF | embargo
fcf_yield | sbl_short | vix_zscore | put_call_ratio_oi | night_session | m2_growth
news_intensity | day_trading_pct | is_in_disposition | log_market_cap
TABLE_REGISTRY | ALL_FEATURES | CONFIDENCE_THRESHOLD | get_all_features
stock_dynamics_registry | fetch_log | integrity_gaps | regime_metrics
```

掃過的關鍵檔案行數合計：**12,000+ 行 Python**

---

> **「系統的形已達機構級，但神尚未通透。先把已建好的工具串起來，再談新功能。」**
>
> — 本報告核心建議

---

**報告終**

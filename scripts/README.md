# 《2026 動力學金融矩陣》(2026 Kinetic Finance Matrix)
> **版本 v4.0 (Trinity Edition)**：融合「第一性原則、動力學模型與宏觀場景模擬」的資產管理框架。

本矩陣旨在利用市場動力學衝量 (Kinetic Impulse)、統計學過濾與產業技術結構變化，在 2026 年預期技術週期臨界點到來之際，為核心資產（2330, 2317, 2454 等）提供具備「反脆弱性」的決策支撐。

---

## 🏛️ 核心理念與典範轉移 (Paradigm Shift)

本系統主張投資應從 **「被動應對市場噪音」** 轉向 **「主動駕馭市場動力學」**：

| 特性 | 傳統投資 (Traditional) | 矩陣系統 (Matrix) |
| :--- | :--- | :--- |
| **驅動力** | **噪音驅動**：依賴短期情緒、MACD 等表面指標 | **動力學驅動**：解析訂單流微觀結構與位移能量 |
| **統計假設** | **正態分佈**：忽略尾部風險，導致短期被動反應 | **冪律分佈**：運用 20/60/20 法則捕捉不對稱回報 |
| **時間維度** | **線性思維**：關注明日漲跌，假設趨勢永遠持續 | **多尺度思維**：結合實時數據流與宏觀結構化背景 |
| **行為模式** | **被動應對**：追漲殺跌，被市場情緒左右 | **主動導航**：在潛在機制切換點 (Regime Shift) 進行動態防禦 |

---

## 系統架構

```
tsmc_predictor/
├── config.py               # 全域設定（路徑、超參數、個股客製化配置）
├── data_pipeline.py        # 資料層：從 PostgreSQL 讀取 → 每日寬格式 DataFrame
├── feature_engineering.py  # 特徵工程：5 大類 ~100 個通用特徵 + 動態產業特徵
├── train_evaluate.py       # 訓練核心：Purged Walk-Forward CV + 評估 + 模型儲存
├── parallel_train.py       # 並行管理器：全自動並行訓練 80+ 標的模型
├── parallel_fetch.py       # 並行抓取器：高速抓取全市場技術、籌碼、基本面數據
├── predict.py              # 推論核心：每日執行，輸出趨勢報告
├── model_health_check.py   # 健康檢查：監控資料鮮度與實戰 DA
├── automate_daily.py       # 自動化流水線：一鍵完成推論、檢查與組合優化
├── models/
```

---

## PostgreSQL 資料表對應

### PostgreSQL 效能優化：
- **複合索引 (Composite Index)**：針對所有核心表建立 `(stock_id, date DESC)` 索引，極大化加速 `build_daily_frame` 中的多表 JOIN 效能。
- **物化視圖 (Materialized View)**：透過 `mv_daily_market_sync` 預處理常用的日線特徵，支援 `CONCURRENTLY` 增量刷新，避免每日全量計算。
- **因子穩定性驗證 (Factor Stability)**：
    - **Rolling IC (Rank IC)**：透過 Spearman 相關係數追蹤每個特徵對未來回報的預測力。
    - **IC Information Ratio (IC IR)**：設定 **IC IR ≥ 0.5** 為強健因子門檻，確保特徵不僅有效，且在不同市場週期下表現穩定。
    - **因子衰減偵測 (Decay Detection)**：自動監控特徵預測力的衰竭速度，作為特徵汰換與模型重訓的領先指標。

| 資料表 | 內容 | 更新頻率 |
|--------|------|---------|
| `stock_price` | 日線股價（open/max/min/close/volume）| 每日 |
| `stock_per` | PER / PBR / 股利殖利率 | 每日 |
| `institutional_investors_buy_sell` | 三大法人買賣超 | 每日 |
| `margin_purchase_short_sale` | 融資融券餘額 | 每日 |
| `shareholding` | 外資持股比率 | 每日 |
| `total_return_index` | TAIEX / TPEx 指數 | 每日 |
| `exchange_rate` | USD/TWD 等匯率 | 每日 |
| `interest_rate` | FED / BOJ / ECB 等央行利率 | 不定期 |
| `month_revenue` | 月營收（YoY / MoM）| 每月 |
| `financial_statements` | 損益表（EPS / 毛利率 / ROE）| 每季 |
| `balance_sheet` | 資產負債表（流動比率 / 負債比）| 每季 |
| `dividend` | 現金股利 / 除息日 | 每次公告 |

---

## 第一性原則特徵分組

| 類別 | 代表特徵 | 主要預測貢獻 |
|------|---------|------------|
| ① 技術動能 | RSI、MACD、BB%、realized_vol | 短期 momentum / 反轉 |
| ② 資金流情緒 | foreign_net、margin_balance | 法人 vs 散戶供需扭曲 |
| ③ 基本面脈衝 | revenue_yoy、gross_margin、eps_ttm | 長期內在價值定錨 |
| ④ 估值錨點 | per_pct_rank、dividend_yield | 資金吸引 / 排斥 |
| ⑤ 宏觀因子 | fed_rate_chg、usd_twd、taiex_rel | 折現率 / 風險偏好 |
| ⑥ 事件驅動 | days_to_ex_dividend、dividend_ex_dummy | 除息前後效應 |
| ⑦ 滾動統計 | skew、kurt、sharpe | 尾部風險 / 動量強度 |

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 確認 PostgreSQL 連線

修改 `data_pipeline.py` 中的 `DB_CONFIG`：
```python
DB_CONFIG = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host": "localhost",
    "port":     "5432",
}
```

連線診斷：
```bash
python data_pipeline.py
```
成功會印出 shape、date range、最後 5 筆關鍵欄及缺失率。

### 3. 資料抓取與更新

```bash
# [推薦] 並行高速抓取 (技術、籌碼、基本面、國際市場、期權)
python scripts/parallel_fetch.py

# 或是手動執行單項抓取 (範例：技術面資料)
python scripts/fetch_technical_data.py --start 2024-01-01
```

### 4. 訓練模型

```bash
# 完整訓練（含 TFT，首次約 2~4 小時）
python train_evaluate.py --start 2010-01-01

# 快速版（僅 XGBoost + LightGBM，約 10~30 分鐘）
python train_evaluate.py --start 2018-01-01 --no-tft

# 只做 Walk-Forward 評估，不儲存最終模型
python train_evaluate.py --no-tft --wf-only
```

訓練輸出：
- `outputs/models/ensemble_final.pkl` — 最終 Ensemble 模型
- `outputs/models/tft_final.ckpt` — TFT checkpoint
- `outputs/wf_fold_metrics.csv` — 各 Fold 評估結果
- `outputs/feature_importance.csv` — 特徵重要性排名
- `outputs/train.log` — 完整訓練日誌

### 4. 每日推論

```bash
# 標準推論（每日收盤後執行）
python predict.py

# 輸出 JSON 報告
python predict.py --output-json outputs/pred_$(date +%Y%m%d).json

# 加上資料漂移偵測
python predict.py --drift-check

# 跳過 TFT（速度較快）
python predict.py --no-tft
```

### 5. 自動化流水線與健康檢查

為了確保大規模生產環境（80+ 標的）的穩定性，系統提供了全自動化管線與監控腳本：

#### 執行全自動流水線
```bash
# 一鍵完成：批次推論 -> 健康檢查 -> 投資組合優化
python scripts/automate_daily.py
```

#### 單獨執行健康檢查
```bash
# 檢查數據鮮度、模型時效性與最近 30 天實戰 DA
python scripts/model_health_check.py
```

健康檢查報告包含：
- **數據流鮮度**：檢查 PostgreSQL 各表最新日期，預防爬蟲失效。
- **模型時效性**：監控各個股模型是否超過 30 天未重訓。
- **預測分佈漂移 (PSI)**：計算最近預測與訓練分佈的 **Population Stability Index**。若 PSI > 0.2，判定為模型失效（Concept Drift），觸發緊急重訓。
- **實戰準確度 (Real-time DA)**：對比歷史預測軌跡與實際價格，監控模型是否因市場 **Regime Shift**（如升息週期、AI 浪潮轉換）而產生效能衰退。

---

## 預期輸出範例

```
══════════════════════════════════════════════════════════════
  台積電 (2330)  30 天趨勢預測報告
  基準日：2026-03-27  →  目標日：2026-04-26
══════════════════════════════════════════════════════════════

  當前收盤價 ：       1,820  TWD

  ▶ 方向預測   ：📈 上漲
  ▶ 上漲機率   ：67.3%
  ▶ 信心等級   ：🟢 高信心
  ▶ 模型離散度 ：σ=0.042  （越小→三模型越一致）

  預期報酬區間（q10 / q50 / q90）：
    低端   -3.2%  →    1,762  TWD
    中位   +4.1%  →    1,895  TWD  ← 點預測
    高端  +11.5%  →    2,029  TWD

  模型分解：
    ensemble    :  0.673  ██████████████░░░░░░
    xgb         :  0.651  █████████████░░░░░░░
    lgb         :  0.668  █████████████░░░░░░░
    tft         :  0.701  ██████████████░░░░░░

  驅動因子 (SHAP)：
    ↑ 支撐上漲：['revenue_yoy', 'foreign_net_ma5', 'rsi_14']
    ↓ 壓制下跌：['per_pct_rank_252', 'fed_rate_chg_30d']
    > [!NOTE]
    > **SHAP 解釋性說明**：SHAP 展示的是特徵與預測目標之間的「統計關聯性」，而非直接的「因果關係」。

  關鍵指標快照：
    RSI(14)=48.3  MACD hist=-0.52
    PER=27.48（歷史 68.0% 分位）
    外資淨買超 MA5=-3,061,107
    *（所有基本面指標均已依據公告延遲進行日期平移，確保無未來資訊洩漏）*
```

---

## Walk-Forward 歷史表現目標

| 指標 | 目標 | 說明 |
|------|------|------|
| 方向正確率（DA） | ≥ 65% | 隨機基準 50%，2330 歷史可達 64~68% |
| 三重障礙勝率 | ≥ 55% | 觸發 +8% 停利 vs -5% 停損的勝率 |
| 損益比 (Payoff) | ≥ 2.0 | 平均獲利 / 平均虧損（確保期望值為正） |
| 期望值 (EV) | > 1.0% | 每筆交易預期淨回報 (含交易成本) |
| 平均淨報酬 (Net) | > 2.0% | 扣除 0.5%~0.8% 摩擦成本後的每筆交易淨回報 |
| IC（Rank IC） | ≥ 0.05 | Spearman 相關係數 |
| Sharpe Ratio | ≥ 1.0 | 基於路徑風險優化後的風險回報比 |

---

## 模型整體架構

```
DB：PostgreSQL 17
    ↓
data_pipeline.build_daily_frame()        ← 12 張資料表 LEFT JOIN + ffill
    ↓
feature_engineering.build_features()    ← ~100 個特徵 + 目標變數
    ↓
    ┌───────────────────────────────────────────────────────┐
    │                  Purged Walk-Forward CV               │
    │                                                       │
    │  [TFT]          [XGBoost]      [LightGBM]            │
    │  序列 + 事件    表格特徵        表格特徵               │
    │  分位輸出       OOF 預測        OOF 預測               │
    │       └──────────────┬──────────────┘                │
    │                      ↓                               │
    │          [Isotonic Calibration]                      │
    │          對齊各模型機率語意，消除過度自信              │
    │                      ↓                               │
    │       3.  **多樣化集成 (Heterogeneous Ensemble)**：
    - **L1 學習器**：XGBoost, LightGBM, TFT (深度學習), ElasticNet (線性因子), 以及基於規則的 Simple Momentum 模型。
    - **動態加權 (Dynamic Weighting)**：使用最近 60 天的 OOF 表現透過 **Softmax (Temperature=0.05)** 進行動態權重分配，自動適應市場機制的轉變。
    - **L2 Stacking**：最終透過 Logistic Regression 進行 meta-features 整合。
    │                      ↓                               │
    │             路徑感知機率 + 信心區間 + SHAP             │
    └───────────────────────────────────────────────────────┘
```

---

## 📅 資料可用性時間表 (Data Availability Calendar)

為了嚴格防止「未來資訊洩漏 (Look-ahead Bias)」，系統對所有非即時資料實施了自動化的公告延遲平移（Publication Lag Shifting）：

| 資料類型 | 名義時間 | 實際可用時間 | 系統處理策略 | 洩漏風險 |
| :--- | :--- | :--- | :--- | :--- |
| **月營收** | 每月底 | 次月 10 日前 | **平移 40 天** (從月初計) | 🔴 極高 (已封堵) |
| **季財報 (Q1-Q3)** | 季末 | 季末後 45 天 | **平移 45 天** | 🔴 極高 (已封堵) |
| **年報 (Q4)** | 年底 | 次年 3 月底 | **平移 90 天** | 🔴 極高 (已封堵) |
| **法人/融資籌碼** | 當日收盤 | 次日開盤前 | **平移 1 天** | 🟡 中等 (已封堵) |
| **股價/技術指標** | 當日收盤 | 即時 | 無需平移 | 🟢 低 |

> [!IMPORTANT]
> - **資料陳舊度 (Data Staleness)**：除了使用 `ffill` 填充基本面數據外，系統額外計算了 `eps_staleness_days` 與 `rev_staleness_days`。這讓模型能識別「目前資訊已落後多久」，解決了 `ffill` 掩蓋數據缺失/陳舊信號的問題，並將「資訊不確定性」轉化為有效的預測因子。
> - **無偏差回測**：所有模型訓練與回測僅使用該日期「在現實中已公告」的資料。例如，2024-11-15 的交易決策只能看到 10 月份的營收與 Q3 的財報。

---

## 注意事項

- **路徑感知標籤**：採用 **三重障礙法 (Triple-Barrier Method)**，預設為 +8% 停利 / -5% 停損 / 30天結算，訓練模型避開路徑風險。
- **資料洩漏防護**：Walk-Forward 使用 `embargo_days=45` 禁區，考慮台股月報 (10號) 與季報公告延遲，確保訓練集無未來資訊。
- **OOF Stacking**：Meta-Learner 僅在 Out-of-Fold 預測上訓練，防止 Level-2 放大 Level-1 的過擬合偏差。
- **結構性斷點標記 (Structural Event Markers)**：
    - **農曆春節效應**：加入 `is_pre_lunar_new_year` 標記封關前的流動性異常。
    - **除息填息模式**：明確區分 `is_pre_dividend` 與 `is_post_dividend`，處理股價不連續性。
    - **ADR 隔夜跳空**：計算 TSM/UMC 等 ADR 隱含價與本地價的 `adr_overnight_gap`，作為領先資訊輸入。
- **免責聲明**：本系統僅供研究參考，不構成投資建議。所有交易決策需結合人工判斷。

---

## 🏛️ 科學性聲明 (Scientific Clarification)
> [!IMPORTANT]
> **本系統所使用的「動力學」、「衝量」與「質量」等術語均為描述市場統計行為的「隱喻 (Metaphors)」**。
> 量化交易本質上是基於歷史數據的**統計學習 (Statistical Learning)**，而非恆久不變的物理定律。系統旨在通過非線性動力學 (Non-linear Dynamics) 模型捕捉訂單流與資金分佈的機率偏離，使用者應始終保持對統計不確定性與市場環境劇變 (Regime Change) 的敬畏。

## 🌌 未來戰略：三位一體決策體系 (The Trinity Architecture)

本系統正朝向 **「微觀動力學、中觀統計優化、宏觀結構模擬」** 三位一體的智能體系演進：

### 1. 第一性原則：市場動力學 (First Principles: Market Dynamics)
> [!CAUTION]
> **宏觀週期與 2026 視窗聲明**
> 本系統參考了「結構化超級週期」作為宏觀場景，但**深知長波理論樣本數極少且具備高度主觀性與後見之明偏誤 (Hindsight Bias)**。
> **2026 年 6 月視窗僅作為壓力測試與場景模擬的參考點**，不具備統計上的精確預測意義。
> 系統所有的實戰買賣信號均嚴格基於具備高統計置信度的**中短期數據流**與**實時機制偵測**。

### 2. 特徵工程層 (Feature Engineering)
系統將原始資料轉化為 7 大類、逾 150 個科學特徵：

1.  **技術動能 (Technical Dynamics)**：Log Return (1/5/10/20/60d), MA Cross, RSI, MACD, BBands。
2.  **微觀結構代理 (Microstructure Proxies)**：Amihud 非流動性、Kyle's Lambda 代理、收盤位置比率 (Close-to-High Ratio)。
3.  **動力學特徵 (Kinetic Features)**：慣性質量 (Log Market Cap)、動能衝量 (Mass x Displacement)、動能能量。
4.  **跨資產關聯 (Cross-Asset)**：與美股連結標的 (如 SOXX, NVDA) 的動態 Beta 偏離與領先落後相關性。
5.  **信號交叉交互 (Interaction)**：外資買超 x RSI 超賣、高波動 x 融資擴張風險。
6.  **基本面陳舊度 (Staleness)**：財報與營收的資訊衰減天數 (EPS/Rev Staleness Days)。
7.  **宏觀機制 (Regime)**：康波週期得分、美債真實收益率 (Kuznets Proxy)、2026 奇點距離。

*   **核心公式**：`動量 (Momentum) = 質量 (Mass) × 位移 (Displacement)`
*   **力 (Force) = 資訊衝擊**：如盈餘驚奇 (Surprise) 或突發事件，是推動市場脫離慣性的唯一外力。
*   **質量 (Mass) = 資本流動性**：由資產市值與掛單深度構成；質量越大，改變價格所需的「資訊力」越大。
*   **位移 (Displacement) = 價格變動**：力與質量交互作用後的最終觀測結果。
*   **重力井模型 (Gravity Well)**：企業的「內在價值」是系統的**絕對重力中心** (低熵狀態)，而市場價格則是受情緒 (高熵) 驅動的動態路徑。當價格偏離中心越遠，系統的物理引力 (中值回歸力) 就越強。

### 2. 八二法則：資源優化與風險管理 (Optimization: 80/20 Pareto)
系統遵循帕累托法則，將 80% 的資源專注於產生 80% 結果的關鍵 20% 要素：

*   **槓鈴策略 (Barbell Strategy) 與風險邊界**：
    *   **80% 極度保守 (Core)**：配置於重力井中心標的 (如 2330, 0050)。
        *   *集中度限制*：單一核心標的上限 15%，防止過度依賴單一龍頭。
    *   **20% 極度進取 (Kinetic)**：捕捉高凸性、高驚奇值的非線性機會。
        *   *集中度限制*：**單一進取標的上限 5%**，徹底規避單一 AI 概念股（如 NVDA, 3661）暴雷引發的系統性潰敗。
    *   **再平衡機制 (Rebalancing)**：當部位偏離原定比例超過 5% 時，系統自動觸發「動能修剪」或「重力補位」訊號。
    *   **流動性過濾 (Liquidity Gate)**：自動剔除日均成交量低於 **5,000 萬 TWD** 的標的，確保在大規模部位進出時不產生過大的衝擊成本。
*   **20/60/20 數據分流**：
    *   **60% 平穩期**：市場熵值穩定時，執行穩健的趨勢追蹤。
    *   **40% 肥尾期 (上下各 20%)**：當進入極端恐懼或貪婪時，啟動「動力學脈衝 (Kinetic Pulse)」模式，調整損益比至 1:3 以上。
*   **關鍵因子篩選**：自動識別 20% 的黃金特徵，將運算資源優先分配給具備最高 IC 值與解釋力的因子。
*   **風險掃描儀**：針對可能引發 80% 虧損的 20% 關鍵風險因子 (如利率暴跌、匯率衝擊) 進行優先級監控。

### 3. 宏觀導航與結構化場景 (Macro Navigation & Structural Scenarios)
系統利用長週期宏觀趨勢作為「場景背景」，決定長期的資金流向與賽道選擇：

*   **2026 結構性壓力視窗 (2026 Stress Window)**：
    *   **戰略定位**：將 2025-2026 定義為技術架構與債務週期交替的潛在臨界區。
    *   **行為模式**：在此區間，系統自動增加對波動率特徵的監控權重。
*   **新週期核心賽道 (New-Cycle Sectors)**：
    *   **自動過濾**：系統優先掃描具備高生產力潛力的標的：**人工智慧 (AI) 與半導體、生物科技、綠色能源與數位基礎設施**。
*   **長波相位模擬 (Phase Simulation)**：
    *   利用金油比、長短期利差與勞動力生產率，輔助判定市場目前所處的宏觀機制。

### 🤖 多智能體協作 (Multi-Agent Roadmap)
我們正在構建「宏觀導航、價值偵探、風險防衛」三位一體的 Agent 協作機制，將本量化引擎升級為全自動的財富智慧實體。

---

## 🛡️ 戰略執行與組合管理 (Strategic Execution)

針對 2026 年週期轉換窗，本系統實施以下戰略佈局：

### 1. 槓鈴部位管理 (Barbell Allocation)
*   **80% 核心防禦端 (Core Defensive)**：配置於高流動性、低波動之權值龍頭（如 2330 核心持倉），確保在週期交替期間的生存能力。
*   **20% 動力學進取端 (Kinetic Alpha)**：利用「動能衝量」與「驚奇值 (Surprise)」特徵，動態捕捉 AI 與半導體賽道的非線性超額收益。

### 2. 2026 導航點執行 (2026 Pivot Execution)
*   系統將 2026 年定義為「結構性週期轉換」的潛在觀測視窗。
*   **機制調節**：在此區間，系統會自動強化對「市場熵值 (Entropy)」與「分佈漂移 (PSI)」的監控，防範舊週期資產的出清波動。

### 3. 反脆弱機制 (Anti-fragility)
*   **非對稱損益比**：當市場進入極端波動區間時，系統自動啟動 **1 : 3 以上的期望值過濾**，確保在不確定性中保持正向期望值。

---

## ⚖️ 科學性聲明 (Scientific Clarification)

1. **去術語化說明**：本系統所提及之「動力學」、「衝量」、「熵值」等詞彙，皆為金融統計學中之隱喻（Metaphor），本質為**動量因子、不確定性度量與非線性回歸**，與量子物理學或熱力學定律無直接因果關係。
2. **長波週期之限制**：康波週期（Kondratiev Wave）僅作為宏觀情境參考，不具備預測短期價格走勢的統計顯著性，亦非交易訊號的硬觸發條件。
3. **SHAP 解釋權力**：SHAP 展示的是特徵與模型輸出間的**統計關聯性**，而非因果實體。所有基本面特徵均已進行公告延遲平移（15-45天），以確保回測之科學嚴謹性。

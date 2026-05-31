# TFT(Temporal Fusion Transformer)股價預測 多重週期驗證報告 v0.1

**日期**: 2026-05-29
**程式**: `scripts/evaluation/multi_cycle_tft_validation.py` v0.1
**共同比較基準**: `reports/common_model_comparison_baseline_v1.md` v1.0
**動因**: 用戶 2026-05-29 directive —「依此 TFT(時序融合變換器 / Google)模型做法產生個別的程式進行模型產生與驗證，並確認依此模型來做預測股價真的可以賺錢嗎?…以目前核心個股依每一支個股在資料庫存在的個股個別過去最久的在 database 內存在的實際數據，來進行此預測股價模型的多次週期驗證，如 by 週、月、季、年等多重週期…請做相關的精準度與信任度分析。」
**治權**: §一.10 全 DB source-traceable / §一.10 #3 multi-run / §8.5 leakage-safe / §14.7-DC v0.18 source-pure universe / §一.11 三段式標頭 / §一.12 5-min 回報

---

## 一、模型做法 (Model Methodology)

### 1.1 什麼是 TFT
**Temporal Fusion Transformer**(Lim, Arık, Loeff, Pfister 2019 / Google)是專為多水平時間序列預測設計之注意力架構,核心組件:
- **Variable Selection Networks**:逐時點學習各輸入變數之重要性權重。
- **LSTM Encoder–Decoder**:捕捉局部時序型態(local processing)。
- **Interpretable Multi-Head Attention**:跨長距時點之全域依賴(long-range dependency)。
- **Quantile Outputs(QuantileLoss)**:同時輸出 P2/P10/P25/P50/P75/P90/P98 → 天然提供不確定性區間(用於信任度 calibration)。

實作:`pytorch_forecasting.TemporalFusionTransformer`(pf 1.7.0 / torch 2.2.2 / lightning 2.6.4)。**非 surrogate,為原始 TFT 架構**。

### 1.2 如何套用於台股 398 核心股
| 設計點 | 做法 | 理由 / 治權 |
| :--- | :--- | :--- |
| 每股最長歷史 | 每股取 DB 中其全部 daily 報價(close + Trading_money)| 用戶 directive「每一支個股過去最久的實際數據」|
| Weekly 重採樣 | 每 5 交易日聚為 1 weekly bar(block 末日 close + 區間 money 加總)| 降噪 + 計算可行性(daily 全序列 TFT 在 CPU 不可行)|
| 輸入特徵(全 source-pure)| weekly log return / 4-week realized vol / 13-week MA ratio / log dollar volume | §一.10 / §一.13:全為 price/volume 之 mathematical transform,**0 imputed、0 hardcoded knowledge** |
| Pooled panel model | 單一 TFT,`group_ids=["stock_id"]`,398 股共享 + static embedding | TFT 標準用法(非每股一模型)|
| 預測目標 | weekly log return 序列;decoder forecast 未來 52 weeks | leakage-safe:encoder 僅見過去 weeks,decoder 預測未來 |
| 多週期導出 | decoder median 之 cumulative-sum → 第 1/4/12/50 週 = 5/20/60/252 交易日 forward return score | 一個 model 服務 4 horizons |
| Walk-forward refit | expanding window,每 12 個 monthly panel(≈ 年度)refit 一次 | 計算可行 + 避免 look-ahead |
| Encoder / Decoder | encoder 104 weeks(≈2yr)/ decoder 52 weeks | 涵蓋 annual horizon |

### 1.3 Anti-leakage(§8.5)結構性保證
- Encoder 僅含 as_of(該股對應 weekly bar)**之前**之 bars;decoder 預測 as_of **之後** → 預測時點不見未來。
- Refit 之 training cutoff = 各股 ≤ as_of 之最後 bar(`max_time_idx_per_stock`)→ 不混入未來 panel。
- realized forward return(評分真實標的)取自 `label_date > as_of`(與全模型相同 query)。

---

## 二、實際市場價格資料 (Actual Market Price Data — 全來自 DB,§一.10 (b))

### 2.1 Universe 價格涵蓋(`TaiwanStockPriceAdj`,2026-05-29 query)
- 核心股:**398**(v0.18 source-pure pan-historical)。
- 價格列數:**2,320,497**(close>0);日期跨度 **1992-01-04 → 2026-05-22**(~34 年)。
- 每股「最久歷史」起始日:最早 **1992-01-04** / 中位 **2003-01-24** / 最晚 **2015-11-10**。
- 每股交易日數:min **2,559** / median **5,676** / max **8,767**。
  → 確認「每股最長歷史」之 directive 可滿足:**最短者仍有 2,559 交易日(≈10 年)**,足供 TFT encoder(104wk)+ walk-forward。

### 2.2 真實 forward 報酬之市場變化(cross-sectional,398 股,4 regime 抽樣)
> 展示「市場價格在 database 內之實際變化」—— 同一 universe 在不同 regime 之 realized forward log return（mean ± std）。

| as_of | weekly 5d | monthly 20d | quarterly 60d | annual 252d | regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-01-15 | +1.34% ±3.33 | +3.42% ±5.90 | +9.62% ±10.17 | +18.97% ±23.70 | 多頭 |
| 2020-03-16 | **−12.43%** ±8.45 | +2.09% ±9.13 | +18.14% ±13.73 | **+43.17%** ±27.92 | COVID 崩盤→V 反彈 |
| 2022-10-17 | −0.91% ±3.67 | +4.34% ±5.90 | +11.87% ±11.14 | +33.92% ±26.87 | 空頭末段→反彈 |
| 2024-08-15 | +2.53% ±3.73 | −1.49% ±6.29 | +1.99% ±12.90 | **−18.39%** ±21.53 | 近期回檔 |

**觀察**:(1) 報酬離散度隨 horizon 單調放大(weekly ±3% → annual ±20–28%);(2) annual mean 橫跨 +43%(post-COVID)到 −18%(2024→2025)→ **OOS 期間確實涵蓋多 regime**,非單一牛市過擬合。此即 TFT 須在其上證明「能否賺錢」之真實市場。

> 註:as_of 落在週末(如 2022-01-15 週六)之 panel,因 `t0` 無當日報價而被丟棄 —— 此為**全模型共用之 convention**(baseline tree 模型亦同),故仍 apples-to-apples。

---

## 三、共同比較基準 (Common Baseline — 與全模型一致)

完整定義見 `reports/common_model_comparison_baseline_v1.md`。摘要:universe v0.18/398 × 95 monthly panels ×
真實 forward log return(TaiwanStockPriceAdj)× 4 horizons(5/20/60/252d)× top-20 equal-weight long × 0.6% cost ×
{Sharpe, Win, Eff-t, T_CZ-6 gate(annual Eff t≥4.20 / Sharpe≥2.40 / Win≥0.79)}。**TFT 與 10 個既有 tree/ensemble 模型走完全相同協定。**

---

## 四、精準度 / 信任度 / 賺錢能力 框架 (Precision / Trust / Profitability)

| 類別 | 指標 | 回答的問題 |
| :--- | :--- | :--- |
| **精準度** | rank IC / directional accuracy / RMSE / MAE / R² | 「TFT 預測準不準?」|
| **信任度** | Effective t-stat 顯著性 / 多 seed 穩定度(min/median/max/mean)/ P10-P90 calibration coverage | 「TFT 預測可不可信、穩不穩、不確定性誠不誠實?」|
| **賺錢能力** | net-of-cost Sharpe / Win / annualized net return / MDD / T_CZ-6 gate | 「**真的能賺錢嗎?**」|

---

## 五、驗證結果 (Validation Results)

> **狀態:⏳ PENDING — 待真實 3-seed 跑期完成**(per §一.10 #3 multi-run + AP-3 反 aspirational inscription:結果未跑出前不填數字)。

執行計畫:
1. **Smoke**(40 股 × 6 panels × 2 epochs)— plumbing 驗證 + 單 refit 計時(進行中)。
2. 依 smoke 計時估算完整跑期(398 股 × 95 panels × ~8 refits × 3 seeds {5422,7331,1009})。
3. 為避免與 Step G sweep 之 CPU 競爭,**TFT 完整跑期排在 sweep 完成後**啟動(SHMM + §一.12 5-min 回報)。
4. 跑完 → `reports/tft_v0/tft_s{5422,7331,1009}.json` → 聚合 min/median/max/mean → 回填下表。

### 5.1 跨週期結果矩陣(median of 3 seeds)— 待填
| Horizon | n_panels | Eff t | Sig? | Sharpe | Win | NetAnn | rank IC | DirAcc | RMSE | MAE | R² |
| :--- | ---: | ---: | :--: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weekly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| monthly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| quarterly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| annual | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### 5.2 信任度:多 seed 穩定度 + calibration — 待填
### 5.3 T_CZ-6 annual gate 裁決(對 median)— 待填
### 5.4 與既有模型對照(同基準)— 待 Step G sweep 聚合 + TFT 跑完

---

## 六、結論:TFT 預測股價能賺錢嗎? (Can TFT Make Money?)

> **⏳ 待真實跑期完成後誠實裁決。** 不預設立場、不 aspirational inscription(§一.10 #5 hallucination 警示 / AP-3)。
> 裁決將基於:(a) annual net-of-cost Sharpe 與 T_CZ-6 gate;(b) precision(rank IC > 0 且顯著);(c) trust(3-seed median 穩定 + calibration ≈ 0.80)。三者俱足才稱「可靠地能賺錢」。

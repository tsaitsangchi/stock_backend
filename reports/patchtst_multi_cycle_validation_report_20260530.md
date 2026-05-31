# PatchTST(時間 Patch 變換器 / Patch Time Series Transformer)股價預測 多重週期驗證報告 v0.1

**日期**: 2026-05-30
**程式**: `scripts/evaluation/multi_cycle_patchtst_validation.py` v0.1
**共同比較基準**: `reports/common_model_comparison_baseline_v1.md` v1.0(第三實作)
**動因**: 用戶 2026-05-30 directive —「依此 PatchTST(時間 Patch 變換器 / ICLR 2023)模型做法產生個別的程式進行模型產生與驗證，並確認依此模型來做預測股價真的可以賺錢嗎?…以目前核心個股依每一支個股在資料庫存在的個股個別過去最久的在 database 內存在的實際數據，來進行此預測股價模型的多次週期驗證，如 by 週、月、季、年等多重週期…請做相關的精準度與信任度分析。後續仍有其他模型都要依此方式來進行比較驗證，所以需要有相同的比較基準定義。」
**治權**: §一.10 全 DB source-traceable / §一.10 #3 multi-run / §8.5 leakage-safe / §14.7-DC v0.18 source-pure universe / §一.11 三段式標頭 / §一.12 5-min 回報

---

## 一、模型做法 (Model Methodology)

### 1.1 什麼是 PatchTST
**PatchTST**(Nie, Nguyen, Sinthong, Kalagnanam 2023 / ICLR 2023,"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers")對 Transformer 時序預測提出兩個關鍵設計:

| 設計 | 內容 | 效果 |
| :--- | :--- | :--- |
| **(1) Patching(分塊)** | 將序列切成 subseries-level **patches**(連續 P 個時點為一個 patch-token,stride STR 滑動)| token 數 = (L−P)/STR+1 ≪ L → 計算高效;每個 patch 保留**局部語義**(類比 ViT 影像分塊),attention 學 patch 與 patch 間之長距依賴 |
| **(2) Channel-Independence(通道獨立)** | 每個 variate(通道)由**共享權重之同一 backbone 獨立** forward,**通道之間不互相 attention** | 降低過擬合、提升泛化(論文核心發現:CI 反而優於 channel-mixing)|

另含 **RevIN(reversible instance normalization)**:以每個 lookback window 自身的 mean/std 標準化輸入、預測後反標準化輸出,處理分布漂移。

論文標題「64 words」即指:把一段長序列當成由數十個 patch(「字」)組成的「句子」餵給 Transformer。

實作:原始 PatchTST 架構(RevIN + patch-embed + 學習式 positional + `nn.TransformerEncoder` over patches + flatten head),**非 surrogate**。torch 2.2.2(於隔離 numpy 1.26.4 環境執行,不污染共用 venv)。

### 1.2 如何套用於台股 398 核心股(channel = stock,通道獨立)
PatchTST 之「通道(channel)」對映為「**個股**」。Channel-independence 表示:**每支股票的 weekly return 序列各自獨立**經由同一組共享權重 backbone 預測,**股與股之間不做 attention**(與 iTransformer 正相反)。

| 設計點 | 做法 | 理由 / 治權 |
| :--- | :--- | :--- |
| 每股最長歷史 | 每股取 DB 中其全部 daily 報價(adjusted close)| 用戶 directive「每一支個股過去最久的實際數據」|
| Weekly 重採樣 | 依 **ISO 日曆週** 取每週最後一筆 close → log return | 與 iTransformer 同格點,跨模型可比 |
| Return 序列 | 每股一條 weekly log return 序列;**缺週留 NaN,不 forward-fill** | §一.10 / §14.7-DC source-pure:補值=AI 幻像;含缺口之 window 直接剔除 |
| 訓練樣本 | (window × stock) pairs:每個 (結束週, 股) 取 lookback L 週 + target S 週為一筆**獨立**樣本 | channel-independence:共享 backbone,大量單變量樣本 |
| RevIN | 以每筆 lookback window 自身 mean/std 標準化(僅用 ≤ as_of 資料)| §8.5 leakage-safe(不用未來統計量)|
| Patching | P=16 週 / stride=8 → num_patches = (104−16)/8+1 = **12 patches** | 局部語義 + 計算高效 |
| 模型輸出 | 每股未來 S=52 週之 weekly return 序列(point forecast)| flatten patch tokens → linear head |
| 多週期導出 | forecast 序列 cumulative-sum → 第 1/4/12/50 週 = 5/20/60/252 交易日 forward score | 一個 model 服務 4 horizons |
| Walk-forward refit | expanding window,每 12 個 monthly panel(≈ 年度)refit 一次 | 計算可行 + 避免 look-ahead |
| Lookback / Forecast | lookback 104 weeks(≈2yr)/ forecast 52 weeks | 涵蓋 annual horizon |
| 計算預算上限 | 每次 refit 之 (window×stock) 樣本上限(default 40,000,隨機抽樣)| model hyperparameter(如 epochs/batch),不影響比較協定 |

### 1.3 三個 Transformer 模型的差異(為何各建一支)
| | TFT(第一)| iTransformer(第二)| **PatchTST(本報告,第三)** |
| :--- | :--- | :--- | :--- |
| token 是什麼 | 時間點(每股 pooled)| 整段變數序列(每股=一 token)| **序列分塊 patch(每股序列切 12 塊)** |
| attention 跨什麼 | 時間(temporal)| **變數(跨股)** | **patch(序列內,通道獨立)** |
| 跨股關聯? | 共享權重 + static embedding | **是(cross-variate attention)** | **否(channel-independent)** |
| 標準化 | GroupNormalizer | 全域 mu/sigma | **RevIN per-instance** |
| 不確定性 | QuantileLoss → P10/P90(有 calibration)| point(MSE)| **point(MSE)** |
| 賣點 | 變數重要性 + 長距依賴 | 橫截面跨股關聯 | **局部 patch 語義 + 通道獨立泛化** |

> **信任度差異(誠實揭露)**:PatchTST 為 point-forecast,**不產生 P10-P90 區間**,故「calibration coverage」這項信任度指標對本模型 **N/A**(報告中以 `—` 標示)。其餘信任度(顯著性、多 seed 穩定度)照常評估。

### 1.4 Anti-leakage(§8.5)結構性保證
- 每筆訓練樣本之 lookback 僅含 as_of 對應週**之前**(含當週)之 returns;target 全在 as_of **之後**。
- RevIN 標準化統計量僅由該 lookback window 估計(不混入未來)。
- Refit training cutoff = 該 panel 對應之 week_cut → 僅取 ≤ week_cut 之 (window, target) 樣本。
- realized forward return(評分真實標的)取自 `label_date > as_of`(與全模型相同 query,§三)。

---

## 二、實際市場價格資料 (Actual Market Price Data — 全來自 DB,§一.10 (b))

> 本節數字全部來自 2026-05-30 對 PostgreSQL 之實際 query(universe SQL + `TaiwanStockPriceAdj`),非估算、非記憶。

### 2.1 Universe 價格涵蓋(`TaiwanStockPriceAdj`,2026-05-30 query)
- 核心股:**398**(v0.18 source-pure pan-historical,最新 committed snapshot)。
- 價格列數:**2,320,497**(close>0);日期跨度 **1992-01-04 → 2026-05-22**(~34 年)。
- 每股「最久歷史」起始日:最早 **1992-01-04** / 中位 **2003-01-22** / 最晚 **2015-11-10**。
- 每股交易日數:min **2,559** / median **5,667** / max **8,767**。
  → 確認「每股最長歷史」之 directive 可滿足:**最短者仍有 2,559 交易日(≈10 年 ≈ ~520 weekly bars)**,足供 PatchTST lookback(104wk)+ forecast(52wk)+ walk-forward。

### 2.2 真實 forward 報酬之市場變化(cross-sectional,398 股,4 regime 抽樣)
> 展示「市場價格在 database 內之實際變化」—— 同一 universe 在不同 regime 之 realized forward log return（mean ± std,2026-05-30 query）。

| as_of | weekly 5d | monthly 20d | quarterly 60d | annual 252d | regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-01-15 | +1.34% ±3.33 | +3.42% ±5.90 | +9.62% ±10.17 | +18.97% ±23.70 | 多頭 |
| 2020-03-16 | **−12.43%** ±8.45 | +2.09% ±9.13 | +18.14% ±13.73 | **+43.17%** ±27.92 | COVID 崩盤→V 反彈 |
| 2022-10-17 | −0.91% ±3.67 | +4.34% ±5.90 | +11.87% ±11.14 | +33.92% ±26.87 | 空頭末段→反彈 |
| 2024-08-15 | +2.53% ±3.73 | −1.49% ±6.29 | +1.99% ±12.90 | **−18.39%** ±21.53 | 近期回檔 |

**觀察**:(1) 報酬離散度隨 horizon 單調放大(weekly ±3% → annual ±20–28%);(2) annual mean 橫跨 +43%(post-COVID)到 −18%(2024→2025)→ **OOS 期間確實涵蓋多 regime**,非單一牛市過擬合。此即 PatchTST 須在其上證明「能否賺錢」之真實市場。

> 註:as_of 落在週末(無當日報價)之 panel,因 `t0` 無報價而被丟棄 —— 此為**全模型共用之 convention**(baseline tree 模型、TFT、iTransformer 亦同),故仍 apples-to-apples。

---

## 三、共同比較基準 (Common Baseline — 與全模型一致)

完整定義見 `reports/common_model_comparison_baseline_v1.md`(本程式為其**第三實作**,TFT 第一、iTransformer 第二)。摘要:universe v0.18/398 × 95 monthly panels ×
真實 forward log return(`TaiwanStockPriceAdj`)× 4 horizons(5/20/60/252d)× top-20 equal-weight long × 0.6% cost ×
{Sharpe, Win, Eff-t, T_CZ-6 gate(annual Eff t≥4.20 / Sharpe≥2.40 / Win≥0.79)}。

**核心原則**:模型可用各自 natural representation(trees=38 features / TFT=每股 weekly 序列 / iTransformer=跨股 return 矩陣 / **PatchTST=channel-independent patched 單變量序列**),但「評估協定」與「真實標的」完全相同 —— 比較點在 **OUTPUT 預測品質**,不在 input 表徵。PatchTST 與既有 10 個 tree/ensemble 模型 + TFT + iTransformer 走完全相同協定。

---

## 四、精準度 / 信任度 / 賺錢能力 框架 (Precision / Trust / Profitability)

| 類別 | 指標 | 回答的問題 | PatchTST 適用性 |
| :--- | :--- | :--- | :--- |
| **精準度** | rank IC / directional accuracy / RMSE / MAE / R² | 「PatchTST 預測準不準?」| ✅ 全適用 |
| **信任度** | Effective t-stat 顯著性 / 多 seed 穩定度(min/median/max/mean)/ P10-P90 calibration | 「可不可信、穩不穩、不確定性誠不誠實?」| ⚠️ calibration **N/A**(point-forecast 無區間);其餘適用 |
| **賺錢能力** | net-of-cost Sharpe / Win / annualized net return / MDD / T_CZ-6 gate | 「**真的能賺錢嗎?**」| ✅ 全適用 |

---

## 五、驗證結果 (Validation Results)

> **狀態:⏳ PENDING — 待真實 3-seed 跑期完成**(per §一.10 #3 multi-run + AP-3 反 aspirational inscription:結果未跑出前不填數字)。

執行計畫:
1. **Smoke**(40 股 × 6 panels × 2 epochs)— plumbing 驗證(於隔離 numpy 1.26.4 環境)。**smoke 數字為 throwaway,不入結果表**(AP-3)。
2. 依 smoke 計時估算完整跑期(398 股 × 95 panels × ~8 refits × 3 seeds {5422,7331,1009})。
3. 為避免與 Step G sweep + TFT + iTransformer 跑期之 CPU 競爭,**PatchTST 完整跑期排在其後**啟動(SHMM + §一.12 5-min 回報)。
4. 跑完 → `reports/patchtst_v0/ptst_s{5422,7331,1009}.json` → 聚合 min/median/max/mean → 回填下表。

### 5.1 跨週期結果矩陣(median of 3 seeds)— 待填
| Horizon | n_panels | Eff t | Sig? | Sharpe | Win | NetAnn | rank IC | DirAcc | RMSE | MAE | R² |
| :--- | ---: | ---: | :--: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weekly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| monthly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| quarterly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| annual | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### 5.2 信任度:多 seed 穩定度 + calibration — 待填
- 多 seed 穩定度(min/median/max/mean/spread):⏳
- Calibration(P10-P90 coverage):**N/A**(PatchTST 為 point-forecast,不產生 quantile 區間)。

### 5.3 T_CZ-6 annual gate 裁決(對 median)— 待填
### 5.4 與既有模型 + TFT + iTransformer 對照(同基準)— 待 Step G sweep 聚合 + 三 neural 模型跑完

---

## 六、結論:PatchTST 預測股價能賺錢嗎? (Can PatchTST Make Money?)

> **⏳ 待真實跑期完成後誠實裁決。** 不預設立場、不 aspirational inscription(§一.10 #5 hallucination 警示 / AP-3)。
> 裁決將基於:(a) annual net-of-cost Sharpe 與 T_CZ-6 gate;(b) precision(rank IC > 0 且顯著);(c) trust(3-seed median 穩定;calibration 因 point-forecast N/A,以顯著性 + 穩定度替代)。三者俱足才稱「可靠地能賺錢」。

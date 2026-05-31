# iTransformer(倒置變換器 / Inverted Transformer)股價預測 多重週期驗證報告 v0.1

**日期**: 2026-05-30
**程式**: `scripts/evaluation/multi_cycle_itransformer_validation.py` v0.1
**共同比較基準**: `reports/common_model_comparison_baseline_v1.md` v1.0(第二實作)
**動因**: 用戶 2026-05-30 directive —「依此 iTransformer(倒置變換器 / ICLR 2024)模型做法產生個別的程式進行模型產生與驗證，並確認依此模型來做預測股價真的可以賺錢嗎?…以目前核心個股依每一支個股在資料庫存在的個股個別過去最久的在 database 內存在的實際數據，來進行此預測股價模型的多次週期驗證，如 by 週、月、季、年等多重週期…請做相關的精準度與信任度分析。後續仍有其他模型都要依此方式來進行比較驗證，所以需要有相同的比較基準定義。」
**治權**: §一.10 全 DB source-traceable / §一.10 #3 multi-run / §8.5 leakage-safe / §14.7-DC v0.18 source-pure universe / §一.11 三段式標頭 / §一.12 5-min 回報

---

## 一、模型做法 (Model Methodology)

### 1.1 什麼是 iTransformer
**iTransformer**(Liu, Wang, Wu, Hu, Dong, Long 2024 / ICLR 2024,"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting")對標準 Transformer 做了一個關鍵的**倒置(inversion)**:

| | 標準 Transformer 時序做法 | **iTransformer(倒置)** |
| :--- | :--- | :--- |
| token 是什麼 | 每個**時間點**(含所有變數)為一個 token | 每個**變數(variate)的整段 lookback 序列**為一個 token |
| attention 跨什麼 | 跨時間點(temporal attention)| **跨變數(cross-variate attention)** → 學變數間相關性 |
| FFN 作用維度 | 跨變數混合 | 在 token(變數)表徵維度上,逐變數獨立學時序型態 |
| 輸出 | 投影回未來時點 | 由變數 token 投影出該變數之未來 horizon |

**核心價值**:倒置後,self-attention 直接建模**變數與變數之間的關聯**——在金融場景中,這正是「個股之間的橫截面關聯(cross-sectional correlation)」。

實作:原始 inverted-Transformer 架構(`nn.TransformerEncoder` + 倒置 embedding/head),**非 surrogate**。torch 2.2.2(於隔離 numpy 1.26.4 環境執行,不污染共用 venv)。

### 1.2 如何套用於台股 398 核心股(variates = stocks)
本驗證將 iTransformer 之「變數(variate)」對映為「**個股**」——即把 N 支核心股的 weekly return 序列當作 N 個 variate-token,讓 cross-variate attention 直接學**跨股關聯**(此即論文核心價值在金融的落點)。

| 設計點 | 做法 | 理由 / 治權 |
| :--- | :--- | :--- |
| 每股最長歷史 | 每股取 DB 中其全部 daily 報價(adjusted close)| 用戶 directive「每一支個股過去最久的實際數據」|
| Weekly 重採樣 | 依 **ISO 日曆週**(year, week)取每週最後一筆 close,組成共同週格點 | 跨股對齊到同一時間軸(cross-variate attention 需共格點)|
| Return 矩陣 | `ret_mat[W×N]` = 各週 log return;**缺週留 NaN,不 forward-fill** | §一.10 / §14.7-DC source-pure:補值=AI 幻像,跨缺口之 window 直接剔除 |
| 輸入特徵 | **純 weekly log return 序列**(每股 = 一個 variate token)| §一.13:price 之 mathematical transform,0 imputed、0 hardcoded knowledge |
| 模型輸出 | 每股未來 S=52 週之 weekly return 序列(point forecast)| 倒置 head:variate token → 未來 horizon |
| 多週期導出 | decoder 序列 cumulative-sum → 第 1/4/12/50 週 = 5/20/60/252 交易日 forward score | 一個 model 服務 4 horizons |
| Walk-forward refit | expanding window,每 12 個 monthly panel(≈ 年度)refit 一次 | 計算可行 + 避免 look-ahead |
| Lookback / Forecast | lookback 104 weeks(≈2yr)/ forecast 52 weeks | 涵蓋 annual horizon |
| 變動變數數 | attention `key_padding_mask` 遮蔽 lookback 不完整之股 | iTransformer 天然支援變動 variate 數 |

### 1.3 與 TFT 的差異(為何要兩個模型)
| | TFT(第一實作)| **iTransformer(本報告)** |
| :--- | :--- | :--- |
| 表徵 | 每股獨立 weekly 序列,pooled panel(共享權重 + static embedding)| **跨股 return 矩陣,股與股之間 attention** |
| attention | interpretable multi-head(跨時間)| **跨變數(跨股)** |
| 不確定性 | QuantileLoss → P10/P50/P90 → 有 calibration | **point forecast(MSE)→ 無 quantile → calibration N/A** |
| 賣點 | 變數重要性 + 長距時序依賴 | **橫截面跨股關聯** |

> **信任度差異(誠實揭露)**:iTransformer 為 point-forecast,**不產生 P10-P90 區間**,故「calibration coverage」這項信任度指標對本模型 **N/A**(報告中以 `—` 標示)。其餘信任度(顯著性、多 seed 穩定度)照常評估。

### 1.4 Anti-leakage(§8.5)結構性保證
- Lookback 僅含 as_of 對應週**之前**(含當週)之 returns;forecast 預測 as_of **之後** → 預測時點不見未來。
- 全域標準化(mu/sigma)僅由 ≤ week_cut 之過去 returns 估計,不混入未來。
- Refit training cutoff = 各 panel 對應之 week_cut → 不混入未來週。
- realized forward return(評分真實標的)取自 `label_date > as_of`(與全模型相同 query,§三)。

---

## 二、實際市場價格資料 (Actual Market Price Data — 全來自 DB,§一.10 (b))

> 本節數字全部來自 2026-05-30 對 PostgreSQL 之實際 query(universe SQL + `TaiwanStockPriceAdj`),非估算、非記憶。

### 2.1 Universe 價格涵蓋(`TaiwanStockPriceAdj`,2026-05-30 query)
- 核心股:**398**(v0.18 source-pure pan-historical,最新 committed snapshot)。
- 價格列數:**2,320,497**(close>0);日期跨度 **1992-01-04 → 2026-05-22**(~34 年)。
- 每股「最久歷史」起始日:最早 **1992-01-04** / 中位 **2003-01-22** / 最晚 **2015-11-10**。
- 每股交易日數:min **2,559** / median **5,667** / max **8,767**。
  → 確認「每股最長歷史」之 directive 可滿足:**最短者仍有 2,559 交易日(≈10 年 ≈ ~520 weekly bars)**,足供 iTransformer lookback(104wk)+ forecast(52wk)+ walk-forward。

### 2.2 真實 forward 報酬之市場變化(cross-sectional,398 股,4 regime 抽樣)
> 展示「市場價格在 database 內之實際變化」—— 同一 universe 在不同 regime 之 realized forward log return（mean ± std,2026-05-30 query）。

| as_of | weekly 5d | monthly 20d | quarterly 60d | annual 252d | regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-01-15 | +1.34% ±3.33 | +3.42% ±5.90 | +9.62% ±10.17 | +18.97% ±23.70 | 多頭 |
| 2020-03-16 | **−12.43%** ±8.45 | +2.09% ±9.13 | +18.14% ±13.73 | **+43.17%** ±27.92 | COVID 崩盤→V 反彈 |
| 2022-10-17 | −0.91% ±3.67 | +4.34% ±5.90 | +11.87% ±11.14 | +33.92% ±26.87 | 空頭末段→反彈 |
| 2024-08-15 | +2.53% ±3.73 | −1.49% ±6.29 | +1.99% ±12.90 | **−18.39%** ±21.53 | 近期回檔 |

**觀察**:(1) 報酬離散度隨 horizon 單調放大(weekly ±3% → annual ±20–28%);(2) annual mean 橫跨 +43%(post-COVID)到 −18%(2024→2025)→ **OOS 期間確實涵蓋多 regime**,非單一牛市過擬合。此即 iTransformer 須在其上證明「能否賺錢」之真實市場。

> 註:as_of 落在週末(無當日報價)之 panel,因 `t0` 無報價而被丟棄 —— 此為**全模型共用之 convention**(baseline tree 模型、TFT 亦同),故仍 apples-to-apples。

---

## 三、共同比較基準 (Common Baseline — 與全模型一致)

完整定義見 `reports/common_model_comparison_baseline_v1.md`(本程式為其**第二實作**,TFT 為第一實作)。摘要:universe v0.18/398 × 95 monthly panels ×
真實 forward log return(`TaiwanStockPriceAdj`)× 4 horizons(5/20/60/252d)× top-20 equal-weight long × 0.6% cost ×
{Sharpe, Win, Eff-t, T_CZ-6 gate(annual Eff t≥4.20 / Sharpe≥2.40 / Win≥0.79)}。

**核心原則**:模型可用各自 natural representation(trees=38 features / TFT=每股 weekly 序列 / **iTransformer=跨股 return 矩陣**),但「評估協定」與「真實標的」完全相同 —— 比較點在 **OUTPUT 預測品質**,不在 input 表徵。iTransformer 與既有 10 個 tree/ensemble 模型 + TFT 走完全相同協定。

---

## 四、精準度 / 信任度 / 賺錢能力 框架 (Precision / Trust / Profitability)

| 類別 | 指標 | 回答的問題 | iTransformer 適用性 |
| :--- | :--- | :--- | :--- |
| **精準度** | rank IC / directional accuracy / RMSE / MAE / R² | 「iTransformer 預測準不準?」| ✅ 全適用 |
| **信任度** | Effective t-stat 顯著性 / 多 seed 穩定度(min/median/max/mean)/ P10-P90 calibration | 「可不可信、穩不穩、不確定性誠不誠實?」| ⚠️ calibration **N/A**(point-forecast 無區間);其餘適用 |
| **賺錢能力** | net-of-cost Sharpe / Win / annualized net return / MDD / T_CZ-6 gate | 「**真的能賺錢嗎?**」| ✅ 全適用 |

---

## 五、驗證結果 (Validation Results)

> **狀態:⏳ PENDING — 待真實 3-seed 跑期完成**(per §一.10 #3 multi-run + AP-3 反 aspirational inscription:結果未跑出前不填數字)。

執行計畫:
1. **Smoke**(40 股 × 6 panels × 2 epochs)— ✅ **2026-05-30 通過**(exit 0,於隔離 numpy 1.26.4 環境):pipeline 全程驗證(universe→return-matrix→train→predict→4-horizon metrics→T_CZ-6→JSON)。單 refit 訓練 **8.1s**(return-matrix 載入 40 股 108.8s 為一次性成本)。**smoke 數字為 throwaway(40 股 / 2 epochs),不入結果表**(AP-3)。
2. 依 smoke 計時估算完整跑期(398 股 × 95 panels × ~8 refits × 3 seeds {5422,7331,1009}):iTransformer 訓練極輕(refit 8.1s vs TFT 369s),主成本為 398 股 return-matrix 一次性載入(~15min)。
3. 為避免與 Step G sweep + TFT 跑期之 CPU 競爭,**iTransformer 完整跑期排在 sweep 完成後**啟動(SHMM + §一.12 5-min 回報)。
4. 跑完 → `reports/itransformer_v0/itr_s{5422,7331,1009}.json` → 聚合 min/median/max/mean → 回填下表。

### 5.1 跨週期結果矩陣(median of 3 seeds)— 待填
| Horizon | n_panels | Eff t | Sig? | Sharpe | Win | NetAnn | rank IC | DirAcc | RMSE | MAE | R² |
| :--- | ---: | ---: | :--: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weekly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| monthly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| quarterly | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| annual | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### 5.2 信任度:多 seed 穩定度 + calibration — 待填
- 多 seed 穩定度(min/median/max/mean/spread):⏳
- Calibration(P10-P90 coverage):**N/A**(iTransformer 為 point-forecast,不產生 quantile 區間)。

### 5.3 T_CZ-6 annual gate 裁決(對 median)— 待填
### 5.4 與既有模型 + TFT 對照(同基準)— 待 Step G sweep 聚合 + iTransformer/TFT 跑完

---

## 六、結論:iTransformer 預測股價能賺錢嗎? (Can iTransformer Make Money?)

> **⏳ 待真實跑期完成後誠實裁決。** 不預設立場、不 aspirational inscription(§一.10 #5 hallucination 警示 / AP-3)。
> 裁決將基於:(a) annual net-of-cost Sharpe 與 T_CZ-6 gate;(b) precision(rank IC > 0 且顯著);(c) trust(3-seed median 穩定;calibration 因 point-forecast N/A,以顯著性 + 穩定度替代)。三者俱足才稱「可靠地能賺錢」。

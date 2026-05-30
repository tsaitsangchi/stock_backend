# Chronos(Amazon 時序基礎模型 / Time-Series Foundation Model)股價預測 多重週期驗證報告 v0.1

**日期**: 2026-05-30
**程式**: `scripts/evaluation/multi_cycle_chronos_validation.py` v0.1
**共同比較基準**: `reports/common_model_comparison_baseline_v1.md` v1.0(**第四實作**)
**動因**: 用戶 2026-05-30 兩請求合一 ——「依此 **Foundation Models(2024-2026 新興時序基礎模型)** 模型做法…」+「依此 **TimesFM(時序基礎模型 / Google)** 模型做法產生個別的程式進行模型產生與驗證,並確認依此模型來做預測股價真的可以賺錢嗎?…以目前核心個股依每一支個股在資料庫存在的個股個別過去最久的在 database 內存在的實際數據…by 週、月、季、年等多重週期…請做相關的精準度與信任度分析…需要有相同的比較基準定義。」
**治權**: §一.10 **INPUT** 全 DB source-traceable / §一.10 #3 multi-run / §8.5 leakage-safe / §14.7-DC v0.18 source-pure universe / §一.11 三段式標頭 / §一.12 5-min 回報

---

## ⚠️ 0. 治權前置揭露 — 外部預訓練先驗 (MANDATORY CAVEAT — External-Pretrained Prior)

> **本模型與先前三支(TFT / iTransformer / PatchTST)在治權上有本質差異,用戶於 2026-05-30 explicit 選擇「Real Chronos + disclose caveat」並要求醒目揭露,故置於報告最前。**

| 維度 | from-scratch 模型(TFT / iTransformer / PatchTST)| **Chronos(本報告)** |
| :--- | :--- | :--- |
| 模型權重學自 | **只學自本 DB**(從零訓練於 TaiwanStockPriceAdj)| ⚠️ **預訓練於本 DB 之外之天量外部時序語料**(Amazon 釋出之 pretrained checkpoint)|
| INPUT context | 本 DB weekly 序列 | 本 DB weekly close 序列(**100% source-pure**)|
| 預測先驗(prior)| DB-traceable(權重由 DB 資料導出)| ⚠️ **NOT DB/FinMind/FRED-traceable**(編碼外部知識)|
| §一.10 / §14.7-DC 對齊 | 完全對齊(INPUT + 權重皆 DB)| **部分**:INPUT 對齊;**權重為新 source 類別「外部預訓練先驗」** |

**白話**:Chronos 是一個「已經在外部海量時序上學過」的基礎模型,我們**沒有**用本 DB 重新訓練它(zero-shot),只把每支台股的真實 weekly 收盤序列餵進去、要它預測未來。
- ✅ **餵進去的數字 100% 來自本 DB**(TaiwanStockPriceAdj,§一.10 (b)),無 imputed、無 forward-fill。
- ⚠️ **但模型「為什麼這樣預測」的那套先驗,來自本 DB 以外的知識** —— 這在本專案九次強調的 source-purity 教義下,是一個**新的、非 DB 可溯源的 source 類別**。用戶已 explicit 知情並授權,授權條件即「醒目揭露本 caveat」。

**關於 TimesFM(Google)**:用戶原請求之 **TimesFM 無法在本機(Intel Mac x86_64)安裝** —— PyPI 上 `timesfm==1.0.0` 依賴 `paxml → lingvo==0.12.7`,而 `lingvo` 僅有 Linux wheel(本機探測 `from versions: none`)。故以**真正之 Amazon Chronos**(可安裝 + checkpoint 可下載,2026-05-30 探測 HuggingFace HTTP 200)作為「時序基礎模型族」之**代表實作**,同時回應 Foundation Models 與 TimesFM 兩請求。**本程式為真正之 pretrained Chronos,非 from-scratch surrogate**(不冒充)。

---

## 一、模型做法 (Model Methodology)

### 1.1 什麼是 Chronos / 時序基礎模型
**Chronos**(Ansari et al. 2024,Amazon,"Chronos: Learning the Language of Time Series")是一個**時序基礎模型(time-series foundation model)**:把連續數值序列經 **scaling + quantization** 轉成「token」,再以**語言模型**(T5 backbone)方式學「下一個 token」。它在**外部天量真實 + 合成時序語料**上**預訓練**,使用時 **zero-shot**(免在目標資料上重訓)即可對任意新序列做**機率預測**。

| 設計 | 內容 |
| :--- | :--- |
| Tokenization | 序列值 scaling(以 context 自身尺度)後量化成有限 vocabulary tokens |
| Backbone | T5(encoder-decoder)/ Bolt 變體(direct multi-step,更快)|
| 預訓練 | ⚠️ 外部語料(見 §0 caveat),非本 DB |
| 使用方式 | **zero-shot**:給 context → 直接 forecast 未來 N 步之**分位數(P10/P50/P90)** |
| 機率性 | ✅ 機率預測 → **有 P10/P90 → calibration 可計算**(與 TFT 同) |

> 本實作預設 `amazon/chronos-bolt-small`(Chronos-Bolt,2024,direct multi-step 分位輸出,CPU 友善);smoke 用 `chronos-bolt-tiny`。torch 2.2.2 於**隔離 venv_fm**(numpy 1.26.4)執行,**完全不污染主 venv**。

### 1.2 如何套用於台股 398 核心股(zero-shot,每股單變量 price)
| 設計點 | 做法 | 理由 / 治權 |
| :--- | :--- | :--- |
| 每股最長歷史 | 每股取 DB 全部 daily adjusted close | 用戶 directive「每一支個股過去最久的實際數據」|
| Weekly 重採樣 | 依 **ISO 日曆週** 取每週最後一筆 close(**價格**,非 return)| 與前三模型同日曆格點 |
| Context | 每股取最近 **≤256 個真實觀測週**(歷史愈長 context 愈完整)| §一.5 longest-available;**缺週不補值,只取真實觀測** |
| 預測 | `predict_quantiles` → 未來 52 週 price path 之 **P10 / P50 / P90** | 機率預測契約 §9.1 |
| Score(預測 forward 報酬)| `score = ln(median_price[第 k 週] / last_close)` | 與 realized log-return 標的同尺度 |
| 多週期導出 | 取 forecast 第 **1 / 4 / 12 / 50** 週 = 5/20/60/252 交易日 | 一個 model 服務 4 horizons |
| Calibration | realized return ∈ [`ln(P10/last)`, `ln(P90/last)`] 之比率 | ✅ **Chronos 可算**(機率模型)|
| Walk-forward | **zero-shot → 無 refit、無訓練**(基礎模型先驗固定)| 基礎模型特性 |

### 1.3 四個「序列型」模型的差異(為何各建一支)
| | TFT(第一)| iTransformer(第二)| PatchTST(第三)| **Chronos(本報告,第四)** |
| :--- | :--- | :--- | :--- | :--- |
| 類型 | 從零訓練 | 從零訓練 | 從零訓練 | ⚠️ **預訓練基礎模型(zero-shot)** |
| 學自 | 本 DB | 本 DB | 本 DB | ⚠️ **外部語料** |
| token 是什麼 | 時間點 | 整段變數序列(每股一 token)| 序列分塊 patch | 量化後之序列值 token |
| 跨股關聯? | 共享權重 | **是(cross-variate)** | 否(channel-independent)| 否(每股獨立 zero-shot)|
| INPUT | weekly 序列 | 跨股 return 矩陣 | channel-independent patched 序列 | **每股 weekly close** |
| 不確定性 | QuantileLoss → P10/P90(**有 calibration**)| point(MSE)| point(MSE)| **機率分位 → 有 calibration** |

> **信任度差異(誠實揭露)**:Chronos 為**機率模型**,**可產生 P10-P90 區間** → 「calibration coverage」這項信任度指標**對本模型可計算**(與 TFT 同;與 iTransformer / PatchTST 之 point-forecast N/A 不同)。
> ⚠️ **但多 seed 穩定度**:Chronos-Bolt zero-shot 為 **~deterministic**(直接分位輸出,無 autoregressive sampling)→ 跨 seed spread ≈ 0;此 determinism 本身為一 trust 觀察(無 seed 變異),報告將誠實揭露,不以 spread 假裝 stochastic 穩定度。

### 1.4 Anti-leakage(§8.5)結構性保證
- Context 僅含 week_date **≤ as_of** 之真實觀測 close;forecast target 全在 as_of **之後**。
- 每股 scaling 統計量由 Chronos 以該 context 自身估計(不混入未來)。
- realized forward return(評分真實標的)取自 `label_date > as_of`(與全模型相同 query,§三)。
- zero-shot → 無「用未來資料訓練」之可能(模型權重在見到台股前即固定)。

---

## 二、實際市場價格資料 (Actual Market Price Data — 全來自 DB,§一.10 (b))

> 本節數字全部來自 2026-05-30 對 PostgreSQL 之實際 query(universe SQL + `TaiwanStockPriceAdj`),非估算、非記憶。Chronos 餵入之 context 即下列真實 weekly close。

### 2.1 Universe 價格涵蓋(`TaiwanStockPriceAdj`,2026-05-30 query)
- 核心股:**398**(v0.18 source-pure pan-historical,最新 committed snapshot)。
- 價格列數:**2,320,497**(close>0);日期跨度 **1992-01-04 → 2026-05-22**(~34 年)。
- 每股「最久歷史」起始日:最早 **1992-01-04** / 中位 **2003-01-22** / 最晚 **2015-11-10**。
- 每股交易日數:min **2,559** / median **5,667** / max **8,767**。
  → 確認「每股最長歷史」directive 可滿足:**最短者仍有 2,559 交易日(≈10 年 ≈ ~520 weekly bars)**,遠超 Chronos context 下限(≥60 觀測週)。

### 2.2 真實 forward 報酬之市場變化(cross-sectional,398 股,4 regime 抽樣)
> 同一 universe 在不同 regime 之 realized forward log return（mean ± std,2026-05-30 query）—— 此即 Chronos 須證明「能否賺錢」之真實市場。

| as_of | weekly 5d | monthly 20d | quarterly 60d | annual 252d | regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-01-15 | +1.34% ±3.33 | +3.42% ±5.90 | +9.62% ±10.17 | +18.97% ±23.70 | 多頭 |
| 2020-03-16 | **−12.43%** ±8.45 | +2.09% ±9.13 | +18.14% ±13.73 | **+43.17%** ±27.92 | COVID 崩盤→V 反彈 |
| 2022-10-17 | −0.91% ±3.67 | +4.34% ±5.90 | +11.87% ±11.14 | +33.92% ±26.87 | 空頭末段→反彈 |
| 2024-08-15 | +2.53% ±3.73 | −1.49% ±6.29 | +1.99% ±12.90 | **−18.39%** ±21.53 | 近期回檔 |

**觀察**:(1) 報酬離散度隨 horizon 單調放大;(2) annual mean 橫跨 +43%(post-COVID)到 −18%(2024→2025)→ **OOS 期間確實涵蓋多 regime**。

> 註:as_of 落在週末(無當日報價)之 panel,因 `t0` 無報價而被丟棄 —— 此為**全模型共用之 convention**,仍 apples-to-apples。

---

## 三、共同比較基準 (Common Baseline — 與全模型一致)

完整定義見 `reports/common_model_comparison_baseline_v1.md`(本程式為其**第四實作**;TFT 第一、iTransformer 第二、PatchTST 第三)。摘要:universe v0.18/398 × 95 monthly panels ×
真實 forward log return(`TaiwanStockPriceAdj`)× 4 horizons(5/20/60/252d)× top-20 equal-weight long × 0.6% cost ×
{Sharpe, Win, Eff-t, T_CZ-6 gate(annual Eff t≥4.20 / Sharpe≥2.40 / Win≥0.79)}。

**核心原則**:模型可用各自 natural representation(trees=38 features / TFT=每股序列 / iTransformer=跨股矩陣 / PatchTST=patched 序列 / **Chronos=每股 zero-shot price 序列**),但「評估協定」與「真實標的」完全相同 —— 比較點在 **OUTPUT 預測品質**,不在 input 表徵,**亦不在「模型如何取得其權重」**(此即 §0 caveat 之所在:評估協定相同,但 Chronos 之權重來源與他模型不同,須並列揭露)。

---

## 四、精準度 / 信任度 / 賺錢能力 框架 (Precision / Trust / Profitability)

| 類別 | 指標 | 回答的問題 | Chronos 適用性 |
| :--- | :--- | :--- | :--- |
| **精準度** | rank IC / directional accuracy / RMSE / MAE / R² | 「Chronos 預測準不準?」| ✅ 全適用 |
| **信任度** | Effective t-stat 顯著性 / 多 seed 穩定度 / **P10-P90 calibration** | 「可不可信、不確定性誠不誠實?」| ✅ **calibration 可算**(機率模型);⚠️ 多 seed:Bolt deterministic → spread≈0(誠實揭露)|
| **賺錢能力** | net-of-cost Sharpe / Win / annualized net / MDD / T_CZ-6 gate | 「**真的能賺錢嗎?**」| ✅ 全適用 |
| **治權** | 外部預訓練先驗揭露 | 「這個 alpha 是本 DB 來的嗎?」| ⚠️ 見 §0 —— 即使賺錢,須揭露 alpha 部分來自外部先驗 |

---

## 五、驗證結果 (Validation Results)

> **狀態:✅ 完成** — 真實跑期 2026-05-30 完成,3 seeds {5422,7331,1009} 全跑完(source: `reports/chronos_v0/chronos_s{5422,7331,1009}.json`,§一.10 (a) 程式輸出)。

**跑期 metadata**:`amazon/chronos-bolt-small` × 398 股 × 95 monthly panels × context 256wk / pred 52wk / batch 32;3 seeds run_at 07:45 / 08:00 / 08:14;每 seed ~906s(~15 min)。

**⚠️ 多 seed determinism(誠實揭露,非 spread 假裝穩定)**:3 seeds 之 **4-horizon metric body 經 md5 比對為 byte-identical**(`2c02a719…`,3 個不同 run_at 證實為 3 次獨立跑期)。Chronos-Bolt 為 **direct 分位輸出、無 autoregressive sampling** → seed **不影響輸出**。故 §一.10 #3「≥3 runs」之 multi-run 要求**已滿足**,但 **spread = 0 為 by-construction**(deterministic model),**非** single-run anchor 偽裝;下表 median = min = max = mean。

### 5.1 跨週期結果矩陣(3 seeds 一致 → median = mean,spread=0)
| Horizon | n_panels | Eff t | Sig p<.05? | Sharpe | Win | NetAnn | rank IC | DirAcc | RMSE | MAE | R² | Calib(P10-P90)|
| :--- | ---: | ---: | :--: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weekly | 66 | +0.72 | ❌ | +0.69 | 65.2% | +0.89% | **−0.0080** | 49.9% | 0.0503 | 0.0348 | **−0.169** | **0.871** |
| monthly | 66 | +1.47 | ❌ | +0.98 | 62.1% | **+14.30%** | +0.0049 | 51.2% | 0.0943 | 0.0653 | **−0.107** | 0.849 |
| quarterly | 65 | +0.29 | ❌ | +0.86 | 63.1% | +7.44% | +0.0254 | 52.1% | 0.1643 | 0.1151 | **−0.186** | 0.833 |
| annual | 61 | **−0.70** | ❌ | +1.39 | 68.9% | +7.03% | +0.0241 | 54.1% | 0.4178 | 0.2753 | **−0.769** | 0.760 |

> Eff t = effective t-stat(annual 經 overlap 調整 n_eff=7.26;quarterly n_eff=32.5)。**4 個 horizon 全部 not significant at p<0.05**(|Eff t| 全 < 1.97)。

### 5.2 信任度:calibration + 多 seed
- **Calibration(P10-P90 coverage,理想≈0.80)= Chronos 唯一真正強項**:weekly **0.871** / monthly **0.849** / quarterly **0.833** / annual **0.760**。四週期皆**接近理想 0.80**(weekly 略偏寬、annual 略偏窄),代表其機率區間**誠實**——「說 80% 信心區間」實際涵蓋率確實 ~76-87%。此為唯一勝過 from-scratch point models(iTransformer/PatchTST 無區間可算)之維度。
- **多 seed 穩定度**:spread = **0**(deterministic,§5 揭露)。穩定但**非** stochastic 穩健性之證據。

### 5.3 T_CZ-6 annual gate 裁決
| 門檻 | 要求 | Chronos annual 實測 | 通過? |
| :--- | :--- | :--- | :--: |
| Effective t-stat | ≥ 4.20 | **−0.70** | ❌ |
| Sharpe | ≥ 2.40 | +1.39 | ❌ |
| Win rate | ≥ 0.79 | 0.689 | ❌ |

> **裁決:❌ FAIL — 三項全不過**。annual Eff t 為**負**(−0.70):annual long 組合 **α = −2.45%/panel、IR = −0.90** → **跑輸單純持有 universe 等權基準**(雖未顯著)。annual 之 +1.39 Sharpe / +7.03% net 僅為 long leg 之原始報酬,**α 為負**。

### 5.4 與既有模型對照(同基準)— Step H 全模型聚合時補完整表
> 本報告 §5.4 之完整跨模型表待 TFT / iTransformer / PatchTST 之真實 3-seed 跑期完成後於 v0.22 Step H 一次性聚合(AP-3:他模型結果未跑出前不填代填數字)。**唯一可立即定錨之共同標尺 = T_CZ-6 gate**:Chronos 與全 tree/transformer 模型同樣以「annual Eff t≥4.20 / Sharpe≥2.40 / Win≥0.79」評斷 → **Chronos 三項全 FAIL**(§5.3)。
> ⚠️ **不在此處填既有 tree 模型數字**:既有 charter 內 tree 模型 metrics(如 v0.7 XGBoost dedicated annual Eff t 4.37)係 **prior-machine ad-hoc run on universe v0.16/1002**,**未在本 baseline(v0.18/398)reproduce**(per §14.7-DC v0.8 揭露);填入即構成 §一.10 #5 之 asymmetric-source 對比(一邊真跑 / 一邊跨機跨 universe 記憶值)。Step H 須**全模型在 v0.18/398 同 baseline 重跑**後才可並列。

---

## 六、結論:Chronos 預測股價能賺錢嗎? (Can Chronos Make Money?)

> **裁決:❌ 在本核心股 universe(v0.18/398)以 zero-shot 方式,Chronos 不能可靠賺錢。** 基於真實 3-seed 跑期(§5,deterministic),逐維度誠實裁決:

**1. 賺錢能力(profitability)— ❌ 不過**
- **T_CZ-6 annual gate 三項全 FAIL**(Eff t −0.70 / Sharpe 1.39 / Win 0.689,§5.3)。
- annual horizon **α 為負(−2.45%/panel,IR −0.90)**→ **跑輸單純等權持有 universe**;表面 +7.03% net 全來自 beta(大盤本身漲),非選股 alpha。
- 最佳 horizon 為 monthly(+14.30% net / Sharpe 0.98),但 **Eff t +1.47 仍不顯著**(< 1.97)→ 無統計信心斷言其非運氣。

**2. 精準度(precision)— ❌ 弱**
- **R² 四週期全為負**(−0.107 ~ −0.769)→ 其 point 預測**比直接猜平均值還差**。
- **directional accuracy ~50%**(49.9% ~ 54.1%)→ 漲跌方向近乎擲硬幣。
- **rank IC 極小且 weekly 為負**(−0.0080 ~ +0.0254)→ 選股排序能力微弱。

**3. 信任度(trust)— ⚠️ 混合,僅 calibration 為真強項**
- ✅ **Calibration 0.76-0.87(理想 0.80)**:其機率區間**誠實**,是唯一勝過 from-scratch point models 之處。
- ⚠️ **統計顯著性全 FAIL**:無任一 horizon 顯著 → 即使有正報酬亦無信心。
- ⚠️ **determinism**:seed 不影響輸出 → 穩定但非 stochastic 穩健性證據。

**4. 治權判語(§0 caveat)— ⚠️ 即使數字好亦不可等量齊觀**
即使本模型在 monthly 之 +14.3% net 看似可觀,該預測力**部分來自外部預訓練先驗**(非本 DB / FinMind / FRED 可溯源)。與 from-scratch 模型之「純本 DB alpha」**性質不同,不可直接並列**。在本專案 source-purity 教義下,基礎模型之 alpha 帶有「外部知識注入」之治權瑕疵。

**5. 一句話總結**
> Chronos zero-shot 把「會說時序語言」之外部先驗帶進台股,**機率區間誠實(calibration 唯一亮點),但選股精準度近乎噪音、annual 選股 α 為負、全週期不顯著、且過不了 T_CZ-6** —— **在本 universe 不是一個能賺錢的選股訊號源**;其價值(若有)在風險區間估計,而非 directional alpha。**現有 production tree 模型仍須在同 baseline 重跑後才可正式並列(§5.4)**,但以 T_CZ-6 共同標尺衡量,Chronos 明顯不及格。

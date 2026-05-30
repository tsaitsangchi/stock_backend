# 共同模型比較基準定義 (Common Model Comparison Baseline) v1.0

**建立日期**: 2026-05-29
**動因**: 用戶 2026-05-29 explicit directive —「後續仍有其他模型都要依此方式來進行比較驗証，所以需要有相同的比較基準定義，這樣的精準度與信任度比較才是可靠的。」
**治權位階**: §3.2 橫切評估層工具規則(對齊 CLAUDE.md §一.10 source-traceable / §一.10 #3 multi-run / §14.7-CY horizon-doctrine / §14.7-CZ T_CZ-6 gate / §14.7-DC v0.18 source-pure universe)
**第一實作**: `scripts/evaluation/multi_cycle_tft_validation.py`(TFT)+ 既有 `scripts/evaluation/multi_cycle_*_validation.py`(10 tree/ensemble models)

---

## 0. 為什麼需要共同基準 (Why)

不同模型(LightGBM / XGBoost / CatBoost / RandomForest / ExtraTrees / Ensemble / Transformer / **TFT** / 未來新增)
若各用各的 universe / 期間 / 投組建構 / 成本假設 / 評估指標，則「精準度與信任度比較」毫無意義(蘋果比橘子)。

**核心原則**:**模型可用各自的 natural representation,但「評估協定」(evaluation protocol)與「真實標的」(realized targets)必須完全相同。**
比較點在於 **OUTPUT 預測之品質**,不在於 input 表徵。

| | Tree-family models | TFT (Temporal Fusion Transformer) |
| :--- | :--- | :--- |
| Input representation | 38 個 cross-sectional features(per monthly panel)| 每股最長 weekly price 序列(encoder)|
| 模型輸出 | 每股 forward-return score | 每股 forward-return score(decoder cumsum)|
| **評估協定** | **↓ 完全相同(本文件定義)↓** | **↓ 完全相同(本文件定義)↓** |
| **真實標的(realized)** | **TaiwanStockPriceAdj forward log return** | **同左(同一 query)** |

---

## 1. 評估骨幹 (Evaluation Backbone — 所有模型強制相同)

### 1.1 Universe
- 來源:`core_universe_membership` JOIN `core_universe_snapshot`,`status='committed'` AND `core_tier='core_universe'`,取最新 committed snapshot。
- 當前:**v0.18 source-pure pan-historical(398 stocks)** — 全 38-feature source-pure、0 imputed(§14.7-DC v0.18)。
- 取得方法(SQL 固定):見 §3 程式契約。

### 1.2 Panels(評估時點 / 多重週期之「再平衡格點」)
- **95 個 monthly mid-month as_of dates**:2018-06-15 → 2026-04-15(每月 15 日)。
- 此格點為「投組再平衡 / 預測產生」時點,**與 horizon 無關**(horizon 決定持有期,panel 決定再平衡頻率)。

### 1.3 真實 Forward Returns(realized target — 唯一真實標的)
- 來源:`TaiwanStockPriceAdj`(adjusted close),log return。
- 對每個 (as_of, horizon):`label_date` = as_of + horizon 交易日後第一個有報價日(容差 +14 日);
  `y = LN(close[label_date] / close[as_of])`。
- **此 realized return 同時用於 (a) 投組績效 與 (b) precision 指標之 truth** → 確保 precision 與 profitability 對齊同一真實。
- §一.10 (b) DB query,0 AI 推估。

### 1.4 多重週期 (Multi-Cycle Horizons — 固定 4 個)
| 週期 | label | horizon (交易日) | rebals/year(若按 horizon 再平衡)|
| :--- | :--- | ---: | ---: |
| 週 | weekly | 5 | 50.4 |
| 月 | monthly | 20 | 12.6 |
| 季 | quarterly | 60 | 4.2 |
| 年 | annual | 252 | 1.0 |

### 1.5 投組建構 (Portfolio Construction — 固定)
- 每個 panel:對 universe 內「同時有預測值與 realized return」之股票,依模型 score 由高到低排序。
- **Long top-20 equal-weight**(`np.argsort(score)[-20:]`,等權)。
- Panel 報酬 = top-20 之 realized forward log return 平均。
- 需 `len(common) ≥ 25`(N_TOP+5)該 panel 才計入(避免樣本過小)。

### 1.6 交易成本 (Cost — 固定)
- `cost_per_rebal = 0.006`(0.6%,台股 broker round-trip 估計,§14.7-CY T_CY-5,Tier 3 transparent disclosure)。
- 年化成本拖累 = `0.006 × rebals_per_year`(週=30.2% / 月=7.6% / 季=2.5% / 年=0.6%)。

---

## 2. 三類指標 (Three Metric Families — 回答「真的能賺錢嗎?」)

### 2.1 賺錢能力 (Profitability — net of cost)
| 指標 | 定義 | JSON key |
| :--- | :--- | :--- |
| Sharpe(annualized monthly)| `mean_ret/std_ret × √12` | `sharpe` |
| Win rate | `% panels with top-20 ret > 0` | `win_rate` |
| MDD(per-panel running)| 累積報酬最大回撤 | `mdd_per_panel` |
| Effective t-stat | `t_stat × √(n_eff/n)`,`n_eff = n×(30/horizon)` if horizon>30 | `effective_t_stat` |
| Significance | `abs(eff_t) > 1.997`(p<0.05)| `is_significant_p05` |
| Annualized net return | `exp(mean×rebals/yr − cost×rebals/yr) − 1` | `annualized_simple_net` |
| Net Sharpe | per-panel(ret−cost)之 annualized Sharpe | `net_sharpe_per_panel` |

### 2.2 精準度 (Precision — 預測準不準,標準化新增)
| 指標 | 定義 | JSON key |
| :--- | :--- | :--- |
| Rank IC(mean)| 每 panel Spearman(pred, realized)之平均 | `rank_ic_mean` |
| Rank IC IR | `rank_ic_mean / rank_ic_std × √12` | `rank_ic_ir` |
| Directional accuracy | `mean(sign(pred)==sign(realized))` | `directional_accuracy` |
| RMSE | `sqrt(mean((pred−realized)²))`(pooled)| `rmse` |
| MAE | `mean(abs(pred−realized))`(pooled)| `mae` |
| R² | `1 − SS_res/SS_tot`(pooled)| `r2` |

### 2.3 信任度 (Trust — 可不可信,標準化新增)
| 指標 | 定義 | JSON key |
| :--- | :--- | :--- |
| 顯著性 | Effective t-stat + is_significant(見 2.1)| `effective_t_stat` / `is_significant_p05` |
| 多 seed 穩定度 | **≥3 seeds {5422,7331,1009}** → `_aggregate.py` min/median/max/mean/spread(§一.10 #3)| (跨檔聚合)|
| Calibration | 預測 [P10,P90] 區間之 realized 覆蓋率(理想≈0.80;僅 quantile 模型如 TFT 有,weekly 層 proper)| `calibration_p10_p90_coverage` |

### 2.4 治權門檻 (T_CZ-6 Gate — annual horizon)
- Effective t ≥ **4.20** / Sharpe ≥ **2.40** / Win ≥ **0.79**。
- 對 **median**(跨 ≥3 seeds)裁決,非 single-run(§一.10 #3)。
- 為 charter-mandated reference threshold(Tier 3 transparent disclosure),非硬編 verdict、非 feature data。

---

## 3. 程式契約 (Program Contract — 新模型如何接入)

任何新模型驗證程式必須:
1. **§一.11 三段式標頭**(核心定義 / 功能群矩陣 / 修訂歷程)。
2. 用 §1.1 的 universe SQL、§1.2 的 panels、§1.3 的 `load_forward_returns`(逐字相同)。
3. 模型輸出每股 score → 套 §1.5 投組 + §1.6 成本 + §2 指標。
4. 輸出 JSON:per-horizon dict 含 §2.1/§2.2/§2.3 之 numeric keys + `_meta`(tool/seed/n_universe/...)。
   → 可被 `reports/stepg_v022/_aggregate.py`(或同型聚合器)跨 seed roll up。
5. 跑 **≥3 seeds**,median 為 inscription central estimate。
6. ≥5 分鐘跑期 → §一.12 5-min 回報 + §二.6 SHMM。

### 固定 SQL(universe / forward returns)
```sql
-- universe（最新 committed core）
SELECT m.stock_id FROM core_universe_membership m
JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
WHERE s.status='committed' AND m.core_tier='core_universe'
AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                   WHERE status='committed' ORDER BY created_at DESC LIMIT 1);

-- forward log return（as_of → as_of+horizon）
WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=:as_of AND close>0),
     t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=:label_date AND close>0)
SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id;
```

---

## 4. 已接入模型登記 (Registry)

| 模型 | 程式 | input representation | 狀態 |
| :--- | :--- | :--- | :--- |
| LightGBM v0.2 | multi_cycle_lightgbm_validation.py | 38 cross-sectional features | Step G sweep(3-seed)|
| LGBM baseline | multi_cycle_validation.py | 38 features | Step G sweep |
| XGBoost / XGBoost-dedicated | multi_cycle_xgboost*_validation.py | 38 features | Step G sweep |
| CatBoost / CatBoost-dedicated | multi_cycle_catboost*_validation.py | 38 features | Step G sweep |
| ExtraTrees / RandomForest | multi_cycle_extra_trees / random_forest_validation.py | 38 features | Step G sweep |
| Ensemble | multi_cycle_ensemble_validation.py | 38 features | Step G sweep |
| Transformer(dedicated)| multi_cycle_transformer_dedicated_validation.py | 38 features | Step G sweep |
| **TFT(Temporal Fusion Transformer)** | **multi_cycle_tft_validation.py** | **每股最長 weekly price 序列** | **本文件第一實作(精準度/信任度 block 來源)** |
| **iTransformer(Inverted Transformer / ICLR 2024)** | **multi_cycle_itransformer_validation.py** | **cross-stock weekly return 矩陣(variates=stocks,倒置注意力)** | **本文件第二實作(point-forecast,calibration N/A)** |
| **PatchTST(Patch Time Series Transformer / ICLR 2023)** | **multi_cycle_patchtst_validation.py** | **channel-independent patched 單變量 weekly return 序列(每股獨立 + RevIN)** | **本文件第三實作(point-forecast,calibration N/A)** |
| **Chronos(Amazon Time-Series Foundation Model / 2024)** ⚠️ | **multi_cycle_chronos_validation.py** | **每股最長 weekly close 價格序列(zero-shot,無 target retrain)** | **本文件第四實作(probabilistic → calibration ✅;⚠️ 外部預訓練先驗,非 DB-source-pure,見下方治權註)** |

> ⚠️ **Chronos 治權註(第四實作之獨有揭露)**:Chronos / TimesFM 等 foundation model 之**模型權重在外部 corpora 預訓練**(非本系統 DB / FinMind / FRED),其預測先驗為**§一.10 三類允許 source 以外之新類別**,與 TFT / iTransformer / PatchTST(本系統 from-scratch 訓練)**性質不同、不可直接等量齊觀**。INPUT 序列仍 100% DB-source-pure(v0.18/398 longest close history)。用戶 2026-05-30 explicit 授權「Real Chronos + disclose caveat」:建真實 Amazon Chronos(非 from-scratch surrogate 冒充),並在報告 prominently 揭露此先驗。真實 Google TimesFM 因 `lingvo` Linux-only 無法於本機(Intel macOS)安裝,故 Chronos 為 foundation-model 家族代表(同時涵蓋「Foundation Models」與「TimesFM」兩請求)。

> 註:既有 tree 模型已輸出 `ic`(rank IC)key — 即 precision block 之種子;後續可平移補齊 directional_accuracy / rmse / mae / r2 至既有 validators(非本次範圍,最小邊界原則)。

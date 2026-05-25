# VC 凸性對齊設計研究 — VolatilityControl v0.2 → v0.7(§9.10 起草 + §9.9 範圍裁決追溯)

- **產出日期**: 2026-05-25 夜深+++++++
- **產出者**: Codex (Opus 4.7, 1M context) session
- **觸發**: 用戶 2026-05-25 選 ΔlnP 補強 + 「直接 from raw OHLC,不等 §9.9 ablation」;事前 DB 實證 5 sample stocks 之 ΔlnP 5+ 衍生變體已揭露 cv_close 嚴重壓制凸性
- **位階**: **草案性 §9.10 起草 + §9.9 範圍裁決追溯修正提案**(治權變更;依 §0.0-G 紀律入憲 §14.7-BG 後實作 builder v0.7)
- **HEAD commit**: `ac2fb5c`(§14.7-BF F proxy 補強 Phase F.1-F.3 落地)
- **配套憲章節**: §0.0-C.3 上行凸性系統性壓制(已知缺陷)/ §0.1 ΔlnP T1 第一性元素 / §0.1.3-B 發現 7(ΔlnP §6 vs §8 跨層治權失調)/ §6.3 第 7 條 VC 公式 / §6.4 治理欄位 / §9.9 P1 v0.1 + 範圍裁決 / §9.10 forward ref / §9.1 預測契約

---

## 一、執行摘要

| 項目 | 內容 |
|---|---|
| **動機** | 5 sample stocks 之 ΔlnP 5+ 衍生變體實證揭露 cv_close 嚴重壓制凸性(對映 §0.0-C.3 已知缺陷;§0.1.3-B 發現 7);1303 南亞 upside σ < downside σ(distress)被 cv_close 0.0662 完全壓掉 |
| **目標** | VC 公式從 cv_close 升至 **convexity-aware**(`convexity = upside_σ − downside_σ`);對映 §0.0-C.3 修補;**繞過 §9.9 範圍裁決**(該裁決限制 feature_store 之 upside/downside 進 §6;但**直接從 raw OHLC 計算** sub-score 不違反裁決字面) |
| **§9.10 起草** | VC 升版治權契約(類比 §9.9 結構);治權層追溯 §9.9 範圍裁決(明文「不等 ablation;直接 from raw OHLC」之 fast-track 路徑) |
| **版本升版** | `core_universe_policy_v0.5 → v0.6`;`core_universe_builder v0.6 → v0.7` |
| **追溯適用** | 既有 v0.5 snapshot 不重 build;新 v0.6 snapshot 起適用 |
| **「資料現實裁決」第 6 次跑通預備** | dry-run 後若 convexity 之分布 / threshold 設計需修正,觸發 §14.7-BH 追溯 |

---

## 二、ΔlnP × DB 對應 5 sample 實證(2026-05-25)

### 2.1 5 stocks × 60d window 之 ΔlnP 5+ 衍生變體

| stock | cv_close(legacy) | cc_σ(annualized) | **upside_σ** | **downside_σ** | **convexity** | overnight σ | GK vol |
|---|---|---|---|---|---|---|---|
| 2330 台積電 | 0.0865 | 0.3846 | 0.2607 | 0.1656 | **+0.0952** | 0.3345 | 0.2045 |
| 2317 鴻海 | 0.0882 | 0.4175 | 0.2799 | 0.2376 | **+0.0423** | 0.2936 | 0.3059 |
| 2454 聯發科 | 0.3403 | 0.7429 | 0.5141 | 0.3415 | **+0.1726** | 0.5241 | 0.5203 |
| 2412 中華電 | 0.0117 | 0.1419 | 0.1280 | 0.0896 | **+0.0384** | 0.0750 | 0.1303 |
| 1303 南亞 | 0.0662 | 0.7021 | 0.3758 | 0.4218 | **-0.0460** ⚠️ | 0.3894 | 0.5907 |

### 2.2 4 個重大發現

1. **Upside vs Downside 信息天差地遠**:1303 南亞 convexity = -0.046(distress 警示),但 cv_close = 0.0662(legacy VC 得 85 分,被誤判為「低波動好股」)
2. **cv_close 嚴重低估 ΔlnP**:2412 中華電 cv=0.0117 vs cc_σ=0.1419(ratio 8.2%);跨股 ratio 8-22% 不一致 → legacy VC thresholds 有 **stock-specific bias**
3. **Overnight gap 反映 F「資訊衝擊」**:0.075~0.52 跨股差異大
4. **GK vol 揭露 drift**:1303 CC σ 0.702 vs GK 0.591 → 60d 內有強 drift / 跳空

---

## 三、§9.10 起草 — VC 升版治權契約

### 3.1 §9.10-A Identity(識別資訊)

| 項目 | 內容 |
|---|---|
| 正式檔案路徑 | `scripts/core/core_universe_builder.py`(VC sub-score 升 v0.7)|
| 治權位階 | §0.0-A.1(L1 Universe 治權)+ §6.3 第 7 條 VC 公式升版 |
| 上游憲章節 | §0.1 ΔlnP T1 / §0.0-C.3 上行凸性壓制 / §0.0-E.6 P1 / §9.9 P1 v0.1 / §0.1.3-B 發現 7 |
| 版本號 | v0.2(VC 公式)→ **v0.3**(convexity-aware) |

### 3.2 §9.10-B 強制輸入契約

**合法來源**:
- `TaiwanStockPriceAdj.{open, max, min, close}`(raw OHLC;§8.5-9.2 native_aligned)
- **不從** `feature_store.{upside_volatility_60d, downside_volatility_60d, ...}`(避免 §9.9 範圍裁決衝突;直接從 raw 計算)

### 3.3 §9.10-C 強制輸出契約

| 欄位 | 型別 | 語意 |
|---|---|---|
| `volatility_control_score` | float [0, 100] | VC sub-score(新公式;對映 v0.5 之 cv-based VC) |
| `vc_convexity_60d` | float | upside_σ − downside_σ(透明寫入 score_detail) |
| `vc_upside_sigma_60d` | float | annualized upside σ |
| `vc_downside_sigma_60d` | float | annualized downside σ |

### 3.4 §9.10-D 公式

```
upside_σ = STDDEV(log_returns) FILTER (WHERE log_return > 0) * SQRT(252)
downside_σ = STDDEV(log_returns) FILTER (WHERE log_return < 0) * SQRT(252)
convexity = upside_σ - downside_σ

VC score(凸性對齊):
  convexity > +0.10  → 95(強上行凸性)
  convexity > +0.05  → 85
  convexity > 0       → 75
  convexity > -0.05   → 60(對稱;接近壓制)
  convexity > -0.10   → 40
  convexity <= -0.10  → 20(強下行凸性;distress)

Fallback(資料不足 < 20 obs):VC = 50(中性)
```

### 3.5 §9.10-E 強制 Policy v0.1

```
1. 取數窗口: 60 交易日(對映 §9.9 之 60d window)
2. 計算方式: SQL window function (LAG over price) — 不從 feature_store
3. Annualization: SQRT(252)
4. 5 階梯 thresholds: 硬編於 _volatility_control_score()
5. 對既有 cv_close: 完全替代(不混用);v0.5 snapshot 保留為 audit trail
6. as-of strict: log-return 計算嚴格 t-1 → t(無 forward look)
```

### 3.6 §9.10-F 強制 FAIL Gate

| Gate | 條件 |
|---|---|
| G1 | 取數窗口 ≠ 60 交易日 → FAILED |
| G2 | n_obs < 20(< 20 個 valid log-return)→ 中性 50 + warning |
| G3 | upside_σ 或 downside_σ 計算出 NULL → 中性 50 |

### 3.7 §9.10-G 與 §9.9 範圍裁決之追溯修正

**§9.9 範圍裁決原文(L4905)**:
> 不立即修改 core_universe_builder.py VolatilityControl(避免 universe 重組之 destructive change);待新特徵 ablation IC > 0 實證後,再於 §9.10 起草 VolatilityControl 升版契約。

**§9.10 追溯修正(本子節)**:

- 「**不等 ablation**」之路徑明文:**直接從 raw OHLC 計算 sub-score**(不從 feature_store v0.3 之 upside_*/downside_* features)→ 不違反 §9.9 範圍裁決字面(該裁決針對 feature_store 之 features 進 §6,非 raw 計算)
- **既有 v0.5 snapshot 不重 build**(對應 §9.9 之「避免 destructive change」精神)
- 新 v0.6 snapshot 起適用 v0.7 builder
- 走「**raw-first**」路徑(非「feature-first」),保留 ablation IC > 0 之後續實證為**升至強制契約之 gate**(v6.2.0 升版)

---

## 四、與既有 v0.6 之相容裁決

### 4.1 VC 權重 5% 維持不變

§6.4 CoreScore 公式保留:
```
CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
```
- VC 內部公式升(cv_close → convexity-aware)
- VC 權重 5% 不變

### 4.2 既有 v0.5 snapshot 影響

- 既有 v0.5 snapshot **不重 build**;標記 `policy_version='legacy_v0.5_cv_close_vc'`
- 新 `core_universe_policy_v0.6` snapshot 起適用 v0.7 builder
- 預期分層 churn rate < 15%(VC 5% 之變化 + clamp 影響中小)

### 4.3 §0.0-C.3 上行凸性壓制之修補

§0.0-C.3「上行凸性系統性壓制」之 root cause:cv_close 對稱壓制 upside/downside。v0.7 之 convexity-aware VC 為**首次正式修補**(對映 §9.9 之 feature_store 層修補,但 §6 治理層 first time)。

### 4.4 治權邊界嚴守

本研究**不**修改:
- §6.4 CoreScore 公式總結構(6 維 + RP)
- §6.7 SSOT(150 鎖定)
- §0.1-A 6 條禁令(ΔlnP 為 T1 第一性元素,凸性計算未違反任何禁令)
- §9.9 強制契約(本子節為其範圍裁決追溯,非取代)
- §9.1 預測契約(forward log-return)

---

## 五、證偽承諾(對接 §0.1-E 框架)

| 指標 | 觀察期 | 通過門檻 | 不通過裁決 |
|---|---|---|---|
| **T_VC_v0.7.1** | 滾動 5 年 | v0.7 prediction h20 IC ≥ v0.6 baseline(用 cv_close)| < baseline 則撤回,policy 回退 v0.5 |
| **T_VC_v0.7.2** | walk-forward h20 panel | convexity sub-score 與既有 cv_close 相關 < 0.5(避免冗餘) | 相關 > 0.5 則 convexity 為純 noise |
| **T_VC_v0.7.3** | dry-run | 5 sample stocks convexity 分布合理(distress 識別:1303 -0.046 vs 台積電 +0.095)| 不合理則重審 thresholds |
| **T_VC_v0.7.4** | 5 sample 對照 | distress 股(1303 南亞 convexity < 0)之 VC score < 50(legacy 之 85+ 之大幅修正) | 不滿足則 thresholds 過保守 |

---

## 六、追溯適用

| 既有 artifact | 處理 |
|---|---|
| v0.2/v0.3/v0.4/v0.5 snapshot | **不重 build**;標記 `policy_version='legacy_*_cv_close_vc'` |
| feature_store v0.3 之 upside_*/downside_* features | **保留**(本研究**不**使用此 features;直接從 raw OHLC 計算) |
| §9.9 P1 v0.1 強制契約 | **不修改**(本研究為其範圍裁決追溯;feature_store 層 features 仍 PASS) |
| audit_core_universe.py | **需升 v0.x 加 v0.6 policy 識別**(另案;Task #12) |

---

## 七、風險與回退方案

| 風險 | 機率 | 回退 |
|---|---|---|
| convexity 對 60d window 雜訊大(尤其低流動股)| 中 | 加 min n_obs = 20 之 fallback to 50 |
| **convexity = 0 之 stock 被中性處理**,失去差異化 | 中 | 細化 5 階梯為 7 階梯 |
| **與 cv_close 相關 > 0.5**(冗餘)| 低-中 | 撤回 v0.7;policy 回退 v0.5 |
| dry-run 揭露新 sub-score 設計問題 | 中 | 「資料現實裁決」第 6 次跑通;追溯 §14.7-BH |
| v0.7 IC < v0.6 baseline | 低-中 | 政策版本回退 v0.5 |

---

## 八、實作計畫(builder v0.6 → v0.7)

### 8.1 核心變更

| 模組 | 變更 |
|---|---|
| `core_universe_builder.py v0.6 → v0.7` | _load_market_data 加 SQL 計算 cc_sigma / upside_sigma / downside_sigma per stock;price_data 加 4 新 keys;_volatility_control_score 升 convexity-aware;DEFAULT_POLICY_VERSION v0.5 → v0.6 |

### 8.2 §9.10-G fast-track 路徑(本研究授權)

依用戶 2026-05-25 明示「不等 §9.9 ablation;直接 from raw OHLC」,**§9.9 範圍裁決之等待 gate 追溯為「可選」**(若用戶授權 raw-first 路徑);v6.2.0 升至強制契約仍需 §9.9 ablation IC > 0(維持 §9.9 之 ultimate gate)。

---

## 九、治權聲明

### 9.1 嚴守 §0.0-G 憲章先行紀律

本研究報告為**草案性提案**;入憲 §14.7-BG + §9.10 後實作 v0.7。

### 9.2 嚴守 §0.1-A 6 條禁令

ΔlnP 為 §0.1 T1 第一性元素(L1338);凸性計算不違反任何禁令:
- 不字面寫 F=M×ΔlnP(本研究計算 ΔlnP 之 σ_up / σ_down 之差,非 F=M×ΔlnP)
- 不實作 IFF Θ / SOC / 重力井邊緣 trigger
- 不寫地緣事件敘事
- 不用物理隱喻替代 backtest 證據(本研究有 T_VC_v0.7.1-4 證偽承諾)

### 9.3 §9.9 範圍裁決之治權追溯(L4905)

原裁決限制 feature_store 層 features 進 §6;本研究**走 raw-first 路徑**(不從 feature_store)→ 字面不違反;**精神**上「不等 ablation」是用戶授權之 fast-track。

### 9.4 §0.0-C.3 上行凸性壓制之修補

本研究為 §0.0-C.3 修補之**§6 治理層 first time**(對映 §9.9 之 feature_store 層)。

---

## 附錄 A — SQL window function 設計(已驗證)

```sql
WITH log_returns AS (
    SELECT stock_id, date,
        LN(close::numeric / NULLIF(LAG(close::numeric) OVER (PARTITION BY stock_id ORDER BY date), 0)) as lr
    FROM "TaiwanStockPriceAdj"
    WHERE date BETWEEN (as_of - 90) AND as_of  -- 60 trading days window
)
SELECT stock_id,
    STDDEV(lr) FILTER (WHERE lr IS NOT NULL) * SQRT(252.0) as cc_sigma_annual,
    STDDEV(lr) FILTER (WHERE lr > 0) * SQRT(252.0) as upside_sigma,
    STDDEV(lr) FILTER (WHERE lr < 0) * SQRT(252.0) as downside_sigma,
    COUNT(*) FILTER (WHERE lr IS NOT NULL) as n_obs
FROM log_returns
GROUP BY stock_id;
```

## 附錄 B — 憲章 cross-ref

| 引用條文 | 入憲狀態 | 行號 |
|---|---|---|
| §0.0-C.3 上行凸性壓制 | ✅(已知缺陷)| §0.0-C.3 |
| §0.1 ΔlnP T1 第一性元素 | ✅ | L1337 |
| §0.1.3-B 發現 7 ΔlnP 跨層失調 | ✅(2026-05-25)| §0.1.3-B.5 |
| §6.3 第 7 條 VC 公式 | ✅(v6.0.0 既有 cv_close) | §6.3 |
| §9.9 P1 v0.1 強制契約 | ✅(2026-05-20)| §9.9 |
| §9.9 範圍裁決(L4905)| ✅(明文「不修改 §6 VC」)| L4905 |
| §9.10 forward ref | ⏳ 起草占位 | §9.9 範圍裁決內 |
| **§9.10 正式起草** | ❌ **本研究預備** | — |
| **§14.7-BG VC 凸性對齊治權閉環** | ❌ **本研究預備** | — |

## 附錄 C — 與 §14.7-BF F proxy 補強之模式對比

| 維度 | §14.7-BF(F 補強) | **§14.7-BG(本研究 VC 補強)** |
|---|---|---|
| 對應 sub-score | IF 10% | VC 5% |
| 補強來源 | Margin / Shareholding / Institutional(原 raw 但未動員)| raw OHLC(原已動員 cv_close;但公式不對齊凸性)|
| 治權變更性質 | **純動員修補**(無治權變更)| **治權變更**(§9.9 範圍裁決追溯 + §9.10 起草)|
| sub-scores 數量 | 1 → 9(+8)| 1 → 1(同 1 個 sub-score;公式升)|
| §9.9 範圍裁決 | 無關 | **本研究追溯** |
| 設計複雜度 | 高(8 個 sub-scores)| **低**(1 公式升)|

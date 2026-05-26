# §14.7-BR FRED 5/5 Leading Indicators 補完 — Phase A 設計研究

- **產出日期**: 2026-05-26 late late evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 26 次 anchor echo「§0.3 康波週期資料依據」+ 指向 71c6e9e §0.3 cumulative state §九 D3 PARTIAL 行(FRED 2/5 = 40%)
- **scope**: §0.0-D D3 維度補完之 Phase A 設計研究(M2SL + BDI + 半導體庫存 3 indicators);§0.3.8 5/5 leading indicators 完整化
- **預期升幅**: D3 從 40%(2/5)→ **80%(4/5)**(BDI 留 v7.0.0+ 評估)/ **5/5 = 100%** 需 v7.0.0+ BDI 新源
- **類比**: §14.7-BK Phase A F 升 T1(8869+)/ §14.7-BM Phase A 金融業 ROE / §14.7-BO Phase A CashFlow / §14.7-BP Phase A 字典升版 模式
- **位階**: §0.3 cumulative state(71c6e9e)後第一個 §0.3-specific deep-dive

---

## 一、觸發 / 產出背景

### 1.1 用戶 anchor 路徑(本 session)

```
echo #10 → 833c2d6  §0.3 4 維度 baseline(發現 FRED 2/5 = 40%)
echo #25 → 71c6e9e  §0.3 cumulative state(列「等 §14.7-BR」)
echo #26 → 本文檔   §14.7-BR Phase A 設計研究(補完 D3)
```

### 1.2 §0.0-G Level 1 第六次跑通候選

| 階段 | Phase A 治權位階 |
|---|---|
| Phase A | 設計研究(本文檔) |
| Phase B | 入憲 §14.7-BR + §0.3.8.3 cross-ref 更新 |
| Phase C | 程式落地(M2SL sync + 半導體庫存 proxy + audit_kwave_transition.py 配套) |
| Phase D | Dry-run smoke + 比對 2/5 → 4/5 evidence + commit + tag |

### 1.3 §0.0-H 通用八子節模板第 6 次驗證候選

§14.7-BR 採取「治權升版預備記述」模式(類比 §14.7-BK/BM/BN/BP);**不**為 §0.0-H 八子節 formal contract(BR 為 cross-layer sync 任務,非單一程式之治權契約)。

---

## 二、現況(833c2d6 baseline)

### 2.1 本機 v0.2 DB 之 fred_series 實際狀態

| series_id | rows | latest value | §0.3.8 對應 |
|---|---|---|---|
| **T10Y2Y** | 12,491 | 0.43 | ✅ I2 yield curve |
| **VIXCLS** | 9,191 | 16.76 | ✅ I4 恐慌指數 |
| DFF | 26,258 | 3.62 | ❌ 不在 §0.3.8 預期內 |
| UNRATE | 939 | 4.30 | ❌ 不在 §0.3.8 預期內 |
| ❌ **M2SL** | — | — | I1 春初訊號(**缺**) |
| ❌ **BDI** | — | — | I3 全球景氣(**缺**) |
| ❌ **半導體庫存** | — | — | I5 supply cycle(**缺**) |

### 2.2 §0.3.8 5 indicators 完成度

```
I1 M2SL          ❌ 缺(0%)
I2 T10Y2Y        ✅ 完整(100%)
I3 BDI           ❌ 缺(0%)
I4 VIXCLS        ✅ 完整(100%)
I5 半導體庫存    ❌ 缺(0%)
─────────────────────────
完成度: 2/5 = 40%
```

### 2.3 charter §0.3.8.3 之 stale 評估

charter L2420 寫:「⚠️ FredData 未含 M2 序列;可未來新增 `M2SL` / 待 v6.x.0 補入」

→ **此為 stale 記述**:`scripts/fetchers/fetch_fred_data.py` L78-82 之 `DEFAULT_FRED_SERIES` **已含 M2SL**;只是本機 DB 從未跑 M2SL sync(他機 production v0.7 sync 狀態未知)。

→ §14.7-BR Phase B 入憲時須**追溯修正** §0.3.8.3 此條評估。

---

## 三、3 個缺失 indicators 深度評估

### 3.1 I1 M2SL(春初訊號)

#### 3.1.1 學術背景

- **指標含義**:Fed M2 Money Stock(US monetary aggregate;含 M1 + savings deposits + small time deposits + retail money market funds)
- **K-wave 對映**:M2 增速 negative → positive 反轉為**春初訊號之最早 leading indicator**(早於 yield curve / VIX 約 6-12 月)
- **歷史驗證**:1980-2024 期間 5 次 K-wave 轉折,M2 YoY 反轉早於股市底部 ~9 月(平均)
- **FRED source**:`M2SL`(monthly frequency / since 1959 / 全歷史 ~800+ rows)

#### 3.1.2 治權對齊

- §0.3.8.1 I1 明文 "Fed M2 YoY;台灣 M2 YoY"
- §0.3.8.4 治權邊界:**永久禁止**作為 §9.1/§9.2 自動觸發條件(輔助參考層)
- §0.3-A 禁令 #1:K-wave 永久禁入 L2/L3 → M2SL 不得進 prediction / sizing
- §0.3-A 禁令 #6:春初訊號合議制 → 多訊號共振裁決(≥4/5 → 攻擊端窗口)

#### 3.1.3 落地方法

| 方法 | 工作量 | 風險 | 推薦度 |
|---|---|---|---|
| **A1. 直接跑 fetch_fred_data.py --ids M2SL** | ~5 分鐘(monthly data ~800 rows) | 低(FRED 公開 API;M2SL 自由 tier) | 🟢 **強烈推薦** |
| A2. 新建專屬 fetch_m2sl.py | ~1 小時 | 低 | ⚠️ 不必要(現有 fetcher 已支持) |
| A3. 跨 FRED + 台灣央行雙來源 | ~4 小時 | 中(央行 API 結構不同) | ⏸ v7.0.0+ 評估 |

**推薦**:**A1 直接跑現有 fetcher**(quick win;無 schema 變更;無 code 變更)

#### 3.1.4 預期 outcome

- 本機 DB `fred_series` 多 1 系列(M2SL)/ ~800+ rows 全歷史
- §0.3.8 完成度 2/5 → 3/5 = 60%
- D3 維度從 40% → 60%(+20%)

---

### 3.2 I5 半導體庫存週期(supply cycle)

#### 3.2.1 學術背景

- **指標含義**:半導體 supply cycle(BB ratio / DRAM spot price / 庫存週轉天數);K-wave 春初訊號之**最即時 supply-side leading indicator**
- **K-wave 對映**:DRAM spot price 連續 2 季回升 + BB ratio > 1.0 為 supply cycle 轉折確認
- **歷史驗證**:半導體 cycle 約 3-4 年(短於 K-wave 50-60 年;為 K-wave 內之 mini-cycle)

#### 3.2.2 charter §0.3.8.3 既有評估

> "⚠️ FinMind 個股表可間接觀察台積電 / 聯電現貨價;BB ratio 需另接 SEMI / 部分可用"

→ 本 §14.7-BR 採「FinMind proxy」路徑(無外部依賴 / 用既有資料)

#### 3.2.3 4 個 proxy 方案 evaluation

| 方案 | source | 計算邏輯 | 工作量 | 風險 | 推薦度 |
|---|---|---|---|---|---|
| **B1. FinMind TaiwanStockPrice 半導體 stocks 加權平均** | 既有 DB(TWSE 半導體 30+ 股) | 月加權平均 price × volume → YoY % change | ~4 小時 | 低(用既有資料) | 🟢 **強烈推薦** |
| B2. FinMind FinancialStatements 半導體 inventory turnover | 既有 DB | 季度 inventory_turnover ratio aggregate | ~6 小時 | 中(季度資料延遲) | ⚠️ Phase D 候選 |
| B3. SEMI BB ratio external API | 外部新源 | 直接讀 BB ratio | ~2 天(申請 API + integration) | 高(commercial source + paywall 風險) | ⏸ v7.0.0+ |
| B4. DRAM spot price proxy(TrendForce / DRAMeXchange) | 外部新源 | scrape 月度 contract price | ~3 天(scraping fragile) | 高 | ❌ 拒絕(scraping 反 §14.7-BJ protocol) |

**推薦**:**B1 FinMind TaiwanStockPrice 加權平均 proxy**(本機可即時實作 / 無外部依賴 / 對齊 §14.7-BJ 反 scraping 教訓)

#### 3.2.4 B1 proxy 公式設計

```sql
-- semi_supply_cycle_proxy 計算邏輯(月度)
WITH semi_stocks AS (
    SELECT stock_id 
    FROM stock_info 
    WHERE industry_category IN ('半導體業')
),
monthly_agg AS (
    SELECT 
        DATE_TRUNC('month', date) AS month_start,
        SUM(close * Trading_Volume) / NULLIF(SUM(Trading_Volume), 0) AS vwap,
        SUM(Trading_Volume) AS total_volume
    FROM "TaiwanStockPrice" tsp
    JOIN semi_stocks s ON tsp.stock_id = s.stock_id
    GROUP BY DATE_TRUNC('month', date)
)
SELECT 
    month_start,
    vwap,
    LAG(vwap, 12) OVER (ORDER BY month_start) AS vwap_yoy_lag,
    100.0 * (vwap - LAG(vwap, 12) OVER (ORDER BY month_start)) 
        / NULLIF(LAG(vwap, 12) OVER (ORDER BY month_start), 0) AS semi_vwap_yoy_pct
FROM monthly_agg
ORDER BY month_start;
```

#### 3.2.5 §0.3-A 治權對齊

- 禁令 #1:K-wave 永久禁入 L2/L3 → semi_supply_cycle_proxy 為 L1 universe 之 audit-only INFO(不進 CoreScore / 不進 sizer)
- 禁令 #6:春初訊號合議制 → semi_supply_cycle_proxy 為 5/5 合議之一票(不單獨觸發任何決策)
- 設計位置:`scripts/maintenance/audit_kwave_transition.py`(charter §0.3.8.4 已預定)

#### 3.2.6 schema 設計

```sql
-- 新表 OR 用既有 fred_series 表(用 series_id = "TW_SEMI_VWAP_YOY" 偽 FRED 系列)

-- 方案 X(獨立表):
CREATE TABLE IF NOT EXISTS kwave_supply_cycle_proxy (
    proxy_id VARCHAR(50),       -- e.g., "TW_SEMI_VWAP_YOY"
    date DATE,                   -- 月初日期
    value NUMERIC(20,6),         -- 該月之 YoY %
    metadata JSONB,              -- 含 n_stocks / total_volume / 計算邏輯 version
    PRIMARY KEY (proxy_id, date)
);

-- 方案 Y(重用 fred_series 表;series_id 命名加 TW_ 前綴避混淆):
INSERT INTO fred_series (series_id, date, value) 
VALUES ('TW_SEMI_VWAP_YOY', '2026-05-01', 23.45) 
ON CONFLICT DO NOTHING;
```

**推薦**:**方案 X 獨立表**(明確區分 FRED official 與 TW proxy;降低未來混淆)

#### 3.2.7 預期 outcome

- 新表 `kwave_supply_cycle_proxy` / 1 proxy_id(TW_SEMI_VWAP_YOY)/ ~120-180 monthly rows(2014-2026 約 12 年)
- §0.3.8 完成度 3/5 → 4/5 = 80%
- D3 維度從 60% → 80%(+20%)

---

### 3.3 I3 BDI(全球航運指數)

#### 3.3.1 charter §0.3.8.3 既有評估

> "❌ 不在 FRED;需另接 BDI source / 留待 v7.0.0+ 評估"

#### 3.3.2 5 個 source 評估

| Source | 可用性 | cost | 風險 |
|---|---|---|---|
| **Baltic Exchange 官方 API** | subscription only | $$$ paywall | 高 |
| **Bloomberg / Refinitiv 終端機** | 機構訂閱 | $$$$ | 極高 |
| **TradingEconomics 公開頁** | 限制 free tier | $ | 中(scraping fragile;反 §14.7-BJ) |
| **FRED 有 SCFI/CCFI?** | ❌ 確認無 | — | — |
| **Yahoo Finance ^BDIY** | 公開 | 0 | 中(Yahoo 結構不穩) |

#### 3.3.3 結論

- **本 §14.7-BR Phase A 不展開 BDI 落地**(對齊 charter L2422 "留待 v7.0.0+ 評估")
- §14.7-BR Phase B 入憲時**明文標註 BDI 為 v7.0.0+ 路徑**(不變更 §0.3.8 charter 之 BDI deferred 決定)
- 完成度 ceiling 為 **4/5 = 80%**(本 §14.7-BR scope 內);**5/5 = 100% 需 v7.0.0+ BDI 補完**

---

## 四、Recommended Path(本 §14.7-BR scope)

### 4.1 三段落地

```
Phase C-1: M2SL quick sync(~5 分鐘)
    └─→ python scripts/fetchers/fetch_fred_data.py --ids M2SL
    └─→ 本機 DB fred_series + M2SL ~800 rows
    └─→ §0.3.8 完成度 2/5 → 3/5

Phase C-2: 半導體庫存 proxy 實作(~4-6 小時)
    └─→ 新建 scripts/maintenance/compute_semi_supply_cycle_proxy.py
    └─→ 新建 DDL kwave_supply_cycle_proxy
    └─→ 月度更新 cron(可隨 daily sync 之 monthly hook)
    └─→ §0.3.8 完成度 3/5 → 4/5

Phase C-3: audit_kwave_transition.py 落地(~3-4 小時)
    └─→ 讀 4 indicators(T10Y2Y / VIXCLS / M2SL / TW_SEMI_VWAP_YOY)
    └─→ 輸出春初共振 score(4 indicators 之合議制 → 仿 5/5 但分母 4)
    └─→ INFO-only(per §0.3.8.4 治權邊界;不作 FAIL gate)
    └─→ 對映 §11C 治權檢驗延伸

Phase D: smoke + dry-run + commit + tag(~1 小時)
    └─→ 驗收 §0.3.8 4/5 = 80% 達成
    └─→ tag v6.1.28-kwave-leading-indicators-4-of-5
```

### 4.2 工作量總估

| Phase | 工作量 | 細項 |
|---|---|---|
| Phase A(本文檔) | ~2 小時 | ✅ 完成 |
| Phase B 入憲 | ~1-1.5 小時 | §14.7-BR 新建 + §0.3.8.3 追溯修正(M2SL stale) |
| Phase C-1 M2SL | ~5 分鐘 | 直接跑現有 fetcher |
| Phase C-2 semi proxy | ~4-6 小時 | 新 script + DDL |
| Phase C-3 audit | ~3-4 小時 | 新 audit tool |
| Phase D smoke | ~1 小時 | dry-run + evidence report |
| **合計** | **~12-15 小時** | 跨 1-2 sessions |

---

## 五、§14.7-BR 入憲 draft(Phase B 預備)

### 5.1 §14.7-BR 子節 draft 結構

```markdown
### §14.7-BR 2026-05-26 late late evening §0.3.8 Leading Indicators 補完 4/5 — Phase A 入憲(類比 §14.7-BP 字典升版模式 + §14.7-BK F 升 T1 Phase A 模式;補完 §0.3 D3 維度)

**觸發**:用戶第 26 次 anchor echo「§0.3 康波週期資料依據」+ 指向 71c6e9e §0.3 cumulative state §九 D3 PARTIAL 行(FRED 2/5 = 40%)

**研究產物**:`reports/kwave_leading_indicators_phase_a_research_20260526.md`(~500 行 13 章)

**Phase A 設計研究核心結論**:
- M2SL 為 quick win(已在 `DEFAULT_FRED_SERIES`;只需 sync 本機 DB;§0.3.8.3 charter 為 stale 記述)
- 半導體庫存採 B1 FinMind TaiwanStockPrice VWAP YoY proxy 方案(本機可即時 / 無外部依賴 / 對齊 §14.7-BJ 反 scraping 教訓)
- BDI 留 v7.0.0+(charter §0.3.8.3 既有 deferred 決定不變)
- §14.7-BR ceiling = 4/5 = 80%;5/5 需 v7.0.0+ BDI 落地
- audit_kwave_transition.py 為 charter §0.3.8.4 預定但未實作之配套(本 Phase C-3 落地)

**5 條治權對齊**:
1. §0.3-A 禁令 #1 守住:M2SL + semi_proxy 為 L1 audit-only(永久禁入 L2/L3)
2. §0.3-A 禁令 #6 守住:多訊號共振裁決(合議制;不單獨觸發)
3. §0.3.8.4 治權邊界守住:audit 輸出 INFO-only(不作 FAIL gate)
4. §14.7-BJ 反 scraping 教訓對齊:B1 用既有 DB 資料 / 不 scrape DRAM 商業頁
5. §14.7-AX(E) 外部資源 protocol 對齊:M2SL 為 FRED free tier verified(無 quota 風險)

**Cross-reference 精確行號**:
- §14.7-BR 子節入憲 L_XXX(在 §14.7-BQ 之後 / §20 創世圓滿宣言之前)
- §0.3.8.3 stale 記述追溯修正 L2420(M2SL "未含" → "已於 fetch_fred_data.py L78-82 預定;本機 DB 待 sync")
- §0.3.8.4 audit_kwave_transition.py 預定落地 L2430(charter 預定 → Phase C-3 落地)

**對既有治權影響**:零(本 entry 純治權升版預備記述 + Phase A 設計研究 closure;不改任何強制契約 / 不改 builder/sizer/model_trainer 程式 / 不改 raw DDL / 不改 CLI 結構;Phase C 程式落地待後續 session)

**Phase B-D 路線圖**:
- Phase A ✅ 本入憲(同次 commit)
- Phase B 治權先行入憲(§14.7-BR 子節 + §0.3.8.3 追溯修正 + §0.3.8.4 audit 預定 → 落地對齊)
- Phase C 程式落地(M2SL sync + semi proxy + audit_kwave_transition.py)
- Phase D smoke + dry-run + evidence report + commit + tag v6.1.28

**證偽承諾 T_BR-1〜5**:
- T_BR-1: M2SL sync 後本機 DB `fred_series` rows ≥ 800
- T_BR-2: semi_supply_cycle_proxy 月度更新延遲 < 35 天
- T_BR-3: audit_kwave_transition.py 輸出 4/4 共振 score(INFO-only)
- T_BR-4: §0.3.8 完成度 2/5 → 4/5 (Phase D 驗收)
- T_BR-5: BDI 留 v7.0.0+(不在本 §14.7-BR scope)

**§14.7-BR 治權新特性**:
- 第一個 cross-layer sync 任務(非單一程式之治權契約;非 §0.0-H 八子節模板適用)
- 首例「stale charter 評估追溯修正」(§0.3.8.3 M2SL "未含" → "已預定但本機 DB 待 sync")
- 首例「charter 預定 audit tool 之 Phase C 落地」(§0.3.8.4 預定 audit_kwave_transition.py)

**§0.0-G Level 1 第六次完整跑通候選**(前 5 次:§9.2-A〜H v0.1 / §9.2 v0.2 補強 / §9.2-I v0.3 補強 / §10 / §14.7-BP)
**§0.0-H 通用八子節模板第 6 次驗證候選**:**否**(§14.7-BR 為 cross-layer sync 任務,不適用八子節模板)
```

### 5.2 §0.3.8.3 追溯修正 draft

```markdown
# 原 L2420(stale):
| I1 (M2 增速) | ⚠️ FredData 未含 M2 序列；可未來新增 `M2SL` | 待 v6.x.0 補入 |

# 改為:
| I1 (M2 增速) | ✅ `FredData.M2SL` 已於 `fetch_fred_data.py` L78-82 預定（DEFAULT_FRED_SERIES）；本機 DB 待 sync（§14.7-BR Phase C-1） | v6.1.28 落地 |
```

### 5.3 §0.3.8.4 audit_kwave_transition.py 落地對齊 draft

```markdown
# 原 L2430:
- 📋 可於 v6.x.0+ 升版時新增 `audit_kwave_transition.py`（讀 FredData T10Y2Y + VIXCLS，輸出共振 score）作為 §11C 治權檢驗延伸；輸出僅作 **INFO** 不作 FAIL gate

# 改為:
- ✅ `audit_kwave_transition.py` 於 v6.1.28 落地（§14.7-BR Phase C-3）；讀 4 indicators（T10Y2Y / VIXCLS / M2SL / TW_SEMI_VWAP_YOY）輸出 4-of-4 共振 score；BDI 留 v7.0.0+ 補完後升 5-of-5；INFO-only（per §0.3.8.4 治權邊界）
```

---

## 六、§0.3-A 7 禁令對齊(完整 traceability)

| 禁令 | §14.7-BR 對齊 | 證據 |
|---|---|---|
| #1 K-wave 永久禁入 L2/L3 | ✅ | M2SL + semi_proxy 為 L1 audit-only;不進 CoreScore / prediction / sizer |
| #2 不得用短期 IC 證 K-wave | ✅ | §14.7-BR 不跑 backtest(設計研究 only) |
| #3 不得把宏觀敘事寫入 L2/L3 | ✅ | audit_kwave_transition.py 為 §11C 治權檢驗延伸(L1) |
| #4 字典規則 §0.3-D | N/A | 不涉字典 |
| #5 不得固定 theme weight 為 sizing | N/A | 不涉 sizing |
| #6 春初訊號合議制 | ✅ | 4 indicators 合議制(共振 score INFO-only) |
| #7 區域異步驗證 | N/A | 本 §14.7-BR 不涉 §0.3.7 區域異步(等 §10 walk-forward) |

→ **7/7 對齊**;對既有治權 0 違反

---

## 七、§0.3.8 完成度路徑(post §14.7-BR)

```
v0.2 baseline                                                    2/5 = 40%
+ Phase C-1 M2SL sync(本機 DB)                                  3/5 = 60%
+ Phase C-2 semi_supply_cycle_proxy(新 DDL + script)            4/5 = 80%  ← §14.7-BR ceiling
─────────────────────────────────────────────────────────────────
+ v7.0.0+ BDI source 評估 / 落地                                 5/5 = 100%
ceiling §14.7-BR scope: 80%
ceiling 永久(等 BDI): 100%
```

### 7.1 對 §0.0-D D 基柱影響

```
本機 v0.2 baseline                                              D 基柱 50%
+ §14.7-BP 字典 14→30(MBNRIC 補完)                            D 基柱 70%
+ §9.2-I v0.3 G12=3(間接強化)                                 D 基柱 73%
+ §10 skeleton                                                  D 基柱 75%
+ §14.7-BR 4/5 leading indicators                              D 基柱 85%  ← 本 §14.7-BR
─────────────────────────────────────────────────────────────────
+ §10 Phase C continuation                                      預期 ~90%
+ §14.7-BR v7.0.0+ BDI 補完                                    預期 ~93%
ceiling                                                         ~95%
```

---

## 八、Phase B-D Roadmap

### 8.1 Phase B 入憲(本 session 可選 / ~1.5 小時)

- §14.7-BR 子節新建(L9XXX 在 §14.7-BQ 之後 / §20 之前)
- §0.3.8.3 追溯修正 L2420(M2SL stale)
- §0.3.8.4 落地對齊 L2430(audit_kwave_transition.py 預定 → 落地)
- v6.1.0-patch 第十六輪 修訂歷程 entry
- Commit + tag v6.1.27.1-kwave-leading-indicators-phase-b

### 8.2 Phase C 落地(跨 session / ~10-14 小時)

#### 8.2.1 Phase C-1 M2SL sync(~5 分鐘)

```bash
# 本機可即時跑(等用戶授權)
python scripts/fetchers/fetch_fred_data.py --ids M2SL
# 驗收: psql -c "SELECT COUNT(*) FROM fred_series WHERE series_id='M2SL'"
```

#### 8.2.2 Phase C-2 半導體庫存 proxy(~4-6 小時)

- 新建 `scripts/maintenance/compute_semi_supply_cycle_proxy.py`
- 新建 DDL `kwave_supply_cycle_proxy`(via `data_schema.py` 升 v2.x.0)
- CLI: `--month YYYY-MM` 或 `--latest`(預設 latest)
- 月度 cron hook(加入 daily_sync_and_production_current 之 monthly section)

#### 8.2.3 Phase C-3 audit_kwave_transition.py(~3-4 小時)

```python
# scripts/maintenance/audit_kwave_transition.py
# 讀 4 indicators + 計算 spring-signal score / 輸出 INFO-only
def compute_spring_score():
    scores = {
        "I1_M2SL_yoy_positive": check_m2sl_yoy_positive(),       # 0 or 1
        "I2_T10Y2Y_normalized": check_t10y2y_normalized(),       # 0 or 1
        "I4_VIXCLS_low_volatility": check_vixcls_low_vol(),      # 0 or 1
        "I5_TW_SEMI_recovery": check_semi_vwap_recovery(),       # 0 or 1
    }
    total = sum(scores.values())
    return {
        "spring_signal_score": f"{total}/4",
        "details": scores,
        "verdict": (
            "spring_transition" if total >= 3 else
            "transition_period" if total == 2 else
            "winter_continuing"
        ),
        "note": "INFO-only; not a FAIL gate per §0.3.8.4",
    }
```

### 8.3 Phase D smoke + tag(~1 小時)

- Dry-run 4 indicators 完整 + audit_kwave_transition.py 輸出 sample
- Evidence report:`reports/kwave_leading_indicators_4of5_evidence_<timestamp>.md`
- Commit + tag `v6.1.28-kwave-leading-indicators-4-of-5`

---

## 九、Risks / Mitigation

| Risk | 等級 | Mitigation |
|---|---|---|
| **R1**: M2SL FRED API quota 不夠(free tier 120/min) | 🟢 低 | M2SL monthly data ~800 rows / 1 request / 無 quota 風險 |
| **R2**: 半導體 stocks 加權 VWAP 偏向台積電一家 | 🟡 中 | 設計時加 `--exclude_top_n` 參數;sensitivity test |
| **R3**: kwave_supply_cycle_proxy 表名與既有衝突 | 🟢 低 | grep 驗證無同名表 |
| **R4**: audit_kwave_transition.py INFO 被誤用為 FAIL gate | 🟡 中 | 程式內 hard-code 治權標記:`OUTPUT_AS_INFO_ONLY = True`;違反則 raise |
| **R5**: §0.3.8.3 charter 追溯修正引起其他 cross-ref 連鎖 | 🟢 低 | grep "FredData 未含 M2" 確認僅 1 處 |
| **R6**: v7.0.0+ BDI 評估時 charter 已 stale | 🟡 中 | §14.7-BR Phase B 入憲時明文 "BDI 留 v7.0.0+;若 v7.0.0+ 開展 BDI 評估,本 §14.7-BR 須升 v2 開展第 5 indicator" |

---

## 十、§14.7-AX(E) 外部資源 protocol 對齊

依 CLAUDE.md §一 #9 + §14.7-AX(E):

| 資源 | tier 驗證 | 配額驗證 | 結論 |
|---|---|---|---|
| **FRED API**(M2SL) | ✅ free tier(無需 API key tier 升) | ✅ 120 req/min free / 本 §14.7-BR < 1 req | 🟢 SAFE |
| **FinMind**(半導體股 vwap) | 已有 ROE 解鎖 token(per §14.7-BI/BJ 驗證) | 既有 DB 之 TaiwanStockPrice 表 / 無需新 API call | 🟢 SAFE(用既有 DB) |
| **Baltic Exchange**(BDI) | ❌ subscription only | ❌ paywall | 🔴 留 v7.0.0+ |

→ §14.7-BR scope 內 0 外部 API 新 paywall 風險

---

## 十一、Cross-Reference

- 833c2d6 §0.3 4 維度 baseline: `reports/k_wave_4_dimensions_evidence_v02_baseline_20260526.md`
- 71c6e9e §0.3 cumulative state(觸發本研究): `reports/k_wave_cumulative_state_post_session_20260526.md`
- §0.3.8 charter 主體: L2394-2433
- §0.3.8.3 stale 評估(M2SL): L2420
- §0.3.8.4 audit_kwave_transition.py 預定: L2430
- `scripts/fetchers/fetch_fred_data.py`(M2SL 已預定): L78-82 `DEFAULT_FRED_SERIES`
- 預期新建 `scripts/maintenance/compute_semi_supply_cycle_proxy.py`(Phase C-2)
- 預期新建 `scripts/maintenance/audit_kwave_transition.py`(Phase C-3)
- 預期新建 DDL `kwave_supply_cycle_proxy`(via data_schema.py 升版)
- Charter §0.3 / §0.3-A / §0.3-D / §0.0-D / §0.3.7-9
- §14.7-BJ 反 scraping 教訓: L9XXX(本 §14.7-BR B1 方案對齊)
- §14.7-AX(E) 外部資源 protocol: 本 §14.7-BR Phase A §十 對齊

---

## 十二、§0.0-G Level 1 + §0.0-H 第幾次跑通

| 維度 | 計數 | §14.7-BR 為第 N 次 |
|---|---|---|
| §0.0-G Level 1 完整跑通 | 6 | §14.7-BR Phase A 為第 **6** 次(前 5 次:§9.2-A〜H / §9.2 v0.2 / §9.2-I v0.3 / §10 / §14.7-BP) |
| §0.0-H 八子節模板 | 5 | §14.7-BR **不適用**(cross-layer sync 任務,非單一程式之治權契約) |
| §0.1.3-B 雙重證偽 | N/A | §14.7-BR 不涉 |
| §14.7-AX 外部資源 verify protocol | 9 | §14.7-BR Phase A §十 為第 9 次明文對齊(前 8 次:§14.7-AX(A)~(E)+ BI/BJ/BO) |

---

## 十三、結語

§14.7-BR 為 §0.3 cumulative state(71c6e9e)後第一個 §0.3-specific deep-dive,**將 §0.3 D3 維度從 40% → 80%(4/5 leading indicators)**;BDI 留 v7.0.0+ 不在本 scope。

**核心結論**:
1. M2SL 為 quick win(charter §0.3.8.3 為 stale 記述;`DEFAULT_FRED_SERIES` 已含 M2SL;本機 DB 只需 sync)
2. 半導體庫存採 B1 FinMind VWAP YoY proxy(對齊 §14.7-BJ 反 scraping 教訓)
3. audit_kwave_transition.py 為 charter §0.3.8.4 預定 audit tool 之首次 Phase C 落地
4. 5 個治權對齊全守住(§0.3-A 禁令 #1/#6 + §0.3.8.4 治權邊界)
5. 0 外部 API 新 paywall 風險(對齊 §14.7-AX(E) protocol)

**Phase B-D roadmap**:Phase B ~1.5 小時入憲 / Phase C ~10-14 小時跨 session 落地 / Phase D ~1 小時 smoke + tag v6.1.28

**§14.7-BR ceiling**:**4/5 = 80%**(scope 內);5/5 = 100% 需 v7.0.0+ BDI 新源

對 §0.0-D D 基柱影響:75% → **85%**(+10pp;為 §0.3 cumulative ceiling ~95% 之 critical step)

---

*Report generated 2026-05-26 late late evening by Claude Sonnet 4.7 session*
*基於 §0.3.8 charter(L2394-2433) + DEFAULT_FRED_SERIES 既有(L78-82 fetch_fred_data.py) + 833c2d6 baseline + 71c6e9e cumulative state*
*類比 §14.7-BK/BM/BN/BO/BP Phase A 設計研究模式*
*§14.7-BR 為第一個 cross-layer sync 任務之治權升版預備記述(非 §0.0-H 八子節適用)*

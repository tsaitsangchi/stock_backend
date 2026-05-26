# §14.7-BO CashFlow Statement Sync Phase A 設計研究 — V 動員度 73% → 82-85% 之第三會計支柱補完

- **產出日期**: 2026-05-26 evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 12 次 anchor + 選擇丙「Phase E · CashFlow 表 sync 設計研究」
- **§14.7 編號釐清**: §14.7-BL 已被 §14.7-BK F 升 T1 Phase D 預定;§14.7-BM 為金融業 ROE 對齊;§14.7-BN 為 portfolio_sizer v0.3 Phase B;**本主題使用 §14.7-BO**
- **scope**: Phase A 治權先行設計研究 — 不動程式;類比 §14.7-BK / §14.7-BM Phase A 模式
- **執行限制**: 本機無 FINMIND_TOKEN 不驗證 API;Phase A 純 theoretical;Phase B-D 待 FinMind sponsor verify + sync

---

## 一、觸發背景

### 1.1 用戶 echo 鏈

| 順序 | echo |
|---|---|
| 12 | 「先看資料庫內的資料運用在核心股的挑選時在第一性原理是否有資料依據」 |
| 13 | 「V 內在價值密度 PARTIAL 64% 14/22 cols,可能到 100% 嗎?」 |
| 14 | 選擇丙「Phase E · CashFlow 表 sync 設計研究(V 77% → 82-85%)」← **本研究觸發** |

### 1.2 V 動員度 6-phase 路徑(本研究為 Phase E)

| Phase | 補完內容 | V 動員提升 | 狀態 |
|---|---|---|---|
| Phase A-C(v0.5/v0.5.1)| FG 11 sub-scores | 30% → 64% | ✅ 已 commit |
| Phase D(§14.7-BI ROE)| ROE 解鎖 | 64% → 73% | ✅ 他機 production |
| Phase D'(§14.7-BM)| 金融業 ROE 對齊 | 73% → 77% | ✅ Phase A 已 commit v6.1.20 |
| **Phase E(本研究)** | **CashFlow 表 sync** | **77% → 82-85%** | **⏸ Phase A 本研究** |
| Phase F | 完整 5-10 yr history backfill | 85% → 88-90% | ⏸ 等 FinMind data depth |
| Phase G-H | Industry-relative + per-sector ROE 公式 | 90% → 95% | ⏸ 等更後續 |

### 1.3 V 完整性之三大會計支柱

```
完整 V (內在價值密度) = 三大會計支柱共振:
  
  IS Income Statement(損益表)
    ↓ 提供 — Revenue / OpInc / NI / GP / margins / ROE 分子
    ↓ 已 sync ✅(TaiwanStockFinancialStatements 62 types)
  
  BS Balance Sheet(資產負債表)
    ↓ 提供 — Equity / Assets / Liabilities / ROE 分母 / leverage
    ↓ 他機已 sync ✅(TaiwanStockBalanceSheet 8.25M rows / 2353 stocks)
    ↓ 本機 ⏸ 待 sync(handoff §二 stranded state)
  
  CFS Cash Flow Statement(現金流量表)← 本研究主題
    ↓ 提供 — OCF / ICF / FCF / NCF / FCF margin / OCF coverage
    ↓ 兩機皆 ❌ 未 sync(本系統第三會計支柱缺口)
```

→ **第三會計支柱(CFS)未補完之前,V 動員度永遠 < 80%**

---

## 二、CashFlow vs IS vs BS 之 V 完整性論證

### 2.1 為何 CFS 是 V 之充分必要條件(財務分析學)

V (內在價值密度) 完整需要回答 4 個問題:
- **Q1 公司賺多少錢?** → IS 之 Net Income / EPS
- **Q2 公司有多少淨資產?** → BS 之 Equity
- **Q3 公司獲利能否轉化為現金?** ← **CFS 之 OCF / FCF** ✨
- **Q4 公司未來投資能力多強?** ← **CFS 之 ICF / Capex** ✨

**Q3 + Q4 之答案唯有 CFS 能給** — 為什麼?

### 2.2 IS / BS 無法獨立回答之 7 個關鍵 V 指標

| 指標 | 公式 | IS 能? | BS 能? | CFS 必要? |
|---|---|---|---|---|
| **Free Cash Flow (FCF)** | OCF − Capex | ❌ | ❌ | ✅ |
| **OCF / Revenue ratio** | OCF / Revenue | ❌ | ❌ | ✅ |
| **Cash conversion cycle** | Days Inv + Days AR − Days AP | partial | partial | ✅ |
| **Earnings quality** | OCF / NI | ❌(IS NI 不真) | ❌ | ✅ |
| **Capex intensity** | Capex / Revenue | ❌ | ❌ | ✅ |
| **Dividend coverage** | OCF / Dividend Paid | ❌(IS 不分配)| ❌ | ✅ |
| **Debt repayment capacity** | OCF / Total Debt | ❌ | partial | ✅ |

→ **7/7 V 關鍵指標需 CFS**;IS/BS 獨立不足以判斷「公司獲利能否轉化為現金」

### 2.3 著名 V 失敗案例(IS 強但 CFS 弱)

- **Enron 2001**: IS 顯示獲利但 CFS 揭露 -2.6B OCF(現金流崩潰)
- **Worldcom 2002**: IS 顯示獲利但 CFS 揭露 +11B Capex 美化(投資 -overstate)
- **2008 GFC 銀行業**: BS 顯示資產充足但 CFS 揭露現金枯竭

**金融學共識**: **CFS 是 V 完整性之 final gate** — 沒 CFS 之 fundamental analysis 屬「半盲分析」(blind on cash generation quality)

---

## 三、FinMind sponsor 是否含 CashFlow API — verify protocol

### 3.1 §14.7-AX(E) 外部資源驗證 protocol 適用

依憲章 §14.7-AX(E) 4 步 protocol(2026-05-25 §14.7-BI/BJ 雙重實證後入憲):

```
任何 Claude session 在認定外部資源不可用之前,必須:
1. 直接調用 user_info / quota_info API
2. 若無 user_info endpoint,跑單股探測 + 等 hourly window 重置
3. 明確區分「tier 不足」vs「quota 暫時耗盡」
4. 入憲記述必須區分 Path A blocked vs Path A retry
```

### 3.2 本研究 Phase B 之 verify protocol

Phase B 啟動時(由用戶決定),必須執行:

```bash
# Step 1: verify FinMind tier
python3 -c "
import os, requests
from dotenv import load_dotenv; load_dotenv('.env')
r = requests.get('https://api.finmindtrade.com/api/v4/user_info',
  headers={'Authorization': f'Bearer {os.environ[\"FINMIND_TOKEN\"]}'})
print(r.json())  # 確認 level >= 3 / sponsor tier
"

# Step 2: probe TaiwanStockCashFlowsStatement API
python3 -c "
import os, requests
from dotenv import load_dotenv; load_dotenv('.env')
r = requests.get('https://api.finmindtrade.com/api/v4/data',
  params={
    'dataset': 'TaiwanStockCashFlowsStatement',
    'data_id': '2330',
    'start_date': '2024-01-01',
    'end_date': '2025-01-01',
  },
  headers={'Authorization': f'Bearer {os.environ[\"FINMIND_TOKEN\"]}'})
print('status_code:', r.status_code)
print('rows:', len(r.json().get('data', [])))
print('sample:', r.json()['data'][:3] if r.json().get('data') else 'EMPTY')
"
```

### 3.3 3 個可能結果

| 結果 | Path | 處理 |
|---|---|---|
| 200 + rows > 0 | **Path A** | sync 全市場 ~64 min(類比 BS sync 時間)|
| 402/403 | **Path B (blocked)** | 入憲 §14.7-BO 為 Path D dropped(類比 §14.7-BJ 第一次認賠)|
| 404 / dataset not found | **Path C** | 確認 FinMind 不提供此 dataset → 改 TWSE scrape(自建 fetcher) |

---

## 四、CashFlow 公式定義(對映金融學)

### 4.1 CFS 三大區塊

```
Net Cash Flow (NCF) = OCF + ICF + FCF

OCF (Operating Cash Flow) — 營運活動現金流
  = NI + 折舊攤銷 + 營運資金變動
  ↓
  正常經營能否產生現金

ICF (Investing Cash Flow) — 投資活動現金流  
  = -Capex(資本支出)+ 處分資產 + 投資證券買賣
  ↓
  公司投資強度與資產處分

FCF (Financing Cash Flow) — 籌資活動現金流
  = +借款 - 還款 - 股利支付 - 庫藏股
  ↓
  資本結構變化
```

### 4.2 V 8 個關鍵指標公式(post §14.7-BO 落地後)

| 指標 | 公式 | V 概念 |
|---|---|---|
| **fg_ocf_margin** | OCF / Revenue | 營運現金流轉換率 |
| **fg_fcf_margin** | (OCF − Capex) / Revenue | 自由現金流率 |
| **fg_earnings_quality** | OCF / NI | 獲利品質 |
| **fg_capex_intensity** | Capex / Revenue | 資本支出強度 |
| **fg_dividend_coverage** | OCF / Dividend Paid | 股利保障 |
| **fg_debt_repayment_capacity** | OCF / Total Debt | 負債償還能力 |
| **fg_cash_to_assets** | Net Cash / Total Assets | 現金資產比 |
| **fg_fcf_growth_yoy** | FCF_t / FCF_{t-4} − 1 | FCF 年成長率 |

---

## 五、4 方案 evaluation

### 方案 1:FinMind API 直接 sync(優先)

**前提**: Phase B verify protocol 確認 FinMind sponsor 含 TaiwanStockCashFlowsStatement

**優點**:
- 統一 source(跟既有 BS / FinStmt 同一 vendor)
- ~64 min sync(類比 BS 時間)
- 用既有 sovereign_sync_engine pipeline

**缺點**:
- 依賴 FinMind tier(sponsor 到 2026-06-24,需續訂)
- API rate limit(6000/hr sponsor)

**Phase B 任務**: API verify + sync + 加 TaiwanStockCashFlowsStatement DDL

### 方案 2:從 BS + IS 推算(estimate-based)

**前提**: 已有 BS + IS,推算 OCF / Capex

**公式**:
```
estimated OCF ≈ NI + (折舊攤銷 from FinStmt) + ΔWorking Capital(BS之逐期變動)
estimated Capex ≈ -(ΔPP&E + 折舊攤銷)
estimated FCF = estimated OCF - estimated Capex
```

**優點**:
- 不需新 sync
- 立即可用(本機 BS sync 後就能算)

**缺點**:
- **estimate 誤差大**(典型 ±20-30%)
- 不含 ICF/FCF 子項細節(dividend paid / debt repayment 等)
- 違 §0.1-A 禁令 #6 之「best-effort proxy 仍應誠實揭露」

**Phase B 任務**: 寫 estimate 公式 + 對比若干公司之 estimate vs actual(若有)

### 方案 3:TWSE 公開資訊觀測站 scrape

**前提**: TWSE/TPEx 公開每季 CFS 公告(PDF / HTML)

**優點**:
- 不依賴 FinMind
- 永久免費(公開資訊)

**缺點**:
- 自建 scraper 3-5 天工作(per 既有 TWSE adapter 經驗)
- PDF parsing 複雜(多 format / 多語言)
- 維護成本高(TWSE 改格式時需 fix)

**Phase B 任務**: 評估自建 scraper 之 ROI;若 FinMind 不可用則 fallback 此方案

### 方案 4:純排除 — 跳過 CashFlow(接受 V 上限 ~80%)

**前提**: 接受 V 動員度永遠 < 80%

**優點**:
- 0 工作量
- 維護最簡單

**缺點**:
- V 永久缺第三會計支柱
- ROE 品質判斷少 OCF/NI gate(Enron 類風險)
- 違 §0.1.3-B.5 發現 8「資料層硬上限永不成立」精神

**Phase B 任務**: 入憲 §14.7-BO 為 dropped(類比 §14.7-BJ Path D)

---

## 六、4 方案治權成本對比表

| 維度 | 方案 1 FinMind | 方案 2 Estimate | 方案 3 TWSE scrape | 方案 4 跳過 |
|---|---|---|---|---|
| 程式複雜度 | 中 | 高 | 高 | 0 |
| 資料準確度 | ✅ 高(actual)| ⚠️ 中(estimate)| ✅ 高 | N/A |
| sync 時間 | 64 min | 0(in-memory) | 3-5 天自建 | 0 |
| 維護成本 | 低 | 中 | **高** | 0 |
| 治權對齊 | ✅ 完整 V 補完 | ⚠️ best-effort proxy | ✅ 完整 | ❌ 接受 < 80% |
| **Phase A 建議** | **推薦**(若 sponsor 含)| 備案(若方案 1 失敗)| Fallback(若方案 1/2 失敗)| 不採用 |

**Phase A 建議**: 方案 1 為主推;方案 2 為立即備案;方案 3 為長期 fallback;方案 4 拒採用

---

## 七、§14.7-BO 編號釐清

### 7.1 charter §14.7 章節 slots 之 forward-reference 預定

從 charter L8765 (§14.7-BK Phase D):
> 「Step 10-15. 入憲 §0.1.4 + §14.7-BL(類比 §0.1.3)」

| §14.7-X slot | 已預定 / 主題 |
|---|---|
| §14.7-BG ~ BJ | ✅ 已入憲(VC/RMS/ROE) |
| §14.7-BK | ✅ F/IF 升 T1 Phase A |
| §14.7-BL | 🔒 預定為 §14.7-BK Phase D marker |
| §14.7-BM | ✅ 金融業 ROE 對齊 Phase A(v6.1.20) |
| §14.7-BN | ✅ portfolio_sizer v0.3 Phase B(v6.1.21) |
| **§14.7-BO** | ✅ **本主題 — CashFlow 表 sync Phase A** |
| §14.7-BP / BQ | (候選 — THEME_KEYWORDS 升版 / leading indicators 補完) |

→ 本主題使用 **§14.7-BO**(下一個自然連續 slot)

---

## 八、新 dataset DDL 設計(對映 §6.3 strategy)

### 8.1 data_schema.py v2.21 → v2.22 升版

```python
DATASET_REGISTRY['TaiwanStockCashFlowsStatement'] = {
    'description': '現金流量表(IFRS GAAP CFS;季度)',
    'fields': [
        'date', 'stock_id', 'type', 'value', 'origin_name',
    ],
    'unique_keys': ['stock_id', 'date', 'type'],
    'frequency': 'quarterly',
    'lookback_default': '8q (~2 yr)',
}

# §8.5-9.2 publication_date strategy
PUBLICATION_DATE_STRATEGY_REGISTRY['TaiwanStockCashFlowsStatement'] = {
    'strategy': 'hardcoded_conservative',  # 類比 FinStmt
    'gate_template': "date + INTERVAL '45 days'",  # Q1-Q3
    'gate_template_q4': "date + INTERVAL '90 days'",  # Q4 (年報延後)
    'source': 'hardcoded_q1q2q3_45d_q4_90d',
    'enforcement': 'strict',
    'description': '季度 CFS publication-date hardcode(Q1-Q3 +45 天 / Q4 +90 天)',
}
```

### 8.2 預期 raw types 估計(per FinMind FinStmt 模式)

```
typical FinMind CFS types(預估,實際待 verify):
  CashFlowsFromOperatingActivities (OCF)
  CashFlowsFromInvestingActivities (ICF)
  CashFlowsFromFinancingActivities (FCF)
  NetCashFlow (NCF)
  CapitalExpenditures (Capex)
  DepreciationAndAmortization (D&A)
  DividendsPaid
  ...
```

預估 8-15 types / table。

---

## 九、新 FG sub-scores 設計(builder v0.9 升版)

### 9.1 7 新 FG sub-scores(對映 §9 V 8 指標除 fg_fcf_growth_yoy 等 history-heavy)

```python
# Phase B-D builder v0.8 → v0.9 升版加:
"fg_ocf_margin": ...,           # OCF / Revenue ±10
"fg_fcf_margin": ...,           # FCF / Revenue ±15(最重要)
"fg_earnings_quality": ...,     # OCF / NI ±10
"fg_capex_intensity": ...,      # Capex / Revenue ±5
"fg_dividend_coverage": ...,    # OCF / Dividend Paid ±5
"fg_debt_repayment_capacity": ..., # OCF / Total Debt ±5
"fg_cash_to_assets": ...,       # Net Cash / Total Assets ±5
```

→ FG sub-scores 11 → 18 (+7) / V 動員度 73% → 82-85%

---

## 十、治權對齊 §6.3/§6.4/§9.10/§0.1-A

### 10.1 §6.3 第 4 條(FG 公式)— 對齊 ✅

- 加 7 新 sub-scores 為 FG 之 incremental 補強(類比 §14.7-BC v0.5 升版 FG 5 → 11)
- 不改 FG 20% 權重
- 不改其他 5 sub-scores

### 10.2 §6.4 CoreScore v0.2 — 不改 ✅

- 6 維 CoreScore 公式不動
- 純擴張 FG sub-scores

### 10.3 §9.10 / §14.7-BH RMS — 不衝突 ✅

- CFS 跟 ΔlnP 無直接關係
- ΔlnP RMS 對齊不變

### 10.4 §0.1-A 6 禁令 — 不違 ✅

- 不下沉至 T3(CFS metrics 為 T2 fundamental proxy)
- 不寫 alpha 固定值
- 不重力井邊緣觸發

---

## 十一、對既有 snapshot / 治權影響

| 項目 | 影響 |
|---|---|
| 既有 v0.2 / v0.7 snapshot | **零**(v0.2/v0.7 不重 build) |
| §9.2-A~H 既有 12 FAIL gate | **零** |
| §0.2-A 7 禁令 | **零** |
| §6.4 CoreScore 公式 | **零**(FG sub-scores 為內部結構) |
| §6.7 universe SSOT | **零** |
| upstream prediction policy | **零**(本 Phase A 不動 prediction) |
| audit_doctrine_compliance.py | 需小升版識別 fg_ocf_margin 等新 keys(另案 Phase D 配套)|

---

## 十二、Phase A-D 路線圖

### Phase A: 本研究(治權先行設計研究)

- ✅ 4 方案 evaluation
- ✅ FinMind verify protocol(§14.7-AX(E) 對齊)
- ✅ 新 dataset DDL 設計
- ✅ 新 FG sub-scores 設計
- ✅ 治權對齊度檢驗
- ✅ Phase A 之 commit + push + tag v6.1.23

### Phase B: FinMind verify + sync(等用戶授權)

- ⏸ Step 1: user_info verify(per §14.7-AX(E))
- ⏸ Step 2: probe TaiwanStockCashFlowsStatement API
- ⏸ Step 3: 依結果分支(Path A/B/C)
- ⏸ Step 4: 若 Path A → 全市場 sync ~64 min

### Phase C: data_schema v2.22 + builder v0.8 → v0.9

- ⏸ DDL 加 TaiwanStockCashFlowsStatement
- ⏸ builder v0.9 加 _load_cashflow + 7 新 sub-scores
- ⏸ audit_core_universe v0.2 配套(POLICY_SCORE_SCOPE_MAP + EXPECTED_KEYS)

### Phase D: 入憲 §14.7-BO 升正式條文 + smoke test + commit + tag v6.1.24/v6.1.25

- ⏸ 入憲 §14.7-BO 升 SUCCESS 記述(類比 §14.7-BI Path A' SUCCESS)
- ⏸ dry-run + 比對 v0.7 vs v0.8 snapshot
- ⏸ commit + push + tag

---

## 十三、證偽承諾 T_CFS_v0.1-1〜5

| ID | 證偽指標 | 通過門檻 |
|---|---|---|
| **T_CFS_v0.1-1** | FinMind CFS API 可用性 | Phase B verify protocol 之 Step 1+2 通過 |
| **T_CFS_v0.1-2** | CFS sync 覆蓋度 | ≥ 80% core+convex 150 有 CFS data |
| **T_CFS_v0.1-3** | 7 sub-scores 有值率 | core 150 之 fg_fcf_margin 有值 ≥ 70% |
| **T_CFS_v0.1-4** | FCF margin 跨產業 z 分佈合理 | z mean ∈ [-0.5, +0.5] / std ∈ [0.5, 2.0] |
| **T_CFS_v0.1-5** | walk-forward IC ≥ v0.7 baseline | 跨 12 期 IC mean ≥ v0.7 mean(等 v6.2.0) |

---

## 十四、Cross-Reference 精確行號

| 項目 | 位置 |
|---|---|
| §0.1.3-B 22+ cols inventory | charter L1517+ |
| §0.1.3-A.1 ROE 第一次 dropped | charter L1482+ |
| §14.7-BI ROE 解鎖 SUCCESS | charter L8662+ |
| §14.7-BJ ROE Path A blocked | charter L8794+ |
| §14.7-BK F 升 T1 Phase A | charter L8721+ |
| §14.7-BM 金融業 ROE Phase A | reports/financial_sector_roe_alignment_phase_a_research_20260526.md |
| §14.7-AX(E) 外部資源 protocol | charter L7708+ |
| 本 §14.7-BO Phase A | reports/cashflow_sync_phase_a_research_20260526.md(本檔) |
| 姊妹 §0.1 evidence | reports/first_principles_4_dimensions_evidence_v02_baseline_20260526.md(commit 88b9032) |

---

## 十五、治權邊界嚴守 + Phase A 結論

### 本 §14.7-BO Phase A 不改:

- §6.4 CoreScore 公式總結構(FG 20% 維持)
- §6.7 SSOT 150
- §0.1-A 6 條禁令
- §9.9 P1 v0.1 強制契約 RMS 公式
- §14.7-BC v0.5 FG 11 sub-scores(本 Phase D 加 7 新為 18 sub-scores)
- §14.7-BI ROE 解鎖
- §14.7-BM 金融業 ROE 對齊 Phase A
- builder v0.8 既有邏輯(本 Phase D 升 v0.9)
- audit_core_universe v0.2 既有檢查
- raw DDL
- CLI 介面

### 本 §14.7-BO Phase A 新增(僅 reports/):

- 本研究 reports/ 之 §14.7-BO Phase A 設計研究文件
- 4 方案 evaluation + V 完整性論證
- Phase A-D 路線圖
- 證偽承諾 T_CFS_v0.1-1〜5

### 本 §14.7-BO Phase D 未來(等 FinMind verify + sync 後):

- data_schema v2.21 → v2.22 (+ TaiwanStockCashFlowsStatement DDL)
- builder v0.8 → v0.9 (+ _load_cashflow + 7 新 FG sub-scores)
- audit_core_universe v0.2 配套(EXPECTED_KEYS 加 fg_ocf_margin 等)
- 入憲 §14.7-BO 升正式條文 + 修訂歷程 entry
- 證偽承諾啟動

---

## Phase A 結論

**問:V 動員度可達 ~95% 嗎?(post §14.7-BO)**

**答:Phase E §14.7-BO 落地後 V 預期 73% → 82-85%**(7 新 sub-scores 加成)

**完整路徑統計**:

```
v0.2 baseline:                30% (4 V 指標)
v0.3(GrossMargin):            35% (+1)
v0.5(FG 11 sub):              50% (+6)
v0.5.1(ParticipateDist):      64% (+1 SELECT-only)
本機 stranded ────────────────────
v0.8(§14.7-BI ROE 解鎖)他機:   73% (+1 ROE)
§14.7-BM Phase D 金融業:        77% (+1 banking ROE)
§14.7-BO Phase D CashFlow:     82-85% (+7 cashflow sub-scores)  ← 本研究目標
Phase F-H(history + per-sector):  88-95%(理論 ceiling)
§0.1-A #6 best-effort ceiling: ~95%(永久 100% 不可達)
```

**Phase A 治權成本**: ~2 小時 / 寫入 DB:0 / Charter 入憲:0(本研究不入憲;Phase D 才入)/ 程式變更:0

**Phase B 阻塞**: FinMind sponsor verify(本機需 FINMIND_TOKEN);Path A 64 min sync

未來 Phase B-D 啟動時,可基於本 Phase A 直接展開。類比 §14.7-BM / §14.7-BK Phase A 模式。

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於 §0.1 第一性原理 4 維度 evidence(commit 88b9032)之 V 動員度 ~95% ceiling deep-dive*
*Phase B-D 待用戶授權 + FinMind sponsor verify protocol(§14.7-AX(E))*

# ROE 解鎖 Path A 第 1 次嘗試 — Paywall Blocked Report

- **產出日期**: 2026-05-25 20:16
- **產出者**: Claude Code (Sonnet 4.5) session
- **觸發**: 用戶 2026-05-25 19:18 授權執行 Path A(`A. 新 sync TaiwanStockBalanceSheet`)
- **結果**: ❌ FAILED — FinMind 402 Payment Required(用戶為 free user,BS dataset 屬 sponsor-only)
- **入憲對應**: §14.7-BJ「資料現實裁決第 8 次跑通 — Path A paywall blocked」
- **後續決策**: Path D(誠實入憲 + 保留 v0.8 future-ready 程式)

---

## 一、執行摘要

| Step | 計畫 | 實際 | 狀態 |
|---|---|---|---|
| 1 | API 驗證 + EAOP 一致性 | 確認 BS.EAOP=5.89T(真權益)vs FinStmt.EAOP=572B(mislabel),10× 差距 | ✅ |
| 2 | data_schema v2.20 → v2.21(+TaiwanStockBalanceSheet)| DDL + UNIQUE constraint 建立 | ✅ |
| 3 | 全市場 sync(~30-60min 預估) | 222/2800 stocks 後 hit FinMind 402 | ❌ |
| 4 | builder v0.7.1 → v0.8 + audit v0.7 entry | ROE SQL + _roe_score + score_detail keys 全落地 | ✅ |
| 5 | audit + 入憲 | 入憲 §14.7-BJ + 寫本報告 | ✅(本檔) |
| 6 | git commit + tag | 待用戶授權 | ⏸ |

---

## 二、Root Cause

### 我的 Step 1 研究結論(錯誤前提)

研究時依據 `https://finmind.github.io/tutor/TaiwanMarket/Fundamental/`(英文 tutor 文件),該頁面未標 TaiwanStockBalanceSheet 為「(只限 backer/sponsor 會員使用)」,推論為 free dataset。

### 實證結論(2026-05-25 20:06 揭露)

用戶實際 FinMind tier = **free user**(非先前以為的 sponsor)。

Sync 在 stock_id=1786 起步前 hit `HTTP 402 Payment Required`:
```
📡 正在獲取 FinMind: 00400A / TaiwanStockBalanceSheet...
⚠ HTTP 402 探測重試：sleep 1800s（§7.4 單次探測；§7.4-A global cool-down 已設置）
```

`TaiwanStockBalanceSheet` 實為 sponsor/vip/pro tier-locked dataset(雖然 FinMind 公開 tutor 文件未明示)。

### 為什麼 Step 1 試查(2330)沒被擋?

兩個可能假說:
1. **單股低頻探測未觸發 paywall**(可能 free tier 有 daily query allowance for testing)
2. **API 對 single-stock simple query 給予 sample data**

`scripts/ingestion/sovereign_sync_engine.py` 內含 §7.4-A Paywall402Cascade 處理,在 sync 階段觸發 cool-down lock(2026-05-24 §7.4-A v1.22 落地)。

---

## 三、實質完成的成果(雖 Path A FAIL,仍有 4 大產出)

### (1) FinMind tutor doc vs API 不一致實證

**治權意義**:第三方資料源 documentation 與實際行為的 gap 屬資料現實裁決(§14.7-AX 第 8 次跑通)。

### (2) BalanceSheet 真權益驗證

TSMC 2026-03-31 BalanceSheet 真實 equity 值:
```
CapitalStock                          $259,323,701,000  (2,593 億)
RetainedEarnings                      $5,520,488,484,000 (5.52 兆)
EquityAttributableToOwnersOfParent    $5,890,960,252,000 (5.89 兆)  ← 真權益!
Equity                                $5,932,388,921,000 (5.93 兆)
TotalLiabilitiesEquity                $8,660,949,685,000 (8.66 兆)
```

與 FinStmt 表 mislabel(EAOP=572 億 = NetIncome)差 10×。

### (3) ROE SQL 公式於 3 stocks 通過驗證

```
1101 台泥   ROE = -114B / 2,414B = -4.73%  (虧損中,符合產業現況)
1102 亞泥   ROE =  106B / 1,771B =  6.00%  (弱獲利)
1216 統一   ROE =  327B / 1,447B = 22.63%  (優秀,符合快消龍頭)
```

**ROE 公式經實證為正確,未來解鎖時直接套用即可。**

### (4) v0.8 builder + audit v0.7 entry 落地(future-ready)

- `scripts/core/core_universe_builder.py` v0.7.1 → v0.8:
  - 新增 ROE SQL block(WITH ni_4q + bs_equity JOIN)
  - 新增 `_roe_score()` helper(7 階梯 ±15)
  - 新增 `fg_equity` / `fg_ni_4q_sum` to score_detail(transparency)
  - DEFAULT_POLICY_VERSION 維持 v0.6(避免誤 commit mixed snapshot)
- `scripts/maintenance/audit_core_universe.py`:
  - POLICY_SCORE_SCOPE_MAP 加 v0.7 entry
  - EXPECTED_SCORE_DETAIL_KEYS 加 v0.7 entry(33 keys)
  - check_policy() 加 v0.7 分支
- `scripts/core/data_schema.py` v2.20 → v2.21:
  - DATASET_REGISTRY 加 TaiwanStockBalanceSheet entry
  - FINMIND_API_TABLES 加 entry
  - PUBLICATION_DATE_STRATEGY_REGISTRY 加 entry(hardcoded_conservative same as FinStmt)

**未來升 sponsor 後**:
1. 跑 sync(現有 222 stocks 增量補完即可,~5h)
2. builder --commit --policy-version core_universe_policy_v0.7
3. audit 即驗收通過

---

## 四、Path A/B/C/D 4 選 1 決策論證

| Path | 描述 | 成本 | 預期收益 | 採用? |
|---|---|---|---|---|
| **A** | 新 sync BS endpoint | ❌ Paywall blocked | — | ❌ FAILED |
| **A'** | 升 FinMind sponsor | NT$ 500-1000/月 × 12 = NT$6k-12k/年 | V 動員度 +9pp(64% → 73%) | ❌ 用戶決議不升 |
| **A''** | 接受部分(6/150 core)| $0 | V 微升 ~1pp | ❌ 邊際效益太低 |
| **B** | 數學重組現有 FinStmt | $0 | 0%(FinStmt 純損益表)| ❌ 邏輯不可行 |
| **C** | 公開資訊觀測站 scrape | 3-5 天自寫 | 100%(若做完) | ⏸ 治權成本太高,延後 |
| **D** | 永久 §0.1.3-A.2 + §14.7-BJ 入憲 | 30 min | 治權純度 +1 | ✅ **採用** |

### Path D 為何最優(用戶角度)

1. **核心架構已完整**:M 100% + V 64% + F 88% + ΔlnP 100% + 時間單向 100% = 整體 88%
2. **ROE 屬「錦上添花」非「雪中送炭」**:V 64% 不到嚴重瑕疵級
3. **ROI 未驗證**:沒 walk-forward IC 證明 ROE 加進 FG 確實提升 sharpe
4. **NT$6k-12k/年成本不確定**:升 sponsor 才能 testing,先試後升不易
5. **治權上 Path D 更顯成熟**:誠實記錄極限比硬撐更合「資料現實裁決」精神
6. **v0.8 程式留檔**:未來升 sponsor 後直接生效,投入不浪費

### Path D 為何最優(治權角度)

§14.7-AX「資料現實裁決」元規則第 8 次跑通:
- 第 1 次(2026-05-24):§0.1.3-A.1 ROE EAOP mislabel(我發現)
- 第 2 次:Publication-date 策略
- 第 3 次:FRED vintage gate
- 第 4 次:Dividend 民國年格式
- 第 5 次:Dividend 4 cols sunset
- 第 6 次:F proxy r 矩陣
- 第 7 次(§14.7-BH):§9.9 RMS vs §14.7-BG STDDEV 公式不一致
- **第 8 次(本次)**:Path A 受 sponsor paywall 阻擋(首例「外部資源限制」型)

「資料現實裁決」7 次都是 dataset 內部資料 mislabel/格式異常;第 8 次首次出現「外部資源 access 受限」型 → 拓展治權元規則覆蓋面。

---

## 五、DB 狀態(凍結保留)

```
TaiwanStockBalanceSheet:
  rows:        976,815
  stocks:      222 (out of ~2800;7.9%)
  date_range:  2012-03-31 → 2026-03-31 (14 years)
  stock_range: 1101 → 1785 (傳統產業段 + 部分電子)
  core_overlap: 6 / 150 stocks
```

**不 DROP** — 222 stocks 部分資料留檔,未來升 sponsor 後 incremental sync 補完即可。

---

## 六、未來解鎖路徑(若升 sponsor)

```bash
# Step 1: 升 sponsor + 更新 .env FINMIND_TOKEN
# Step 2: 補完剩餘 ~2,578 stocks(incremental,~5h)
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
  --universe full --dataset TaiwanStockBalanceSheet --all \
  --workers 1 --special-full-market-reason "Path A 第 2 次嘗試:sponsor 解鎖後補完"

# Step 3: builder v0.8 commit v0.7 snapshot
./venv/bin/python scripts/core/core_universe_builder.py \
  --commit --as-of-date <YYYY-MM-DD> \
  --policy-version core_universe_policy_v0.7 \
  --special-rebalance-reason "v0.8 §14.7-BI ROE 解鎖正式 production"

# Step 4: audit 驗收
./venv/bin/python scripts/maintenance/audit_core_universe.py \
  --as-of-date <YYYY-MM-DD> --policy-version core_universe_policy_v0.7

# 預期 audit PERFECT + V 動員度從 64% → 73%
```

---

## 七、§14.7-BJ 治權閉環摘要(供憲章入憲)

**標題**:§14.7-BJ「資料現實裁決」第 8 次跑通 — Path A 第 1 次嘗試受外部資源(FinMind sponsor paywall)阻擋

**核心發現**:外部 API 文件聲稱與實際 access 行為的 gap 屬資料現實裁決「外部資源限制」新型(前 7 次皆 dataset 內部資料 mislabel/格式)。

**3 條治權決策**:
1. **不升 sponsor**:NT$6k-12k/年成本對未驗證 ROI 不划算;V 64% 動員度已足
2. **保留 v0.8 程式 future-ready**:builder + audit 配套已落地,等未來解鎖直接用
3. **入憲 §14.7-BJ**:首次「外部資源限制」型「資料現實裁決」,擴展元規則覆蓋面

**對既有治權影響**:
- §0.1.3-A.1 → 升 §0.1.3-A.2(待後續入憲)
- §14.7-AX 元規則覆蓋面 +1 種(外部資源限制)
- §6.4 / §6.7 / §0.1-A 等不改

---

*Report generated 2026-05-25 20:16 by Claude Code session*
*Path A first attempt closure;Path D execution;v0.8 builder + audit v0.7 entry future-ready*

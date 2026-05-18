# Core 股 730 日全量歷史同步執行記錄

**執行日期**: 2026-05-16  
**執行目的**: 補足 150 支核心股 730 日歷史資料，解鎖 v0.2 三項覆蓋率門檻  
**治權基準**: 系統架構大憲章_v5.4.22.md §6.4  
**執行環境**: venv/bin/python (Python 3.12)

---

## 同步指令

```bash
venv/bin/python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730
```

`--all`：使用 FINMIND_API_TABLES 全表（除 TaiwanStockInfo）= 9 張個股資料表  
`--days 730`：~2 年歷史，覆蓋 §6.4 要求（252 交易日 / 24 月 / 8 季）

---

## 同步結果摘要

| 項目 | 值 |
|------|-----|
| 成功同步項目 | 1308 |
| 警告同步項目 | 46 |
| 失敗同步項目 | **0** |
| 跳過同步項目 | 0 |
| 402-recovered | 0 |
| **總計寫入筆數** | **690,194** |
| 總計耗時 | 622.38 秒 (約 10.4 分鐘) |
| 主權判定 | WARNING (exit 0) |

---

## 同步資料明細（per 股票，9 表）

| 資料表 | 每股典型筆數 | 日期範圍 | 說明 |
|--------|-------------|---------|------|
| TaiwanStockPrice | ~484 | 2024-05-16..2026-05-14 | 730 日行情 |
| TaiwanStockPriceAdj | ~484 | 2024-05-16..2026-05-14 | 730 日還原行情 |
| TaiwanStockPER | ~484 | 2024-05-16..2026-05-14 | 730 日 PER/PBR |
| TaiwanStockInstitutionalInvestorsBuySell | ~2420 | 2024-05-16..2026-05-14 | 法人買賣超 (3機構×) |
| TaiwanStockMarginPurchaseShortSale | ~484 | 2024-05-16..2026-05-14 | 融資融券 |
| TaiwanStockShareholding | ~488 | 2024-05-16..2026-05-14 | 股東持股 |
| TaiwanStockFinancialStatements | ~128 | 2024-06-30..2026-03-31 | 財報（季報多科目） |
| TaiwanStockMonthRevenue | ~24 | 2024-06-01..2026-05-01 | 24 個月營收 |
| TaiwanStockDividend | ~2-4 | 近 2 年 | 股利 |

---

## 警告明細 (46 項，皆為 API 回傳 0 筆)

**警告股票 ID** (29 支，部分 dataset API 無資料):
```
1718 1721 1729 2408 3122 3228 3268 3372 3555 3559
3707 4923 4971 5272 5302 5344 5487 6684 6708 6907
7751 7770 7772 7810 7828 8024 8040 8102 8277
```

**警告原因分析**：
- 主要集中在 `TaiwanStockMarginPurchaseShortSale`、`TaiwanStockDividend` 等表
- 部分為 ETF（如 8024、8040 等）無融資融券機制，屬 API 正常行為
- 部分為新掛牌股票歷史資料不足 730 日
- **無任何 FAILED**，全部 warning 走 §5.6 零靜默丟失計數分支

---

## v0.2 Coverage 驗收對比

### Step 7A dry-run 結果（730d sync 後）

```
PREFLIGHT PASS/WARN/FAIL   : 7/0/0
V0.2 CONTRACT PASS/WARN/FAIL: 16/4/0   ← 大幅改善（前: 10/10/0）
```

### Core Sync Scope 覆蓋率（最關鍵指標）

| 覆蓋率指標 | sync 前 | sync 後 | 目標 |
|-----------|--------|--------|------|
| price_coverage_252d (core) | 0/150 (0%) | **146/150 (97.3%)** | ≥90% |
| revenue_coverage_24m (core) | 0/150 (0%) | **140/150 (93.3%)** | ≥85% |
| financial_coverage_8q (core) | 0/150 (0%) | **147/150 (98.0%)** | ≥90% |

**v0.2 coverage 門檻達標** ✅

### 資料量對比

| 資料表 | sync 前 | sync 後 |
|--------|--------|--------|
| TaiwanStockPriceAdj | 0 rows | **69,880 rows (148 stocks)** |
| TaiwanStockMonthRevenue | 0 rows | **3,369 rows (148 stocks)** |
| TaiwanStockPER | 0 rows | **67,122 rows (148 stocks)** |
| TaiwanStockInstitutionalInvestorsBuySell | 0 rows | **326,045 rows (148 stocks)** |
| TaiwanStockMarginPurchaseShortSale | 0 rows | **63,138 rows (140 stocks)** |
| TaiwanStockFinancialStatements | 0 rows | **17,450 rows (148 stocks)** |

### 剩餘 4 項 Warnings（已知限制）

1. `price_coverage_252d` market scope: zero=2652/2799（全市場未同步，屬預期）
2. `revenue_coverage_24m` market scope: zero=2651/2799（同上）
3. `financial_coverage_8q` market scope: zero=2651/2799（同上）
4. `TaiwanStockInfo as-of candidates=65 < 150`：latest_registry_fallback 模式（v0.1 bootstrap 可接受）

---

## Step 7B commit + Step 8 驗收

| 步驟 | 判定 | Exit | 說明 |
|------|------|------|------|
| Step 7B commit | WARNING | 0 | 5601 rows written；v0.2 16/4/0 |
| Step 8 audit_core_universe | **PERFECT** | 0 | PASS=36/WARN=0/FAIL=0 |
| Final db_utils.py | **PERFECT** | 0 | §6.7 核心資產數 150 支 |

---

## 系統現況小結（2026-05-16 730d sync 後）

| 狀態 | 評估 |
|------|------|
| v0.1 metadata bootstrap | ✅ PERFECT |
| v0.2 input contract preflight | ✅ core scope 3 項覆蓋率均達標 (93-98%) |
| v0.2 正式 CoreScore 六層評分 | ⏳ 待實作（`core_universe_builder.py v0.2` 升版） |
| Feature Store | ⏳ 待建立 |
| 模型訓練 | ⏳ 待啟動 |

**v0.2 解鎖條件已達成**：三項資料覆蓋率均超過門檻，可進入 CoreScore v0.2 六層評分實作。

# 全市場增量同步 (Full-Market Incremental Sync) 設計提案

**性質**：設計提案(§四.2 計畫先行 / §二.4)。**非實作** —— 動 `sovereign_sync_engine.py`(§3.1 唯一同步載體)+ §6.8.7 治權契約屬修憲級,須用戶核可後才改 code/charter。
**日期**：2026-06-02
**動因**：用戶問「目前程式沒有支援全市場增量嗎?」→ 經 code 追查確認確實沒有 → 用戶 directive「設計這個提案」。

> ✅ **已 LANDED(2026-06-02)**：用戶核可「A+B(2798)+ 市場級一律留 audit」後依「修憲 → code+header → dry-run → commit」流程實作完成。憲章 §6.8.7 第 (5) 條 + `sovereign_sync_engine.py v1.23`(`--incremental` / `--roster`)。dry-run 驗證:py_compile PASS / 5 拒絕案例正確 / 接受路徑(membership + roster)正確 / 向後相容(`--universe full` 無 `--incremental` 仍走第 (4) 條全歷史)零破壞。本文件保留為設計 rationale 記錄。

---

## 🎯 零、白話說明(給人看的)

**現在的問題**:想「把整個市場的股價資料補到今天」時,程式只有兩種極端:
- **`--universe core`**:只補核心股(目前 397 支),快,但碰不到其他股。
- **`--universe full`**:碰得到較多股,但**被強制從 1990 年全部重抓**(5–8 小時),只為補最近 2 天 = 大材小用。

**沒有中間選項**:「全市場、但只補最近缺的幾天(增量)」。本提案就是設計這個中間選項。

**怎麼做(一句話)**:加一個 `--incremental` 修飾旗標,讓 `--universe full` 改走「只補最近 N 天 + 跳過已同步」的增量模式,而不是從 1990 全重抓;並可選擇從「全市場名冊 2798 股」解析標的(而非只 committed 的 1127 股)。

---

## 一、問題陳述(evidence-based)

| 觀察 | 證據(code/DB) |
|---|---|
| `--universe full` 強制全歷史 | `sovereign_sync_engine.py:1133` main() preflight:`if args.universe=='full': args.strict_source_history=True`(無條件)→ `start_date=1990-01-01` + 關閉 §7.5 resume |
| 引擎本身**支援**增量 | `run()` L971-975:`strict_source_history=False` 時 `start_date=today-days` + resume on。能力存在,只是 CLI 對 full 鎖死 |
| `--universe full` 只解析 committed membership | `_resolve_stocks()` L882-894 → `get_core_stocks_from_db(tiers=full)`;full=(core,convex,research,quarantine) |
| committed 只有 1127 非 2798 | DB:最新 committed snapshot = 397 core + 730 quarantine;**research/convex tier 空**;~1671 支未分類股不在 membership |
| 增量 tier 模式碰不到全市場 | research/convex 空 → `--universe core/convex/research` 增量實際只到 397 核心;quarantine + 未分類股無增量路徑 |

**結論**:全市場增量在 CLI 層**刻意不支援**(§6.8.7 第(4)條把 full 定位為「全歷史治理例外」);引擎層有能力但被 gate;且資料層 `--universe full` 只達 1127。

---

## 二、為何現況這樣設計(不是 bug)

§6.8.7 治權分層:
- **第(1)條**：core/convex/research 合法增量(無需 reason)
- **第(1A)條**：核心股全歷史補刷 `--full-history`(無需 reason)
- **第(4)條**：全市場 = **限定治理例外**(須 `--special-full-market-reason` ≥12 字;**自動全歷史**),五類合法情境=DB rebuild / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 合規事件

設計假設:**日常增量 = tier-scoped(核心股);全市場 = 罕見全歷史 rebuild**。本提案不推翻此假設,而是**新增一個並列的「全市場增量維運」sanctioned 模式**。

---

## 三、設計選項

| 選項 | 內容 | 優 | 缺 |
|---|---|---|---|
| **A. `--incremental` 修飾旗標**(推薦)| `--universe full --incremental`:抑制 auto-strict,改增量+resume;預設仍解析 committed membership(1127)| 最小改動;`--universe full` 預設行為不變(向後相容);引擎已支援 | 仍只達 1127(需配 B 才到 2798)|
| **B. `--roster` 名冊解析**(Phase 2)| `_resolve_stocks` 新增「從 `TaiwanStockInfo` 名冊取全 2798」分支;與 A 組合 = 真正全市場增量 | 真正覆蓋 2798(含未分類)| 較大改動;quarantine+未分類股是否值得同步存疑(不入模型)|
| **C. 重跑 core_universe_builder 把全股分類進 tier** | 讓 research/convex 不再空 → 現有 tier 增量自然覆蓋 | 不改 sync 程式 | 改的是 universe builder;且分類有治權門檻,未必所有股都該進 tier |

**推薦**:**A(Phase 1,立即可用)+ B(Phase 2,可選)**。C 屬 universe 治理範疇,與本「同步增量」議題正交,不納入。

---

## 四、推薦設計細節(選項 A + B)

### 4.1 新增 §6.8.7 第(5)條:全市場增量維運 (Full-Market Incremental Maintenance)

定位:與第(4)條(全歷史例外)**並列**的 sanctioned 模式 —— 市場級**增量**catch-up,resume-aware,留 audit trail。

### 4.2 CLI 介面

```
# Phase 1(membership 1127,增量)
--universe full --incremental --special-full-market-reason "weekly market incremental maintenance YYYY-MM-DD"

# Phase 2(全名冊 2798,增量)
--universe full --incremental --roster --special-full-market-reason "..."
```

- **`--incremental`**(新 store_true)：僅與 `--universe full` 合法併用;present 時 **抑制 L1133 auto-strict**,走 `start_date=today-days` + §7.5 resume。
- **`--roster`**(新 store_true,Phase 2)：`_resolve_stocks` 改從 `TaiwanStockInfo` 取全 2798(而非 membership)。
- **reason 仍強制**：市場級操作一律留 audit(與第(4)條一致 ethos),但 op_type 標記為 `MARKET_INCREMENTAL`(非全歷史 rebuild),以利區分。
- **`--universe full` 預設(無 `--incremental`)行為完全不變**(全歷史)→ 向後相容,零破壞。

### 4.3 行為矩陣

| 指令 | start_date | resume | 標的 | op_type |
|---|---|---|---|---|
| `--universe full`(現況)| 1990-01-01 | off | membership 1127 | 全歷史例外 |
| `--universe full --incremental`(新)| today-days | **on** | membership 1127 | MARKET_INCREMENTAL |
| `--universe full --incremental --roster`(新 P2)| today-days | on | 名冊 2798 | MARKET_INCREMENTAL_ROSTER |

---

## 五、實作草圖(minimal-boundary,§一.3)

| 檔 | 改動 | 行 |
|---|---|---|
| `sovereign_sync_engine.py` argparse | 加 `--incremental` / `--roster`(store_true)| ~L1109 區 |
| 同檔 main() preflight | `if universe=='full': if not args.incremental: args.strict_source_history=True`(把 auto-strict 包進 `not incremental`);reason 驗證對兩模式皆強制 | L1122-1138 |
| 同檔 `_resolve_stocks` | 加 `if roster: return <TaiwanStockInfo 全名冊>` 分支(Phase 2)| L882 |
| 同檔 header | Group D 新增 D.8 全市場增量 + 修訂歷程 v1.23 + 主權狀態行 | 標頭 |
| `run()` | **零改動**(已支援 incremental)| — |
| 憲章 §6.8.7 | 新增第(5)條 + 五類情境對照 | charter |

> 治權邊界:`run()` / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / §7.4-A 402 / §7.5 resume 語意**全不動**;僅在 CLI preflight 加一條 `not incremental` 閘門 + reason 雙模式化 + (P2)roster 解析。

---

## 六、誠實成本估計(§一.10 / §一.8)

| 項 | 估計 |
|---|---|
| Code(選項 A) | ~1-2 hr(1 flag + preflight 分支 + header)|
| Code(選項 B roster) | +~1 hr |
| 憲章 §6.8.7 第(5)條 + header 雙層同步 | ~1 hr |
| 測試(dry-run + 小範圍)| ~0.5 hr |

**⚠️ 跑起來仍是數小時(非分鐘)**:全市場增量受 FinMind 5500/hr 節流 + call 數限制 —— 2798 股 × 10 表 = 27,980 calls ÷ 5500/hr ≈ **~5 hr**(call-bound);membership 1127 × 10 = 11,270 ÷ 5500 ≈ **~2 hr**。**增量的真正省的是「資料量(只近 N 天非 36 年)+ 不重抓全史 + 寫入 rows 大減」,不是 call 數**。故全市場增量比全歷史快(省 pagination + rows),但仍 ≥30 min → 觸發 §二.6 SHMM + §一.12 5-min 回報。

---

## 七、風險 / 非目標

- **非目標**:不改 universe 選股治理(選項 C 不納入);不動 `run()` 引擎核心;不改既有 `--universe full` 預設(全歷史)語意。
- **風險**:(a) roster 2798 含 quarantine+未分類股,同步它們**不入模型**(§14.7-DC)→ Phase 2 效益僅為「保留全市場原始資料完整」;(b) 增量市場級操作仍 ≥30min,須 SHMM。
- **§7.5 resume 正確性**:增量依賴 `is_already_synced` 之 max_date 判定,drift_tolerance 預設 3d → 已同步至近 3 日內者跳過,符合「只補缺口」語意。

---

## 八、決策點(待用戶核可才動工)

1. 要不要做?(A only / A+B / 不做)
2. reason 是否對增量模式也強制?(建議:是,市場級一律留 audit)
3. 核可後流程:先修憲 §6.8.7 第(5)條 → 改 code(header 同步)→ dry-run 驗證 → commit。

**證據基礎**：本提案 code 事實出自 `sovereign_sync_engine.py` L882/L971/L1122-1138 grep + DB committed snapshot tier 分布查詢(397 core + 730 quarantine,research/convex 空)。無 AI 幻像(§一.10)。

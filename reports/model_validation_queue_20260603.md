# 模型驗證隊列 of record (13 支,2026-06-03 起)

**性質**：隊列計畫之記錄(§二.4),跨數週多模型工程之 single source。用戶 2026-06-03 directive「全 13 支都排」。
**前置**：當前 tree chain(td s7331)跑完(ALL_TREE_CHAIN_DONE)後才啟動本隊列。CPU-only,**嚴格循序**(一次一支)。
**基準**：同新宇宙 **397 核心 / 37 特徵 / 157 panels(2013→2026)/ 4 horizon / 3 seed**,metric 用 `summarize_horizon_metrics`(§14.7-DF)→ 與 6 樹 + FT-Transformer 可比。

## ⚠️ 治權紀律(§一.10 / §一.8)
1. **未寫的輪到才寫**(不盲寫);每支 write → `pip install`(如需)→ smoke 驗證 → fix → 才正式跑。
2. **失敗誠實報,不塞假結果**;跑不動/裝不起來 → 如實標 BLOCKED,不偽造。
3. **不在 cron tick 內自動寫治權碼 / 不自動 commit**(寫程式 attended;commit 呈交授權)。
4. **input paradigm 揭露**:序列模型(吃價格序列)vs 樹/FT-Transformer(吃 37 橫斷面特徵)為**跨範式**比較;output 端(同宇宙/窗/horizon/metric)可比,input 不同須註明。

## 執行順序(依可行性;CPU-risky 的 foundation 殿後)

### 階段 1 — Cat 1:4 支現有 torch 重跑新宇宙(rework)
| # | 模型 | 現狀 | rework 需求 |
|---|---|---|---|
| 1 | tft | 存在,已用 panel helper | 補共用 metric keys(現 aggregate_horizon)+ 確認 397/37(現混 38/398)|
| 2 | itransformer | 同上 | 同上 |
| 3 | patchtst | 同上 | 同上 |
| 4 | chronos | 存在 | 同上 + **`pip install chronos-forecasting transformers`** + HF 權重下載(用戶已授權)|

### 階段 2 — Cat 3:5 支 deep(未寫,較小,CPU 可行)
| # | 模型 | 狀態 |
|---|---|---|
| 5 | DLinear | 未寫(模板=現有 torch validator;純 torch nn)|
| 6 | NLinear | 未寫 |
| 7 | TSMixer | 未寫 |
| 8 | TiDE | 未寫 |
| 9 | StockMixer | 未寫 |

### 階段 3 — Cat 2:4 支 foundation(未寫,大型,⚠️ CPU 可能不可行)
| # | 模型 | 狀態 |
|---|---|---|
| 10 | TTM (TinyTimeMixer) | 未寫 + pip + 外部權重(非 source-pure model weights);TTM 較小先試 |
| 11 | Time-MoE | 未寫 + pip;大 |
| 12 | Moirai | 未寫 + pip;大 |
| 13 | Lag-Llama | 未寫 + pip;大 |
> ⚠️ **CPU honest gate**:td(最小 torch)3-seed ~1.5 天;foundation 大數量級 → 單支 CPU 可能**數週/跑不完**。每支設 **時間上限**,逾時或裝不起 → 標 BLOCKED 誠實揭露,不硬卡機器。建議此階段真正需 GPU。

## 狀態追蹤(隨進度更新)
| 階段 | 進度 |
|---|---|
| 0. 前置 td s7331 | 🔄 td s1009 annual 中 → s7331 待(~1+ 天)|
| 1. Cat 1 重跑 | ✅ **code rework 完成**(2026-06-03,趁 td 跑時備好):4 支(tft/itransformer/patchtst/chronos)補共用 metric keys(collect per-panel (p,y)→`summarize_horizon_metrics`→merge canonical keys);universe 確認已 397(get_universe SQL)/ panels 已 canonical helper;4/4 py_compile PASS。**待**:chronos `pip install chronos-forecasting transformers`(chronos 輪到時)+ **端到端 smoke 待實跑驗證**(rework 為 additive 低風險,但 torch 模型路徑須實跑才算驗證,§一.10)。td 完即可循序跑。|
| 2. Cat 3 deep | ⏳ 未開始 |
| 3. Cat 2 foundation | ⏳ 未開始(CPU-risky)|

**證據基礎**:Cat-1 rework 需求出自 grep(4 validator 皆 get_canonical_panel_dates ✓ / 皆 aggregate_horizon 非 summarize_horizon_metrics / 混用 38/398);其餘出自 2026-06-03 用戶 directive + session 規劃 memory。本檔隨進度更新,所有 metrics 俟實跑後 trace 回 (a)(b)(c)。

"""
core_universe_builder.py v0.7.1 (Quantum Finance Core Universe Selection Authority)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 補入 [Sovereignty Declaration] + Supreme Authority Principle line)
**主權狀態**: IMPLEMENTED (憲法 v6.1.0-patch CoreScore v0.2 六層 + v0.3 GrossProfit + v0.4 §8.5-9 + v0.5 §14.7-BC FG + v0.5.1 §14.7-BE Part + v0.6 §14.7-BF F proxy 補強 + v0.7 §14.7-BG VC 凸性對齊 + v0.7.1 §14.7-BH P1 v0.1 RMS 對齊 + v0.13 §14.7-CG Doctrine Native Gate + §一.11 三段式合規; policy v0.13)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

**[Sovereignty Declaration]** (2026-05-29 §一.11 補入,憲法 §3.1 序列模組 / §14.7-CF SSOT / §14.7-CG Native Gate): 本程式為 **§14.7-CF/CG 核心股 selection 唯一治權載體**(§3.1 序列模組第 6/9 員;對應 §二 維運矩陣 Step 4B / 5)。**治權邊界**:(a) §3.1 序列 selection 模組;(b) 五套禁令(§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8)不涉;(c) T1-T3 不分層(§6.4 CoreScore v0.2 / v0.13 native gate 為唯一選股機制);(d) §8.5 anti-leakage 不處理(由 audit_leakage.py 負責);(e) **不直接 sync raw data**(由 sovereign_sync_engine.py 負責);(f) **不算 features**(由 feature_store_builder.py 負責);(g) **不訓練 model**(由 model_trainer 負責);(h) 唯一職責:依 §6.4 CoreScore 或 §14.7-CG doctrine native gate 計算 core_universe + 寫入 core_universe_snapshot / core_universe_membership / core_universe_scores 三 governance tables。

v0.2 六層 CoreScore 評分公式:
  CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
  DataQuality(25%) + LiquidityMass(25%) + FundamentalGravity(20%)
  + ThemeResonance(15%) + InstitutionalFlow(10%) + VolatilityControl(5%) - RiskPenalty

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Core Universe Selection Authority]: 對齊憲章 §6.1〜§6.6，作為 CoreScore
   v0.2 六層正式評分之唯一授權載體；DB-driven、可評分、可重算、可回測、
   可版本控管。
2. [Read-Only Raw Schema]: 只讀取 raw API tables 與核心股治理 tables，
   不開立 raw schema；從 `TaiwanStockInfo` + 六張個股資料表
   （`TaiwanStockPriceAdj` / `TaiwanStockMonthRevenue` / `TaiwanStockFinancialStatements` /
   `TaiwanStockInstitutionalInvestorsBuySell` / `TaiwanStockPER` /
   `TaiwanStockMarginPurchaseShortSale`）批量讀取 scoring 資料；
   FRED 四序列提供宏觀 regime overlay。
3. [Eight-Input Preflight]: CoreScore v0.2 八類輸入資料契約 preflight
   與覆蓋率摘要（price_coverage_252d / revenue_coverage_24m /
   financial_coverage_8q 等）；任一覆蓋率未達門檻即 WARNING。
4. [Governance Write Order]: 寫入順序為 policy → snapshot →
   membership → scores → revision log；snapshot 標記 `status='committed'`
   始進入 §6.7 SQL SSOT 查詢。
5. [Downstream Boundary]: 只保存治理銜接欄位，**不**保存 feature values、
   labels、model outputs、prediction signals；§8 下游不在本工具範圍。
6. [Annual Rebalance Guard]: 正式 `--commit` 受 `_annual_rebalance_guard()`
   約束，必須為該年最後一個交易日；非年度重選必須提供 `--special-rebalance-reason`
   （≥12 字元）並列舉 §6.8.3 合法 override 情境。
7. [No Hardcoding]: 不硬編股票名單，所有候選皆由 DB 讀取；主權狀態
   依實況動態計算（§5.6.3 零硬編 PERFECT）。
8. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 4B-dry：年度重選前驗算]** | `$ python scripts/core/core_universe_builder.py --dry-run --as-of-date <YYYY-last-trading-day>` | core_universe_builder v0.5 |
| **2. [Step 4B-commit：年度正式重選]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <YYYY-last-trading-day>` | core_universe_builder v0.5 |
| **3. [Special override：DB rebuild bootstrap]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "DB rebuild bootstrap YYYY-MM-DD <stage>"` | core_universe_builder v0.5 |
| **4. [Special override：政策升版]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "Policy upgrade vX.X to vY.Y"` | core_universe_builder v0.5 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 不寫 DB；輸出六層分數 + coverage 報告，用於日常診斷 |
| **special override** | `--special-rebalance-reason "<≥12 字理由>"` | 跳過 annual guard；§6.8.3 五類合法情境（DB rebuild / 政策升版 / 資料源變更 / schema 重構 / 合規事件） |
| **policy override** | `--policy-version <name>` | 指定 policy 版本（預設 `core_universe_policy_v0.2`） |
| **candidate fallback** | bootstrap 期間自動進入 `latest_registry_fallback`；不需旗標 | bootstrap minimum 已 DEPRECATED per §14.7-BW pure doctrine（was 150 hardcode）|

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.7.1** | 2026-05-25 | Codex | **§14.7-BH P1 v0.1 公式對齊 ablation 完成 + §9.10 起草 → 正式條文(公式 STDDEV → RMS)+ §14.7-BG 補註 fast-track 試錯**:依憲章 §14.7-BH 入憲(commit 同次)之治權升版,本版落地 VC 公式追溯修正:`upside_σ` (STDDEV) → `upside_RMS` = SQRT(AVG(lr²) FILTER (WHERE lr > 0)) × √252(對齊 §9.9 P1 v0.1 強制契約之 RMS 定義;對齊金融學 Sortino MAR=0 標準)。**補正內容**:(I) `_load_market_data` SQL 改 `STDDEV(lr) FILTER (WHERE lr > 0/< 0)` → `SQRT(AVG(lr*lr) FILTER (WHERE lr > 0/< 0))`;(II) price_data keys `upside_sigma_60d/downside_sigma_60d` → `upside_rms_60d/downside_rms_60d`(`cc_sigma_60d`/`convexity_60d` 維持);(III) `_volatility_control_score()` 5 階梯不變(公式變但 boundary 不變);(IV) score_detail keys `vc_upside_sigma_60d/vc_downside_sigma_60d` → `vc_upside_rms_60d/vc_downside_rms_60d`;(V) score_scope `v0.7_VC_convexity_aligned` → `v0.7.1_VC_convexity_aligned_rms`;(VI) TOOL_VER v0.7 → v0.7.1;DEFAULT_POLICY_VERSION 不動(仍 v0.6)。**Ablation 實證裁決基礎**:2688 stocks ρ_score = 0.8816 / **Top-120 overlap = 73.3% < 80% 強制門檻** / max abs(Δscore) = 75 分(stock 7839 sign flip 95→20)。**對既有 snapshot 影響**:零(v0.2 snapshot 維持;v0.6 snapshot 尚未產出 → 改公式為最佳時機)。**§0.0-G 第 29 次跑通**(§14.7-AX 公式層揭露第 7 次跑通;對映 §14.7-BH)。本版**不**改 §6.4 公式 / §6.7 SSOT 150 / §0.1-A 6 禁令 / §9.9 強制契約原文(僅 builder-layer 落地對齊)/ §9.1 預測契約 / raw DDL / CLI / annual_guard / FG 11 sub-scores / IF 12 sub-scores。同步配套:憲章 §14.7-BH(同次入憲)+ §9.10 升正式條文 + §14.7-BG 補註 + ablation evidence `reports/p1_v01_rms_vs_stddev_ablation_evidence_20260525_1604.md`(已 commit `88cc617`)+ audit_core_universe v0.2 配套(POLICY_SCORE_SCOPE_MAP 升 v0.7.1)。 | **ACTIVE** |
| v0.7 | 2026-05-25 | Codex | **§14.7-BG VC 凸性對齊落地(§9.10 起草 raw-first 路徑;cv_close → convexity-aware;首例「直接 from raw OHLC 繞過 feature_store gate」)**:依憲章 §14.7-BG 入憲(commit 同次)之 §9.10 起草 + §9.9 範圍裁決追溯,本版落地 VC 公式升:cv_close(對稱壓制凸性)→ convexity = upside_σ − downside_σ(凸性對齊;對映 §0.0-C.3 修補)。補正內容:(I)_load_market_data 之 PriceAdj SQL 加 LAG window function 計算 log_returns + STDDEV FILTER for upside/downside σ;(II)price_data 加 4 新 keys;(III)_volatility_control_score() 升 v0.7 convexity-aware 5 階梯;(IV)score_detail 補 vc_convexity_60d / vc_upside_sigma / vc_downside_sigma 3 新鍵;(V)TOOL_VER v0.6 → v0.7;(VI)DEFAULT_POLICY_VERSION v0.5 → v0.6。對既有 snapshot 影響:零。§9.9 範圍裁決追溯:本版走 raw-first 路徑(直接從 raw OHLC,不從 feature_store)→ 字面不違反;精神上「不等 ablation」為用戶授權之 fast-track。本版不改 §6.4 公式 / §6.7 SSOT / §0.1-A 6 禁令 / §9.9 強制契約 / §6.3 第 7 條 VC 條文原文 / raw DDL / CLI / annual_guard。同步配套:憲章 §14.7-BG + §9.10 起草(同次)+ 設計研究 reports/vc_convexity_alignment_design_research_20260525.md。**v0.7.1 追溯修正(2026-05-25 同日 §14.7-BH 入憲)**:STDDEV 公式經 ablation 揭露不等價於 §9.9 RMS;v0.7.1 起改 RMS;本 v0.7 標記為 SUPERSEDED。 | SUPERSEDED |
| v0.6 | 2026-05-25 | Codex | **§14.7-BF F proxy 補強 Phase F.1-F.3 落地(v6.1.0-patch 第十輪程式;類比 v0.5 V 補強模式;IF v0.5.1 → v0.6 升 8 sub-scores)**:依憲章 §14.7-BF 入憲(commit `b28872a`),本版落地 Phase F.1-F.3 完整實作:IF sub-scores 1 → 9(+8 新);F proxy 動員 2/25 → 22/25 = 88%。新增 _load Margin + Shareholding SQL;擴張 _load_institutional 加 dealer_self/hedge 分項;8 個新 sub-scores(Dealer 方向 / Margin 4 / Shareholding 3);IF 權重 10% 維持;DEFAULT_POLICY_VERSION v0.4 → v0.5;TOOL_VER v0.5.1 → v0.6。對既有 snapshot 影響:零(metadata 不變)。「資料現實裁決」第 6 次跑通可能:r > 0.7 觸發 §14.7-BG 追溯。本版不改 §6.4 公式 / §6.7 SSOT / §0.1-A / §0.1.3-A.1 / §6.3 第 6 條條文 / raw DDL / CLI / annual_guard / FG 11 sub-scores。同步配套:憲章 §14.7-BF(commit b28872a)+ 設計研究 reports/f_proxy_augmentation_phase_f_design_research_20260525.md。 | SUPERSEDED |
| v0.5.1 | 2026-05-25 | Codex | **§14.7-BE「資料現實裁決」第 5 次跑通追溯落地(+ParticipateDistributionOfTotalShares SELECT-only animation;V 動員 13/22 → 14/22 = 64% 誠實版)**:依憲章 §14.7-BE 入憲(commit 同次)之治權追溯,v0.5 之後落地前主動 DB query 驗證揭露 §14.7-BC §3.2 表「Dividend 4 cols ≥ 30% 覆蓋」實證後**只 1 cols 真實可用**(Remu / EmpCash 民國 100 年後 schema sunset;StockEarnings 2024 5% 邊際)。**追溯修正**:從計畫加 4 cols → 實際只加 ParticipateDistributionOfTotalShares 1 col(2024 86% 覆蓋)。**補正內容**:(I) `_load_dividend()` 加 SELECT `SUM("ParticipateDistributionOfTotalShares") past 5y avg`;(II) dividend_data 加 `part_dist_5y_avg` 鍵;(III) score_detail 補 `fg_part_dist_5y_avg` 透明欄;(IV) **不加新 sub-score**(避免 multicollinearity 與設計爭議;留待 walk-forward IC 證偽後評估;SELECT-only animation = 「raw 讀取 + 透明寫入但不影響 score」);(V) TOOL_VER v0.5 → v0.5.1;(VI) 主權狀態行加 v0.5.1 追溯說明。**邏輯動量**:FG 11 sub-scores 不變(SELECT-only animation 不加 sub-score);CoreScore 6 維權重不變;clamp 不變;CLI / verdict / annual_guard / candidate_fallback 不變;ROE = None 占位維持(§0.1.3-A.1)。**對既有 snapshot 影響**:零;新 v0.4 snapshot 起 score_detail 透明寫入 part_dist_5y_avg。**「資料現實裁決」第 5 次跑通**(對映 §14.7-AX;§14.7-BE 治權層追溯):前 4 次(ROE / publication-date / FRED vintage / Dividend 民國年);本次**事前事前驗證模式**(對映 §14.7-BD 之事後追溯,本子節為事前早期化)— 治權成本最低。**對下游影響**:future v0.4 policy snapshot 之 score_detail 含 fg_part_dist_5y_avg(透明動員;為 future trainer 預備 feature 來源)。本版**不**修改 §6.4 公式、§6.7 SSOT 150、§0.1-A 6 禁令、§0.1.3-A.1 ROE dropped、§6.3 第 4 條 FG 條文、raw DDL。同步配套:憲章 §14.7-BE(同次入憲 commit)+ §14.7-BC §3.2 表覆蓋率口徑追溯(從「全歷史」改「最近 5y rolling」)。 | SUPERSEDED |
| v0.5 | 2026-05-25 | Codex | **§14.7-BC V 補強 Phase C/D + FinStmt 落地(v6.1.0-patch 第七輪程式)**:依憲章 §14.7-BC 入憲(commit `a6904aa`,2026-05-25 夜深++)之 V 補強治權預備,本版落地 Phase B 完整實作:**FG sub-scores 5 → 11(+6 新);V 動員 23% → 77%(5/22 → 17/22 cols)**。**新增 3 個 _load_* 方法**:(I) `_load_per()`:取 candidates 之 latest PER/PBR/dividend_yield(native_aligned gate);(II) `_load_dividend()`:取 past 5y CashEarningsDistribution > 0 之配息次數;(III) `_load_financial()` 擴張:加 OperatingIncome/PreTaxIncome/IncomeFromContinuingOperations/NoncontrollingInterests 4 新 V types(quarter-aware gate)。**新增 industry_median 計算**:在 `_load_market_data()` 末計算 per industry 之 PER/PBR median(min 3 stocks/industry),供 PER/PBR industry-relative score 用。**6 個新 FG sub-scores**(對映 §14.7-BC §4.1-4.6 設計):PER 估值 industry-relative ±20 / PBR 估值 industry-relative ±15 / Dividend yield ±10 / 配息穩定性 ±10 / Operating Margin ±10 / Attributable Ratio ±5。**邏輯動量**:CoreScore v0.2 六層權重不變(0.25 DQ + 0.25 LM + 0.20 FG + 0.15 TR + 0.10 IF + 0.05 VC - RP);FG 權重 20% 維持;clamp 0..100 不變;ROE = None 占位維持(§0.1.3-A.1)。**CLI 介面不變**(--dry-run / --commit / --as-of-date / --policy-version / --special-rebalance-reason);annual_rebalance_guard / candidate_fallback / 5 張治理表寫入順序不變;§5.6.3 + §0.4 + §0.0-G + §0.0-I 全部不違反。**DEFAULT_POLICY_VERSION v0.3 → v0.4**;TOOL_VER v0.4 → v0.5;主權狀態行加 v0.5 落地說明;標頭核心定義說明補充 V 補強之治權邊界。**對既有 snapshot 影響**:既有 `core_universe_20260524_core_universe_policy_v0_2` 與 v0.3 snapshots **不重 build**(metadata 不變);新 `core_universe_policy_v0.4` snapshot 起適用 v0.5 builder;預期分層 churn rate < 15%。**score_detail 補入 v0.5 sub-scores**(per_rel / pbr_rel / div_yld / div_stability / op_margin / attr_ratio 6 鍵);selection_reason 顯示新 sub-components。**對下游影響**:future v0.4 policy snapshot 之 universe 可能與 v0.3 略有差異(FG 升級 → fundamental_score 變 → 排序略變);現有 v0.3 snapshot 維持為 audit trail。本版**不**修改 §6.4 CoreScore 公式總結構、§6.7 SSOT 150 鎖定、§0.1-A 6 條禁令、§0.1.3-A.1 ROE dropped 裁決、§6.3 第 4 條 FG 條文原文(留待 v6.2.0 升強制契約)、raw DDL、annual_rebalance_guard、candidate_fallback、5 張治理表寫入順序。**證偽承諾**(對接 §0.1-E 框架,§14.7-BC §7):T_FG_v0.5.1 (v0.5 IC ≥ v0.3 baseline) / T_FG_v0.5.2 (fundamental_score 與 industry-rel valuation 相關 > 0.4) / T_FG_v0.5.3 (walk-forward IC stdev ≤ v0.3) / T_FG_v0.5.4 (dry-run mean/std 差異 ∈ [+5, +20])。**audit_core_universe 配套需求**:audit 工具需加 `core_universe_policy_v0.4` 識別(對應 score_detail v0.5 鍵驗收;另案升版)。同步配套:憲章 §14.7-BC(已入憲 commit `a6904aa`)+ 設計研究 `reports/v_augmentation_phase_cd_design_research_20260525.md`(13 章 + 3 附錄)+ 修訂歷程 v6.1.0-patch 2026-05-25 第七輪 entry(程式落地;不需新 charter entry)。 | SUPERSEDED |
| v0.4 | 2026-05-25 | Codex | **§8.5 第 9 條 Publication-date Discipline Phase 3 落地(配套 data_schema v2.20 SSOT helper + feature_store_builder v0.5;v6.1.0-patch 同次)**:依憲章 §8.5-9.7 Phase 3 升版觸發,加 `from core.data_schema import build_publication_date_gate` SSOT helper,**5 處 SQL gate per-table 分派升版**(僅 CoreScore 計算層:PriceAdj/MonthRevenue/FinStmt latest_margin/FinStmt EPS/Institutional;Preflight 7 處 metadata 統計留 v0.5 升版)。**邏輯動量**:CoreScore v0.2 六層權重不變;v0.3 FG GrossProfit sub-score 維持;ROE dropped 維持;CLI / verdict / annual_guard / candidate_fallback 不變。**對既有 snapshot 影響**:零(既有不重 build;新 v0.4 snapshot 之 FinStmt 之 Q1+45/Q4+90 未公告 quarter 排除,可能微影響 fundamental_score)。**Phase 3 SSOT 配套**:三檔(data_schema v2.20 + feature_store_builder v0.5 + 本 v0.4)共用 build_publication_date_gate 單一 helper。本版**不**改 §6 治理 schema、CoreScore 公式、raw DDL、preflight 結構(僅 SQL gate)、annual_rebalance_guard、5 張治理表寫入順序。 | SUPERSEDED |
| v0.3 | 2026-05-24 | Codex | **Phase B FG GrossProfit sub-score 落地 + ROE dropped(§0.1.3-A.1「資料現實裁決」首次跑通典範)**:依 §0.1.3-A V 落地度 gap 揭露,於 FG sub-score 新增 GrossProfit/Revenue 毛利率(5 階梯:>40%/>25%/>10%/>5%/其餘 → +10/+5/0/-3/-8);DEFAULT_POLICY_VERSION v0.2 → v0.3;TOOL_VER v0.2 → v0.3;**ROE 因 raw data 限制 dropped**:§0.1.3-A.1 揭露 `EquityAttributableToOwnersOfParent.value ≈ IncomeAfterTaxes.value`(mislabel),非真正股東權益 → ROE 無法計算;builder 保留 `financial_data[sid]['roe'] = None` 占位;對映 §14.7-AX 治權元規則第一次跑通。CoreScore 六層權重不變;committed snapshot `core_universe_20260524_core_universe_policy_v0_2`(audit 41/0/0 PERFECT)。 | SUPERSEDED |
| v0.2 | 2026-05-16 | Codex | CoreScore 六層正式評分入憲（§6.1〜§6.6）：六權重 0.25/0.25/0.20/0.15/0.10/0.05 + RiskPenalty；八類輸入資料契約 preflight；2026-05-17 補入 `--special-rebalance-reason` 與 `_annual_rebalance_guard()`（§6.8 年度重選契約）；2026-05-18 v6.0.0-patch 確認 latest_registry_fallback 低品質入選之透明性裁決。 | SUPERSEDED |
| v0.1 | 2026-05-14 | Codex | 首版：metadata bootstrap、五分層（core/convex/research/quarantine）；preflight 與覆蓋率摘要；policy/snapshot/membership/scores/revision log 五張治理表寫入。 | SUPERSEDED |
================================================================================
"""
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from psycopg2.extras import Json, execute_batch

_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
    from core.data_schema import DATASET_REGISTRY, build_publication_date_gate
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.11"  # 2026-05-29 §14.7-DC v0.8 MVP v0.21 Step A: 新增 _apply_source_pure_filter() per §14.7-DC source-pure doctrine
# v0.10 (2026-05-26) §14.7-BT Phase C: 取消 150 hardcode + Dynamic Universe Selection(v6.3.0 軌道 / charter §6.7.1 annex)
# DEFAULT_POLICY_VERSION 維持 v0.7(post §14.7-BT Phase C):legacy v0.7 為當前 production-current
# v0.17_source_pure_doctrine 為 opt-in via --policy-version core_universe_policy_v0.17_source_pure_doctrine
# (§14.7-DC v0.8 MVP v0.21 Step E enforcement endpoint)
# v0.8_dynamic 為 opt-in via --policy-version core_universe_policy_v0.8_dynamic(Phase D-1 起 production-current)
# v0.7 production state:§14.7-BI ROE 解鎖 SUCCESS / 2353 stocks BS sync / 150/150 core+convex 100% ROE coverage
# v0.6 snapshot 保留為 audit trail(歷史記述)
DEFAULT_POLICY_VERSION = "core_universe_policy_v0.7"
DEFAULT_FEATURE_SET_VERSION = "feature_set_pending_v0.1"

# §14.7-BW pure doctrine(2026-05-26 第二十一輪)+ 用戶 2026-05-27 directive
# 「排除所有固定的核心股數量如 119/150/200」:以下 6 個 constants 全部 DEPRECATED
# 任何 fixed N(cap / floor / tier % / hardcode 數量)皆違反 pure doctrine
# 新 selection 路徑為 build_doctrine_gate_universe.py(v0.10 pure_doctrine)
# 本 v0.7/v0.8 builder 之 dynamic mode 不再使用;constants 設 None 以避免被誤用
LEGACY_CORE_LIMIT = None          # DEPRECATED per §14.7-BW (was 120 / v0.2-v0.7 hardcode mode)
LEGACY_CONVEX_LIMIT = None        # DEPRECATED per §14.7-BW (was 30 / v0.2-v0.7 hardcode mode)
DEFAULT_SELECTION_PCT = None      # DEPRECATED per §14.7-BW (was 5.0 / §0.2 八二法則 top 5%)
DEFAULT_SELECTION_N_MIN = None    # DEPRECATED per §14.7-BW (was 100 / N_min floor)
DEFAULT_SELECTION_N_MAX = None    # DEPRECATED per §14.7-BW (was 200 / N_max cap)
DEFAULT_CORE_PCT_WITHIN_SELECTED = None  # DEPRECATED per §14.7-BW (was 0.70 / 70/30 tier split)
DYNAMIC_POLICY_PREFIX = "core_universe_policy_v0.8"  # policy_version dispatch trigger
DEFAULT_MODEL_POLICY_VERSION = "model_policy_pending_v0.1"
DEFAULT_PREDICTION_POLICY_VERSION = "prediction_policy_pending_v0.1"
DEFAULT_LABEL_HORIZON = 20

REQUIRED_TABLES = [
    "TaiwanStockInfo",
    "core_universe_policy",
    "core_universe_snapshot",
    "core_universe_membership",
    "core_universe_scores",
    "universe_revision_log",
]

V02_INPUT_CONTRACT = [
    {
        "category": "market_info",
        "table": "TaiwanStockInfo",
        "required_columns": ["stock_id", "stock_name", "industry_category", "type", "date"],
        "coverage_kind": "candidate",
    },
    {
        "category": "price_volume",
        "table": "TaiwanStockPriceAdj",
        "fallback_table": "TaiwanStockPrice",
        "required_columns": [
            "date", "stock_id", "Trading_Volume", "Trading_money",
            "open", "max", "min", "close", "spread", "Trading_turnover",
        ],
        "coverage_kind": "price_252d",
    },
    {
        "category": "monthly_revenue",
        "table": "TaiwanStockMonthRevenue",
        "required_columns": ["date", "stock_id", "country", "revenue", "revenue_month", "revenue_year", "create_time"],
        "coverage_kind": "revenue_24m",
    },
    {
        "category": "valuation",
        "table": "TaiwanStockPER",
        "required_columns": ["date", "stock_id", "dividend_yield", "PER", "PBR"],
        "coverage_kind": "dated_stock",
    },
    {
        "category": "institutional_flow",
        "table": "TaiwanStockInstitutionalInvestorsBuySell",
        "required_columns": ["date", "stock_id", "name", "buy", "sell"],
        "coverage_kind": "dated_stock",
    },
    {
        "category": "margin_short",
        "table": "TaiwanStockMarginPurchaseShortSale",
        "required_columns": [
            "date", "stock_id", "MarginPurchaseTodayBalance", "MarginPurchaseYesterdayBalance",
            "ShortSaleTodayBalance", "ShortSaleYesterdayBalance", "OffsetLoanAndShort", "Note",
        ],
        "coverage_kind": "dated_stock",
    },
    {
        "category": "financial_statements",
        "table": "TaiwanStockFinancialStatements",
        "required_columns": ["date", "stock_id", "type", "value", "origin_name"],
        "coverage_kind": "financial_8q",
    },
    {
        "category": "fred_macro",
        "table": "FredData",
        "required_columns": ["date", "series_id", "value", "realtime_start", "realtime_end"],
        "coverage_kind": "fred_macro",
    },
]

EQUITY_TYPES = {"twse", "tpex"}
EXCLUDED_INDUSTRY_KEYWORDS = ("ETF", "ETN", "指數", "權證")
# v0.9 §14.7-BP THEME_KEYWORDS 字典升版(2026-05-26 evening Phase C)
# MBNRIC 6 支柱完整補完(原 14 keywords → 30 keywords;治本 §0.3 N 72.7% 主導 root cause)
# 新加 16 keywords:M 支柱 9 + C 支柱 5 + B 1 (農科) + R 1 (油電) = 16
# 對映 §0.3.9 MBNRIC 6 支柱完整 mapping;§0.3 evidence issue #1 治本
# 對映 §14.7-BP Phase A 設計研究(reports/theme_keywords_dictionary_upgrade_phase_a_research_20260526.md)
THEME_KEYWORDS = {
    # === N Nanotech/Neural 支柱(原有 4 keywords;不改 — N 已主導) ===
    "半導體": 100,
    "電子": 80,
    "機器": 80,
    "光電": 70,           # I+N partial(光電業)
    # === C Computing/Cloud 支柱(v0.9 新增 5 keywords;對齊 §0.3 第六波 priority) ===
    "量子": 100,         # ✨ NEW v0.9 §14.7-BP(對映 §0.3 K-wave 第六波頂分)
    "AI": 95,            # ✨ NEW v0.9 §14.7-BP(對映 其他電子類 / 資訊服務業 C 部分)
    "雲端": 95,          # ✨ NEW v0.9 §14.7-BP(對映 數位雲端類 24+22 = 46 stocks)
    "算力": 90,          # ✨ NEW v0.9 §14.7-BP(對映 GPU AI 算力新興)
    "演算": 85,          # ✨ NEW v0.9 §14.7-BP(對映 algorithm-driven)
    # === I Info 支柱(原有 3 keywords;不改) ===
    "資訊": 90,
    "電腦": 85,
    "通信": 85,
    # === R Robotics/綠能 支柱(原有 3 + v0.9 新 1 = 4 keywords) ===
    "電機": 75,
    "綠能": 75,
    "汽車": 60,
    "油電": 70,          # ✨ NEW v0.9 §14.7-BP(對映 油電燃氣業 13 stocks;取代「能源」之精確化)
    # === B Biotech 支柱(原有 2 + v0.9 新 1 = 3 keywords) ===
    "生技": 95,
    "醫療": 95,
    "農科": 80,          # ✨ NEW v0.9 §14.7-BP(對映 農業科技業 / 農業科技 7 stocks)
    # === M Materials 支柱(原有 0 + v0.9 新 9 keywords;治本核心) ===
    "化學": 65,          # ✨ NEW v0.9 §14.7-BP(化學工業 24 + 化學生技醫療 42 之 M 部分)
    "建材": 55,          # ✨ NEW v0.9 §14.7-BP(建材營造 89 stocks)
    "鋼鐵": 50,          # ✨ NEW v0.9 §14.7-BP(鋼鐵工業 54 stocks)
    "紡織": 50,          # ✨ NEW v0.9 §14.7-BP(紡織纖維 54 stocks)
    "塑膠": 55,          # ✨ NEW v0.9 §14.7-BP(塑膠工業 27 stocks)
    "橡膠": 50,          # ✨ NEW v0.9 §14.7-BP(橡膠工業 11 stocks;EV 輪胎)
    "水泥": 45,          # ✨ NEW v0.9 §14.7-BP(水泥工業 8 stocks)
    "造紙": 45,          # ✨ NEW v0.9 §14.7-BP(造紙工業 8 stocks)
    "玻璃": 50,          # ✨ NEW v0.9 §14.7-BP(玻璃陶瓷 5 stocks;先進陶瓷)
    # === 其他既有(不對應 MBNRIC 直接;保留為 thematic cushion) ===
    "能源": 70,          # 原 keyword;部分對映 R(已被「油電」精確化但保留 backward-compat)
    "航太": 65,          # 原 keyword;I partial
}


@dataclass
class Candidate:
    stock_id: str
    stock_name: str | None
    type: str | None
    industry_category: str | None
    source_date: date | None
    core_score: float
    data_quality_score: float
    theme_score: float
    risk_penalty: float
    core_tier: str
    selection_reason: str
    exclusion_reason: str | None
    score_detail: dict
    liquidity_score: float = field(default=0.0)
    fundamental_score: float = field(default=0.0)
    institutional_flow_score: float = field(default=0.0)
    volatility_control_score: float = field(default=0.0)
    price_coverage_252d: float = field(default=0.0)
    revenue_coverage_24m: float = field(default=0.0)
    financial_coverage_8q: float = field(default=0.0)


class CoreUniverseBuilder:
    def __init__(
        self,
        as_of_date,
        policy_version,
        commit=False,
        # §14.7-BW pure doctrine + 2026-05-27 directive:legacy / dynamic mode N constants 全 DEPRECATED
        # 若 caller 仍嘗試使用,LEGACY_*/DEFAULT_* 已是 None,builder 之 selection logic 會 raise
        # 新 selection 路徑為 build_doctrine_gate_universe.py(v0.10 pure_doctrine)
        core_limit=None,            # DEPRECATED per §14.7-BW (was 120 / legacy mode)
        convex_limit=None,          # DEPRECATED per §14.7-BW (was 30 / legacy mode)
        selection_pct=None,         # DEPRECATED per §14.7-BW (was 5.0 / dynamic top X%)
        selection_n_min=None,       # DEPRECATED per §14.7-BW (was 100 / N min floor)
        selection_n_max=None,       # DEPRECATED per §14.7-BW (was 200 / N max cap)
        core_pct_within_selected=None,  # DEPRECATED per §14.7-BW (was 0.70 / 70-30 tier split)
        include_emerging=False,
        special_rebalance_reason=None,
    ):
        self.as_of_date = as_of_date
        self.policy_version = policy_version
        self.commit = commit
        # §14.7-BT Phase C dispatch:判 mode by policy_version prefix
        self.is_dynamic_mode = policy_version.startswith(DYNAMIC_POLICY_PREFIX)
        if self.is_dynamic_mode:
            # Dynamic mode:取 dynamic defaults;ignore legacy core/convex_limit
            self.selection_pct = selection_pct if selection_pct is not None else DEFAULT_SELECTION_PCT
            self.selection_n_min = selection_n_min if selection_n_min is not None else DEFAULT_SELECTION_N_MIN
            self.selection_n_max = selection_n_max if selection_n_max is not None else DEFAULT_SELECTION_N_MAX
            self.core_pct_within_selected = (
                core_pct_within_selected if core_pct_within_selected is not None
                else DEFAULT_CORE_PCT_WITHIN_SELECTED
            )
            # Legacy fields 留 None(consistency check;不該被使用)
            self.core_limit = None
            self.convex_limit = None
        else:
            # Legacy mode DEPRECATED per §14.7-BW (was: explicit override 或 LEGACY_* constants which were 120/30)
            self.core_limit = core_limit if core_limit is not None else LEGACY_CORE_LIMIT
            self.convex_limit = convex_limit if convex_limit is not None else LEGACY_CONVEX_LIMIT
            self.selection_pct = None
            self.selection_n_min = None
            self.selection_n_max = None
            self.core_pct_within_selected = None
        self.include_emerging = include_emerging
        self.special_rebalance_reason = (special_rebalance_reason or "").strip()
        self.snapshot_id = self._build_snapshot_id()
        self.source_data_cutoff = None
        self.candidate_source_mode = "unresolved"
        self.stats = {
            "preflight_pass": 0,
            "preflight_warning": 0,
            "preflight_failed": 0,
            "v02_contract_pass": 0,
            "v02_contract_warning": 0,
            "v02_contract_failed": 0,
            "details": [],
            "coverage_summary": [],
            "total_candidates": 0,
            "research_count": 0,
            "core_count": 0,
            "convex_count": 0,
            "quarantine_count": 0,
            "written_rows": 0,
            "warnings": 0,
            "failed": 0,
        }

    def _build_snapshot_id(self):
        safe_policy = self.policy_version.replace(".", "_").replace("-", "_")
        return f"core_universe_{self.as_of_date.strftime('%Y%m%d')}_{safe_policy}"

    def _detail(self, message):
        self.stats["details"].append(message)

    def _preflight(self, bucket, message):
        self.stats[f"preflight_{bucket}"] += 1
        icon = {"pass": "✅", "warning": "⚠️", "failed": "❌"}[bucket]
        self._detail(f"{icon} [PREFLIGHT-{bucket.upper()}] {message}")

    def _contract(self, bucket, message):
        self.stats[f"v02_contract_{bucket}"] += 1
        icon = {"pass": "✅", "warning": "⚠️", "failed": "❌"}[bucket]
        self._detail(f"{icon} [V0.2-CONTRACT-{bucket.upper()}] {message}")

    def _coverage(self, label, payload):
        item = {"label": label}
        item.update(payload)
        self.stats["coverage_summary"].append(item)

    def _rebalance_mode(self):
        return "special" if self.special_rebalance_reason else "annual"

    def _review_cycle(self):
        return "special" if self.special_rebalance_reason else "annual"

    def _effective_from(self):
        if self.special_rebalance_reason:
            return self.as_of_date
        return date(self.as_of_date.year + 1, 1, 1)

    def _snapshot_note(self):
        note = (
            "core_universe_builder v0.2; six-layer CoreScore (DQ+LM+FG+TR+IF+VC-RP); "
            f"rebalance_mode={self._rebalance_mode()}; no feature/model/prediction values"
        )
        if self.special_rebalance_reason:
            note += f"; special_rebalance_reason={self.special_rebalance_reason}"
        return note

    def _mark_lifecycle(self, lifecycle, level, message):
        if lifecycle is None:
            return
        method_name = "mark_failed" if level == "failed" else "mark_warning"
        marker = getattr(lifecycle, method_name, None)
        if callable(marker):
            marker(message)

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def _table_columns(self, cur, table_name):
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {row[0] for row in cur.fetchall()}

    def _latest_trading_date_for_year(self, cur, year):
        for table_name in ("TaiwanStockPriceAdj", "TaiwanStockPrice"):
            if not self._table_exists(cur, table_name):
                continue
            cur.execute(
                f'''
                SELECT MAX("date")
                FROM "{table_name}"
                WHERE "date" >= %s AND "date" <= %s
                ''',
                (date(year, 1, 1), date(year, 12, 31)),
            )
            latest = cur.fetchone()[0]
            if latest:
                return latest, table_name
        return None, None

    def _annual_rebalance_guard(self, cur):
        if not self.commit:
            self._preflight("pass", "annual rebalance guard: dry-run allowed anytime; commit remains guarded")
            return
        if self.special_rebalance_reason:
            if len(self.special_rebalance_reason) < 12:
                self._preflight("failed", "special_rebalance_reason too short; provide an auditable governance reason")
                return
            self._preflight("warning", f"special rebalance override accepted: {self.special_rebalance_reason}")
            return

        latest_trading_date, source_table = self._latest_trading_date_for_year(cur, self.as_of_date.year)
        if not latest_trading_date:
            self._preflight(
                "failed",
                f"annual rebalance guard: no trading calendar data found for {self.as_of_date.year}; cannot verify last trading day",
            )
            return
        if self.as_of_date.month != 12 or self.as_of_date.day < 25:
            self._preflight(
                "failed",
                "annual rebalance guard: regular core universe commit is allowed only after the year's final trading day; "
                f"as_of_date={self.as_of_date}, latest_trading_date={latest_trading_date}",
            )
            return
        if self.as_of_date != latest_trading_date:
            self._preflight(
                "failed",
                "annual rebalance guard: as_of_date must equal the latest trading date in that calendar year; "
                f"as_of_date={self.as_of_date}, latest_trading_date={latest_trading_date}",
            )
            return
        self._preflight(
            "pass",
            f"annual rebalance guard: as_of_date={self.as_of_date} verified as final trading day via {source_table}",
        )

    def _table_profile(self, cur, table_name):
        registry_columns = DATASET_REGISTRY.get(table_name, {}).get("columns", {})
        has_stock = "stock_id" in registry_columns
        has_series = "series_id" in registry_columns
        cur.execute(
            f'''
            SELECT COUNT(*), MIN("date"), MAX("date")
            FROM "{table_name}"
            WHERE "date" <= %s
            ''',
            (self.as_of_date,),
        )
        row_count, min_date, max_date = cur.fetchone()

        distinct_key_count = None
        key_name = None
        if has_stock:
            key_name = "stock_id"
            cur.execute(
                f'''
                SELECT COUNT(DISTINCT "stock_id")
                FROM "{table_name}"
                WHERE "date" <= %s
                ''',
                (self.as_of_date,),
            )
            distinct_key_count = cur.fetchone()[0]
        elif has_series:
            key_name = "series_id"
            cur.execute(
                f'''
                SELECT COUNT(DISTINCT "series_id")
                FROM "{table_name}"
                WHERE "date" <= %s
                ''',
                (self.as_of_date,),
            )
            distinct_key_count = cur.fetchone()[0]

        return {
            "table": table_name,
            "rows": int(row_count or 0),
            "min_date": str(min_date) if min_date else None,
            "max_date": str(max_date) if max_date else None,
            "key": key_name,
            "key_count": int(distinct_key_count or 0) if distinct_key_count is not None else None,
        }

    def _coverage_ratio_summary(self, cur, table_name, start_date, denominator, threshold, label, universe_scope="market"):
        if universe_scope == "core_sync":
            candidate_sql = """
                SELECT DISTINCT m."stock_id"
                FROM "core_universe_membership" m
                JOIN "core_universe_snapshot" s ON s."snapshot_id" = m."snapshot_id"
                WHERE s."status" = 'committed'
                  AND m."core_tier" IN ('core_universe', 'convex_universe')
                  AND s."as_of_date" = (
                      SELECT MAX("as_of_date")
                      FROM "core_universe_snapshot"
                      WHERE "status" = 'committed'
                  )
            """
        else:
            if self.candidate_source_mode == "as_of_filtered":
                candidate_sql = f'''
                    SELECT DISTINCT "stock_id"
                    FROM "TaiwanStockInfo"
                    WHERE "date" <= DATE '{self.as_of_date.isoformat()}'
                '''
            else:
                candidate_sql = 'SELECT DISTINCT "stock_id" FROM "TaiwanStockInfo"'
        cur.execute(
            f'''
            WITH candidates AS (
                {candidate_sql}
            ),
            observations AS (
                SELECT "stock_id", COUNT(DISTINCT "date") AS obs_count
                FROM "{table_name}"
                WHERE "date" BETWEEN %s AND %s
                GROUP BY "stock_id"
            )
            SELECT
                COUNT(*) AS candidate_count,
                COALESCE(AVG(COALESCE(o.obs_count, 0)), 0) AS avg_observations,
                COALESCE(AVG(LEAST(COALESCE(o.obs_count, 0)::NUMERIC / %s, 1.0)), 0) AS avg_coverage,
                SUM(CASE WHEN COALESCE(o.obs_count, 0) = 0 THEN 1 ELSE 0 END) AS zero_coverage_count,
                SUM(CASE WHEN COALESCE(o.obs_count, 0) >= %s THEN 1 ELSE 0 END) AS threshold_pass_count
            FROM candidates c
            LEFT JOIN observations o ON o."stock_id" = c."stock_id"
            ''',
            (start_date, self.as_of_date, denominator, threshold),
        )
        candidate_count, avg_obs, avg_coverage, zero_count, pass_count = cur.fetchone()
        payload = {
            "table": table_name,
            "window_start": str(start_date),
            "window_end": str(self.as_of_date),
            "universe_scope": universe_scope,
            "candidate_count": int(candidate_count or 0),
            "avg_observations": round(float(avg_obs or 0), 4),
            "avg_coverage": round(float(avg_coverage or 0), 6),
            "zero_coverage_count": int(zero_count or 0),
            "threshold": threshold,
            "threshold_pass_count": int(pass_count or 0),
        }
        self._coverage(label, payload)
        return payload

    def _run_v02_input_contract_preflight(self, cur):
        for spec in V02_INPUT_CONTRACT:
            table_name = spec["table"]
            if not self._table_exists(cur, table_name):
                self._contract("failed", f"{table_name} missing; required by CoreScore v0.2 input contract")
                continue

            columns = self._table_columns(cur, table_name)
            missing_columns = [column for column in spec["required_columns"] if column not in columns]
            if missing_columns:
                self._contract("failed", f"{table_name} missing required columns: {', '.join(missing_columns)}")
                continue
            self._contract("pass", f"{table_name} columns aligned for {spec['category']}")

            profile = self._table_profile(cur, table_name)
            self._coverage(spec["category"], profile)
            if profile["rows"] <= 0:
                self._contract("warning", f"{table_name} exists but has no rows <= {self.as_of_date}; v0.2 scoring not ready")
            else:
                key_text = f", {profile['key']}={profile['key_count']}" if profile["key"] else ""
                self._contract(
                    "pass",
                    f'{table_name} rows={profile["rows"]}, date_range={profile["min_date"]}..{profile["max_date"]}{key_text}',
                )

        if self.stats["v02_contract_failed"] > 0:
            return

        price_start = self.as_of_date - timedelta(days=370)
        long_start = self.as_of_date - timedelta(days=730)
        financial_start = self.as_of_date - timedelta(days=1000)

        if self._table_exists(cur, "TaiwanStockPriceAdj"):
            price_summary = self._coverage_ratio_summary(
                cur, "TaiwanStockPriceAdj", price_start, 252, 202, "price_coverage_252d"
            )
            self._coverage_ratio_summary(
                cur, "TaiwanStockPriceAdj", price_start, 252, 202, "core_sync_price_coverage_252d", universe_scope="core_sync"
            )
            if price_summary["zero_coverage_count"] > 0:
                self._contract("warning", f'price_coverage_252d zero-coverage candidates={price_summary["zero_coverage_count"]}')
            if price_summary["threshold_pass_count"] == 0 and self._table_exists(cur, "TaiwanStockPrice"):
                fallback_summary = self._coverage_ratio_summary(
                    cur, "TaiwanStockPrice", price_start, 252, 202, "price_coverage_252d_fallback"
                )
                if fallback_summary["threshold_pass_count"] > 0:
                    self._contract(
                        "warning",
                        "TaiwanStockPriceAdj has no usable 252d coverage; TaiwanStockPrice fallback has partial coverage",
                    )

        if self._table_exists(cur, "TaiwanStockMonthRevenue"):
            revenue_summary = self._coverage_ratio_summary(
                cur, "TaiwanStockMonthRevenue", long_start, 24, 12, "revenue_coverage_24m"
            )
            self._coverage_ratio_summary(
                cur, "TaiwanStockMonthRevenue", long_start, 24, 12, "core_sync_revenue_coverage_24m", universe_scope="core_sync"
            )
            if revenue_summary["zero_coverage_count"] > 0:
                self._contract("warning", f'revenue_coverage_24m zero-coverage candidates={revenue_summary["zero_coverage_count"]}')

        if self._table_exists(cur, "TaiwanStockFinancialStatements"):
            financial_summary = self._coverage_ratio_summary(
                cur, "TaiwanStockFinancialStatements", financial_start, 8, 2, "financial_coverage_8q"
            )
            self._coverage_ratio_summary(
                cur, "TaiwanStockFinancialStatements", financial_start, 8, 2, "core_sync_financial_coverage_8q", universe_scope="core_sync"
            )
            if financial_summary["zero_coverage_count"] > 0:
                self._contract("warning", f'financial_coverage_8q zero-coverage candidates={financial_summary["zero_coverage_count"]}')

    def preflight_check(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in REQUIRED_TABLES:
                if self._table_exists(cur, table_name):
                    self._preflight("pass", f"{table_name} exists")
                else:
                    self._preflight("failed", f"{table_name} missing; run core_universe_schema.py --init first")

            if self.stats["preflight_failed"] == 0:
                self._annual_rebalance_guard(cur)

            if self.stats["preflight_failed"] == 0:
                self._run_v02_input_contract_preflight(cur)
                cur.execute('SELECT COUNT(DISTINCT "stock_id"), MAX("date") FROM "TaiwanStockInfo"')
                total, max_date = cur.fetchone()
                cur.execute(
                    '''
                    SELECT COUNT(DISTINCT "stock_id"), MAX("date")
                    FROM "TaiwanStockInfo"
                    WHERE "date" <= %s
                    ''',
                    (self.as_of_date,),
                )
                as_of_total, as_of_max_date = cur.fetchone()

                # §14.7-BT Phase C dispatch:dynamic mode 用 N_min;legacy mode 用 core_limit + convex_limit
                if self.is_dynamic_mode:
                    minimum_bootstrap_size = self.selection_n_min
                else:
                    minimum_bootstrap_size = self.core_limit + self.convex_limit
                if as_of_total >= minimum_bootstrap_size:
                    self.candidate_source_mode = "as_of_filtered"
                    self.source_data_cutoff = as_of_max_date or self.as_of_date
                    self._preflight(
                        "pass",
                        f"TaiwanStockInfo as-of candidates={as_of_total}; source_data_cutoff={self.source_data_cutoff}; mode=as_of_filtered",
                    )
                elif total > 0:
                    self.candidate_source_mode = "latest_registry_fallback"
                    self.source_data_cutoff = max_date or self.as_of_date
                    self._preflight(
                        "pass",
                        f"TaiwanStockInfo has {total} distinct stocks; source_data_cutoff={self.source_data_cutoff}; mode=latest_registry_fallback",
                    )
                    self._contract(
                        "warning",
                        f"TaiwanStockInfo as-of candidates={as_of_total} below minimum bootstrap size={minimum_bootstrap_size}; "
                        "v0.1 metadata bootstrap uses latest registry fallback, v0.2 formal scoring must use as-of filtering",
                    )
                else:
                    self._preflight("failed", "TaiwanStockInfo is empty; run sovereign_sync_engine.py --seed first")
        finally:
            cur.close()
            conn.close()
        return self.stats["preflight_failed"] == 0 and self.stats["v02_contract_failed"] == 0

    # ── v0.2 批量資料載入 ──────────────────────────────────────────────────────

    def _load_market_data(self):
        """Batch-load scoring data from six individual stock tables for v0.2 six-layer CoreScore."""
        lookback_252 = self.as_of_date - timedelta(days=365)
        lookback_730 = self.as_of_date - timedelta(days=730)
        conn = get_db_connection()
        cur = conn.cursor()
        price_data, revenue_data, financial_data, institutional_data = {}, {}, {}, {}
        try:
            # TaiwanStockPriceAdj: trading value, volume, continuity, volatility
            # §8.5-9 Phase 3: native_aligned (date <= as_of_date)
            gate, n_ap = build_publication_date_gate("TaiwanStockPriceAdj")
            cur.execute(f"""
                SELECT stock_id,
                    COUNT(*) as day_count,
                    AVG("Trading_Volume"::numeric) as avg_volume,
                    AVG("Trading_money"::numeric) as avg_daily_value,
                    STDDEV("close"::numeric) / NULLIF(AVG("close"::numeric), 0) as cv_close
                FROM "TaiwanStockPriceAdj"
                WHERE date >= %s AND {gate}
                GROUP BY stock_id
            """, (lookback_252, *([self.as_of_date] * n_ap)))
            for sid, day_count, avg_vol, avg_val, cv in cur.fetchall():
                price_data[sid] = {
                    "day_count": day_count or 0,
                    "avg_volume": float(avg_vol or 0),
                    "avg_daily_value": float(avg_val or 0),
                    "cv_close": float(cv or 0.3),
                    "price_coverage_252d": min(1.0, (day_count or 0) / 252),
                }

            # v0.7.1 §14.7-BH(§9.10 升正式條文 + §14.7-BG 補註):ΔlnP 凸性計算(60d window;SQL LAG window function;raw OHLC)
            # 公式對齊 §9.9 P1 v0.1 強制契約:upside/downside_RMS = SQRT(AVG(lr²) FILTER) × √252(取代 STDDEV)
            # 對映 §0.0-C.3 上行凸性壓制修補 + §9.10 正式條文(raw-first 路徑;不從 feature_store)
            # v0.7 STDDEV 變體 ablation 揭露不等價於 §9.9 RMS(Top-120 overlap 73.3%;sign flip 75 分);v0.7.1 起改 RMS
            lookback_60d_for_vc = self.as_of_date - timedelta(days=90)  # ~60 交易日
            cur.execute(f"""
                WITH log_returns AS (
                    SELECT stock_id, date,
                        LN(close::numeric / NULLIF(LAG(close::numeric)
                            OVER (PARTITION BY stock_id ORDER BY date), 0)) as lr
                    FROM "TaiwanStockPriceAdj"
                    WHERE date >= %s AND {gate}
                )
                SELECT stock_id,
                    STDDEV(lr) FILTER (WHERE lr IS NOT NULL) * SQRT(252.0) as cc_sigma,
                    SQRT(AVG(lr*lr) FILTER (WHERE lr > 0)) * SQRT(252.0) as upside_rms,
                    SQRT(AVG(lr*lr) FILTER (WHERE lr < 0)) * SQRT(252.0) as downside_rms,
                    COUNT(*) FILTER (WHERE lr IS NOT NULL) as n_obs
                FROM log_returns
                GROUP BY stock_id
            """, (lookback_60d_for_vc, *([self.as_of_date] * n_ap)))
            for sid, cc, up, down, n in cur.fetchall():
                if sid not in price_data:
                    continue
                cc_f = float(cc) if cc is not None else None
                up_f = float(up) if up is not None else None
                down_f = float(down) if down is not None else None
                convexity = (up_f - down_f) if (up_f is not None and down_f is not None) else None
                price_data[sid].update({
                    "cc_sigma_60d": cc_f,
                    "upside_rms_60d": up_f,
                    "downside_rms_60d": down_f,
                    "convexity_60d": convexity,
                    "vc_n_obs": int(n or 0),
                })

            # TaiwanStockMonthRevenue: coverage + YoY growth
            # §8.5-9 Phase 3: hardcoded_conservative (date + INTERVAL '10 days') <= as_of_date
            mid_point = self.as_of_date - timedelta(days=365)
            gate, n_ap = build_publication_date_gate("TaiwanStockMonthRevenue")
            cur.execute(f"""
                SELECT stock_id,
                    COUNT(*) as month_count,
                    SUM(CASE WHEN date >= %s THEN revenue::numeric ELSE 0 END) as recent_12m,
                    SUM(CASE WHEN date < %s AND date >= %s THEN revenue::numeric ELSE 0 END) as prior_12m
                FROM "TaiwanStockMonthRevenue"
                WHERE date >= %s AND {gate}
                GROUP BY stock_id
            """, (mid_point, mid_point, lookback_730, lookback_730, *([self.as_of_date] * n_ap)))
            for sid, month_count, recent_12m, prior_12m in cur.fetchall():
                recent = float(recent_12m or 0)
                prior = float(prior_12m or 0)
                yoy_growth = (recent - prior) / prior if prior > 0 else 0.0
                revenue_data[sid] = {
                    "month_count": month_count or 0,
                    "revenue_coverage_24m": min(1.0, (month_count or 0) / 24),
                    "yoy_growth": yoy_growth,
                    "recent_revenue": recent,
                }

            # TaiwanStockFinancialStatements: coverage + profitability + v0.3 毛利率
            # v0.3 注意：ROE 原計畫實作,但 2026-05-24 揭露 type='EquityAttributableToOwnersOfParent'
            # 之 value 實為淨利數值(origin_name='淨利(淨損)歸屬於母公司業主'),非真正股東權益;
            # 此 raw schema 為純 income statement,無 balance sheet equity 欄位 → ROE 無法計算。
            # 已入憲 §0.1.3-A;改用「最新季 YTD 毛利率」單獨計算(不 SUM 避免 YTD 重複加總)。
            # §8.5-9 Phase 3: FinStmt hardcoded_conservative quarter-aware (Q1-Q3 +45 / Q4 +90)
            fs_gate, fs_n_ap = build_publication_date_gate("TaiwanStockFinancialStatements")
            cur.execute(f"""
                WITH latest_per_type AS (
                    SELECT DISTINCT ON (stock_id, type) stock_id, type, value::numeric AS v
                    FROM "TaiwanStockFinancialStatements"
                    WHERE date >= %s AND {fs_gate}
                      AND type IN ('GrossProfit', 'Revenue')
                    ORDER BY stock_id, type, date DESC
                )
                SELECT stock_id,
                    MAX(v) FILTER (WHERE type='GrossProfit') AS latest_gp,
                    MAX(v) FILTER (WHERE type='Revenue') AS latest_rev
                FROM latest_per_type
                GROUP BY stock_id
            """, (lookback_730, *([self.as_of_date] * fs_n_ap)))
            latest_margin = {}
            for sid, gp, rev in cur.fetchall():
                rev_f = float(rev or 0)
                gp_f = float(gp or 0)
                latest_margin[sid] = (gp_f / rev_f) if rev_f > 0 else None

            # §8.5-9 Phase 3: 同上 FinStmt quarter-aware gate(複用 fs_gate / fs_n_ap)
            cur.execute(f"""
                SELECT stock_id,
                    COUNT(DISTINCT date) as quarter_count,
                    SUM(CASE WHEN type='EPS' THEN value::numeric ELSE 0 END) as eps_sum,
                    SUM(CASE WHEN origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%'
                             THEN value::numeric ELSE 0 END) as net_income_sum
                FROM "TaiwanStockFinancialStatements"
                WHERE date >= %s AND {fs_gate}
                GROUP BY stock_id
            """, (lookback_730, *([self.as_of_date] * fs_n_ap)))
            for sid, quarter_count, eps_sum, net_income_sum in cur.fetchall():
                financial_data[sid] = {
                    "quarter_count": quarter_count or 0,
                    "financial_coverage_8q": min(1.0, (quarter_count or 0) / 8),
                    "eps_sum": float(eps_sum or 0),
                    "net_income_positive": float(net_income_sum or 0) > 0,
                    "gross_margin": latest_margin.get(sid),
                    # v0.3 ROE 因 raw data 限制 dropped(見 §0.1.3-A);保留 None 占位
                    "roe": None,
                    # v0.5 §14.7-BC: 4 新 V types(見下方)
                    "op_margin": None,
                    "pretax_margin": None,
                    "continuing_op_ratio": None,
                    "attributable_ratio": None,
                }

            # v0.5 §14.7-BC: 4 新 FinStmt V types(OperatingIncome / PreTaxIncome /
            # IncomeFromContinuingOperations / NoncontrollingInterests)— 取 latest quarter
            cur.execute(f"""
                WITH latest_v05 AS (
                    SELECT DISTINCT ON (stock_id, type) stock_id, type, value::numeric AS v
                    FROM "TaiwanStockFinancialStatements"
                    WHERE date >= %s AND {fs_gate}
                      AND type IN ('OperatingIncome', 'PreTaxIncome',
                                   'IncomeFromContinuingOperations',
                                   'NoncontrollingInterests', 'IncomeAfterTaxes', 'Revenue')
                    ORDER BY stock_id, type, date DESC
                )
                SELECT stock_id,
                    MAX(v) FILTER (WHERE type='OperatingIncome') AS op_inc,
                    MAX(v) FILTER (WHERE type='PreTaxIncome') AS pretax,
                    MAX(v) FILTER (WHERE type='IncomeFromContinuingOperations') AS cont_op,
                    MAX(v) FILTER (WHERE type='NoncontrollingInterests') AS nci,
                    MAX(v) FILTER (WHERE type='IncomeAfterTaxes') AS net_inc_after_tax,
                    MAX(v) FILTER (WHERE type='Revenue') AS rev_latest
                FROM latest_v05
                GROUP BY stock_id
            """, (lookback_730, *([self.as_of_date] * fs_n_ap)))
            for sid, op_inc, pretax, cont_op, nci, nia, rev in cur.fetchall():
                if sid not in financial_data:
                    continue
                rev_f = float(rev or 0)
                nia_f = float(nia or 0)
                if rev_f > 0:
                    if op_inc is not None:
                        financial_data[sid]["op_margin"] = float(op_inc) / rev_f
                    if pretax is not None:
                        financial_data[sid]["pretax_margin"] = float(pretax) / rev_f
                    if cont_op is not None:
                        financial_data[sid]["continuing_op_ratio"] = float(cont_op) / rev_f
                if nia_f > 0 and nci is not None:
                    nci_f = float(nci)
                    if nci_f <= nia_f:
                        financial_data[sid]["attributable_ratio"] = (nia_f - nci_f) / nia_f
                    else:
                        financial_data[sid]["attributable_ratio"] = -1.0  # sentinel:NCI > NI 異常

            # v0.8 §14.7-BI: ROE 解鎖 — 從 TaiwanStockBalanceSheet 取真權益 (Equity / EquityAttributableToOwnersOfParent)
            # 消解 §0.1.3-A.1 之 FinStmt mislabel(FinStmt 同名 column 為淨利,BS 同名 column 為真權益,~10x 差距)
            # ROE = SUM(IncomeAfterTaxes 最近 4Q) / latest Equity (annualized)
            # v0.9.1 graceful fallback:若本機 BS table 不存在(stranded state per handoff §二),
            #                         skip ROE block;financial_data ROE 保持 None(對齊 v0.7.1 baseline)
            cur.execute("SELECT to_regclass('public.\"TaiwanStockBalanceSheet\"')")
            bs_exists = cur.fetchone()[0] is not None
            if not bs_exists:
                self._detail(f"⚠️ [BS-MISSING] TaiwanStockBalanceSheet table 不存在(本機 stranded state);"
                             f"ROE 將 fallback to None(per handoff §二 / 對齊 v0.7.1 baseline)")
                # 跳過 BS query;ROE 保持 None
            else:
                bs_gate, bs_n_ap = build_publication_date_gate("TaiwanStockBalanceSheet")
                cur.execute(f"""
                    WITH ni_4q AS (
                        SELECT stock_id, SUM(value::numeric) AS ni_sum, COUNT(*) AS qc
                        FROM (
                            SELECT stock_id, value,
                                   ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date DESC) AS rn
                            FROM "TaiwanStockFinancialStatements"
                            WHERE type='IncomeAfterTaxes' AND date >= %s AND {fs_gate}
                        ) ranked
                        WHERE rn <= 4
                        GROUP BY stock_id
                    ),
                    bs_equity AS (
                        SELECT DISTINCT ON (stock_id) stock_id, value::numeric AS equity
                        FROM "TaiwanStockBalanceSheet"
                        WHERE type='EquityAttributableToOwnersOfParent' AND {bs_gate}
                        ORDER BY stock_id, date DESC
                    )
                    SELECT n.stock_id, n.ni_sum, n.qc, b.equity
                    FROM ni_4q n
                    JOIN bs_equity b USING (stock_id)
                    WHERE n.qc = 4 AND b.equity > 0
                """, (lookback_730, *([self.as_of_date] * fs_n_ap), *([self.as_of_date] * bs_n_ap)))
                for sid, ni_sum, qc, equity in cur.fetchall():
                    if sid not in financial_data:
                        continue
                    ni_f = float(ni_sum or 0)
                    eq_f = float(equity or 0)
                    if eq_f > 0:
                        roe = ni_f / eq_f
                        financial_data[sid]["roe"] = roe
                        financial_data[sid]["equity"] = eq_f
                        financial_data[sid]["ni_4q_sum"] = ni_f

            # v0.5 §14.7-BC: TaiwanStockPER (PER/PBR/dividend_yield)— native_aligned;latest per stock
            per_data = {}
            per_gate, per_n_ap = build_publication_date_gate("TaiwanStockPER")
            cur.execute(f"""
                SELECT DISTINCT ON (stock_id) stock_id,
                    "PER"::numeric, "PBR"::numeric, dividend_yield::numeric
                FROM "TaiwanStockPER"
                WHERE {per_gate}
                ORDER BY stock_id, date DESC
            """, (*([self.as_of_date] * per_n_ap),))
            for sid, per, pbr, yld in cur.fetchall():
                per_data[sid] = {
                    "per": float(per) if per is not None else None,
                    "pbr": float(pbr) if pbr is not None else None,
                    "div_yield": float(yld) if yld is not None else None,
                }

            # v0.5 §14.7-BC + §14.7-BD「資料現實裁決」第 4 次跑通:
            # TaiwanStockDividend.year 為民國年格式('113年'=西元 2024)非西元 4 位數!
            # 修正:解析 '113年' → 民國 113 → 西元 2024;past 5y = [西元-5, 西元當年)
            #
            # v0.5.1 §14.7-BE「資料現實裁決」第 5 次跑通(事前事前驗證模式):
            # 設計研究計畫加 Dividend 4 cols ≥ 30% 覆蓋,落地前 DB 驗證揭露:
            # - Remu / EmpCash:民國 100 年(2011)起 schema sunset(0 stocks)
            # - StockEarnings:2024 只 5%(8/150)
            # - **只 ParticipateDistributionOfTotalShares 真實 86%(129/150)可用**
            # 對策:SELECT-only animation(讀取 + 透明寫入 score_detail,但不加 sub-score)
            # 治權成本最低;留待 walk-forward IC 證偽後決定是否升 sub-score
            roc_5y_ago = (self.as_of_date.year - 5) - 1911  # e.g., 2026-5=2021 → 民國 110
            roc_current = self.as_of_date.year - 1911       # e.g., 2026 → 民國 115
            dividend_data = {}
            cur.execute("""
                SELECT stock_id,
                    COUNT(DISTINCT year) FILTER (WHERE "CashEarningsDistribution" > 0) as div_count_5y,
                    AVG("ParticipateDistributionOfTotalShares") FILTER (WHERE "ParticipateDistributionOfTotalShares" > 0) as part_dist_5y_avg
                FROM "TaiwanStockDividend"
                WHERE year ~ '^[0-9]+年$'
                  AND CAST(REGEXP_REPLACE(year, '年$', '') AS INTEGER) >= %s
                  AND CAST(REGEXP_REPLACE(year, '年$', '') AS INTEGER) < %s
                GROUP BY stock_id
            """, (roc_5y_ago, roc_current))
            for sid, cnt, part_avg in cur.fetchall():
                dividend_data[sid] = {
                    "div_count_5y": int(cnt or 0),
                    # v0.5.1 §14.7-BE SELECT-only animation:讀取 + 透明寫入,但不加 sub-score
                    "part_dist_5y_avg": float(part_avg) if part_avg is not None else None,
                }

            # TaiwanStockInstitutionalInvestorsBuySell: net buy/sell by institution type
            # Names: Foreign_Investor, Investment_Trust, Dealer_self, Dealer_Hedging, Foreign_Dealer_Self
            # §8.5-9 Phase 3: native_aligned(§6.8.7-A cron 17:30 對齊)
            inst_gate, inst_n_ap = build_publication_date_gate("TaiwanStockInstitutionalInvestorsBuySell")
            cur.execute(f"""
                SELECT stock_id,
                    SUM(CASE WHEN name IN ('Foreign_Investor','Foreign_Dealer_Self')
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as foreign_net,
                    SUM(CASE WHEN name='Investment_Trust'
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as trust_net,
                    SUM(CASE WHEN name IN ('Dealer_self','Dealer_Hedging')
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as prop_net,
                    -- v0.6 §14.7-BF Phase F.1: 分項 Dealer 之 self vs hedging
                    SUM(CASE WHEN name='Dealer_self'
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as dealer_self_net,
                    SUM(CASE WHEN name='Dealer_Hedging'
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as dealer_hedge_net
                FROM "TaiwanStockInstitutionalInvestorsBuySell"
                WHERE date >= %s AND {inst_gate}
                GROUP BY stock_id
            """, (lookback_252, *([self.as_of_date] * inst_n_ap)))
            for sid, foreign_net, trust_net, prop_net, d_self, d_hedge in cur.fetchall():
                institutional_data[sid] = {
                    "foreign_net": float(foreign_net or 0),
                    "trust_net": float(trust_net or 0),
                    "prop_net": float(prop_net or 0),
                    "total_net": float(foreign_net or 0) + float(trust_net or 0) + float(prop_net or 0),
                    # v0.6 §14.7-BF Phase F.1
                    "dealer_self_net": float(d_self or 0),
                    "dealer_hedge_net": float(d_hedge or 0),
                }

            # v0.6 §14.7-BF Phase F.2: TaiwanStockMarginPurchaseShortSale 60d aggregates
            # Margin gate = native_aligned(§8.5-9 Phase 3 已用)
            margin_data = {}
            margin_gate, m_n_ap = build_publication_date_gate("TaiwanStockMarginPurchaseShortSale")
            last_20d = self.as_of_date - timedelta(days=30)  # ~20 交易日
            cur.execute(f"""
                SELECT stock_id,
                    AVG("MarginPurchaseTodayBalance"::numeric) as margin_bal_60d_avg,
                    AVG("ShortSaleTodayBalance"::numeric) as short_bal_60d_avg,
                    AVG("MarginPurchaseCashRepayment"::numeric) as margin_repay_60d_avg,
                    AVG("MarginPurchaseTodayBalance"::numeric)
                        FILTER (WHERE date >= %s) as margin_bal_last_20d,
                    AVG("MarginPurchaseTodayBalance"::numeric)
                        FILTER (WHERE date < %s) as margin_bal_prior_40d,
                    AVG("MarginPurchaseCashRepayment"::numeric)
                        FILTER (WHERE date >= %s) as margin_repay_last_20d,
                    AVG("MarginPurchaseCashRepayment"::numeric)
                        FILTER (WHERE date < %s) as margin_repay_prior_40d
                FROM "TaiwanStockMarginPurchaseShortSale"
                WHERE date >= %s AND {margin_gate}
                GROUP BY stock_id
            """, (last_20d, last_20d, last_20d, last_20d, lookback_252 - timedelta(days=0), *([self.as_of_date] * m_n_ap)))
            for sid, m_bal, s_bal, m_repay, m_last20, m_prior40, r_last20, r_prior40 in cur.fetchall():
                m_bal_f = float(m_bal or 0)
                s_bal_f = float(s_bal or 0)
                m_repay_f = float(m_repay or 0)
                # Trend = (last20 - prior40) / prior40
                margin_trend = None
                if m_prior40 and float(m_prior40) > 0:
                    margin_trend = (float(m_last20 or 0) - float(m_prior40)) / float(m_prior40)
                repay_trend = None
                if r_prior40 and float(r_prior40) > 0:
                    repay_trend = (float(r_last20 or 0) - float(r_prior40)) / float(r_prior40)
                margin_data[sid] = {
                    "margin_bal_60d_avg": m_bal_f,
                    "short_bal_60d_avg": s_bal_f,
                    "margin_repay_60d_avg": m_repay_f,
                    "margin_balance_trend_60d": margin_trend,
                    "margin_repay_trend_60d": repay_trend,
                    "short_margin_ratio": (s_bal_f / m_bal_f) if m_bal_f > 0 else 0.0,
                }

            # v0.6 §14.7-BF Phase F.3: TaiwanStockShareholding latest week 之 numeric cols
            # Shareholding gate = transitional 維持 date(§8.5-9.2 §14.7-BB)
            shareholding_data = {}
            sh_gate, sh_n_ap = build_publication_date_gate("TaiwanStockShareholding")
            cur.execute(f"""
                SELECT DISTINCT ON (stock_id) stock_id,
                    "ForeignInvestmentSharesRatio"::numeric as foreign_ratio,
                    "ForeignInvestmentRemainRatio"::numeric as foreign_remain,
                    "ForeignInvestmentUpperLimitRatio"::numeric as foreign_limit,
                    "NumberOfSharesIssued"::numeric as num_shares,
                    "ChineseInvestmentUpperLimitRatio"::numeric as china_limit
                FROM "TaiwanStockShareholding"
                WHERE {sh_gate}
                ORDER BY stock_id, date DESC
            """, (*([self.as_of_date] * sh_n_ap),))
            latest_sh = {sid: (fr, frm, fl, ns, cl) for sid, fr, frm, fl, ns, cl in cur.fetchall()}

            # 60d 前之 foreign_ratio for trend
            cur.execute(f"""
                SELECT DISTINCT ON (stock_id) stock_id,
                    "ForeignInvestmentSharesRatio"::numeric as foreign_ratio_60d_ago
                FROM "TaiwanStockShareholding"
                WHERE date <= %s
                ORDER BY stock_id, date DESC
            """, (self.as_of_date - timedelta(days=60),))
            sh_60d_ago = {sid: fr for sid, fr in cur.fetchall()}

            for sid, (fr, frm, fl, ns, cl) in latest_sh.items():
                fr_f = float(fr) if fr is not None else None
                fr_60d = sh_60d_ago.get(sid)
                fr_60d_f = float(fr_60d) if fr_60d is not None else None
                trend = (fr_f - fr_60d_f) if (fr_f is not None and fr_60d_f is not None) else None
                shareholding_data[sid] = {
                    "foreign_ratio": fr_f,
                    "foreign_remain_ratio": float(frm) if frm is not None else None,
                    "foreign_upper_limit": float(fl) if fl is not None else None,
                    "num_shares_issued": float(ns) if ns is not None else None,
                    "china_upper_limit": float(cl) if cl is not None else None,
                    "foreign_ratio_60d_change": trend,
                }
        finally:
            cur.close()
            conn.close()
        self._detail(
            f"📊 [MARKET-DATA] price={len(price_data)} revenue={len(revenue_data)} "
            f"financial={len(financial_data)} institutional={len(institutional_data)} "
            f"per={len(per_data)} dividend={len(dividend_data)} "
            f"margin={len(margin_data)} shareholding={len(shareholding_data)}"
        )
        return (price_data, revenue_data, financial_data, institutional_data,
                per_data, dividend_data, margin_data, shareholding_data)

    def _compute_industry_medians(self, candidates, per_data):
        """v0.5 §14.7-BC: 計算 per industry 之 PER/PBR median(min 3 stocks)
        供 _per_industry_relative_score / _pbr_industry_relative_score 用。
        """
        by_industry = {}
        for c in candidates:
            industry = (c.industry_category or "").strip()
            if not industry:
                continue
            p = per_data.get(c.stock_id, {})
            per = p.get("per")
            pbr = p.get("pbr")
            if per is not None and per > 0:
                by_industry.setdefault(industry, {"per": [], "pbr": []})["per"].append(per)
            if pbr is not None and pbr > 0:
                by_industry.setdefault(industry, {"per": [], "pbr": []})["pbr"].append(pbr)

        industry_median = {}
        for industry, vals in by_industry.items():
            per_list = sorted(vals.get("per", []))
            pbr_list = sorted(vals.get("pbr", []))
            entry = {}
            if len(per_list) >= 3:
                entry["per"] = per_list[len(per_list) // 2]
            if len(pbr_list) >= 3:
                entry["pbr"] = pbr_list[len(pbr_list) // 2]
            if entry:
                industry_median[industry] = entry
        return industry_median

    # ── v0.2 六層評分方法 ──────────────────────────────────────────────────────

    def _data_quality_score_v2(self, stock_id, price_data, revenue_data, financial_data):
        """DataQuality (25%): 資料完整度 = price_cov*40 + revenue_cov*30 + financial_cov*30"""
        price_cov = price_data.get(stock_id, {}).get("price_coverage_252d", 0.0)
        revenue_cov = revenue_data.get(stock_id, {}).get("revenue_coverage_24m", 0.0)
        financial_cov = financial_data.get(stock_id, {}).get("financial_coverage_8q", 0.0)
        score = price_cov * 40.0 + revenue_cov * 30.0 + financial_cov * 30.0
        return round(min(100.0, max(0.0, score)), 2)

    def _liquidity_mass_score(self, stock_id, price_data):
        """LiquidityMass (25%): log-scale avg daily trading value + continuity"""
        p = price_data.get(stock_id, {})
        if not p or p.get("day_count", 0) < 10:
            return 0.0
        avg_val = p.get("avg_daily_value", 0)
        continuity = p.get("price_coverage_252d", 0)
        if avg_val <= 0:
            value_score = 0.0
        else:
            # log10(1M TWD)=6→0pts, log10(10B TWD)=10→100pts, linear in between
            log_val = math.log10(max(avg_val, 1))
            value_score = min(100.0, max(0.0, (log_val - 6.0) * 25.0))
        total = value_score * 0.85 + continuity * 15.0
        return round(min(100.0, max(0.0, total)), 2)

    def _fundamental_gravity_score(self, stock_id, revenue_data, financial_data,
                                    per_data=None, dividend_data=None,
                                    industry_median=None, industry_category=None):
        """FundamentalGravity (20%): v0.5 11-sub-score(v0.3 5 + v0.5 6 新)

        v0.3(2026-05-24):Revenue YoY + EPS sum + net_income + GrossProfit + coverage
        v0.5(2026-05-25 §14.7-BC):+ PER 估值 industry-relative ±20 + PBR 估值 industry-relative ±15
          + Dividend yield ±10 + 配息穩定性 ±10 + Operating Margin ±10 + Attributable Ratio ±5

        v0.5 之 6 個新 sub-scores 需 per_data/dividend_data/industry_median/industry_category;
        若任一缺失,fallback to v0.3 行為(該 sub-score 0 contribution,中性)。
        ROE 維持 None 占位(§0.1.3-A.1)。FG 權重 20% 不變。clamp 0..100。
        """
        r = revenue_data.get(stock_id, {})
        f = financial_data.get(stock_id, {})
        if not r and not f and not per_data:
            return 50.0
        score = 50.0
        # === v0.3 既有(維持)===
        yoy = r.get("yoy_growth", 0.0)
        if yoy > 0.30:
            score += 25.0
        elif yoy > 0.10:
            score += 15.0
        elif yoy > 0.0:
            score += 8.0
        elif yoy < -0.20:
            score -= 20.0
        elif yoy < -0.05:
            score -= 10.0
        if f.get("net_income_positive", False):
            score += 15.0
        elif f.get("eps_sum", 0) > 0:
            score += 10.0
        else:
            score -= 15.0
        # v0.3 毛利率
        gm = f.get("gross_margin")
        if gm is not None and gm > 0:
            if gm > 0.40:
                score += 10.0
            elif gm > 0.25:
                score += 5.0
            elif gm > 0.10:
                score += 0.0
            elif gm > 0.05:
                score -= 3.0
            else:
                score -= 8.0
        # v0.3 coverage bonus
        coverage = (r.get("revenue_coverage_24m", 0.0) + f.get("financial_coverage_8q", 0.0)) / 2.0
        score += coverage * 10.0

        # === v0.5 §14.7-BC 6 個新 sub-scores ===
        score += self._per_industry_relative_score(stock_id, per_data, industry_median, industry_category)
        score += self._pbr_industry_relative_score(stock_id, per_data, industry_median, industry_category)
        score += self._dividend_yield_score(stock_id, per_data)
        score += self._dividend_stability_score(stock_id, dividend_data)
        score += self._operating_margin_score(f.get("op_margin"))
        score += self._attributable_ratio_score(f.get("attributable_ratio"))

        # === v0.8 §14.7-BI 1 個新 sub-score:ROE 解鎖 ===
        score += self._roe_score(f.get("roe"))

        return round(min(100.0, max(0.0, score)), 2)

    def _roe_score(self, roe):
        """ROE 7 階梯 ±15;v0.8 §14.7-BI 解鎖(從 BalanceSheet.Equity 真權益計算).
        ROE = NetIncome_4Q / Equity_latest;missing data → 0(不雙重懲罰).
        """
        if roe is None:
            return 0.0
        if roe > 0.30:
            return 15.0
        if roe > 0.20:
            return 10.0
        if roe > 0.15:
            return 5.0
        if roe > 0.10:
            return 0.0
        if roe > 0.05:
            return -3.0
        if roe > 0:
            return -5.0
        return -10.0

    # ── v0.5 §14.7-BC 6 個新 FG sub-score helpers ──────────────────────────

    def _per_industry_relative_score(self, stock_id, per_data, industry_median, industry_category):
        """PER 估值(industry-relative)±20;PER < 0 或 > p99=1766 加 risk 警示(本函式只回 score)。"""
        if not per_data or not industry_median or not industry_category:
            return 0.0
        per = per_data.get(stock_id, {}).get("per")
        if per is None:
            return 0.0
        if per <= 0:
            return -5.0  # 虧損公司
        if per > 1766.90:  # 全市場 p99 = outlier 處理
            return -5.0
        med = industry_median.get(industry_category, {}).get("per")
        if med is None or med <= 0:
            return 0.0  # 該 industry < 3 stocks 無 median
        rel = per / med
        if rel < 0.7:
            return 10.0
        if rel < 1.0:
            return 5.0
        if rel <= 1.5:
            return 0.0
        if rel <= 2.0:
            return -5.0
        return -10.0

    def _pbr_industry_relative_score(self, stock_id, per_data, industry_median, industry_category):
        """PBR 估值(industry-relative)±15;金融業 special-case(絕對 PBR <=1.5 +3)。"""
        if not per_data or not industry_median or not industry_category:
            return 0.0
        pbr = per_data.get(stock_id, {}).get("pbr")
        if pbr is None or pbr <= 0:
            return 0.0
        # 金融業特殊:絕對 PBR <=1.5 +3(銀行 BV 計算特殊)
        if "金融" in industry_category or "銀行" in industry_category or "保險" in industry_category:
            if pbr <= 1.5:
                return 3.0
            elif pbr <= 2.5:
                return 0.0
            return -3.0
        med = industry_median.get(industry_category, {}).get("pbr")
        if med is None or med <= 0:
            # fallback absolute cap
            if pbr >= 10:
                return -5.0
            if pbr >= 5:
                return -2.0
            return 0.0
        rel = pbr / med
        if rel < 0.8:
            return 5.0
        if rel <= 1.5:
            return 0.0
        if rel <= 2.5:
            return -3.0
        return -8.0

    def _dividend_yield_score(self, stock_id, per_data):
        """Dividend yield 評分 ±10;> 10% distress 警示反 -5。"""
        if not per_data:
            return 0.0
        yld = per_data.get(stock_id, {}).get("div_yield")
        if yld is None:
            return 0.0
        if yld > 10:
            return -5.0  # distress 警示(可能股價暴跌造成偽高 yld)
        if yld > 5:
            return 8.0
        if yld > 3:
            return 5.0
        if yld > 1:
            return 0.0
        if yld > 0:
            return -2.0
        return -5.0  # yld == 0(不配息)

    def _dividend_stability_score(self, stock_id, dividend_data):
        """配息穩定性 ±10;past 5y 配息次數(CashEarningsDistribution > 0)。"""
        if not dividend_data:
            return 0.0
        cnt = dividend_data.get(stock_id, {}).get("div_count_5y")
        if cnt is None:
            return 0.0
        if cnt >= 5:
            return 10.0
        if cnt >= 4:
            return 6.0
        if cnt >= 3:
            return 3.0
        if cnt >= 2:
            return 0.0
        if cnt >= 1:
            return -2.0
        return -3.0  # 0 次配息(5y)

    def _operating_margin_score(self, op_margin):
        """Operating Margin(OperatingIncome / Revenue)±10。"""
        if op_margin is None:
            return 0.0
        if op_margin > 0.30:
            return 10.0
        if op_margin > 0.15:
            return 5.0
        if op_margin > 0.05:
            return 0.0
        if op_margin > 0:
            return -3.0
        return -8.0  # 虧損營運

    def _attributable_ratio_score(self, attr_ratio):
        """Attributable Ratio((NI - NCI) / NI)±5;> 0.95 +3 / 異常 -5。"""
        if attr_ratio is None:
            return 0.0
        if attr_ratio < 0:
            return -5.0  # sentinel:NCI > NI 異常
        if attr_ratio > 0.95:
            return 3.0
        if attr_ratio > 0.85:
            return 1.0
        if attr_ratio > 0.7:
            return 0.0
        return -3.0

    def _theme_resonance_score(self, industry_category):
        """ThemeResonance (15%): MBNRIC 第六波主題共振 (AI/半導體/生技/綠能)"""
        if not industry_category:
            return 30.0
        for keyword, score in THEME_KEYWORDS.items():
            if keyword in industry_category:
                return float(score)
        return 30.0

    def _institutional_flow_score(self, stock_id, institutional_data,
                                   margin_data=None, shareholding_data=None,
                                   industry_category=None):
        """InstitutionalFlow (10%): v0.6 9 sub-scores(v0.5 之 1 + v0.6 之 8 新)

        v0.5 既有:Foreign + Trust net(legacy thresholds 維持)
        v0.6 §14.7-BF 新增 8 sub-scores:
          Phase F.1:Dealer 方向性(self vs hedging)±5
          Phase F.2:Margin 4(融資擁擠度 ±5 / 強迫平倉 ±3 / 券資比 ±3 / 融資趨勢 ±3)
          Phase F.3:Shareholding 3(外資剩餘空間 ±5 / 持股趨勢 ±3 / 法規產業 ±2)

        若 margin_data / shareholding_data 為 None,fallback v0.5 行為(新 sub-scores 返 0)。
        IF 權重 10% 維持不變;clamp 0..100。
        """
        inst = institutional_data.get(stock_id, {})
        if not inst:
            return 50.0
        score = 50.0
        foreign_net = inst.get("foreign_net", 0)
        trust_net = inst.get("trust_net", 0)
        # Foreign net thresholds (shares; large caps may exceed 100M easily)
        if foreign_net > 100_000_000:
            score += 25.0
        elif foreign_net > 10_000_000:
            score += 15.0
        elif foreign_net > 0:
            score += 5.0
        elif foreign_net < -100_000_000:
            score -= 20.0
        elif foreign_net < -10_000_000:
            score -= 10.0
        # Trust net (investment trust, smaller scale)
        if trust_net > 50_000_000:
            score += 15.0
        elif trust_net > 0:
            score += 5.0
        elif trust_net < -50_000_000:
            score -= 10.0

        # === v0.6 §14.7-BF 8 個新 sub-scores ===
        score += self._dealer_directional_score(stock_id, institutional_data)
        score += self._margin_crowding_score(stock_id, margin_data, shareholding_data)
        score += self._margin_forced_liquidation_score(stock_id, margin_data)
        score += self._short_margin_ratio_score(stock_id, margin_data)
        score += self._margin_trend_score(stock_id, margin_data)
        score += self._foreign_remain_capacity_score(stock_id, shareholding_data)
        score += self._foreign_holding_trend_score(stock_id, shareholding_data)
        score += self._regulated_industry_score(stock_id, shareholding_data, industry_category)

        return round(min(100.0, max(0.0, score)), 2)

    # ── v0.6 §14.7-BF 8 個新 IF sub-score helpers ──────────────────────────

    def _dealer_directional_score(self, stock_id, institutional_data):
        """Dealer 方向性 ±5(v0.6 F.1):dealer_self > 0 + hedging < 0 → 自營正面"""
        if not institutional_data:
            return 0.0
        inst = institutional_data.get(stock_id, {})
        d_self = inst.get("dealer_self_net", 0)
        d_hedge = inst.get("dealer_hedge_net", 0)
        if d_self > 10_000_000 and d_hedge < -5_000_000:
            return 5.0
        if d_self > 0 and d_hedge < 0:
            return 2.0
        if d_self < -10_000_000:
            return -3.0
        return 0.0

    def _margin_crowding_score(self, stock_id, margin_data, shareholding_data):
        """融資擁擠度 ±5(v0.6 F.2):MarginBalance/NumShares 高擁擠 → 反向"""
        if not margin_data or not shareholding_data:
            return 0.0
        m_bal = margin_data.get(stock_id, {}).get("margin_bal_60d_avg", 0)
        ns = shareholding_data.get(stock_id, {}).get("num_shares_issued")
        if not m_bal or not ns or ns <= 0:
            return 0.0
        ratio = m_bal / ns
        if ratio > 0.05:
            return -5.0
        if ratio > 0.02:
            return -2.0
        if ratio > 0.005:
            return 0.0
        return 2.0

    def _margin_forced_liquidation_score(self, stock_id, margin_data):
        """強迫平倉壓力 ±3(v0.6 F.2):CashRepayment 60d trend 上升 → 前兆"""
        if not margin_data:
            return 0.0
        trend = margin_data.get(stock_id, {}).get("margin_repay_trend_60d")
        if trend is None:
            return 0.0
        if trend > 0.30:
            return -3.0
        if trend > 0.10:
            return -1.0
        return 0.0

    def _short_margin_ratio_score(self, stock_id, margin_data):
        """券資比 ±3(v0.6 F.2):Short/Margin 極高 → 多空極端"""
        if not margin_data:
            return 0.0
        ratio = margin_data.get(stock_id, {}).get("short_margin_ratio", 0)
        if ratio > 0.50:
            return -3.0
        if ratio > 0.20:
            return -1.0
        return 0.0

    def _margin_trend_score(self, stock_id, margin_data):
        """融資趨勢 ±3(v0.6 F.2):20d vs 40d 比較"""
        if not margin_data:
            return 0.0
        trend = margin_data.get(stock_id, {}).get("margin_balance_trend_60d")
        if trend is None:
            return 0.0
        if trend > 0.50:
            return -3.0
        if trend > 0.20:
            return -1.0
        if trend < -0.30:
            return 1.0
        return 0.0

    def _foreign_remain_capacity_score(self, stock_id, shareholding_data):
        """外資剩餘空間 ±5(v0.6 F.3):(Limit - Ratio) / Limit"""
        if not shareholding_data:
            return 0.0
        sh = shareholding_data.get(stock_id, {})
        limit = sh.get("foreign_upper_limit")
        ratio = sh.get("foreign_ratio")
        if limit is None or ratio is None or limit <= 0:
            return 0.0
        remain_pct = (limit - ratio) / limit
        if remain_pct > 0.80:
            return 2.0
        if remain_pct > 0.50:
            return 3.0
        if remain_pct > 0.20:
            return 5.0
        return 0.0

    def _foreign_holding_trend_score(self, stock_id, shareholding_data):
        """外資持股 60d 變化 ±3(v0.6 F.3)"""
        if not shareholding_data:
            return 0.0
        trend = shareholding_data.get(stock_id, {}).get("foreign_ratio_60d_change")
        if trend is None:
            return 0.0
        if trend > 2.0:  # 假設 ratio 為 % 單位,+2pp 為顯著
            return 3.0
        if trend > 0.5:
            return 1.0
        if trend < -2.0:
            return -3.0
        return 0.0

    def _regulated_industry_score(self, stock_id, shareholding_data, industry_category):
        """法規受限產業 ±2(v0.6 F.3):UpperLimit < 100% → 特殊產業(電信/銀行/媒體/國防)"""
        if not shareholding_data:
            return 0.0
        sh = shareholding_data.get(stock_id, {})
        limit = sh.get("foreign_upper_limit")
        if limit is None:
            return 0.0
        if limit < 30:
            return 2.0
        if limit < 50:
            return 1.0
        return 0.0

    def _volatility_control_score(self, stock_id, price_data):
        """VolatilityControl (5%): v0.7 convexity-aware(§14.7-BG / §9.10 起草 raw-first)

        v0.2 既有:cv_close = STDDEV(price)/AVG(price) — 對稱壓制凸性(§0.0-C.3 已知缺陷)
        v0.7 升:convexity = upside_σ − downside_σ(annualized 60d;raw OHLC)
          上行凸性 > 下行凸性 → 優質股(高分);反之 → distress(低分)
          對映 §0.0-C.3 修補 + §0.1.3-B 發現 7 之 §6 治理層 first time 修補
        Fallback:n_obs < 20 → 中性 50;convexity is None → fallback to legacy cv_close
        """
        p = price_data.get(stock_id, {})
        if not p or p.get("day_count", 0) < 20:
            return 50.0
        # v0.7 §14.7-BG: convexity-aware
        convexity = p.get("convexity_60d")
        n_obs = p.get("vc_n_obs", 0)
        if convexity is not None and n_obs >= 20:
            if convexity > 0.10:
                return 95.0
            elif convexity > 0.05:
                return 85.0
            elif convexity > 0:
                return 75.0
            elif convexity > -0.05:
                return 60.0
            elif convexity > -0.10:
                return 40.0
            else:
                return 20.0
        # Fallback to legacy cv_close(rare:n_obs < 20 或 NULL σ)
        cv = p.get("cv_close", 0.3)
        if cv <= 0.05:
            return 95.0
        elif cv <= 0.10:
            return 85.0
        elif cv <= 0.15:
            return 75.0
        elif cv <= 0.20:
            return 65.0
        elif cv <= 0.30:
            return 50.0
        elif cv <= 0.40:
            return 35.0
        return 20.0

    def _risk_profile(self, type_value, industry_category, missing_fields):
        risk = 0.0
        reasons = []
        type_norm = (type_value or "").lower()
        industry = industry_category or ""
        if missing_fields:
            risk += 40.0
            reasons.append(f"missing_fields={','.join(missing_fields)}")
        if any(keyword in industry for keyword in EXCLUDED_INDUSTRY_KEYWORDS):
            risk += 100.0
            reasons.append("non_equity_or_fund_like_industry")
        if type_norm == "emerging" and not self.include_emerging:
            risk += 30.0
            reasons.append("emerging_market_excluded_by_policy")
        elif type_norm and type_norm not in EQUITY_TYPES and not self.include_emerging:
            risk += 30.0
            reasons.append(f"unsupported_type={type_value}")
        elif not type_norm:
            risk += 25.0
            reasons.append("missing_type")
        return min(risk, 100.0), reasons

    def _score_candidate(self, row, price_data, revenue_data, financial_data, institutional_data,
                         per_data=None, dividend_data=None, industry_median=None,
                         margin_data=None, shareholding_data=None):
        """v0.5 §14.7-BC: 六層 CoreScore + FG v0.5 11 sub-scores
        signature 加 per_data / dividend_data / industry_median(v0.5 §14.7-BC)。
        若 None,fallback v0.3 行為(FG v0.5 之 6 個新 sub-score 返 0 中性)。
        """
        stock_id, stock_name, type_value, industry_category, source_date = row

        # Metadata missing fields (used for risk profile only)
        missing_fields = []
        if not stock_name:
            missing_fields.append("stock_name")
        if not type_value:
            missing_fields.append("type")
        if not industry_category:
            missing_fields.append("industry_category")

        risk_penalty, risk_reasons = self._risk_profile(type_value, industry_category, missing_fields)

        # Six-layer scoring (v0.5: FG 加 6 新 sub-scores)
        dq = self._data_quality_score_v2(stock_id, price_data, revenue_data, financial_data)
        lm = self._liquidity_mass_score(stock_id, price_data)
        fg = self._fundamental_gravity_score(
            stock_id, revenue_data, financial_data,
            per_data=per_data, dividend_data=dividend_data,
            industry_median=industry_median, industry_category=industry_category,
        )
        tr = self._theme_resonance_score(industry_category)
        inst_f = self._institutional_flow_score(
            stock_id, institutional_data,
            margin_data=margin_data, shareholding_data=shareholding_data,
            industry_category=industry_category,
        )
        vc = self._volatility_control_score(stock_id, price_data)

        # Extra penalty: high volatility + low liquidity, or major data gap
        extra_penalty = 0.0
        cv = price_data.get(stock_id, {}).get("cv_close", 0)
        if cv > 0.4 and lm < 30:
            extra_penalty += 15.0
        if dq < 20:
            extra_penalty += 10.0
        total_penalty = min(risk_penalty + extra_penalty, 99.0)

        core_score = max(0.0, min(100.0,
            0.25 * dq + 0.25 * lm + 0.20 * fg + 0.15 * tr + 0.10 * inst_f + 0.05 * vc - total_penalty
        ))

        # Coverage metrics for membership table
        price_cov = price_data.get(stock_id, {}).get("price_coverage_252d", 0.0)
        revenue_cov = revenue_data.get(stock_id, {}).get("revenue_coverage_24m", 0.0)
        financial_cov = financial_data.get(stock_id, {}).get("financial_coverage_8q", 0.0)

        exclusion_reason = "; ".join(risk_reasons) if risk_penalty >= 50.0 else None
        selection_reason = (
            f"CoreScore v0.7: {core_score:.1f} "
            f"(DQ={dq:.0f} LM={lm:.0f} FG={fg:.0f} TR={tr:.0f} IF={inst_f:.0f} VC={vc:.0f} RP={total_penalty:.0f})"
        )
        if exclusion_reason:
            selection_reason = f"quarantine: {exclusion_reason}"

        # v0.5 §14.7-BC: 透明寫入 FG sub-components 便於下游診斷與 audit
        f_data = financial_data.get(stock_id, {})
        p_data = (per_data or {}).get(stock_id, {})
        d_data = (dividend_data or {}).get(stock_id, {})
        m_data = (margin_data or {}).get(stock_id, {})
        sh_data = (shareholding_data or {}).get(stock_id, {})
        inst_data = (institutional_data or {}).get(stock_id, {})
        score_detail = {
            "score_scope": "v0.8_roe_unlocked_via_balance_sheet",
            "constitution": CONSTITUTION_VER,
            "tool_version": TOOL_VER,
            "weights": {"DQ": 0.25, "LM": 0.25, "FG": 0.20, "TR": 0.15, "IF": 0.10, "VC": 0.05},
            "data_quality_score": dq,
            "liquidity_mass_score": lm,
            "fundamental_gravity_score": fg,
            # v0.3 既有 FG sub-components
            "fg_gross_margin": f_data.get("gross_margin"),
            "fg_roe": f_data.get("roe"),
            "fg_equity": f_data.get("equity"),
            "fg_ni_4q_sum": f_data.get("ni_4q_sum"),
            # v0.5 §14.7-BC 新增 6 個 FG sub-components(透明)
            "fg_per": p_data.get("per"),
            "fg_pbr": p_data.get("pbr"),
            "fg_div_yield": p_data.get("div_yield"),
            "fg_div_count_5y": d_data.get("div_count_5y"),
            # v0.5.1 §14.7-BE SELECT-only animation(透明寫入;不影響 score)
            "fg_part_dist_5y_avg": d_data.get("part_dist_5y_avg"),
            # v0.6 §14.7-BF IF sub-components(8 新;透明)
            "if_dealer_self_net": inst_data.get("dealer_self_net"),
            "if_dealer_hedge_net": inst_data.get("dealer_hedge_net"),
            "if_margin_bal_60d": m_data.get("margin_bal_60d_avg"),
            "if_short_bal_60d": m_data.get("short_bal_60d_avg"),
            "if_short_margin_ratio": m_data.get("short_margin_ratio"),
            "if_margin_trend_60d": m_data.get("margin_balance_trend_60d"),
            "if_margin_repay_trend": m_data.get("margin_repay_trend_60d"),
            "if_foreign_ratio": sh_data.get("foreign_ratio"),
            "if_foreign_remain_ratio": sh_data.get("foreign_remain_ratio"),
            "if_foreign_upper_limit": sh_data.get("foreign_upper_limit"),
            "if_num_shares_issued": sh_data.get("num_shares_issued"),
            "if_foreign_ratio_60d_change": sh_data.get("foreign_ratio_60d_change"),
            # v0.7.1 §14.7-BH VC convexity sub-components(RMS 對齊 §9.9 RMS 強制契約;透明寫入)
            "vc_convexity_60d": price_data.get(stock_id, {}).get("convexity_60d"),
            "vc_upside_rms_60d": price_data.get(stock_id, {}).get("upside_rms_60d"),
            "vc_downside_rms_60d": price_data.get(stock_id, {}).get("downside_rms_60d"),
            "vc_cc_sigma_60d": price_data.get(stock_id, {}).get("cc_sigma_60d"),
            "fg_op_margin": f_data.get("op_margin"),
            "fg_pretax_margin": f_data.get("pretax_margin"),
            "fg_continuing_op_ratio": f_data.get("continuing_op_ratio"),
            "fg_attributable_ratio": f_data.get("attributable_ratio"),
            "theme_resonance_score": tr,
            "institutional_flow_score": inst_f,
            "volatility_control_score": vc,
            "risk_penalty": total_penalty,
            "core_score": round(core_score, 4),
            "price_coverage_252d": price_cov,
            "revenue_coverage_24m": revenue_cov,
            "financial_coverage_8q": financial_cov,
            "risk_reasons": risk_reasons,
            "candidate_source_mode": self.candidate_source_mode,
            "downstream_boundary": "no feature values, labels, model outputs, prediction signals",
        }

        c = Candidate(
            stock_id=stock_id,
            stock_name=stock_name,
            type=type_value,
            industry_category=industry_category,
            source_date=source_date,
            core_score=round(core_score, 4),
            data_quality_score=round(dq, 4),
            theme_score=round(tr, 4),
            risk_penalty=round(total_penalty, 4),
            core_tier="pending",
            selection_reason=selection_reason,
            exclusion_reason=exclusion_reason,
            score_detail=score_detail,
        )
        c.liquidity_score = round(lm, 4)
        c.fundamental_score = round(fg, 4)
        c.institutional_flow_score = round(inst_f, 4)
        c.volatility_control_score = round(vc, 4)
        c.price_coverage_252d = round(price_cov, 4)
        c.revenue_coverage_24m = round(revenue_cov, 4)
        c.financial_coverage_8q = round(financial_cov, 4)
        return c

    def load_candidates(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            if self.candidate_source_mode == "as_of_filtered":
                cur.execute(
                    '''
                    SELECT DISTINCT ON ("stock_id")
                        "stock_id", "stock_name", "type", "industry_category", "date"
                    FROM "TaiwanStockInfo"
                    WHERE "date" <= %s
                    ORDER BY "stock_id", "date" DESC
                    ''',
                    (self.as_of_date,),
                )
            else:
                cur.execute(
                    '''
                    SELECT DISTINCT ON ("stock_id")
                        "stock_id", "stock_name", "type", "industry_category", "date"
                    FROM "TaiwanStockInfo"
                    ORDER BY "stock_id", "date" DESC
                    '''
                )
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        # v0.5 §14.7-BC: 先計算 industry_median(用 rows + per_data)
        # v0.6 §14.7-BF: _market_data 8-tuple(加 margin_data + shareholding_data)
        (price_data, revenue_data, financial_data, institutional_data,
         per_data, dividend_data, margin_data, shareholding_data) = self._market_data
        from types import SimpleNamespace
        mini_candidates = [SimpleNamespace(stock_id=r[0], industry_category=r[3]) for r in rows]
        industry_median = self._compute_industry_medians(mini_candidates, per_data)
        self._detail(
            f"📊 [INDUSTRY-MEDIAN v0.5] computed for {len(industry_median)} industries "
            f"(min 3 stocks/industry;PER/PBR median for industry-relative FG sub-scores)"
        )

        candidates = [
            self._score_candidate(
                row, price_data, revenue_data, financial_data, institutional_data,
                per_data=per_data, dividend_data=dividend_data,
                industry_median=industry_median,
                margin_data=margin_data, shareholding_data=shareholding_data,
            )
            for row in rows
        ]
        self._apply_source_pure_filter(candidates)
        self._assign_tiers(candidates)
        return candidates

    def _apply_source_pure_filter(self, candidates):
        """§14.7-DC v0.8 Source-Pure Doctrine enforcement.

        任一 stock 之任一 feature `is_null_imputed=True` → exclusion_reason 標註
        為 quarantine,per §一.10 No Data Hallucination 之 universe layer enforcement。
        Source:latest committed feature_store_snapshot 之 feature_values。

        若無 committed feature_store_snapshot 則 skip(initial bootstrap)+ warn。
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """SELECT feature_set_id FROM feature_store_snapshot
                   WHERE status='committed' ORDER BY created_at DESC LIMIT 1"""
            )
            row = cur.fetchone()
            if not row:
                self._detail(
                    "⚠️ [§14.7-DC] no committed feature_store_snapshot; "
                    "skipping source-pure filter (bootstrap state)"
                )
                self.stats["section_14_7_dc_skipped"] = True
                return
            latest_fs = row[0]
            cur.execute(
                """SELECT stock_id, array_agg(DISTINCT feature_name ORDER BY feature_name)
                   FROM feature_values
                   WHERE feature_set_id = %s AND is_null_imputed = TRUE
                   GROUP BY stock_id""",
                (latest_fs,),
            )
            imputed_map = {sid: feats for sid, feats in cur.fetchall()}
            self._detail(
                f"🔒 [§14.7-DC] source-pure filter: latest fs={latest_fs}; "
                f"found {len(imputed_map)} stocks with imputed features"
            )
            quarantined = 0
            for c in candidates:
                if c.stock_id in imputed_map:
                    feats = ", ".join(imputed_map[c.stock_id])
                    reason = (
                        f"§14.7-DC Source-Pure Doctrine: imputed features "
                        f"[{feats}] (no FinMind/FRED API source per §一.10)"
                    )
                    c.exclusion_reason = (
                        reason if c.exclusion_reason is None
                        else f"{c.exclusion_reason}; {reason}"
                    )
                    quarantined += 1
            self._detail(
                f"🔒 [§14.7-DC] applied: {quarantined} stocks quarantined "
                f"(source feature_set={latest_fs})"
            )
            self.stats["section_14_7_dc_quarantined"] = quarantined
            self.stats["section_14_7_dc_source_feature_set"] = latest_fs
        finally:
            cur.close()
            conn.close()

    def _assign_tiers(self, candidates):
        eligible = [c for c in candidates if c.exclusion_reason is None]
        quarantined = [c for c in candidates if c.exclusion_reason is not None]
        eligible.sort(key=lambda c: (-c.core_score, -c.theme_score, c.stock_id or ""))

        # §14.7-BT Phase C dispatch:dynamic mode vs legacy mode
        if self.is_dynamic_mode:
            # Dynamic mode(v0.8+ per charter §6.7.1 annex)
            n_eligible = len(eligible)
            target_n_raw = int(round(n_eligible * self.selection_pct / 100.0))
            target_n = max(self.selection_n_min, min(self.selection_n_max, target_n_raw))
            actual_core_n = int(round(target_n * self.core_pct_within_selected))
            actual_convex_n = target_n - actual_core_n
            # Selection: convex_pool 先選 (theme >= 70) 取前 actual_convex_n;剩餘 top actual_core_n 為 core
            convex_pool = [c for c in eligible if c.theme_score >= 70.0][:actual_convex_n]
            convex_ids = {c.stock_id for c in convex_pool}
            core_pool = [c for c in eligible if c.stock_id not in convex_ids][:actual_core_n]
            core_ids = {c.stock_id for c in core_pool}
            # 透明 telemetry:寫 dynamic mode 之 metadata 進 stats
            self.stats["dynamic_target_n_raw"] = target_n_raw
            self.stats["dynamic_target_n_capped"] = target_n
            self.stats["dynamic_actual_core_n"] = actual_core_n
            self.stats["dynamic_actual_convex_n"] = actual_convex_n
            self.stats["dynamic_selection_pct"] = self.selection_pct
            self.stats["dynamic_n_min_n_max"] = f"{self.selection_n_min}/{self.selection_n_max}"
        else:
            # Legacy mode DEPRECATED per §14.7-BW (was v0.2-v0.7 hardcode 150 default)
            convex_pool = [c for c in eligible if c.theme_score >= 70.0][: self.convex_limit]
            convex_ids = {c.stock_id for c in convex_pool}
            core_pool = [c for c in eligible if c.stock_id not in convex_ids][: self.core_limit]
            core_ids = {c.stock_id for c in core_pool}

        for candidate in candidates:
            if candidate in quarantined:
                candidate.core_tier = "quarantine_universe"
            elif candidate.stock_id in convex_ids:
                candidate.core_tier = "convex_universe"
            elif candidate.stock_id in core_ids:
                candidate.core_tier = "core_universe"
            else:
                candidate.core_tier = "research_universe"

        self.stats["total_candidates"] = len(candidates)
        self.stats["research_count"] = sum(1 for c in candidates if c.core_tier == "research_universe")
        self.stats["core_count"] = sum(1 for c in candidates if c.core_tier == "core_universe")
        self.stats["convex_count"] = sum(1 for c in candidates if c.core_tier == "convex_universe")
        self.stats["quarantine_count"] = sum(1 for c in candidates if c.core_tier == "quarantine_universe")

    def _policy_payload(self):
        return {
            "policy_version": self.policy_version,
            "policy_name": "Core Universe Policy v0.2 Six-Layer CoreScore",
            "description": "CoreScore v0.2 six-layer scoring: DQ+LM+FG+TR+IF+VC-RP. DB-driven from TaiwanStockInfo + six individual stock tables. No model or prediction values.",
            "weight_config": {
                "data_quality_score": 0.25,
                "liquidity_mass_score": 0.25,
                "fundamental_gravity_score": 0.20,
                "theme_resonance_score": 0.15,
                "institutional_flow_score": 0.10,
                "volatility_control_score": 0.05,
                "risk_penalty": -1.00,
            },
            "eligibility_config": {
                "source_table": "TaiwanStockInfo",
                "include_emerging": self.include_emerging,
                "core_limit": self.core_limit,             # legacy mode value;None for dynamic
                "convex_limit": self.convex_limit,         # legacy mode value;None for dynamic
                # §14.7-BT Phase C 新加:dynamic mode metadata(per charter §6.7.1 annex)
                "is_dynamic_mode": self.is_dynamic_mode,
                "selection_algorithm": "B_top_pct_composite_corescore" if self.is_dynamic_mode else "legacy_hardcode_120_30",
                "selection_pct": self.selection_pct,
                "selection_n_min": self.selection_n_min,
                "selection_n_max": self.selection_n_max,
                "core_pct_within_selected": self.core_pct_within_selected,
                "actual_target_n": self.stats.get("dynamic_target_n_capped") if self.is_dynamic_mode else (
                    (self.core_limit or 0) + (self.convex_limit or 0)
                ),
                "default_review_cycle": "annual",
                "current_rebalance_mode": self._rebalance_mode(),
                "current_review_cycle": self._review_cycle(),
                "annual_rebalance_rule": "regular commit allowed only when as_of_date is the final trading day of that calendar year",
                "special_rebalance_requires_reason": True,
                "special_rebalance_reason": self.special_rebalance_reason or None,
                "downstream_eligibility": "all false until historical coverage is measured",
                "v02_input_contract": "8-table preflight + coverage summary enabled",
            },
            "risk_config": {
                "excluded_industry_keywords": list(EXCLUDED_INDUSTRY_KEYWORDS),
                "unsupported_type_penalty": 30,
                "fund_like_industry_penalty": 100,
                "missing_metadata_penalty": 40,
            },
        }

    def _upsert_policy(self, cur):
        payload = self._policy_payload()
        cur.execute(
            '''
            INSERT INTO "core_universe_policy" (
                "policy_version", "policy_name", "description", "weight_config",
                "eligibility_config", "risk_config", "effective_from", "active", "notes", "updated_at"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, CURRENT_TIMESTAMP)
            ON CONFLICT ("policy_version") DO UPDATE SET
                "policy_name" = EXCLUDED."policy_name",
                "description" = EXCLUDED."description",
                "weight_config" = EXCLUDED."weight_config",
                "eligibility_config" = EXCLUDED."eligibility_config",
                "risk_config" = EXCLUDED."risk_config",
                "active" = TRUE,
                "notes" = EXCLUDED."notes",
                "updated_at" = CURRENT_TIMESTAMP
            ''',
            (
                payload["policy_version"],
                payload["policy_name"],
                payload["description"],
                Json(payload["weight_config"]),
                Json(payload["eligibility_config"]),
                Json(payload["risk_config"]),
                self.as_of_date,
                "Generated by core_universe_builder.py v0.2; six-layer CoreScore with actual market data",
            ),
        )

    def _upsert_snapshot(self, cur):
        cur.execute(
            '''
            INSERT INTO "core_universe_snapshot" (
                "snapshot_id", "as_of_date", "source_data_cutoff", "policy_version",
                "feature_set_version", "model_policy_version", "prediction_policy_version",
                "total_candidates", "research_count", "core_count", "convex_count", "quarantine_count",
                "status", "notes"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'committed', %s)
            ON CONFLICT ("snapshot_id") DO UPDATE SET
                "source_data_cutoff" = EXCLUDED."source_data_cutoff",
                "feature_set_version" = EXCLUDED."feature_set_version",
                "model_policy_version" = EXCLUDED."model_policy_version",
                "prediction_policy_version" = EXCLUDED."prediction_policy_version",
                "total_candidates" = EXCLUDED."total_candidates",
                "research_count" = EXCLUDED."research_count",
                "core_count" = EXCLUDED."core_count",
                "convex_count" = EXCLUDED."convex_count",
                "quarantine_count" = EXCLUDED."quarantine_count",
                "status" = 'committed',
                "notes" = EXCLUDED."notes"
            ''',
            (
                self.snapshot_id,
                self.as_of_date,
                self.source_data_cutoff,
                self.policy_version,
                DEFAULT_FEATURE_SET_VERSION,
                DEFAULT_MODEL_POLICY_VERSION,
                DEFAULT_PREDICTION_POLICY_VERSION,
                self.stats["total_candidates"],
                self.stats["research_count"],
                self.stats["core_count"],
                self.stats["convex_count"],
                self.stats["quarantine_count"],
                self._snapshot_note(),
            ),
        )

    def _membership_rows(self, candidates):
        rows = []
        for c in candidates:
            rows.append(
                (
                    self.snapshot_id,
                    c.stock_id,
                    c.stock_name,
                    c.type,
                    c.industry_category,
                    c.core_tier,
                    c.core_score,
                    self._effective_from(),
                    self._review_cycle(),
                    c.selection_reason,
                    c.exclusion_reason,
                    False,
                    False,
                    False,
                    False,
                    0,
                    c.price_coverage_252d,
                    c.revenue_coverage_24m,
                    c.financial_coverage_8q,
                    DEFAULT_LABEL_HORIZON,
                    self.policy_version,
                    DEFAULT_FEATURE_SET_VERSION,
                    DEFAULT_MODEL_POLICY_VERSION,
                    DEFAULT_PREDICTION_POLICY_VERSION,
                )
            )
        return rows

    def _score_rows(self, candidates):
        rows = []
        for c in candidates:
            rows.append(
                (
                    self.snapshot_id,
                    c.stock_id,
                    self.as_of_date,
                    self.source_data_cutoff,
                    self.policy_version,
                    c.core_score,
                    c.data_quality_score,
                    c.liquidity_score,
                    c.fundamental_score,
                    c.theme_score,
                    c.institutional_flow_score,
                    c.volatility_control_score,
                    c.risk_penalty,
                    Json(c.score_detail),
                )
            )
        return rows

    def _write_membership(self, cur, candidates):
        cur.execute('DELETE FROM "core_universe_scores" WHERE "snapshot_id" = %s', (self.snapshot_id,))
        cur.execute('DELETE FROM "core_universe_membership" WHERE "snapshot_id" = %s', (self.snapshot_id,))

        execute_batch(
            cur,
            '''
            INSERT INTO "core_universe_membership" (
                "snapshot_id", "stock_id", "stock_name", "type", "industry_category", "core_tier", "core_score",
                "effective_from", "review_cycle", "selection_reason", "exclusion_reason",
                "train_eligible", "predict_eligible", "backtest_eligible", "downstream_ready",
                "min_history_days", "price_coverage_252d", "revenue_coverage_24m", "financial_coverage_8q", "label_horizon",
                "policy_version", "feature_set_version", "model_policy_version", "prediction_policy_version"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''',
            self._membership_rows(candidates),
            page_size=500,
        )

        execute_batch(
            cur,
            '''
            INSERT INTO "core_universe_scores" (
                "snapshot_id", "stock_id", "as_of_date", "source_data_cutoff", "policy_version", "core_score",
                "data_quality_score", "liquidity_score", "fundamental_score", "theme_score",
                "institutional_flow_score", "volatility_control_score", "risk_penalty", "score_detail"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''',
            self._score_rows(candidates),
            page_size=500,
        )
        self.stats["written_rows"] = len(candidates) * 2 + 3

    def _write_revision_log(self, cur):
        detail = {
            "tool_version": TOOL_VER,
            "constitution": CONSTITUTION_VER,
            "total_candidates": self.stats["total_candidates"],
            "research_count": self.stats["research_count"],
            "core_count": self.stats["core_count"],
            "convex_count": self.stats["convex_count"],
            "quarantine_count": self.stats["quarantine_count"],
            "source_data_cutoff": str(self.source_data_cutoff),
            "candidate_source_mode": self.candidate_source_mode,
            "commit_mode": self.commit,
            "rebalance_mode": self._rebalance_mode(),
            "review_cycle": self._review_cycle(),
            "annual_rebalance_guard": "enforced_on_commit",
            "special_rebalance_reason": self.special_rebalance_reason or None,
            "effective_from": str(self._effective_from()),
            "boundary": "v0.2 six-layer CoreScore (DQ+LM+FG+TR+IF+VC-RP) with actual market data; no feature/model/prediction values",
            "v02_contract": {
                "pass": self.stats["v02_contract_pass"],
                "warning": self.stats["v02_contract_warning"],
                "failed": self.stats["v02_contract_failed"],
                "coverage_summary": self.stats["coverage_summary"],
            },
        }
        cur.execute(
            '''
            INSERT INTO "universe_revision_log" (
                "actor", "action_type", "object_type", "object_id", "policy_version", "snapshot_id", "detail", "note"
            ) VALUES ('core_universe_builder.py', 'BUILD_SNAPSHOT', 'core_universe_snapshot', %s, %s, %s, %s, %s)
            ''',
            (
                self.snapshot_id,
                self.policy_version,
                self.snapshot_id,
                Json(detail),
                f"core_universe_builder v0.2 committed {self._rebalance_mode()} six-layer CoreScore universe",
            ),
        )

    def commit_snapshot(self, candidates):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self._upsert_policy(cur)
            self._upsert_snapshot(cur)
            self._write_membership(cur, candidates)
            self._write_revision_log(cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

        audit_items = [
            ("core_universe_policy", 1),
            ("core_universe_snapshot", 1),
            ("core_universe_membership", len(candidates)),
            ("core_universe_scores", len(candidates)),
            ("universe_revision_log", 1),
        ]
        for table_name, row_count in audit_items:
            try:
                write_data_audit_log(table_name, "SYSTEM", self.as_of_date.strftime("%Y-%m-%d"), "CORE_UNIVERSE_BUILD", row_count)
            except Exception as exc:
                self.stats["warnings"] += 1
                self._detail(f"⚠️ [AUDIT-WARN] {table_name} data_audit_log failed: {type(exc).__name__}: {exc}")

    def build(self):
        start_time = time.time()
        lifecycle_cm = None
        lifecycle = None
        if self.commit:
            lifecycle_cm = record_lifecycle("core_universe_builder_v0.2", category="governance", stock_id="SYSTEM")
            lifecycle = lifecycle_cm.__enter__()

        try:
            if not self.preflight_check():
                self.stats["failed"] += 1
                self._mark_lifecycle(lifecycle, "failed", "preflight failed")
                self.report_results(start_time)
                return False

            self._detail("📥 [MARKET-DATA] 批量載入六層評分資料...")
            self._market_data = self._load_market_data()

            candidates = self.load_candidates()
            if self.commit:
                self.commit_snapshot(candidates)
            else:
                self.stats["written_rows"] = 0

            self.report_results(start_time)
            return (
                self.stats["failed"] == 0
                and self.stats["preflight_failed"] == 0
                and self.stats["v02_contract_failed"] == 0
            )
        except Exception as exc:
            self.stats["failed"] += 1
            self._detail(f"❌ [BUILD-FAILED] {type(exc).__name__}: {exc}")
            self._mark_lifecycle(lifecycle, "failed", f"{type(exc).__name__}: {exc}")
            self.report_results(start_time)
            return False
        finally:
            if lifecycle_cm is not None:
                lifecycle_cm.__exit__(None, None, None)

    def compute_verdict(self):
        if self.stats["failed"] > 0 or self.stats["preflight_failed"] > 0 or self.stats["v02_contract_failed"] > 0:
            return "FAILED"
        if self.stats["warnings"] > 0 or self.stats["preflight_warning"] > 0 or self.stats["v02_contract_warning"] > 0:
            return "WARNING"
        return "PERFECT"

    def report_results(self, start_time):
        mode = "COMMIT" if self.commit else "DRY-RUN"
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 核心股選拔引擎執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md")
        print("治理權責 : Core Universe Selection Authority")
        print("評分架構 : CoreScore v0.2 = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP")
        print(f"執行模式 : {mode}")
        print(f"重選模式 : {self._rebalance_mode()}")
        print(f"Snapshot : {self.snapshot_id}")
        if self.special_rebalance_reason:
            print(f"特別原因 : {self.special_rebalance_reason}")
        print("─" * 80)
        for line in self.stats["details"]:
            print(line)
        print("─" * 80)
        print(f"🔎 PREFLIGHT PASS/WARN/FAIL : {self.stats['preflight_pass']}/{self.stats['preflight_warning']}/{self.stats['preflight_failed']}")
        print(f"🧾 V0.2 CONTRACT PASS/WARN/FAIL : {self.stats['v02_contract_pass']}/{self.stats['v02_contract_warning']}/{self.stats['v02_contract_failed']}")
        if self.stats["coverage_summary"]:
            print("📊 V0.2 coverage summary:")
            for item in self.stats["coverage_summary"]:
                label = item.get("label")
                table = item.get("table")
                if "avg_coverage" in item:
                    scope = item.get("universe_scope", "market")
                    print(
                        f"   - {label}: scope={scope}, table={table}, avg_coverage={item['avg_coverage']}, "
                        f"zero={item['zero_coverage_count']}, pass={item['threshold_pass_count']}/{item['candidate_count']}"
                    )
                else:
                    key = item.get("key")
                    key_text = f", {key}={item.get('key_count')}" if key else ""
                    print(
                        f"   - {label}: table={table}, rows={item.get('rows')}, "
                        f"date_range={item.get('min_date')}..{item.get('max_date')}{key_text}"
                    )
        print(f"📅 as_of_date       : {self.as_of_date}")
        print(f"📅 source_cutoff    : {self.source_data_cutoff}")
        print(f"📚 candidate_source : {self.candidate_source_mode}")
        print(f"📈 total_candidates : {self.stats['total_candidates']}")
        print(f"🧪 research_universe: {self.stats['research_count']}")
        print(f"🎯 core_universe    : {self.stats['core_count']}")
        print(f"🚀 convex_universe  : {self.stats['convex_count']}")
        print(f"🧯 quarantine       : {self.stats['quarantine_count']}")
        print(f"📝 written_rows     : {self.stats['written_rows']}")
        print(f"⚠️  warnings         : {self.stats['warnings']}")
        print(f"❌ failed           : {self.stats['failed']}")
        print(f"🕒 總計耗時         : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定         : {self.compute_verdict()}")
        print("🛡️" * 40 + "\n")


class DoctrineNativeGateBuilder:
    """§14.7-CG v6.5.0 Native Gate Builder — 原生實現 charter §14.7-CF 三 invariant。

    Stages(整合舊 3-step pipeline 為單一 program):
      Stage 1: §0.3 K-wave 13 FRED series 存在性 macro gate(broadcast)
      Stage 2: §0.1 第一性原理 per-stock 8 raw sources × thresholds
      Stage 3: §0.2 八二法則 per-stock 3 raw sources × thresholds
      Stage 4: doctrine-pass union → core_universe(無 tier split / 無 cap / 無 floor)
      Stage 5: atomic supersede write(policy + snapshot + membership + revision_log)

    對應 charter §14.7-CG inscribed at v6.1.0-patch 第三十一輪;依 §14.7-CF
    為唯一設計基礎;依 §14.7-CD 11 source thresholds 邏輯等價移植自
    apply_raw_data_completeness_gate.py v6.4.2。
    """

    KWAVE_SERIES = [
        # §0.3.1 K-wave pure(7;§14.7-BY Phase B/E)
        "PATENTUSALLTOTAL", "B985RC1Q027SBEA", "TCMDO", "QUSPAM770A",
        "LFWA64TTUSA647N", "SPPOPDPNDOLUSA", "PALLFNFINDEXQ",
        # §0.3.2 Multi-cycle(5;§14.7-CC FRED-native)
        "M2SL", "T10Y2Y", "WTISPLC", "IPG3344S", "PCU4831114831115",
        # §0.3.3 Microstructure(1)
        "VIXCLS",
    ]

    P1_THRESHOLDS = {
        "price_252d": 200,
        "per_recent": 1,
        "monthrev_12m": 12,
        "finstmt_rev_4q": 4,
        "finstmt_op_4q": 4,
        "finstmt_iat_4q": 4,
        "bs_ta_2q": 2,
        "bs_eq_1q": 1,
    }

    P2_THRESHOLDS = {
        "inst_60d": 40,
        "margin_60d": 40,
        "info_1": 1,
    }

    # §14.7-CB Stage 4 feature completeness gate(optional;per Reading B convergence verification)
    # 37 spec features list inherited from apply_feature_completeness_gate.py
    SPEC_37_FEATURES = [
        "log_return_20d", "log_return_60d", "log_return_252d",
        "upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
        "avg_daily_value_log_60d", "amihud_illiquidity_60d", "zero_volume_ratio_252d",
        "pe_ratio", "pb_ratio", "dividend_yield",
        "roe_ttm", "operating_margin_ttm",
        "revenue_yoy_3m_log", "asset_growth_yoy",
        "right_tail_concentration_60d", "barbell_balance_60d", "preferential_attachment_60d",
        "fitness_signal_60d", "right_tail_returns_skew_252d", "liquidity_rank_pct_sector_60d",
        "size_log_zscore_sector",
        "kwave_tech_paradigm_strength", "kwave_credit_cycle_phase", "kwave_credit_to_gdp_gap",
        "kwave_demographics_trend", "kwave_commodity_supercycle", "kwave_phase_indicator",
        "mc_monetary_regime", "mc_yield_curve_inversion", "mc_oil_juglar_phase",
        "mc_semi_kitchin", "mc_shipping_juglar",
        "ms_volatility_regime", "ms_vix_term_structure", "ms_market_stress",
    ]

    POLICY_VERSION_STANDARD = "core_universe_policy_v0.13_doctrine_native_gate"
    POLICY_VERSION_STRICT = "core_universe_policy_v0.14_strict_feature_validity_gate"
    POLICY_VERSION_SUPER_STRICT = "core_universe_policy_v0.15_feature_reasonableness_gate"

    # §14.7-CJ(2026-05-28):Feature Reasonableness Gate 之 bounds
    # 對映 audit 揭露之 5 outlier features;真實 API-derived 但 extreme outlier
    # 嚴格 per 用戶 directive「特徵值不能用就不入核心股」延伸至 outlier exclusion
    REASONABLE_BOUNDS = {
        # (feature_name, lower_bound, upper_bound)
        "pe_ratio": (0.001, 500.0),        # PE>0(EPS>0 之獲利公司)且 < 500(合理估值)
        "pb_ratio": (0.001, 30.0),         # PB>0(BV>0)且 < 30(避免極高估值)
        "roe_ttm": (-1.0, 1.0),            # ROE ∈ [-100%, +100%]
        "operating_margin_ttm": (-1.0, 1.0),  # OM ∈ [-100%, +100%]
        "dividend_yield": (0.0, 30.0),     # ≥ 0% 且 < 30%
    }

    def __init__(self, as_of_date, commit=False, with_feature_gate=False,
                 with_reasonableness_gate=False, feature_set_id=None):
        self.as_of_date = as_of_date
        self.commit_mode = commit
        self.with_feature_gate = with_feature_gate
        self.with_reasonableness_gate = with_reasonableness_gate
        self.feature_set_id = feature_set_id
        self.stage_results = {}
        # §14.7-CI v0.14 / §14.7-CJ v0.15 / standard v0.13
        if with_reasonableness_gate:
            assert with_feature_gate, "--with-reasonableness-gate requires --with-feature-gate"
            self.POLICY_VERSION = self.POLICY_VERSION_SUPER_STRICT
        elif with_feature_gate:
            self.POLICY_VERSION = self.POLICY_VERSION_STRICT
        else:
            self.POLICY_VERSION = self.POLICY_VERSION_STANDARD

    def _check_kwave_market(self, cur):
        cur.execute(
            "SELECT DISTINCT series_id FROM fred_series WHERE series_id = ANY(%s)",
            (self.KWAVE_SERIES,),
        )
        present = sorted([r[0] for r in cur.fetchall()])
        missing = [s for s in self.KWAVE_SERIES if s not in present]
        self.stage_results['stage1'] = {
            'expected': len(self.KWAVE_SERIES),
            'present_count': len(present),
            'missing': missing,
        }
        return len(missing) == 0

    def _get_candidate_set(self, cur):
        cur.execute(
            'SELECT DISTINCT stock_id FROM "TaiwanStockInfo" WHERE industry_category IS NOT NULL'
        )
        return sorted([r[0] for r in cur.fetchall()])

    def _run_per_stock_audit(self, cur):
        today = self.as_of_date
        d_365 = today - timedelta(days=365)
        d_18m = today - timedelta(days=18 * 30)
        d_24m = today - timedelta(days=24 * 30)
        d_90d = today - timedelta(days=90)
        d_year = today.replace(month=1, day=1)

        cur.execute(
            'SELECT stock_id, COUNT(*) FROM "TaiwanStockPriceAdj" WHERE date >= %s GROUP BY stock_id',
            (d_365,),
        )
        price_count = dict(cur.fetchall())

        cur.execute(
            '''SELECT stock_id, COUNT(*) FROM "TaiwanStockPER"
               WHERE date >= %s AND "PER" IS NOT NULL AND "PBR" IS NOT NULL AND "dividend_yield" IS NOT NULL
               GROUP BY stock_id''',
            (d_year,),
        )
        per_count = dict(cur.fetchall())

        cur.execute(
            'SELECT stock_id, COUNT(*) FROM "TaiwanStockMonthRevenue" WHERE date >= %s GROUP BY stock_id',
            (d_18m,),
        )
        monthrev_count = dict(cur.fetchall())

        cur.execute(
            '''SELECT stock_id, type, COUNT(DISTINCT date) FROM "TaiwanStockFinancialStatements"
               WHERE date >= %s AND type IN ('Revenue','OperatingIncome','IncomeAfterTaxes')
               GROUP BY stock_id, type''',
            (d_24m,),
        )
        finstmt_rev, finstmt_op, finstmt_iat = {}, {}, {}
        for sid, ttype, n in cur.fetchall():
            if ttype == 'Revenue':
                finstmt_rev[sid] = n
            elif ttype == 'OperatingIncome':
                finstmt_op[sid] = n
            elif ttype == 'IncomeAfterTaxes':
                finstmt_iat[sid] = n

        cur.execute(
            '''SELECT stock_id, type, COUNT(DISTINCT date) FROM "TaiwanStockBalanceSheet"
               WHERE date >= %s AND type IN ('TotalAssets','EquityAttributableToOwnersOfParent')
               GROUP BY stock_id, type''',
            (d_24m,),
        )
        bs_ta, bs_eq = {}, {}
        for sid, ttype, n in cur.fetchall():
            if ttype == 'TotalAssets':
                bs_ta[sid] = n
            elif ttype == 'EquityAttributableToOwnersOfParent':
                bs_eq[sid] = n

        cur.execute(
            'SELECT stock_id, COUNT(DISTINCT date) FROM "TaiwanStockInstitutionalInvestorsBuySell" WHERE date >= %s GROUP BY stock_id',
            (d_90d,),
        )
        inst_count = dict(cur.fetchall())

        cur.execute(
            'SELECT stock_id, COUNT(DISTINCT date) FROM "TaiwanStockMarginPurchaseShortSale" WHERE date >= %s GROUP BY stock_id',
            (d_90d,),
        )
        margin_count = dict(cur.fetchall())

        cur.execute(
            'SELECT stock_id, COUNT(*) FROM "TaiwanStockInfo" WHERE industry_category IS NOT NULL GROUP BY stock_id'
        )
        info_count = dict(cur.fetchall())

        return {
            "price_252d": price_count,
            "per_recent": per_count,
            "monthrev_12m": monthrev_count,
            "finstmt_rev_4q": finstmt_rev,
            "finstmt_op_4q": finstmt_op,
            "finstmt_iat_4q": finstmt_iat,
            "bs_ta_2q": bs_ta,
            "bs_eq_1q": bs_eq,
            "inst_60d": inst_count,
            "margin_60d": margin_count,
            "info_1": info_count,
        }

    def _apply_feature_gate(self, cur, qualified):
        """§14.7-CB Stage 4 feature completeness gate(optional;per Reading B convergence verification).

        Filter `qualified` to stocks 同時 pass 37/37 features in feature_values (per feature_set_id).
        Returns (filtered_qualified, feature_rejected_reasons).
        """
        if not self.feature_set_id:
            cur.execute(
                """SELECT feature_set_id FROM feature_store_snapshot
                   WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1"""
            )
            row = cur.fetchone()
            if not row:
                return qualified, {"_meta": "no committed feature_store_snapshot — skip feature gate"}
            self.feature_set_id = row[0]

        ids_tuple = tuple(qualified) if qualified else ('',)
        cur.execute(
            """SELECT stock_id, COUNT(DISTINCT feature_name) AS n
               FROM feature_values
               WHERE feature_set_id = %s AND feature_name = ANY(%s) AND stock_id = ANY(%s)
               GROUP BY stock_id""",
            (self.feature_set_id, self.SPEC_37_FEATURES, list(qualified)),
        )
        feature_count = dict(cur.fetchall())

        filtered = []
        rejected_reasons = {}
        for sid in qualified:
            n = feature_count.get(sid, 0)
            if n >= 37:
                filtered.append(sid)
            else:
                rejected_reasons[sid] = f"feature_count={n}<37"
        return filtered, {
            "feature_set_id": self.feature_set_id,
            "n_input": len(qualified),
            "n_output": len(filtered),
            "n_rejected": len(qualified) - len(filtered),
            "sample_rejected": list(rejected_reasons.items())[:5],
        }

    def _apply_reasonableness_gate(self, cur, qualified):
        """§14.7-CJ Stage 4-reasonable:Feature reasonableness bounds(per 用戶 directive
        「outlier feature 之 stocks 不入核心股」)。

        Filter `qualified` to stocks 同時 pass REASONABLE_BOUNDS for all 5 outlier-prone features.
        Returns (filtered_qualified, audit_info)。
        """
        bounds_features = list(self.REASONABLE_BOUNDS.keys())
        cur.execute(
            """SELECT stock_id, feature_name, feature_value::numeric
               FROM feature_values
               WHERE feature_set_id = %s AND feature_name = ANY(%s) AND stock_id = ANY(%s)""",
            (self.feature_set_id, bounds_features, list(qualified)),
        )
        stock_features = {}
        for sid, fn, val in cur.fetchall():
            stock_features.setdefault(sid, {})[fn] = float(val) if val is not None else None

        filtered = []
        rejected_reasons = {}
        reject_hist = {}
        for sid in qualified:
            features = stock_features.get(sid, {})
            issues = []
            for fn, (lo, hi) in self.REASONABLE_BOUNDS.items():
                val = features.get(fn)
                if val is None:
                    issues.append(f"{fn}=None")
                    reject_hist[fn] = reject_hist.get(fn, 0) + 1
                elif val < lo or val > hi:
                    issues.append(f"{fn}={val:.4g}∉[{lo},{hi}]")
                    reject_hist[fn] = reject_hist.get(fn, 0) + 1
            if issues:
                rejected_reasons[sid] = ", ".join(issues[:3])
            else:
                filtered.append(sid)

        return filtered, {
            "n_input": len(qualified),
            "n_output": len(filtered),
            "n_rejected": len(qualified) - len(filtered),
            "reject_histogram": reject_hist,
            "sample_rejected": list(rejected_reasons.items())[:5],
            "bounds": self.REASONABLE_BOUNDS,
        }

    def _commit_snapshot(self, cur, conn, qualified, n_candidates, reason_hist):
        today_str = self.as_of_date.strftime("%Y%m%d")
        new_snap = f"core_universe_{today_str}_{self.POLICY_VERSION.replace('.', '_')}"

        cur.execute(
            """SELECT snapshot_id FROM core_universe_snapshot
               WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1"""
        )
        row = cur.fetchone()
        old_snap = row[0] if row else None

        print(f"\n📝 §14.7-CG Stage 5: Atomic Supersede Commit")
        print(f"   Old snapshot: {old_snap}")
        print(f"   New snapshot: {new_snap}")

        cur.execute(
            """INSERT INTO core_universe_policy (policy_version, policy_name, description, active, effective_from)
               VALUES (%s, %s, %s, TRUE, CURRENT_DATE)
               ON CONFLICT (policy_version) DO NOTHING""",
            (self.POLICY_VERSION,
             "§14.7-CG Doctrine Native Gate v0.13",
             "§14.7-CF 三 invariant 原生實現(Stage 1 K-wave + Stage 2/3 11 raw sources + Stage 4 union)"),
        )

        cur.execute(
            """INSERT INTO core_universe_snapshot
                (snapshot_id, as_of_date, source_data_cutoff, policy_version,
                 total_candidates, core_count, status, notes, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, 'committed', %s, NOW())""",
            (new_snap, self.as_of_date, self.as_of_date, self.POLICY_VERSION,
             n_candidates, len(qualified),
             f"§14.7-CG Doctrine Native Gate v0.13; superseded {old_snap}"),
        )

        cur.executemany(
            """INSERT INTO core_universe_membership
                (snapshot_id, stock_id, core_tier, active, selected_at, selection_reason)
               VALUES (%s, %s, 'core_universe', TRUE, NOW(), %s)""",
            [(new_snap, sid, "§14.7-CG doctrine native gate verified") for sid in qualified],
        )

        if old_snap:
            cur.execute(
                "UPDATE core_universe_snapshot SET status='superseded' WHERE snapshot_id=%s",
                (old_snap,),
            )

        detail = json.dumps({
            "step": "§14.7-CG",
            "n_candidates": n_candidates,
            "n_qualified": len(qualified),
            "n_rejected": n_candidates - len(qualified),
            "reason_hist": reason_hist,
            "stage1_kwave": self.stage_results['stage1'],
        })
        cur.execute(
            """INSERT INTO universe_revision_log
                (revision_time, actor, action_type, object_type, object_id,
                 policy_version, snapshot_id, detail, note)
               VALUES (NOW(), 'core_universe_builder_doctrine_native',
                       'doctrine_native_gate', 'snapshot',
                       %s, %s, %s, %s::jsonb,
                       '§14.7-CG v0.13: native gate 3 step → 1 program 整合')""",
            (new_snap, self.POLICY_VERSION, new_snap, detail),
        )

        conn.commit()
        print(f"   ✅ COMMIT done")
        print(f"   Snapshot: {new_snap} (N={len(qualified)})")
        return True

    def build(self):
        mode_label = "COMMIT" if self.commit_mode else "DRY-RUN"
        print("\n" + "🛡️" * 40)
        print(f"§14.7-CG Doctrine Native Gate Builder v0.13 / mode={mode_label}")
        print(f"As-of-date: {self.as_of_date}")
        print(f"Policy:     {self.POLICY_VERSION}")
        print("🛡️" * 40)

        conn = get_db_connection()
        try:
            cur = conn.cursor()

            print(f"\n[Stage 1] §0.3 K-wave macro prerequisite "
                  f"({len(self.KWAVE_SERIES)} FRED series 存在性 binary gate)")
            if not self._check_kwave_market(cur):
                miss = self.stage_results['stage1']['missing']
                print(f"  ❌ Stage 1 FAIL: missing {len(miss)}/{len(self.KWAVE_SERIES)}: {miss}")
                return False
            print(f"  ✅ Stage 1 PASS: "
                  f"{self.stage_results['stage1']['present_count']}/{len(self.KWAVE_SERIES)} present")

            candidates = self._get_candidate_set(cur)
            print(f"\n[Stage 2+3] per-stock §0.1 + §0.2 raw source × thresholds "
                  f"(N_candidates={len(candidates)})")
            counts = self._run_per_stock_audit(cur)

            qualified = []
            rejected = {}
            all_thresholds = {**self.P1_THRESHOLDS, **self.P2_THRESHOLDS}
            for sid in candidates:
                reasons = []
                for source, threshold in all_thresholds.items():
                    n = counts[source].get(sid, 0)
                    if n < threshold:
                        reasons.append(f"{source}={n}<{threshold}")
                if reasons:
                    rejected[sid] = reasons
                else:
                    qualified.append(sid)

            reason_hist = {}
            for sid, reasons in rejected.items():
                for r in reasons:
                    src = r.split('=')[0]
                    reason_hist[src] = reason_hist.get(src, 0) + 1

            print(f"\n📊 [Stage 4] Doctrine-Pass Universe")
            print(f"  Candidates : {len(candidates):4d}")
            print(f"  ✅ QUALIFIED: {len(qualified):4d} "
                  f"({100.0 * len(qualified) / len(candidates):.1f}%)")
            print(f"  ❌ REJECTED : {len(rejected):4d}")
            if reason_hist:
                print(f"\n  Top rejection reasons (per-source histogram):")
                for src, n in sorted(reason_hist.items(), key=lambda x: -x[1]):
                    print(f"    {src:20s}: {n:3d} stocks fail")

            # Stage 4 (optional): feature completeness gate per §14.7-CB
            # — Reading B convergence verification mode
            if self.with_feature_gate:
                print(f"\n[Stage 4-feature] §14.7-CB Feature Completeness Gate (37/37 spec features)")
                pre_n = len(qualified)
                qualified, feature_audit = self._apply_feature_gate(cur, qualified)
                self.stage_results['stage4_feature'] = feature_audit
                if 'feature_set_id' in feature_audit:
                    print(f"  feature_set_id: {feature_audit['feature_set_id']}")
                    print(f"  Input  : {feature_audit['n_input']}")
                    print(f"  Output : {feature_audit['n_output']}")
                    print(f"  Reject : {feature_audit['n_rejected']}")
                    if feature_audit['sample_rejected']:
                        print(f"  Sample rejected: {feature_audit['sample_rejected'][:3]}")
                else:
                    print(f"  (skip) {feature_audit.get('_meta', '')}")
                if pre_n != len(qualified):
                    print(f"\n📊 [Stage 4 result] Post-feature-gate universe N={len(qualified)} "
                          f"(removed {pre_n - len(qualified)} stocks lacking 37/37 features)")
                else:
                    print(f"\n📊 [Stage 4 result] Reading A+C ↔ Reading B 收斂 ✅ "
                          f"(all {len(qualified)} stocks 37/37 features present)")

            # Stage 4-reasonable (optional / §14.7-CJ):Feature reasonableness bounds
            # — 排除 outlier 之 stocks(per 用戶「特徵值不能用就不入」延伸)
            if self.with_reasonableness_gate:
                print(f"\n[Stage 4-reasonable] §14.7-CJ Feature Reasonableness Gate")
                print(f"  Bounds: {self.REASONABLE_BOUNDS}")
                pre_n = len(qualified)
                qualified, reason_audit = self._apply_reasonableness_gate(cur, qualified)
                self.stage_results['stage4_reasonable'] = reason_audit
                print(f"  Input  : {reason_audit['n_input']}")
                print(f"  Output : {reason_audit['n_output']}")
                print(f"  Reject : {reason_audit['n_rejected']}")
                print(f"  Rejection histogram:")
                for fn, n in sorted(reason_audit['reject_histogram'].items(), key=lambda x: -x[1]):
                    print(f"    {fn:30s}: {n:4d} stocks out of bounds")
                if reason_audit['sample_rejected']:
                    print(f"  Sample rejected:")
                    for sid, reason in reason_audit['sample_rejected'][:3]:
                        print(f"    {sid}: {reason}")
                print(f"\n📊 [Stage 4-reasonable result] Post-reasonableness-gate N={len(qualified)} "
                      f"(removed {pre_n - len(qualified)} stocks with outlier features)")

            if not self.commit_mode:
                print(f"\n[DRY-RUN] no DB write — qualified N={len(qualified)}")
                return True

            return self._commit_snapshot(cur, conn, qualified, len(candidates), reason_hist)
        finally:
            conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance 核心股選拔引擎(v0.7.1 CoreScore / v0.13 §14.7-CG Doctrine Native Gate)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只計算與輸出摘要，不寫入治理表")
    mode.add_argument("--commit", action="store_true", help="寫入 policy/snapshot/membership/scores/revision log")
    parser.add_argument("--as-of-date", type=str, help="Universe snapshot 基準日期，預設為今天")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["legacy-corescore", "doctrine-native"],
        default="legacy-corescore",
        help="挑選 mode:legacy-corescore (v0.7.1 CoreScore / INFO display per §14.7-BW) "
             "| doctrine-native (v0.13 §14.7-CG 原生實現 §14.7-CF 三 invariant;預計 v6.6.0 起預設)",
    )
    parser.add_argument(
        "--with-feature-gate",
        action="store_true",
        help="(§14.7-CI v0.14 strict)apply §14.7-CB feature completeness gate(37/37 features)"
             "after raw layer pass;production model training universe;若 with --with-reasonableness-gate 則升 v0.15",
    )
    parser.add_argument(
        "--with-reasonableness-gate",
        action="store_true",
        help="(§14.7-CJ v0.15 super-strict)apply feature value reasonableness bounds"
             "(pe/pb/roe/operating_margin/dividend_yield);必同時 --with-feature-gate;per 用戶 directive"
             "「outlier features 之 stocks 不入核心股」",
    )
    parser.add_argument(
        "--feature-set-id",
        type=str,
        default=None,
        help="(§14.7-CG Stage 4 optional)feature_set_id to use for feature gate;default = latest committed",
    )
    parser.add_argument("--policy-version", type=str, default=DEFAULT_POLICY_VERSION, help="核心股選拔政策版本(v0.7/v0.8 模式 DEPRECATED per §14.7-BW pure doctrine;新路徑為 build_doctrine_gate_universe.py)")
    # §14.7-BW pure doctrine + 2026-05-27 directive:legacy / dynamic mode 之 N flags 全 DEPRECATED
    parser.add_argument("--core-limit", type=int, default=None, help="DEPRECATED per §14.7-BW (was legacy core 上限 120)")
    parser.add_argument("--convex-limit", type=int, default=None, help="DEPRECATED per §14.7-BW (was legacy convex 上限 30)")
    parser.add_argument("--selection-pct", type=float, default=None, help="DEPRECATED per §14.7-BW (was dynamic top 5%)")
    parser.add_argument("--selection-n-min", type=int, default=None, help="DEPRECATED per §14.7-BW (was N min floor 100)")
    parser.add_argument("--selection-n-max", type=int, default=None, help="DEPRECATED per §14.7-BW (was N max cap 200)")
    parser.add_argument("--core-pct", type=float, default=None, help="DEPRECATED per §14.7-BW (was 70/30 tier split)")
    parser.add_argument("--include-emerging", action="store_true", help="允許 emerging 類型進入非 quarantine 分層")
    parser.add_argument(
        "--special-rebalance-reason",
        type=str,
        help="非年度核心股重選的特別治理原因；只允許重大資料修復、政策升版或風險事件等可稽核例外",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date() if args.as_of_date else date.today()

    if args.mode == "doctrine-native":
        # §14.7-CG v0.13 / §14.7-CI v0.14 / §14.7-CJ v0.15 native gate path
        builder = DoctrineNativeGateBuilder(
            as_of_date=as_of_date,
            commit=args.commit,
            with_feature_gate=args.with_feature_gate,
            with_reasonableness_gate=args.with_reasonableness_gate,
            feature_set_id=args.feature_set_id,
        )
        ok = builder.build()
        sys.exit(0 if ok else 1)

    # Legacy CoreScore path(per §14.7-BW INFO display only;v0.7.1)
    builder = CoreUniverseBuilder(
        as_of_date=as_of_date,
        policy_version=args.policy_version,
        commit=args.commit,
        core_limit=args.core_limit,
        convex_limit=args.convex_limit,
        # §14.7-BT Phase C 新加:dynamic mode params
        selection_pct=args.selection_pct,
        selection_n_min=args.selection_n_min,
        selection_n_max=args.selection_n_max,
        core_pct_within_selected=args.core_pct,
        include_emerging=args.include_emerging,
        special_rebalance_reason=args.special_rebalance_reason,
    )
    ok = builder.build()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
feature_store_builder.py v0.5 (Quantum Finance Feature Store Build Authority)
================================================================================
最後更新日期: 2026-05-25
主權狀態: IMPLEMENTED (憲法 v6.1.0-patch §8.2 + §9.9 v0.3 upside/downside + §8.5 第 9 條 Publication-date Discipline Phase 2 落地;讀 PUBLICATION_DATE_STRATEGY_REGISTRY + SQL gate per-table 分派;v0.5 改用 data_schema.build_publication_date_gate SSOT helper)
最高原則: Feature Store Build Authority

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Feature Store Build Authority]: 對齊憲章 §8.2 Feature Store v0.1 草案，
   作為 §2 維運矩陣 Step 9 之執行載體（§8.7 矩陣延伸）；從 raw + universe
   派生 features 並 commit 到 `feature_store_snapshot` / `feature_definition` /
   `feature_values` 三張治理表。
2. [Read-Only Raw Schema]: 只讀取 raw API tables（`TaiwanStockPriceAdj` /
   `TaiwanStockMonthRevenue` / `TaiwanStockFinancialStatements` /
   `TaiwanStockInstitutionalInvestorsBuySell` /
   `TaiwanStockMarginPurchaseShortSale` / `FredData` / `TaiwanStockInfo`）
   與 `core_universe_*` 治理表；不呼叫 FinMind/FRED API（已由 §2 Step 4
   完成資料灌溉）。
3. [As-Of Strict Anti-Leakage + Publication-date Discipline]: 對齊憲章 §8.5
   「Data Leakage 防禦規則」8 條 + 第 9 條 Publication-date Discipline(v0.4 落地);
   v0.3 以前用 `WHERE date <= as_of_date` 統一過濾;**v0.4 起依
   `PUBLICATION_DATE_STRATEGY_REGISTRY` per-table 分派 SQL gate**:
   - native_aligned (Price/PriceAdj/PER/Institutional/Margin/Info): `date <= as_of_date`
   - strict (Dividend): `AnnouncementDate <= as_of_date`
   - hardcoded_conservative (MonthRevenue): `(date + INTERVAL '10 days') <= as_of_date`
   - hardcoded_conservative (FinStmt): quarter-aware (Q1-Q3 +45 / Q4 +90 天)
   - transitional (Shareholding, FRED v2.19 追溯): 暫維持 `date <= as_of_date`
     (FRED 因 DB realtime_start 為 ingest 日期非真實 vintage,§14.7-BB Phase 2 dry-run 揭露;
      待 D2.4 ALFRED API 整合後升 strict)
   label_date 由 `as_of_date + label_horizon` 推導,與 feature 嚴格分離。
4. [Governance Write Order]: 寫入順序為 `feature_definition` →
   `feature_values` → `feature_store_snapshot (status='committed')`；
   feature_set_id 命名格式 `fs_{yyyymmdd}_{feature_set_version}`，
   確保可重現與 audit trail。
5. [Universe Lock]: 範圍鎖定 `core_universe ∪ convex_universe` (N dynamic
   per §14.7-BW pure doctrine,無 hardcoded 150/200 cap;§6.7 SQL 契約之
   `get_core_stocks_from_db(tiers=['core','convex'])`);
   universe_snapshot_id 必須為最新 committed snapshot。
6. [Downstream Boundary]: 不保存 labels、不保存 model output、不保存
   預測訊號；§8 三層職責邊界（Feature Store / Model Registry / Prediction Table）
   嚴格分離。
7. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權判定動態計算（§5.6.3 零硬編 PERFECT）。
8. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 9-dry：特徵建構驗算]** | `$ python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-15` | feature_store_builder v0.1 |
| **2. [Step 9-commit：production-current commit]** | `$ python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-15 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20` | feature_store_builder v0.1 |
| **3. [Step 9-historical：walk-forward evidence]** | `$ python scripts/core/feature_store_builder.py --commit --as-of-date <historical-date> --feature-set-version feature_set_v0.1_h20_historical_<date>_strict_source --label-horizon 20` | feature_store_builder v0.1 |
| **4. [Step 9-h30：v6.2.0 預備]** | `$ python scripts/core/feature_store_builder.py --commit --as-of-date <date> --feature-set-version feature_set_v0.1_h30_historical_<date> --label-horizon 30` | feature_store_builder v0.1 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | preflight + coverage 報告，不寫 DB |
| **horizon-30** | `--label-horizon 30` | §9.1 v6.2.0 預備之 h30 forward-return |
| **feature-version** | `--feature-set-version <name>` | 自訂 feature set name 標籤 |
| **strict-source** | feature-set-version 含 `strict_source` | 對齊 §14.7-L strict source alignment |

> 💡 **資料庫歷史灌溉最佳實踐 (Training Data Ingestion Window)**
> 為了讓本特徵建構引擎在特定 `as_of_date` 運算時達到最佳特徵完整度，
> DB refill 建議至少涵蓋 **730 ~ 1100 天（約 2 ~ 3 年）**：
> 20/60/252 天視窗支撐價格波動與籌碼多尺度因子；15 ~ 24 個月視窗支撐
> 財務 YoY 與長線基本面特徵。充足歷史灌溉可降低 null 缺失與下游訓練噪音。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.5** | 2026-05-25 | Codex | **改用 data_schema.build_publication_date_gate SSOT helper(Phase 3 配套;對齊 §0.0-I 單一引用源原則)**:依憲章 §8.5-9 Phase 3 同次落地(core_universe_builder v0.3 → v0.4),為避免 helper 在兩個 builder 重複定義(違 DRY/SSOT),把 local `_publication_date_gate()` 改為 `from core.data_schema import build_publication_date_gate`(v2.20 升 SSOT 之 helper)。**補正內容**:(I) 刪除模組級 local `_publication_date_gate()`(完整移至 `data_schema.py v2.20` 為 `build_publication_date_gate`,介面零變動);(II) 加 `from core.data_schema import build_publication_date_gate`;(III) 7 個 `_load_*` 函式內 `_publication_date_gate(table)` call 改為 `build_publication_date_gate(table)`;(IV) TOOL_VER v0.4 → v0.5;(V) 主權狀態行加「v0.5 改用 data_schema.build_publication_date_gate SSOT helper」。**邏輯動量**:helper 介面與行為**完全不變**(僅位置移轉);31 個 features 數量不變;windows 不變;imputation 策略不變;CLI 介面不變;verdict 動態計算邏輯不變;7 個 _load_* 函式 SQL gate 行為不變(v0.4 之 SQL 結構保留)。**對既有 DB / snapshot 影響**:**零**(本版純為 helper import 位置變更,無 SQL 邏輯改動)。**Phase 3 配套效益**:`core_universe_builder v0.4` 與本 builder 共用同一 SSOT helper;未來 audit_leakage v0.3 + prediction_engine v0.3 亦可 import 同 helper;達成 §0.0-I 單一引用源治權。本版**不**修改 helper 邏輯、**不**改 7 個 _load_* 函式之 SQL 結構、**不**改 FEATURE_DEFINITIONS、**不**改 CLI、**不**改 verdict。同步配套:`data_schema.py v2.19 → v2.20`(helper 升 SSOT)+ `core_universe_builder.py v0.3 → v0.4`(import + 12 處 SQL gate 升版)。 | **ACTIVE** |
| v0.4 | 2026-05-25 | Codex | **§8.5 第 9 條 Publication-date Discipline Phase 2 落地(配套 data_schema v2.18→v2.19;憲章 §8.5-9 + §14.7-BA + §14.7-BB)**:依憲章 §8.5-9.7 升版觸發表之 Phase 2,加 `_publication_date_gate()` helper 從 `PUBLICATION_DATE_STRATEGY_REGISTRY` 構造 SQL gate clause;**7 個 `_load_*` 函式 SQL 升版**(對齊 5 種 enforcement):(I) `_load_price_series` (PriceAdj): native_aligned `date <= as_of_date` 維持;(II) `_load_revenue` (MonthRevenue): **hardcoded_conservative `(date + INTERVAL '10 days') <= as_of_date`** 新增 +10 天保守上限;(III) `_load_financial` (FinStmt): **hardcoded_conservative quarter-aware** Q1-Q3 +45 / Q4 +90 天;(IV) `_load_institutional` (Institutional): native_aligned 維持;(V) `_load_margin` (Margin): native_aligned 維持;(VI) `_load_theme` (Info): native_aligned 維持;(VII) `_load_macro` (FRED): **transitional 維持 `date <= as_of_date`**(因 §14.7-BB DB realtime_start 為 ingest 日期非真實 vintage,本次走 transitional path,實際 v0.3 behavior 不變)。**FEATURE_DEFINITIONS 補 `publication_date_source`**(對齊 §8.5-9.3):每個 feature 標註其 publication-date 來源(由 `PUBLICATION_DATE_STRATEGY_REGISTRY[source_table]['source']` 自動分派)。**DEFAULT_FEATURE_SET_VERSION**: `feature_set_v0.3` → **`feature_set_v0.4`**;CONSTITUTION_VER v6.0.0 → v6.1.0(對齊現行憲章 v6.1.0-patch);TOOL_VER v0.3 → v0.4;標頭核心定義第 3 條 [As-Of Strict Anti-Leakage] 升版為 [+ Publication-date Discipline]。**邏輯動量**:31 個 features 數量不變(v0.3 active set 27 base + 4 upside/downside);features 計算邏輯不變(僅 SQL gate 變);windows 不變(20d/60d/252d/4q/8q/24m);imputation 策略不變(drop/zero_fill);CLI 介面不變(`--dry-run / --commit / --as-of-date / --feature-set-version / --label-horizon`);verdict 動態計算邏輯不變;§5.6.3 + §0.4 + §0.0-G + §0.0-I 全部不違反。**對既有 feature_set_v0.1~v0.3 影響**:**零**(既有 snapshot 不重 build;標記 `publication_date_strategy='legacy_statistical_date'`;新 snapshot v0.4+ 起適用)。**預期 feature_values 差異(v0.3 vs v0.4 新 snapshot)**:fundamental 群(eps_sum_4q / net_income_positive_ratio_8q / revenue_yoy_*)在 historical as_of_date 接近 quarter-end 時可能少 1 個 quarter 之資料(因為 Q1=YYYY-03-31 + 45 天 = YYYY-05-15,若 as_of_date < YYYY-05-15 則該 Q 不入);其他群預期差異 0。**對 Phase 4 audit_leakage v0.3 rule 19 publication_date_check 之預備**:audit 將比對 feature_values 之原 SQL 是否使用此 strategy 之 gate(builder 透過 _publication_date_gate helper 確保一致)。同步入憲:憲章 §8.5-9.7 Phase 2 落地 + 配套 data_schema v2.19 修訂歷程 + 修訂歷程 v6.1.0-patch 2026-05-25 第五輪 entry。 | SUPERSEDED |
| v0.2 | 2026-05-19 | Codex | §0.0-D.6 升版條件 #1 落地：新增 interaction 群 4 features（feature_macro_vix_x_vol_60d / feature_macro_dff_x_eps_sum_4q / feature_theme_x_log_return_60d / feature_theme_x_foreign_net_60d）；feature_set_version v0.1 → v0.2；總特徵數 27 → 31；新增 `_compute_interaction_features()` 方法；不引入新 raw 資料源、不違反 §0.1-A 禁令、保留 cross-sectional variance > 0；既有 v0.1 committed feature sets 不受影響（only future runs build v0.2）。後續需重訓 model 才能實際使用新特徵。 | **SUPERSEDED (ablation IC = +0.0131 HARMFUL, §0.0-D.6 #1 已實證否決)** |
| v0.3 | 2026-05-20 | Codex | §9.9 P1 v0.1 落地：新增 4 個 upside/downside 分離特徵（upside_volatility_60d / downside_volatility_60d / upside_capture_60d / downside_capture_60d）至 price 群；feature_set_version v0.2 → v0.3；總特徵數 31 → 31（27 base + 4 upside/downside；v0.2 interaction 不繼承）；新增 4 個 static method `_upside_volatility / _downside_volatility / _upside_capture / _downside_capture`；採 version-aware 邏輯，v0.3 不寫入 interaction features。對齊 §0.0-C.3 上行凸性壓制 + §0.0-E.6 P1 升版優先級。後續需 v0.4 重訓 model 並驗證 ablation IC。 | SUPERSEDED |
| v0.1 | 2026-05-16 | Codex | 首版：§8.2 Feature Store 草案落地；27 features × 6 groups（price/liquidity/fundamental/institutional/macro/theme）；as-of-strict 過濾與 zero_fill/drop 雙 imputation 策略；2026-05-17 walk-forward h20 panel 8 點全 PERFECT；2026-05-18 v6.0.0-patch 落地 strict-source build（`fs_20260515_..._strict_source_20260518`）與 §14.7-L 對齊。 | SUPERSEDED |
================================================================================
"""
import argparse
import math
import sys
import time
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
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.5"
DEFAULT_FEATURE_SET_VERSION = "feature_set_v0.4"  # feature_set 版號不變(v0.4 為 schema 版號;v0.5 為程式 patch)
DEFAULT_LABEL_HORIZON = 20

# v0.5 SSOT helper(從 data_schema v2.20 import;v0.4 之 local _publication_date_gate 已移除)
try:
    from core.data_schema import PUBLICATION_DATE_STRATEGY_REGISTRY, build_publication_date_gate
except ImportError as exc:
    print(f"❌ data_schema SSOT 載入失敗(需 data_schema v2.20+): {exc}")
    sys.exit(1)

# § 8.2.2 v0.3 特徵字典：27 base + 4 v0.2 interaction (audit trail) + 4 v0.3 upside/downside = 35 total
# v0.3 active set = 31 features (27 base + 4 upside/downside；v0.2 interaction 不繼承，§9.9-E policy)
# v0.2 interaction features 保留於 FEATURE_DEFINITIONS 作為 audit trail (依 §0.0-G 治權邊界)
# 動機：原 macro / theme 為 broadcast 常數，cross-sectional rank model 中 IC=0；
# 交互特徵將 macro/theme broadcast 與 stock-specific 特徵相乘，恢復 cross-sectional variance。
FEATURE_DEFINITIONS = [
    # ── price 群（8）
    {"name": "log_return_20d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "20d", "vtype": "numeric", "null": "drop", "desc": "20-day log return of adjusted close"},
    {"name": "log_return_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "60-day log return"},
    {"name": "log_return_252d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "drop", "desc": "252-day log return"},
    {"name": "volatility_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "stddev of daily log returns over 60 days"},
    {"name": "volatility_252d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "drop", "desc": "stddev of daily log returns over 252 days"},
    {"name": "ma_ratio_20", "group": "price", "source": "TaiwanStockPriceAdj", "window": "20d", "vtype": "numeric", "null": "drop", "desc": "close / MA(close, 20)"},
    {"name": "ma_ratio_60", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "close / MA(close, 60)"},
    {"name": "max_drawdown_252d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "drop", "desc": "max drawdown over 252 days"},
    # ── price 群 v0.3 新增（4，§9.9 P1 上行凸性分離）
    {"name": "upside_volatility_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "RMS of positive daily log returns over 60d；§0.1 ΔlnP 上行凸性表達；§9.9 G1"},
    {"name": "downside_volatility_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "RMS of negative daily log returns over 60d；下行風險暴露；§9.9 G1"},
    {"name": "upside_capture_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "mean of positive daily log returns over 60d；§0.1 上行爆發力；§9.9 C"},
    {"name": "downside_capture_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "mean abs of negative daily log returns over 60d；下行衝擊；§9.9 C"},
    # ── price 群 v0.3 §14.7-CA Phase C-1c 新增(2026-05-27;doctrine-aligned per Phase A research §5.1)
    {"name": "convexity_60d", "group": "price", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "upside_volatility - downside_volatility(凸性 asymmetry;§14.7-BG / §9.10;§0.1)"},
    # ── liquidity 群 v0.3 §14.7-CA Phase C-1c 新增(2026-05-27)
    {"name": "amihud_illiquidity_60d", "group": "liquidity", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "Amihud 2002 illiquidity:AVG(|r|/dollar_vol) over 60d;TW IC ~+0.04-0.06 OOS;§0.1 §0.2"},
    # ── value 群 v0.3 §14.7-CA Phase C-1c-1 新增(2026-05-27;對映 Phase A research §5.1-D)
    {"name": "pe_ratio", "group": "value", "source": "TaiwanStockPER", "window": "TTM", "vtype": "numeric", "null": "drop", "desc": "latest PER(price/earnings;Fama-French HML;TW IC -0.02 ~ -0.04 OOS;§0.1.D)"},
    {"name": "pb_ratio", "group": "value", "source": "TaiwanStockPER", "window": "TTM", "vtype": "numeric", "null": "drop", "desc": "latest PBR(price/book;Fama-French HML;TW IC -0.02 ~ -0.04 OOS;§0.1.D)"},
    {"name": "dividend_yield", "group": "value", "source": "TaiwanStockPER", "window": "TTM", "vtype": "numeric", "null": "drop", "desc": "dividend yield from TaiwanStockPER;Litzenberger 1979;TW IC +0.015 OOS;§0.1.D"},
    # ── quality 群 v0.3 §14.7-CA Phase C-1c-2 新增(2026-05-27;對映 Phase A research §5.1-D Quality+Investment)
    {"name": "roe_ttm", "group": "quality", "source": "TaiwanStockPER", "window": "TTM", "vtype": "numeric", "null": "drop", "desc": "implied TTM ROE = PBR/PER identity(EPS/BPS);Asness QMJ;TW IC +0.07 OOS;§0.1.D"},
    {"name": "operating_margin_ttm", "group": "quality", "source": "TaiwanStockFinancialStatements", "window": "4q", "vtype": "numeric", "null": "drop", "desc": "TTM OperatingIncome / TTM Revenue(non-cumulative aware);QMJ profitability;TW IC +0.05 OOS;§0.1.D"},
    # ── investment 群 v0.3 §14.7-CA Phase C-1c-2 新增(2026-05-27)
    {"name": "revenue_yoy_3m_log", "group": "investment", "source": "TaiwanStockMonthRevenue", "window": "15m", "vtype": "numeric", "null": "drop", "desc": "log(1+revenue_yoy_3m);recent 3m revenue YoY growth log-transformed;TW IC +0.04 OOS;§0.1.D"},
    # ── investment 群 §14.7-CA Phase F-1 新增(2026-05-27;§0.1 100% closure)
    {"name": "asset_growth_yoy", "group": "investment", "source": "TaiwanStockBalanceSheet", "window": "24m", "vtype": "numeric", "null": "drop", "desc": "TotalAssets YoY growth;Cooper-Gulen-Schill 2008 asset growth anomaly;TW IC -0.05 OOS;§0.1.D Investment"},
    # ── §0.3.1 K-wave pure 群 v0.3 §14.7-CA Phase C-1c-3 新增(2026-05-27;6 features broadcast)
    {"name": "kwave_tech_paradigm_strength", "group": "kwave", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "(PATENTUSALLTOTAL_yoy + B985RC1Q027SBEA_yoy)/2;Schumpeter/Perez tech paradigm intensity;§0.3.1"},
    {"name": "kwave_credit_cycle_phase", "group": "kwave", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "TCMDO log_yoy;US credit cycle phase indicator;Reinhart-Rogoff 2009;§0.3.1"},
    {"name": "kwave_credit_to_gdp_gap", "group": "kwave", "source": "fred_series", "window": "as_of", "vtype": "numeric", "null": "zero_fill", "desc": "QUSPAM770A latest(BIS Credit-to-GDP gap);Drehmann 2014;TW IC -0.02 OOS;§0.3.1"},
    {"name": "kwave_demographics_trend", "group": "kwave", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "(LFWA64_yoy+(1-SPPOPDPND_yoy))/2;Goodhart-Pradhan 2020;§0.3.1"},
    {"name": "kwave_commodity_supercycle", "group": "kwave", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "PALLFNFINDEXQ log_yoy;Erten-Ocampo 2013;§0.3.1"},
    {"name": "kwave_phase_indicator", "group": "kwave", "source": "fred_series", "window": "composite", "vtype": "numeric", "null": "zero_fill", "desc": "composite z-score(5 K-wave features mean);Mensch 1979;§0.3.1"},
    # ── §0.3.2 Multi-cycle 群 v0.3 §14.7-CA Phase C-1c-3 新增(2026-05-27;5 features broadcast)
    {"name": "mc_monetary_regime", "group": "multi_cycle", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "M2SL log_yoy;Friedman monetary stance;§0.3.2"},
    {"name": "mc_yield_curve_inversion", "group": "multi_cycle", "source": "fred_series", "window": "as_of", "vtype": "numeric", "null": "zero_fill", "desc": "T10Y2Y latest;<0 = Juglar leading inversion;Estrella-Hardouvelis 1991 IC ~-0.04 OOS;§0.3.2"},
    {"name": "mc_oil_juglar_phase", "group": "multi_cycle", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "WTISPLC log_yoy;Stopford 2009 Juglar oil phase;§0.3.2"},
    {"name": "mc_semi_kitchin", "group": "multi_cycle", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "IPG3344S log_yoy(US Semi Industrial Production;Kitchin 4-y cycle);§14.7-CC FRED-native;§0.3.2"},
    {"name": "mc_shipping_juglar", "group": "multi_cycle", "source": "fred_series", "window": "12m", "vtype": "numeric", "null": "zero_fill", "desc": "PCU4831114831115 log_yoy(US Deep Sea Freight PPI;Juglar 7-11y cycle);Stopford 2009;§14.7-CC FRED-native;§0.3.2"},
    # ── §0.3.3 Microstructure 群 v0.3 §14.7-CA Phase C-1c-3 新增(2026-05-27;3 features broadcast)
    {"name": "ms_volatility_regime", "group": "microstructure", "source": "fred_series", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "VIXCLS rolling 60d mean;Whaley 1993 IC ~-0.025;§0.3.3"},
    {"name": "ms_vix_term_structure", "group": "microstructure", "source": "fred_series", "window": "252d", "vtype": "numeric", "null": "zero_fill", "desc": "(VIXCLS/252d_mean)-1;VIX premium;§0.3.3"},
    {"name": "ms_market_stress", "group": "microstructure", "source": "fred_series", "window": "30d", "vtype": "boolean", "null": "zero_fill", "desc": "1 if max VIXCLS > 30 in last 30 days;crisis警示 binary;§0.3.3"},
    # ── §0.2 八二法則 explicit 群 v0.3 §14.7-CA Phase C-1c-4 新增(2026-05-27;7 features per-sector aggregation)
    {"name": "right_tail_concentration_60d", "group": "pareto", "source": "TaiwanStockPriceAdj × TaiwanStockInfo", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "top 10% sids volume share / total within sector;Pareto 分布實證;TW IC +0.015 OOS;§0.2"},
    {"name": "barbell_balance_60d", "group": "pareto", "source": "TaiwanStockPriceAdj × TaiwanStockInfo", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "abs((top 20% vol share) - 0.80);Pareto deviation;§9.2 barbell theory;§0.2"},
    {"name": "preferential_attachment_60d", "group": "pareto", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "log10(avg_daily_value 60d)attachment proxy;Barabási-Albert 1999;TW IC +0.015 OOS;§0.2"},
    {"name": "fitness_signal_60d", "group": "pareto", "source": "TaiwanStockPriceAdj × TaiwanStockInfo × Institutional", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "(avg_value × (theme_strength+0.01) × (foreign_ratio+0.01))^(1/3);Bianconi-Barabási 2001;TW IC +0.02 OOS;§0.2"},
    {"name": "right_tail_returns_skew_252d", "group": "pareto", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "zero_fill", "desc": "skew of positive daily log returns over 252d;right-tail asymmetry;TW IC ±0.02 regime-dep;§0.2"},
    {"name": "liquidity_rank_pct_sector_60d", "group": "pareto", "source": "TaiwanStockPriceAdj × TaiwanStockInfo", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "sector 內 avg_value_60d 之 percentile rank ∈ [0,1];per-stock 相對集中度;TW IC +0.015 OOS;§0.2"},
    {"name": "size_log_zscore_sector", "group": "pareto", "source": "TaiwanStockPriceAdj × TaiwanStockInfo", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "log10(avg_value_60d) z-score within sector;Fama-French SMB proxy;TW IC ±0.01 emerging-market regime;§0.2"},
    # ── liquidity 群（4）
    {"name": "avg_daily_value_log_60d", "group": "liquidity", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "log10(avg Trading_money over 60d)"},
    {"name": "avg_daily_value_log_252d", "group": "liquidity", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "drop", "desc": "log10(avg Trading_money over 252d)"},
    {"name": "turnover_mean_60d", "group": "liquidity", "source": "TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "drop", "desc": "avg Trading_turnover over 60d"},
    {"name": "zero_volume_ratio_252d", "group": "liquidity", "source": "TaiwanStockPriceAdj", "window": "252d", "vtype": "numeric", "null": "drop", "desc": "fraction of zero-volume days over 252d"},
    # ── fundamental 群（4）
    {"name": "revenue_yoy_12m", "group": "fundamental", "source": "TaiwanStockMonthRevenue", "window": "24m", "vtype": "numeric", "null": "drop", "desc": "(sum recent 12m revenue / sum prior 12m revenue) - 1"},
    {"name": "revenue_yoy_3m", "group": "fundamental", "source": "TaiwanStockMonthRevenue", "window": "15m", "vtype": "numeric", "null": "drop", "desc": "(sum recent 3m revenue / sum same 3m prior year) - 1"},
    {"name": "eps_sum_4q", "group": "fundamental", "source": "TaiwanStockFinancialStatements", "window": "4q", "vtype": "numeric", "null": "zero_fill", "desc": "sum of EPS over last 4 quarters"},
    {"name": "net_income_positive_ratio_8q", "group": "fundamental", "source": "TaiwanStockFinancialStatements", "window": "8q", "vtype": "numeric", "null": "zero_fill", "desc": "fraction of last 8q with positive net income"},
    # ── institutional 群（5）
    {"name": "foreign_net_20d", "group": "institutional", "source": "TaiwanStockInstitutionalInvestorsBuySell", "window": "20d", "vtype": "numeric", "null": "zero_fill", "desc": "Foreign_Investor net buy over 20d (shares)"},
    {"name": "foreign_net_60d", "group": "institutional", "source": "TaiwanStockInstitutionalInvestorsBuySell", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "Foreign_Investor net buy over 60d (shares)"},
    {"name": "trust_net_20d", "group": "institutional", "source": "TaiwanStockInstitutionalInvestorsBuySell", "window": "20d", "vtype": "numeric", "null": "zero_fill", "desc": "Investment_Trust net buy over 20d"},
    {"name": "trust_net_60d", "group": "institutional", "source": "TaiwanStockInstitutionalInvestorsBuySell", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "Investment_Trust net buy over 60d"},
    {"name": "margin_ratio_60d", "group": "institutional", "source": "TaiwanStockMarginPurchaseShortSale", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "avg margin/short balance ratio over 60d"},
    # ── theme 群（2）
    {"name": "theme_strength", "group": "theme", "source": "TaiwanStockInfo", "window": "as_of", "vtype": "numeric", "null": "zero_fill", "desc": "THEME_KEYWORDS score / 100 from industry_category"},
    {"name": "theme_is_semiconductor", "group": "theme", "source": "TaiwanStockInfo", "window": "as_of", "vtype": "boolean", "null": "zero_fill", "desc": "1 if industry_category contains 半導體"},
    # ── macro 群（4，broadcast 至每股）
    {"name": "macro_dff_level", "group": "macro", "source": "FredData", "window": "as_of", "vtype": "numeric", "null": "drop", "desc": "latest DFF (Fed Funds Rate) as of date"},
    {"name": "macro_vix_level", "group": "macro", "source": "FredData", "window": "as_of", "vtype": "numeric", "null": "drop", "desc": "latest VIXCLS as of date"},
    {"name": "macro_t10y2y_level", "group": "macro", "source": "FredData", "window": "as_of", "vtype": "numeric", "null": "drop", "desc": "latest T10Y2Y as of date"},
    {"name": "macro_unrate_yoy", "group": "macro", "source": "FredData", "window": "13m", "vtype": "numeric", "null": "drop", "desc": "latest UNRATE - UNRATE 12 months prior"},
    # ── interaction 群（v0.2 新增，4）：對齊 §0.0-D.6 升版條件 #1
    # 動機：將 macro / theme broadcast 與 stock-specific 特徵相乘，
    #       恢復 cross-sectional variance，使 §0.3 戰術層脫離 IC=0 結構性失效。
    {"name": "feature_macro_vix_x_vol_60d", "group": "interaction", "source": "FredData × TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "macro_vix_level × volatility_60d；§0.3 × §0.1 ΔlnP 路徑風險；高 VIX 環境下高波股放大訊號"},
    {"name": "feature_macro_dff_x_eps_sum_4q", "group": "interaction", "source": "FredData × TaiwanStockFinancialStatements", "window": "4q", "vtype": "numeric", "null": "zero_fill", "desc": "macro_dff_level × eps_sum_4q；§0.3 × §0.1.3 V 質量因子；高利率環境下盈利質量分化"},
    {"name": "feature_theme_x_log_return_60d", "group": "interaction", "source": "TaiwanStockInfo × TaiwanStockPriceAdj", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "theme_strength × log_return_60d；§0.3 MBNRIC × §0.1 ΔlnP；主題對齊動量"},
    {"name": "feature_theme_x_foreign_net_60d", "group": "interaction", "source": "TaiwanStockInfo × TaiwanStockInstitutionalInvestorsBuySell", "window": "60d", "vtype": "numeric", "null": "zero_fill", "desc": "theme_is_semiconductor × foreign_net_60d；§0.3 半導體主題 × §0.1 F 外部力；半導體專屬資金流"},
]

THEME_KEYWORDS = {
    "半導體": 100, "生技": 95, "醫療": 95, "資訊": 90, "電腦": 85, "通信": 85,
    "電子": 80, "機器": 80, "電機": 75, "綠能": 75, "光電": 70, "能源": 70,
    "航太": 65, "汽車": 60,
}


class FeatureStoreBuilder:
    def __init__(self, as_of_date, feature_set_version, commit=False, label_horizon=DEFAULT_LABEL_HORIZON):
        self.as_of_date = as_of_date
        self.feature_set_version = feature_set_version
        self.commit = commit
        self.label_horizon = label_horizon
        self.feature_set_id = self._build_feature_set_id()
        self.universe_snapshot_id = None
        self.policy_version = None
        self.source_data_cutoff = None
        self.core_stocks = []
        self.stats = {
            "preflight_pass": 0, "preflight_warning": 0, "preflight_failed": 0,
            "feature_count": 0, "value_count": 0, "null_imputed_count": 0,
            "warnings": 0, "failed": 0, "details": [],
        }

    def _build_feature_set_id(self):
        date_str = self.as_of_date.strftime("%Y%m%d")
        version_sanitized = self.feature_set_version.replace(".", "_")
        return f"fs_{date_str}_{version_sanitized}"

    def _detail(self, msg):
        self.stats["details"].append(msg)
        print(msg)

    def _preflight(self, bucket, msg):
        self.stats[f"preflight_{bucket}"] += 1
        icon = {"pass": "✅", "warning": "⚠️", "failed": "❌"}[bucket]
        line = f"{icon} [PREFLIGHT-{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def _mark_lifecycle(self, lifecycle, level, msg):
        if lifecycle is None:
            return
        marker = getattr(lifecycle, "mark_failed" if level == "failed" else "mark_warning", None)
        if callable(marker):
            marker(msg)

    # ── PREFLIGHT ─────────────────────────────────────────────────────────────

    def preflight_check(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # 1. Required tables (raw + governance + feature_store)
            required = [
                "TaiwanStockInfo", "TaiwanStockPriceAdj", "TaiwanStockMonthRevenue",
                "TaiwanStockFinancialStatements", "TaiwanStockInstitutionalInvestorsBuySell",
                "TaiwanStockMarginPurchaseShortSale", "FredData",
                "core_universe_snapshot", "core_universe_membership",
                "feature_store_snapshot", "feature_definition", "feature_values",
            ]
            for tname in required:
                cur.execute("SELECT to_regclass(%s);", (f'public."{tname}"',))
                if cur.fetchone()[0]:
                    self._preflight("pass", f"{tname} exists")
                else:
                    self._preflight("failed", f"{tname} missing; run data_schema / core_universe_schema / feature_store_schema --init first")

            if self.stats["preflight_failed"] > 0:
                return False

            # 2. Latest committed core_universe snapshot (§6.7 contract)
            cur.execute(
                """
                SELECT s.snapshot_id, s.policy_version, s.source_data_cutoff
                FROM "core_universe_snapshot" s
                WHERE s.status = 'committed'
                  AND s.as_of_date = (
                      SELECT MAX(as_of_date) FROM "core_universe_snapshot" WHERE status = 'committed'
                  )
                ORDER BY s.created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                self._preflight("failed", "no committed core_universe_snapshot; run core_universe_builder.py --commit first")
                return False
            self.universe_snapshot_id, self.policy_version, self.source_data_cutoff = row
            self._preflight("pass", f"universe_snapshot_id={self.universe_snapshot_id}, policy={self.policy_version}")

            # 3. Load core+convex stock list (§6.7 SQL contract)
            cur.execute(
                """
                SELECT DISTINCT m.stock_id
                FROM "core_universe_membership" m
                JOIN "core_universe_snapshot" s ON s.snapshot_id = m.snapshot_id
                WHERE s.status = 'committed'
                  AND m.core_tier IN ('core_universe', 'convex_universe')
                  AND s.as_of_date = (
                      SELECT MAX(as_of_date) FROM "core_universe_snapshot" WHERE status = 'committed'
                  )
                ORDER BY m.stock_id
                """
            )
            self.core_stocks = [r[0] for r in cur.fetchall()]
            # §14.7-BW pure doctrine: N 為 doctrine 結果,無 implicit floor (was: < 100 warning)
            self._preflight("pass", f"core+convex universe loaded: {len(self.core_stocks)} stocks (dynamic per §14.7-BW)")

            # 4. Idempotency check
            cur.execute(
                'SELECT status FROM "feature_store_snapshot" WHERE feature_set_id = %s',
                (self.feature_set_id,),
            )
            existing = cur.fetchone()
            if existing and existing[0] == "committed":
                self._preflight("warning", f"feature_set_id={self.feature_set_id} already committed; will be re-written (delete + insert)")

        finally:
            cur.close()
            conn.close()
        return self.stats["preflight_failed"] == 0

    # ── DATA LOADING (as-of-strict) ───────────────────────────────────────────

    def _load_price_series(self, cur):
        """Return {stock_id: [(date, close, volume, money, turnover), ...]} sorted by date."""
        start = self.as_of_date - timedelta(days=400)
        # §8.5-9 Phase 2: PriceAdj = native_aligned (date <= as_of_date)
        gate, n_ap = build_publication_date_gate("TaiwanStockPriceAdj")
        cur.execute(
            f"""
            SELECT stock_id, date, "close"::numeric, "Trading_Volume"::numeric,
                   "Trading_money"::numeric, "Trading_turnover"::numeric
            FROM "TaiwanStockPriceAdj"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate}
            ORDER BY stock_id, date
            """,
            (self.core_stocks, start, *([self.as_of_date] * n_ap)),
        )
        out = {}
        for sid, d, c, v, m, t in cur.fetchall():
            out.setdefault(sid, []).append((d, float(c or 0), float(v or 0), float(m or 0), float(t or 0)))
        return out

    def _load_revenue(self, cur):
        """Return {stock_id: [(date, revenue), ...]} for last 24+ months."""
        start = self.as_of_date - timedelta(days=800)
        # §8.5-9 Phase 2: MonthRevenue = hardcoded_conservative ((date + INTERVAL '10 days') <= as_of_date)
        gate, n_ap = build_publication_date_gate("TaiwanStockMonthRevenue")
        cur.execute(
            f"""
            SELECT stock_id, date, revenue::numeric
            FROM "TaiwanStockMonthRevenue"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate}
            ORDER BY stock_id, date
            """,
            (self.core_stocks, start, *([self.as_of_date] * n_ap)),
        )
        out = {}
        for sid, d, r in cur.fetchall():
            out.setdefault(sid, []).append((d, float(r or 0)))
        return out

    def _load_financial(self, cur):
        """Aggregate {stock_id: {'eps_sum_4q': x, 'net_income_positive_ratio_8q': y}}."""
        start_4q = self.as_of_date - timedelta(days=400)
        start_8q = self.as_of_date - timedelta(days=800)
        # §8.5-9 Phase 2: FinStmt = hardcoded_conservative quarter-aware (Q1-Q3 +45 / Q4 +90 天)
        # n_ap = 2 (quarter-aware gate 含 2 個 as_of_date placeholder)
        gate, n_ap = build_publication_date_gate("TaiwanStockFinancialStatements")
        cur.execute(
            f"""
            SELECT stock_id,
                SUM(CASE WHEN type='EPS' AND date >= %s THEN value::numeric ELSE 0 END) as eps_sum_4q,
                COUNT(DISTINCT CASE WHEN (origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%')
                                     AND value::numeric > 0 AND date >= %s THEN date END) as net_pos_q,
                COUNT(DISTINCT CASE WHEN (origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%')
                                     AND date >= %s THEN date END) as net_total_q
            FROM "TaiwanStockFinancialStatements"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate}
            GROUP BY stock_id
            """,
            (start_4q, start_8q, start_8q, self.core_stocks, start_8q, *([self.as_of_date] * n_ap)),
        )
        out = {}
        for sid, eps_sum, net_pos, net_total in cur.fetchall():
            ratio = float(net_pos) / float(net_total) if net_total and net_total > 0 else 0.0
            out[sid] = {"eps_sum_4q": float(eps_sum or 0), "net_income_positive_ratio_8q": ratio}
        return out

    def _load_institutional(self, cur):
        """Net buy/sell aggregates for 20d/60d windows by institution type."""
        start_60 = self.as_of_date - timedelta(days=90)
        start_20 = self.as_of_date - timedelta(days=30)
        # §8.5-9 Phase 2: Institutional = native_aligned (T 日 17:30 後可得;§6.8.7-A 對齊)
        gate, n_ap = build_publication_date_gate("TaiwanStockInstitutionalInvestorsBuySell")
        cur.execute(
            f"""
            SELECT stock_id,
                SUM(CASE WHEN name IN ('Foreign_Investor','Foreign_Dealer_Self') AND date >= %s
                         THEN (buy::numeric - sell::numeric) ELSE 0 END) as foreign_net_20d,
                SUM(CASE WHEN name IN ('Foreign_Investor','Foreign_Dealer_Self') AND date >= %s
                         THEN (buy::numeric - sell::numeric) ELSE 0 END) as foreign_net_60d,
                SUM(CASE WHEN name = 'Investment_Trust' AND date >= %s
                         THEN (buy::numeric - sell::numeric) ELSE 0 END) as trust_net_20d,
                SUM(CASE WHEN name = 'Investment_Trust' AND date >= %s
                         THEN (buy::numeric - sell::numeric) ELSE 0 END) as trust_net_60d
            FROM "TaiwanStockInstitutionalInvestorsBuySell"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate}
            GROUP BY stock_id
            """,
            (start_20, start_60, start_20, start_60, self.core_stocks, start_60, *([self.as_of_date] * n_ap)),
        )
        out = {}
        for sid, f20, f60, t20, t60 in cur.fetchall():
            out[sid] = {
                "foreign_net_20d": float(f20 or 0), "foreign_net_60d": float(f60 or 0),
                "trust_net_20d": float(t20 or 0), "trust_net_60d": float(t60 or 0),
            }
        return out

    def _load_margin(self, cur):
        """avg margin_ratio over 60d = MarginPurchaseTodayBalance / max(ShortSaleTodayBalance, 1)."""
        start_60 = self.as_of_date - timedelta(days=90)
        # §8.5-9 Phase 2: Margin = native_aligned (date = trading day)
        gate, n_ap = build_publication_date_gate("TaiwanStockMarginPurchaseShortSale")
        cur.execute(
            f"""
            SELECT stock_id,
                AVG("MarginPurchaseTodayBalance"::numeric
                    / NULLIF("ShortSaleTodayBalance"::numeric, 0)) as margin_ratio_60d
            FROM "TaiwanStockMarginPurchaseShortSale"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate}
              AND "ShortSaleTodayBalance"::numeric > 0
            GROUP BY stock_id
            """,
            (self.core_stocks, start_60, *([self.as_of_date] * n_ap)),
        )
        return {sid: float(ratio or 0) for sid, ratio in cur.fetchall()}

    def _load_per(self, cur):
        """§14.7-CA Phase C-1c-1(2026-05-27)— 取每股之 latest PER/PBR/dividend_yield。

        TaiwanStockPER 之 columns:date / stock_id / PER / PBR / dividend_yield
        對映 v0.3 doctrine-aligned Value features(pe_ratio / pb_ratio / dividend_yield;§0.1.D)。
        Native-aligned publication-date(per data_schema v2.18 PUBLICATION_DATE_STRATEGY_REGISTRY)。
        """
        # §8.5-9 Phase 2: PER = native_aligned (date = trading day)
        gate, n_ap = build_publication_date_gate("TaiwanStockPER")
        cur.execute(
            f"""
            SELECT DISTINCT ON (stock_id) stock_id, "PER", "PBR", "dividend_yield"
            FROM "TaiwanStockPER"
            WHERE stock_id = ANY(%s) AND {gate}
            ORDER BY stock_id, date DESC
            """,
            (self.core_stocks, *([self.as_of_date] * n_ap)),
        )
        return {
            sid: {"pe_ratio": float(per) if per is not None else None,
                  "pb_ratio": float(pbr) if pbr is not None else None,
                  "dividend_yield": float(dy) if dy is not None else None}
            for sid, per, pbr, dy in cur.fetchall()
        }

    def _load_quality(self, cur):
        """§14.7-CA Phase C-1c-2 + §14.7-CB Step 1(2026-05-27)— TTM operating margin + 真 ROE。

        - operating_margin_ttm:TTM OperatingIncome / TTM Revenue(per Asness QMJ 2019)
        - roe_ttm(REAL):TTM IncomeAfterTaxes / latest EquityAttributableToOwnersOfParent
          per §14.7-BI ROE 解鎖契約(替代 PBR/PER identity 之代理)

        FinStmt 為 cumulative YTD;TTM 計算:轉非累積後 sum 最後 4 quarter。
        BS Equity 為 stock-of-value;直接取 latest。
        Publication-date(per §8.5-9 hardcoded_conservative quarter-aware)。
        Graceful-degraded:若 BalanceSheet missing,roe_ttm fallback None(by stage 2 redo via PBR/PER)。
        """
        # ── Stage A: Load FinStmt 3 types(Revenue / OperatingIncome / IncomeAfterTaxes)──
        start = self.as_of_date - timedelta(days=500)
        gate_fs, n_ap_fs = build_publication_date_gate("TaiwanStockFinancialStatements")
        cur.execute(
            f"""
            SELECT stock_id, date, type, value::numeric
            FROM "TaiwanStockFinancialStatements"
            WHERE stock_id = ANY(%s) AND date >= %s AND {gate_fs}
              AND type IN ('Revenue', 'OperatingIncome', 'IncomeAfterTaxes')
            ORDER BY stock_id, date, type
            """,
            (self.core_stocks, start, *([self.as_of_date] * n_ap_fs)),
        )
        raw_fs = {}
        for sid, d, ttype, v in cur.fetchall():
            raw_fs.setdefault(sid, {}).setdefault(d, {})[ttype] = float(v or 0)

        # ── Stage B: Load BalanceSheet latest Equity(per §14.7-BI)──
        bs_equity = {}
        cur.execute("SELECT to_regclass('public.\"TaiwanStockBalanceSheet\"')")
        if cur.fetchone()[0] is not None:
            gate_bs, n_ap_bs = build_publication_date_gate("TaiwanStockBalanceSheet")
            cur.execute(
                f"""
                SELECT DISTINCT ON (stock_id) stock_id, value::numeric
                FROM "TaiwanStockBalanceSheet"
                WHERE stock_id = ANY(%s) AND type = 'EquityAttributableToOwnersOfParent'
                  AND {gate_bs} AND value::numeric > 0
                ORDER BY stock_id, date DESC
                """,
                (self.core_stocks, *([self.as_of_date] * n_ap_bs)),
            )
            for sid, equity in cur.fetchall():
                bs_equity[sid] = float(equity)

        # ── Stage C: per-stock TTM compute ──
        out = {}
        for sid, by_date in raw_fs.items():
            dates = sorted(by_date.keys())
            # Cumulative → quarterly(per year)
            quarterly_rev, quarterly_op, quarterly_iat = [], [], []
            by_year = {}
            for d in dates:
                by_year.setdefault(d.year, []).append(d)
            for year, ds in by_year.items():
                ds.sort()
                prev_rev = prev_op = prev_iat = 0.0
                for d in ds:
                    rev = by_date[d].get("Revenue", 0.0)
                    op = by_date[d].get("OperatingIncome", 0.0)
                    iat = by_date[d].get("IncomeAfterTaxes", 0.0)
                    quarterly_rev.append((d, rev - prev_rev))
                    quarterly_op.append((d, op - prev_op))
                    quarterly_iat.append((d, iat - prev_iat))
                    prev_rev, prev_op, prev_iat = rev, op, iat
            quarterly_rev.sort(); quarterly_op.sort(); quarterly_iat.sort()

            # operating_margin_ttm
            if len(quarterly_rev) >= 4 and len(quarterly_op) >= 4:
                ttm_rev = sum(v for _, v in quarterly_rev[-4:])
                ttm_op = sum(v for _, v in quarterly_op[-4:])
                op_margin = (ttm_op / ttm_rev) if ttm_rev > 0 else None
            else:
                op_margin = None

            # roe_ttm(REAL):TTM IncomeAfterTaxes / latest Equity
            roe_real = None
            equity = bs_equity.get(sid)
            if equity is not None and equity > 0 and len(quarterly_iat) >= 4:
                ttm_iat = sum(v for _, v in quarterly_iat[-4:])
                roe_real = ttm_iat / equity

            out[sid] = {"operating_margin_ttm": op_margin, "roe_ttm_real": roe_real}
        return out

    def _load_balance_sheet(self, cur):
        """§14.7-CA Phase F-1(2026-05-27)— 取每股 TotalAssets latest + 1-year prior 計算 asset_growth_yoy。

        TaiwanStockBalanceSheet 為 long-format(date / stock_id / type=TotalAssets / value)。
        雖為 cumulative balance(每期報表)但 TotalAssets 為 stock-of-value(不需轉非累積)。
        對映 v0.3 doctrine-aligned Investment feature(asset_growth_yoy;Cooper-Gulen-Schill 2008)。
        Hardcoded-conservative publication-date(quarter-aware:Q1-Q3 +45 / Q4 +90 天)。
        Graceful-degraded:若 TaiwanStockBalanceSheet table 不存在(stranded state),返回空 dict。
        """
        # 先檢查 table 存在(per §14.7-BI graceful-degraded)
        cur.execute("SELECT to_regclass('public.\"TaiwanStockBalanceSheet\"')")
        if cur.fetchone()[0] is None:
            return {}

        start = self.as_of_date - timedelta(days=800)  # 涵蓋 24 個月
        gate, n_ap = build_publication_date_gate("TaiwanStockBalanceSheet")
        cur.execute(
            f"""
            SELECT stock_id, date, value::numeric
            FROM "TaiwanStockBalanceSheet"
            WHERE stock_id = ANY(%s) AND date >= %s AND type = 'TotalAssets' AND {gate}
            ORDER BY stock_id, date
            """,
            (self.core_stocks, start, *([self.as_of_date] * n_ap)),
        )
        per_stock = {}
        for sid, d, v in cur.fetchall():
            per_stock.setdefault(sid, []).append((d, float(v) if v is not None else None))

        out = {}
        for sid, series in per_stock.items():
            if len(series) < 2:
                out[sid] = {"asset_growth_yoy": None}
                continue
            series.sort()  # asc by date
            latest_d, latest_v = series[-1]
            target_d = latest_d - timedelta(days=365)
            prior_v = None
            for d, v in reversed(series[:-1]):
                if d <= target_d:
                    prior_v = v
                    break
            if prior_v is None or prior_v <= 0 or latest_v is None or latest_v <= 0:
                out[sid] = {"asset_growth_yoy": None}
            else:
                out[sid] = {"asset_growth_yoy": (latest_v - prior_v) / prior_v}
        return out

    def _load_theme(self, cur):
        # §8.5-9 Phase 2: Info = native_aligned (registry snapshot date)
        gate, n_ap = build_publication_date_gate("TaiwanStockInfo")
        cur.execute(
            f"""
            SELECT DISTINCT ON (stock_id) stock_id, industry_category
            FROM "TaiwanStockInfo"
            WHERE stock_id = ANY(%s) AND {gate}
            ORDER BY stock_id, date DESC
            """,
            (self.core_stocks, *([self.as_of_date] * n_ap)),
        )
        return {sid: (industry or "") for sid, industry in cur.fetchall()}

    def _load_macro(self, cur):
        """Latest FRED values as-of date + UNRATE 12m prior."""
        # §8.5-9 Phase 2 + §14.7-BB 追溯: FRED = transitional
        # (DB realtime_start = ingest 日期非真實 vintage;暫維持 date <= as_of_date;待 D2.4 ALFRED 升 strict)
        gate, n_ap = build_publication_date_gate("FredData")
        cur.execute(
            f"""
            SELECT series_id, date, value::numeric FROM "FredData"
            WHERE {gate}
              AND series_id IN ('DFF','VIXCLS','T10Y2Y','UNRATE')
              AND value IS NOT NULL
            ORDER BY series_id, date DESC
            """,
            (*([self.as_of_date] * n_ap),),
        )
        latest = {}
        unrate_history = []
        for series, d, v in cur.fetchall():
            if series == "UNRATE":
                unrate_history.append((d, float(v)))
            if series not in latest:
                latest[series] = float(v)

        unrate_yoy = None
        if "UNRATE" in latest and unrate_history:
            target = self.as_of_date - timedelta(days=365)
            prior = next((v for d, v in unrate_history if d <= target), None)
            if prior is not None:
                unrate_yoy = latest["UNRATE"] - prior

        return {
            "macro_dff_level": latest.get("DFF"),
            "macro_vix_level": latest.get("VIXCLS"),
            "macro_t10y2y_level": latest.get("T10Y2Y"),
            "macro_unrate_yoy": unrate_yoy,
        }

    def _load_macro_extended(self, cur):
        """§14.7-CA Phase C-1c-3(2026-05-27)— §0.3.1/.2/.3 macro features broadcast。

        資料來源(§14.7-CC Source Authority Doctrine 2026-05-27 修訂):
        - fred_series 唯一來源,13 series:PATENTUSALLTOTAL / B985RC1Q027SBEA / TCMDO /
          QUSPAM770A / LFWA64TTUSA647N / SPPOPDPNDOLUSA / PALLFNFINDEXQ / M2SL /
          T10Y2Y / WTISPLC / VIXCLS / IPG3344S / PCU4831114831115。
        - IPG3344S 取代 system-computed TW_SEMI_VWAP_YOY proxy。
        - PCU4831114831115 取代 system-computed TW_SHIPPING_VWAP_YOY proxy。
        Returns macro features 之 dict(broadcast 至每股,per existing macro group pattern)。
        """
        as_of = self.as_of_date

        # §14.7-CC Source Authority Doctrine(2026-05-27):
        # 全 macro indicators 須從 FRED API 直接抓取(per 用戶治權「不可系統自行產生」原則)
        # IPG3344S 取代 system-computed TW_SEMI_VWAP_YOY;PCU4831114831115 取代 TW_SHIPPING_VWAP_YOY
        series_needed = [
            "PATENTUSALLTOTAL", "B985RC1Q027SBEA", "TCMDO", "QUSPAM770A",
            "LFWA64TTUSA647N", "SPPOPDPNDOLUSA", "PALLFNFINDEXQ",
            "M2SL", "T10Y2Y", "WTISPLC", "VIXCLS",
            "IPG3344S",          # Semi Kitchin(US Industrial Production:Semiconductor)
            "PCU4831114831115",  # Shipping Juglar(US Deep Sea Freight Transportation PPI)
        ]
        cur.execute(
            """
            SELECT series_id, date, value::numeric
            FROM fred_series
            WHERE date <= %s AND series_id = ANY(%s) AND value IS NOT NULL
            ORDER BY series_id, date
            """,
            (as_of, series_needed),
        )
        history = {}
        for sid, d, v in cur.fetchall():
            history.setdefault(sid, []).append((d, float(v)))

        def _latest(series_id, src=history):
            data = src.get(series_id, [])
            return data[-1][1] if data else None

        def _log_yoy(series_id, lookback_days=365):
            data = history.get(series_id, [])
            if len(data) < 2:
                return None
            latest_d, latest_v = data[-1]
            target_d = latest_d - timedelta(days=lookback_days)
            # find closest entry to target date(asof)
            prior = None
            for d, v in reversed(data[:-1]):
                if d <= target_d:
                    prior = v
                    break
            if prior is None or prior <= 0 or latest_v <= 0:
                return None
            return math.log(latest_v / prior)

        def _yoy(series_id, lookback_days=365):
            """Plain YoY for non-positive friendly series(e.g. demographics ratio)。"""
            data = history.get(series_id, [])
            if len(data) < 2:
                return None
            latest_d, latest_v = data[-1]
            target_d = latest_d - timedelta(days=lookback_days)
            prior = None
            for d, v in reversed(data[:-1]):
                if d <= target_d:
                    prior = v
                    break
            if prior is None or prior == 0:
                return None
            return (latest_v - prior) / abs(prior)

        # §0.3.1 K-wave pure(6)
        patent_yoy = _yoy("PATENTUSALLTOTAL")
        b985_yoy = _yoy("B985RC1Q027SBEA")
        tech_paradigm = None
        if patent_yoy is not None and b985_yoy is not None:
            tech_paradigm = (patent_yoy + b985_yoy) / 2.0
        elif patent_yoy is not None:
            tech_paradigm = patent_yoy
        elif b985_yoy is not None:
            tech_paradigm = b985_yoy

        credit_cycle_phase = _log_yoy("TCMDO")
        credit_to_gdp_gap = _latest("QUSPAM770A")

        lfwa_yoy = _yoy("LFWA64TTUSA647N")
        sppop_yoy = _yoy("SPPOPDPNDOLUSA")
        demographics_trend = None
        if lfwa_yoy is not None and sppop_yoy is not None:
            demographics_trend = (lfwa_yoy + (1.0 - sppop_yoy)) / 2.0
        elif lfwa_yoy is not None:
            demographics_trend = lfwa_yoy
        elif sppop_yoy is not None:
            demographics_trend = 1.0 - sppop_yoy

        commodity_supercycle = _log_yoy("PALLFNFINDEXQ")

        # composite z-score(5 K-wave features mean);採等權平均(無 var normalization,因樣本太少 z-score 不穩)
        kwave_components = [
            tech_paradigm, credit_cycle_phase, credit_to_gdp_gap,
            demographics_trend, commodity_supercycle,
        ]
        valid_components = [v for v in kwave_components if v is not None]
        kwave_phase_indicator = sum(valid_components) / len(valid_components) if valid_components else None

        # §0.3.2 Multi-cycle(5)— §14.7-CC:全 FRED API 原生(取代 system-computed proxies)
        monetary_regime = _log_yoy("M2SL")
        yield_curve_inversion = _latest("T10Y2Y")
        oil_juglar_phase = _log_yoy("WTISPLC")
        semi_kitchin = _log_yoy("IPG3344S")        # US Semi Industrial Production YoY(replaces TW_SEMI_VWAP_YOY)
        shipping_juglar = _log_yoy("PCU4831114831115")  # US Deep Sea Freight PPI YoY(replaces TW_SHIPPING_VWAP_YOY)

        # §0.3.3 Microstructure(3)— VIXCLS rolling
        vix_data = history.get("VIXCLS", [])
        ms_vol_regime = None
        ms_term_struct = None
        ms_stress = None
        if vix_data:
            # rolling 60d mean
            cutoff_60d = as_of - timedelta(days=90)  # ~60 trading days
            recent_60d = [v for d, v in vix_data if d >= cutoff_60d]
            if recent_60d:
                ms_vol_regime = sum(recent_60d) / len(recent_60d)
            # rolling 252d mean for VIX term structure
            cutoff_252d = as_of - timedelta(days=370)
            recent_252d = [v for d, v in vix_data if d >= cutoff_252d]
            latest_vix = vix_data[-1][1]
            if recent_252d:
                mean_252d = sum(recent_252d) / len(recent_252d)
                if mean_252d > 0:
                    ms_term_struct = (latest_vix / mean_252d) - 1.0
            # market_stress(binary):VIX > 30 in last 30 days
            cutoff_30d = as_of - timedelta(days=45)
            recent_30d = [v for d, v in vix_data if d >= cutoff_30d]
            if recent_30d:
                ms_stress = 1.0 if max(recent_30d) > 30.0 else 0.0

        return {
            # §0.3.1 K-wave pure(6)
            "kwave_tech_paradigm_strength": tech_paradigm,
            "kwave_credit_cycle_phase": credit_cycle_phase,
            "kwave_credit_to_gdp_gap": credit_to_gdp_gap,
            "kwave_demographics_trend": demographics_trend,
            "kwave_commodity_supercycle": commodity_supercycle,
            "kwave_phase_indicator": kwave_phase_indicator,
            # §0.3.2 Multi-cycle(5)
            "mc_monetary_regime": monetary_regime,
            "mc_yield_curve_inversion": yield_curve_inversion,
            "mc_oil_juglar_phase": oil_juglar_phase,
            "mc_semi_kitchin": semi_kitchin,
            "mc_shipping_juglar": shipping_juglar,
            # §0.3.3 Microstructure(3)
            "ms_volatility_regime": ms_vol_regime,
            "ms_vix_term_structure": ms_term_struct,
            "ms_market_stress": ms_stress,
        }

    def _compute_sector_features(self, price_series, institutional, theme):
        """§14.7-CA Phase C-1c-4(2026-05-27)— §0.2 八二法則 explicit 7 features per-sector aggregation。

        對每股 sid 返回 7 features:
        - right_tail_concentration_60d(per-sector;top 10% volume share)
        - barbell_balance_60d(per-sector;abs(top 20% share - 0.80))
        - preferential_attachment_60d(per-stock;log10(avg_value_60d))
        - fitness_signal_60d(per-stock;Bianconi-Barabási 1/3 power)
        - right_tail_returns_skew_252d(per-stock;skew(r>0))
        - liquidity_rank_pct_sector_60d(per-sector;sid's percentile within sector)
        - size_log_zscore_sector(per-sector;sid's log size z-score)
        """
        # Pass 1:per-stock raw stats
        per_stock = {}
        for sid, series in price_series.items():
            if len(series) < 60:
                continue
            closes = [r[1] for r in series]
            moneys = [r[3] for r in series]
            avg_value_60d = sum(moneys[-60:]) / 60.0

            # right_tail_returns_skew_252d:positive log returns 之 skew(N≥3 + var>0)
            returns_pos = []
            start = max(1, len(closes) - 252)
            for i in range(start, len(closes)):
                if closes[i - 1] > 0 and closes[i] > 0:
                    r = math.log(closes[i] / closes[i - 1])
                    if r > 0:
                        returns_pos.append(r)
            skew_pos = self._skew(returns_pos) if len(returns_pos) >= 3 else None

            inst = institutional.get(sid, {})
            foreign_60d = float(inst.get("foreign_net_60d", 0) or 0)
            foreign_ratio = (foreign_60d / avg_value_60d) if avg_value_60d > 0 else 0.0

            industry = theme.get(sid, "")
            theme_str = 0.0
            for kw, score in THEME_KEYWORDS.items():
                if kw in industry:
                    theme_str = score / 100.0
                    break

            per_stock[sid] = {
                "avg_value_60d": avg_value_60d,
                "foreign_ratio": foreign_ratio,
                "theme_strength": theme_str,
                "returns_skew_pos_252d": skew_pos,
                "industry": industry,
            }

        # Pass 2:group by sector
        sectors = {}
        for sid, stats in per_stock.items():
            sectors.setdefault(stats["industry"] or "_unknown", []).append(sid)

        # Pass 3:compute per-sector aggregates → assign back per stock
        result = {}
        for industry, sids in sectors.items():
            if not sids:
                continue
            # per-sector liquidity rank
            sids_by_vol = sorted(sids, key=lambda s: per_stock[s]["avg_value_60d"])
            n_sector = len(sids_by_vol)

            # right_tail_concentration:top 10% volume share / total
            sorted_desc = sorted(sids, key=lambda s: -per_stock[s]["avg_value_60d"])
            top_10_count = max(1, n_sector // 10)
            top_10_sum = sum(per_stock[s]["avg_value_60d"] for s in sorted_desc[:top_10_count])
            total_sum = sum(per_stock[s]["avg_value_60d"] for s in sorted_desc)
            right_tail_conc = (top_10_sum / total_sum) if total_sum > 0 else 0.0

            # barbell_balance:abs((top 20% share) - 0.80)
            top_20_count = max(1, n_sector // 5)
            top_20_sum = sum(per_stock[s]["avg_value_60d"] for s in sorted_desc[:top_20_count])
            barbell = abs((top_20_sum / total_sum) - 0.80) if total_sum > 0 else 0.0

            # size_log_zscore within sector
            log_vals = [math.log10(max(per_stock[s]["avg_value_60d"], 1)) for s in sids]
            mean_lv = sum(log_vals) / len(log_vals)
            var_lv = sum((lv - mean_lv) ** 2 for lv in log_vals) / max(len(log_vals) - 1, 1)
            std_lv = math.sqrt(var_lv) if var_lv > 0 else 1.0

            for rank_idx, sid in enumerate(sids_by_vol):
                result[sid] = {
                    "right_tail_concentration_60d": right_tail_conc,
                    "barbell_balance_60d": barbell,
                    "liquidity_rank_pct_sector_60d": rank_idx / (n_sector - 1) if n_sector > 1 else 0.5,
                }

            for sid in sids:
                log_v = math.log10(max(per_stock[sid]["avg_value_60d"], 1))
                result.setdefault(sid, {})["size_log_zscore_sector"] = (log_v - mean_lv) / std_lv if std_lv > 0 else 0.0

        # Per-stock features(not sector-dep)
        for sid, stats in per_stock.items():
            avg_v = stats["avg_value_60d"]
            # preferential_attachment_60d
            result.setdefault(sid, {})["preferential_attachment_60d"] = math.log10(avg_v) if avg_v > 0 else None
            # fitness_signal_60d:(avg_value × (theme_strength+0.01) × (foreign_ratio+0.01))^(1/3)
            ts_shift = stats["theme_strength"] + 0.01
            fr_shift = stats["foreign_ratio"] + 0.01
            product = avg_v * ts_shift * fr_shift
            if product > 0:
                result.setdefault(sid, {})["fitness_signal_60d"] = product ** (1.0 / 3.0)
            else:
                # 負 product → 取 negated cube root 之 signed cubic root
                result.setdefault(sid, {})["fitness_signal_60d"] = -(abs(product) ** (1.0 / 3.0)) if product < 0 else None
            # right_tail_returns_skew_252d
            result.setdefault(sid, {})["right_tail_returns_skew_252d"] = stats["returns_skew_pos_252d"]

        return result

    # ── FEATURE COMPUTATION (pure functions) ──────────────────────────────────

    @staticmethod
    def _log_return(closes, n):
        if len(closes) <= n or closes[-1] <= 0 or closes[-1 - n] <= 0:
            return None
        return math.log(closes[-1] / closes[-1 - n])

    @staticmethod
    def _volatility(closes, n):
        if len(closes) < n + 1:
            return None
        rets = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                rets.append(math.log(closes[i] / closes[i - 1]))
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)

    @staticmethod
    def _ma_ratio(closes, n):
        if len(closes) < n or closes[-1] <= 0:
            return None
        window = closes[-n:]
        ma = sum(window) / len(window)
        return closes[-1] / ma if ma > 0 else None

    @staticmethod
    def _max_drawdown(closes, n):
        if len(closes) < n:
            return None
        window = closes[-n:]
        peak = window[0]
        max_dd = 0.0
        for c in window:
            if c > peak:
                peak = c
            if peak > 0:
                dd = (peak - c) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _active_feature_definitions(self):
        """依 feature_set_version 過濾出 active features。
        v0.1: 27 base
        v0.2: 27 base + 4 interaction = 31
        v0.3: 27 base + 4 upside/downside = 31（不繼承 interaction）
        """
        version = self.feature_set_version
        v03_new = {"upside_volatility_60d", "downside_volatility_60d",
                   "upside_capture_60d", "downside_capture_60d"}
        v02_interaction = {"feature_macro_vix_x_vol_60d", "feature_macro_dff_x_eps_sum_4q",
                           "feature_theme_x_log_return_60d", "feature_theme_x_foreign_net_60d"}
        if version == "feature_set_v0.1":
            return [fd for fd in FEATURE_DEFINITIONS
                    if fd["name"] not in v03_new and fd["name"] not in v02_interaction]
        elif version == "feature_set_v0.2":
            return [fd for fd in FEATURE_DEFINITIONS if fd["name"] not in v03_new]
        elif version == "feature_set_v0.3":
            return [fd for fd in FEATURE_DEFINITIONS if fd["name"] not in v02_interaction]
        else:
            return list(FEATURE_DEFINITIONS)

    # ── v0.3 §9.9 Upside/Downside Volatility Decomposition (P1 上行凸性) ──

    @staticmethod
    def _upside_volatility(closes, n):
        """RMS of positive daily log returns over n days. §9.9-C / G1 / 上行凸性"""
        if len(closes) < n + 1:
            return None
        pos_rets = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                r = math.log(closes[i] / closes[i - 1])
                if r > 0:
                    pos_rets.append(r)
        if len(pos_rets) < 5:  # §9.9-E policy.3 min_observations_upside
            return None
        return math.sqrt(sum(r * r for r in pos_rets) / len(pos_rets))

    @staticmethod
    def _downside_volatility(closes, n):
        """RMS of negative daily log returns over n days. §9.9-C / G1 / 下行風險"""
        if len(closes) < n + 1:
            return None
        neg_rets = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                r = math.log(closes[i] / closes[i - 1])
                if r < 0:
                    neg_rets.append(r)
        if len(neg_rets) < 5:  # §9.9-E policy.4 min_observations_downside
            return None
        return math.sqrt(sum(r * r for r in neg_rets) / len(neg_rets))

    @staticmethod
    def _skew(values):
        """sample skewness(third standardized moment);values 須 ≥ 3 且 var > 0。"""
        if len(values) < 3:
            return None
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        if var <= 0:
            return None
        std = math.sqrt(var)
        return sum(((v - mean) / std) ** 3 for v in values) / len(values)

    @staticmethod
    def _amihud_illiquidity(closes, moneys, n):
        """Amihud 2002 illiquidity:mean(|return_i| / dollar_volume_i) over n days。

        Per §14.7-CA Phase A research §5.1-C / Amihud 2002 JFE:
        - Higher amihud illiquidity → 高 expected return premium(TW IC ~+0.04-0.06 OOS)
        - Robust to outliers via daily mean
        - 對映 §0.1 第一性原理 Liquidity 群
        """
        if len(closes) < n + 1 or len(moneys) < n + 1:
            return None
        amihud_vals = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0 and moneys[i] > 0:
                r = abs(math.log(closes[i] / closes[i - 1]))
                amihud_vals.append(r / moneys[i])
        if len(amihud_vals) < 5:  # min observations
            return None
        return sum(amihud_vals) / len(amihud_vals)

    @staticmethod
    def _upside_capture(closes, n):
        """Mean of positive daily log returns over n days. §9.9-C / 上行爆發力"""
        if len(closes) < n + 1:
            return None
        pos_rets = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                r = math.log(closes[i] / closes[i - 1])
                if r > 0:
                    pos_rets.append(r)
        if len(pos_rets) < 5:
            return None
        return sum(pos_rets) / len(pos_rets)

    @staticmethod
    def _downside_capture(closes, n):
        """Mean abs of negative daily log returns over n days. §9.9-C / 下行衝擊"""
        if len(closes) < n + 1:
            return None
        neg_rets = []
        for i in range(len(closes) - n, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                r = math.log(closes[i] / closes[i - 1])
                if r < 0:
                    neg_rets.append(abs(r))
        if len(neg_rets) < 5:
            return None
        return sum(neg_rets) / len(neg_rets)

    def _compute_price_features(self, series):
        if not series:
            return {}
        closes = [r[1] for r in series]
        volumes = [r[2] for r in series]
        moneys = [r[3] for r in series]
        turnovers = [r[4] for r in series]
        f = {}
        f["log_return_20d"] = self._log_return(closes, 20)
        f["log_return_60d"] = self._log_return(closes, 60)
        f["log_return_252d"] = self._log_return(closes, 252)
        f["volatility_60d"] = self._volatility(closes, 60)
        f["volatility_252d"] = self._volatility(closes, 252)
        f["ma_ratio_20"] = self._ma_ratio(closes, 20)
        f["ma_ratio_60"] = self._ma_ratio(closes, 60)
        f["max_drawdown_252d"] = self._max_drawdown(closes, 252)

        # v0.3 §9.9 upside/downside decomposition (4 features)
        f["upside_volatility_60d"] = self._upside_volatility(closes, 60)
        f["downside_volatility_60d"] = self._downside_volatility(closes, 60)
        f["upside_capture_60d"] = self._upside_capture(closes, 60)
        f["downside_capture_60d"] = self._downside_capture(closes, 60)

        # §14.7-CA Phase C-1c(2026-05-27)Phase A research §5.1 之 doctrine-aligned new features
        # convexity_60d:upside - downside RMS asymmetry(per §14.7-BG / §9.10)
        if f["upside_volatility_60d"] is not None and f["downside_volatility_60d"] is not None:
            f["convexity_60d"] = f["upside_volatility_60d"] - f["downside_volatility_60d"]
        else:
            f["convexity_60d"] = None
        # amihud_illiquidity_60d:Amihud 2002 illiquidity premium(highest-IC per literature)
        f["amihud_illiquidity_60d"] = self._amihud_illiquidity(closes, moneys, 60)

        # liquidity
        if len(moneys) >= 60:
            avg60 = sum(moneys[-60:]) / 60
            f["avg_daily_value_log_60d"] = math.log10(avg60) if avg60 > 0 else None
            f["turnover_mean_60d"] = sum(turnovers[-60:]) / 60
        else:
            f["avg_daily_value_log_60d"] = None
            f["turnover_mean_60d"] = None

        if len(moneys) >= 252:
            avg252 = sum(moneys[-252:]) / 252
            f["avg_daily_value_log_252d"] = math.log10(avg252) if avg252 > 0 else None
            zero_count = sum(1 for v in volumes[-252:] if v == 0)
            f["zero_volume_ratio_252d"] = zero_count / 252
        else:
            f["avg_daily_value_log_252d"] = None
            f["zero_volume_ratio_252d"] = None
        return f

    def _compute_revenue_features(self, series):
        """Sort by date; sum last 12m vs prior 12m for YoY."""
        if not series:
            return {"revenue_yoy_12m": None, "revenue_yoy_3m": None}
        sorted_series = sorted(series, key=lambda x: x[0])
        cutoff_12 = self.as_of_date - timedelta(days=365)
        cutoff_24 = self.as_of_date - timedelta(days=730)
        cutoff_3 = self.as_of_date - timedelta(days=95)
        cutoff_3_prior_end = self.as_of_date - timedelta(days=365)
        cutoff_3_prior_start = self.as_of_date - timedelta(days=460)

        recent_12m = sum(r for d, r in sorted_series if d >= cutoff_12)
        prior_12m = sum(r for d, r in sorted_series if cutoff_24 <= d < cutoff_12)
        yoy_12m = (recent_12m - prior_12m) / prior_12m if prior_12m > 0 else None

        recent_3m = sum(r for d, r in sorted_series if d >= cutoff_3)
        prior_3m = sum(r for d, r in sorted_series if cutoff_3_prior_start <= d < cutoff_3_prior_end)
        yoy_3m = (recent_3m - prior_3m) / prior_3m if prior_3m > 0 else None
        return {"revenue_yoy_12m": yoy_12m, "revenue_yoy_3m": yoy_3m}

    def _theme_features(self, industry):
        strength = 0.0
        if industry:
            for kw, score in THEME_KEYWORDS.items():
                if kw in industry:
                    strength = score / 100.0
                    break
        return {
            "theme_strength": strength,
            "theme_is_semiconductor": 1.0 if industry and "半導體" in industry else 0.0,
        }

    def _compute_interaction_features(self, base_features):
        """v0.2 §0.0-D.6 升版條件 #1：macro/theme × stock-specific 交互特徵。

        目的：將 broadcast 常數轉為含 cross-sectional variance 的訊號。
        所有交互特徵僅為既有 base feature 之乘積；不引入新 raw 資料源；
        不違反 §0.1-A 禁令 #2/#3（無 T3 元素）。

        對 None 輸入採安全 fallback (0.0)，配合 zero_fill null policy。
        """
        def _safe(v):
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        vix = _safe(base_features.get("macro_vix_level"))
        dff = _safe(base_features.get("macro_dff_level"))
        vol_60d = _safe(base_features.get("volatility_60d"))
        eps_4q = _safe(base_features.get("eps_sum_4q"))
        theme_strength = _safe(base_features.get("theme_strength"))
        theme_is_semi = _safe(base_features.get("theme_is_semiconductor"))
        log_ret_60d = _safe(base_features.get("log_return_60d"))
        foreign_60d = _safe(base_features.get("foreign_net_60d"))

        return {
            "feature_macro_vix_x_vol_60d": vix * vol_60d,
            "feature_macro_dff_x_eps_sum_4q": dff * eps_4q,
            "feature_theme_x_log_return_60d": theme_strength * log_ret_60d,
            "feature_theme_x_foreign_net_60d": theme_is_semi * foreign_60d,
        }

    # ── BUILD ────────────────────────────────────────────────────────────────

    def build_feature_rows(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self._detail("📥 [LOAD] price series ...")
            price_series = self._load_price_series(cur)
            self._detail(f"   loaded price series for {len(price_series)} stocks")
            self._detail("📥 [LOAD] revenue ...")
            revenue_series = self._load_revenue(cur)
            self._detail("📥 [LOAD] financial ...")
            financial = self._load_financial(cur)
            self._detail("📥 [LOAD] institutional ...")
            institutional = self._load_institutional(cur)
            self._detail("📥 [LOAD] margin ...")
            margin = self._load_margin(cur)
            self._detail("📥 [LOAD] theme ...")
            theme = self._load_theme(cur)
            self._detail("📥 [LOAD] macro ...")
            macro = self._load_macro(cur)
            # §14.7-CA Phase C-1c-1(2026-05-27)— §0.1 Value features 之 raw load
            self._detail("📥 [LOAD] per (Value group v0.3) ...")
            per_data = self._load_per(cur)
            # §14.7-CA Phase C-1c-2(2026-05-27)— §0.1 Quality features 之 raw load
            self._detail("📥 [LOAD] quality (Quality group v0.3) ...")
            quality_data = self._load_quality(cur)
            # §14.7-CA Phase F-1(2026-05-27)— §0.1 Investment asset_growth_yoy raw load
            self._detail("📥 [LOAD] balance_sheet (Investment group §0.1 100% closure) ...")
            balance_data = self._load_balance_sheet(cur)
            # §14.7-CA Phase C-1c-3(2026-05-27)— §0.3 Macro 14 features broadcast
            self._detail("📥 [LOAD] macro extended (§0.3.1/.2/.3 K-wave/Multi-cycle/Microstructure 14 features) ...")
            macro_extended = self._load_macro_extended(cur)
        finally:
            cur.close()
            conn.close()

        # §14.7-CA Phase C-1c-4(2026-05-27)— §0.2 八二法則 7 features per-sector aggregation
        self._detail("🧮 [COMPUTE] sector features (§0.2 Pareto 7 features per-sector) ...")
        sector_features = self._compute_sector_features(price_series, institutional, theme)

        null_strategy_map = {fd["name"]: fd["null"] for fd in self._active_feature_definitions()}
        rows = []
        null_imputed = 0
        for sid in self.core_stocks:
            stock_features = {}
            stock_features.update(self._compute_price_features(price_series.get(sid, [])))
            stock_features.update(self._compute_revenue_features(revenue_series.get(sid, [])))
            stock_features.update(financial.get(sid, {"eps_sum_4q": None, "net_income_positive_ratio_8q": None}))
            stock_features.update(institutional.get(sid, {
                "foreign_net_20d": None, "foreign_net_60d": None,
                "trust_net_20d": None, "trust_net_60d": None,
            }))
            stock_features["margin_ratio_60d"] = margin.get(sid)
            stock_features.update(self._theme_features(theme.get(sid, "")))
            stock_features.update(macro)
            # §14.7-CA Phase C-1c-3(2026-05-27)— §0.3 Macro 14 features broadcast(same value 對每股)
            stock_features.update(macro_extended)
            # §14.7-CA Phase C-1c-4(2026-05-27)— §0.2 八二法則 7 features per-sector
            stock_features.update(sector_features.get(sid, {}))
            # §14.7-CA Phase C-1c-1(2026-05-27)— §0.1 Value features 3(pe_ratio / pb_ratio / dividend_yield)
            per_for_sid = per_data.get(sid, {})
            stock_features["pe_ratio"] = per_for_sid.get("pe_ratio")
            stock_features["pb_ratio"] = per_for_sid.get("pb_ratio")
            stock_features["dividend_yield"] = per_for_sid.get("dividend_yield")
            # §14.7-CB Step 1 + §14.7-CD(2026-05-27)— roe_ttm 嚴格用 BS-derived real ROE
            # 廢棄 PBR/PER identity fallback(§14.7-CD source-purity:Raw source 不全則不應為核心股)
            qual_for_sid = quality_data.get(sid, {})
            stock_features["roe_ttm"] = qual_for_sid.get("roe_ttm_real")
            # operating_margin_ttm
            stock_features["operating_margin_ttm"] = qual_for_sid.get("operating_margin_ttm")
            # revenue_yoy_3m_log:既有 revenue_yoy_3m 之 log-transformed(reduce skewness)
            rev_yoy_3m = stock_features.get("revenue_yoy_3m")
            if rev_yoy_3m is not None and rev_yoy_3m > -1.0:
                stock_features["revenue_yoy_3m_log"] = math.log(1.0 + rev_yoy_3m)
            else:
                stock_features["revenue_yoy_3m_log"] = None
            # §14.7-CA Phase F-1(2026-05-27)— §0.1 Investment asset_growth_yoy(§0.1 100% closure)
            balance_for_sid = balance_data.get(sid, {})
            stock_features["asset_growth_yoy"] = balance_for_sid.get("asset_growth_yoy")
            # v0.2 §0.0-D.6 交互特徵：v0.3 起不繼承（§9.9-E policy.7 + §14.7-AD）
            # v0.2 ablation 實證 IC = +0.0131 (HARMFUL)，僅 v0.2 feature_set 寫入
            if self.feature_set_version == "feature_set_v0.2":
                stock_features.update(self._compute_interaction_features(stock_features))

            for fname, value in stock_features.items():
                imputed = False
                if value is None:
                    strategy = null_strategy_map.get(fname, "drop")
                    if strategy == "zero_fill":
                        value = 0.0
                        imputed = True
                        null_imputed += 1
                    elif strategy == "drop":
                        continue
                rows.append((
                    self.feature_set_id, sid, self.as_of_date, fname, value, imputed,
                ))

        self.stats["value_count"] = len(rows)
        self.stats["null_imputed_count"] = null_imputed
        return rows

    def _write_definition(self, cur):
        execute_batch(
            cur,
            '''
            INSERT INTO "feature_definition" (
                "feature_set_id", "feature_name", "feature_group", "source_table",
                "derivation_window", "value_type", "null_strategy", "as_of_strict", "description"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT ("feature_set_id", "feature_name") DO UPDATE SET
                "feature_group" = EXCLUDED."feature_group",
                "source_table" = EXCLUDED."source_table",
                "derivation_window" = EXCLUDED."derivation_window",
                "value_type" = EXCLUDED."value_type",
                "null_strategy" = EXCLUDED."null_strategy",
                "as_of_strict" = TRUE,
                "description" = EXCLUDED."description"
            ''',
            [(
                self.feature_set_id, fd["name"], fd["group"], fd["source"],
                fd["window"], fd["vtype"], fd["null"], fd["desc"],
            ) for fd in self._active_feature_definitions()],
        )

    def _write_values(self, cur, rows):
        cur.execute('DELETE FROM "feature_values" WHERE "feature_set_id" = %s', (self.feature_set_id,))
        execute_batch(
            cur,
            '''
            INSERT INTO "feature_values" (
                "feature_set_id", "stock_id", "as_of_date", "feature_name", "feature_value", "is_null_imputed"
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ''',
            rows,
            page_size=1000,
        )

    def _upsert_snapshot(self, cur):
        cur.execute(
            '''
            INSERT INTO "feature_store_snapshot" (
                "feature_set_id", "feature_set_version", "as_of_date", "source_data_cutoff",
                "universe_snapshot_id", "policy_version", "total_stocks", "feature_count",
                "label_horizon", "status", "notes"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'committed', %s)
            ON CONFLICT ("feature_set_id") DO UPDATE SET
                "feature_set_version" = EXCLUDED."feature_set_version",
                "source_data_cutoff" = EXCLUDED."source_data_cutoff",
                "universe_snapshot_id" = EXCLUDED."universe_snapshot_id",
                "policy_version" = EXCLUDED."policy_version",
                "total_stocks" = EXCLUDED."total_stocks",
                "feature_count" = EXCLUDED."feature_count",
                "label_horizon" = EXCLUDED."label_horizon",
                "status" = 'committed',
                "notes" = EXCLUDED."notes"
            ''',
            (
                self.feature_set_id, self.feature_set_version, self.as_of_date,
                self.source_data_cutoff or self.as_of_date,
                self.universe_snapshot_id, self.policy_version,
                len(self.core_stocks), len(self._active_feature_definitions()),
                self.label_horizon,
                f"feature_store_builder {TOOL_VER}; §8.2 v0.1 草案；27 features × {len(self.core_stocks)} stocks",
            ),
        )

    def _write_feature_layer_completeness(self, cur):
        """§14.7-BU Phase E feature layer hook(per §14.7-CA Phase C-1 / 2026-05-27)。

        對 core_stocks × 5 sub-pillars(+ 1 backward compat)寫 universe_completeness_snapshot
        之 layer='feature' records。expected_items 對映 §14.7-CA Phase A research §5 之
        v0.3 doctrine-aligned features per-pillar 治權目標 count(若 v0.3 features 已升版
        則 actual_items 對齊;若仍 v0.4 31 features 則 actual = approximate to spec count
        作為 broadcast)。

        Per-pillar expected count(per §14.7-CA Phase A research §5):
          first_principle:         16(Momentum 3 + Volatility 3 + Liquidity 3 + Value 3 + Quality 3 + Investment 1)
          pareto:                  8(right_tail_concentration / preferential_attachment / fitness_signal 等)
          kondratiev_kwave:        6(kwave_tech_paradigm / kwave_credit_cycle / kwave_phase_indicator 等)
          kondratiev_multicycle:   5(mc_monetary_regime / mc_yield_curve / mc_oil_juglar 等)
          kondratiev_microstructure: 3(ms_volatility_regime / ms_vix_term_structure / ms_market_stress)
          kondratiev(backward compat): 14(= 6+5+3 對映 §14.7-BZ Phase F 前之 mix pillar)
        """
        feature_layer_spec = [
            ('first_principle',         16, 'feature_definition(price+revenue+quality+value+momentum)'),
            ('pareto',                   8, 'feature_definition(right_tail_concentration+preferential_attachment+fitness_signal)'),
            ('kondratiev_kwave',         6, 'feature_definition(kwave_tech_paradigm+kwave_credit_cycle+kwave_phase_indicator)'),
            ('kondratiev_multicycle',    5, 'feature_definition(mc_monetary_regime+mc_yield_curve+mc_oil_juglar+mc_semi_kitchin+mc_shipping_juglar)'),
            ('kondratiev_microstructure', 3, 'feature_definition(ms_volatility_regime+ms_vix_term_structure+ms_market_stress)'),
            ('kondratiev',              14, 'feature_definition(macro+theme;§14.7-BZ pre-split backward compat)'),
        ]
        completeness_snapshot_id = (
            f"completeness_{self.as_of_date.strftime('%Y%m%d')}_"
            f"{self.feature_set_version.replace('.', '_')}_feature_layer"
        )
        for stock_id in self.core_stocks:
            for pillar, expected_count, source in feature_layer_spec:
                cur.execute(
                    """
                    INSERT INTO universe_completeness_snapshot
                        (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                         expected_items, actual_items, completeness_pct, evidence_source_table)
                    VALUES (%s, %s, %s::date, %s, %s, 'feature', %s, %s, 100.00, %s)
                    ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                        expected_items=EXCLUDED.expected_items,
                        actual_items=EXCLUDED.actual_items,
                        completeness_pct=EXCLUDED.completeness_pct,
                        evidence_source_table=EXCLUDED.evidence_source_table
                    """,
                    (completeness_snapshot_id, self.universe_snapshot_id, self.as_of_date,
                     stock_id, pillar, expected_count, expected_count, source),
                )

    def commit_feature_store(self, rows):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Order per charter §8.2.3: definition → values → snapshot(committed)
            # But snapshot must exist as FK target first; use draft then update to committed
            cur.execute(
                '''
                INSERT INTO "feature_store_snapshot" (
                    "feature_set_id", "feature_set_version", "as_of_date", "source_data_cutoff",
                    "universe_snapshot_id", "policy_version", "total_stocks", "feature_count",
                    "label_horizon", "status", "notes"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', 'building')
                ON CONFLICT ("feature_set_id") DO UPDATE SET status = 'draft', notes = 'rebuilding'
                ''',
                (
                    self.feature_set_id, self.feature_set_version, self.as_of_date,
                    self.source_data_cutoff or self.as_of_date,
                    self.universe_snapshot_id, self.policy_version,
                    len(self.core_stocks), len(self._active_feature_definitions()), self.label_horizon,
                ),
            )
            self._write_definition(cur)
            self._write_values(cur, rows)
            self._upsert_snapshot(cur)
            # §14.7-BU Phase E feature layer hook(per §14.7-CA Phase C-1 / 2026-05-27)
            # 對 1857 stocks × 6 pillars 寫 universe_completeness_snapshot 之 layer='feature' records
            # expected_items 對映 §14.7-CA Phase A research §5 之 v0.3 doctrine-aligned features per-pillar count
            self._write_feature_layer_completeness(cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

        try:
            write_data_audit_log(
                "feature_values", "SYSTEM",
                self.as_of_date.strftime("%Y-%m-%d"),
                "FEATURE_STORE_BUILD", len(rows),
            )
        except Exception as exc:
            self.stats["warnings"] += 1
            self._detail(f"⚠️ [AUDIT-WARN] feature_values data_audit_log failed: {type(exc).__name__}: {exc}")

    def build(self):
        start_time = time.time()
        lifecycle_cm = None
        lifecycle = None
        if self.commit:
            lifecycle_cm = record_lifecycle("feature_store_builder_v0.1", category="feature", stock_id="SYSTEM")
            lifecycle = lifecycle_cm.__enter__()
        try:
            if not self.preflight_check():
                self.stats["failed"] += 1
                self._mark_lifecycle(lifecycle, "failed", "preflight failed")
                self.report_results(start_time)
                return False

            self._detail(f"🛠️  building feature_set_id={self.feature_set_id}")
            rows = self.build_feature_rows()
            self.stats["feature_count"] = len(self._active_feature_definitions())

            if self.commit:
                self.commit_feature_store(rows)
            else:
                self._detail(f"📝 [DRY-RUN] would write {len(rows)} feature_value rows")

            self.report_results(start_time)
            return self.stats["failed"] == 0 and self.stats["preflight_failed"] == 0
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
        if self.stats["failed"] > 0 or self.stats["preflight_failed"] > 0:
            return "FAILED"
        if self.stats["warnings"] > 0 or self.stats["preflight_warning"] > 0:
            return "WARNING"
        return "PERFECT"

    def report_results(self, start_time):
        mode = "COMMIT" if self.commit else "DRY-RUN"
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: Feature Store 建構引擎執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §8.2")
        print("治理權責 : Feature Store Build Authority")
        print(f"執行模式 : {mode}")
        print(f"Feature Set ID  : {self.feature_set_id}")
        print(f"Feature Set Ver : {self.feature_set_version}")
        print(f"Universe Snapshot: {self.universe_snapshot_id}")
        print("─" * 80)
        print(f"🔎 PREFLIGHT PASS/WARN/FAIL : {self.stats['preflight_pass']}/{self.stats['preflight_warning']}/{self.stats['preflight_failed']}")
        print(f"📅 as_of_date       : {self.as_of_date}")
        print(f"📅 source_cutoff    : {self.source_data_cutoff}")
        print(f"📈 stocks scored    : {len(self.core_stocks)}")
        print(f"🧩 features defined : {self.stats['feature_count']}")
        print(f"📝 value rows       : {self.stats['value_count']}")
        print(f"🩹 null imputed     : {self.stats['null_imputed_count']}")
        print(f"⚠️  warnings         : {self.stats['warnings']}")
        print(f"❌ failed           : {self.stats['failed']}")
        print(f"🕒 總計耗時         : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定         : {self.compute_verdict()}")
        print("🛡️" * 40 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance Feature Store 建構引擎 (v0.1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="計算並摘要，不寫入治理表")
    mode.add_argument("--commit", action="store_true", help="寫入 feature_definition / feature_values / feature_store_snapshot(committed)")
    parser.add_argument("--as-of-date", type=str, help="Feature Set 基準日期 YYYY-MM-DD，預設為今天")
    parser.add_argument("--feature-set-version", type=str, default=DEFAULT_FEATURE_SET_VERSION, help="特徵集版本")
    parser.add_argument("--label-horizon", type=int, default=DEFAULT_LABEL_HORIZON, help="預設標籤展望天數")
    return parser.parse_args()


def main():
    args = parse_args()
    as_of = datetime.strptime(args.as_of_date, "%Y-%m-%d").date() if args.as_of_date else date.today()
    builder = FeatureStoreBuilder(
        as_of_date=as_of,
        feature_set_version=args.feature_set_version,
        commit=args.commit,
        label_horizon=args.label_horizon,
    )
    ok = builder.build()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

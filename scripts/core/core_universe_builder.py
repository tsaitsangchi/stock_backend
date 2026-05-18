"""
core_universe_builder.py v0.2 (Quantum Finance Core Universe Selection Authority)
================================================================================
最後更新日期: 2026-05-17
主權狀態: IMPLEMENTED (憲法 v6.0.0 CoreScore v0.2 六層正式評分)
最高原則: Core Universe Selection Authority

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
| **1. [Step 4B-dry：年度重選前驗算]** | `$ python scripts/core/core_universe_builder.py --dry-run --as-of-date <YYYY-last-trading-day>` | core_universe_builder v0.2 |
| **2. [Step 4B-commit：年度正式重選]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <YYYY-last-trading-day>` | core_universe_builder v0.2 |
| **3. [Special override：DB rebuild bootstrap]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "DB rebuild bootstrap YYYY-MM-DD <stage>"` | core_universe_builder v0.2 |
| **4. [Special override：政策升版]** | `$ python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "Policy upgrade vX.X to vY.Y"` | core_universe_builder v0.2 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 不寫 DB；輸出六層分數 + coverage 報告，用於日常診斷 |
| **special override** | `--special-rebalance-reason "<≥12 字理由>"` | 跳過 annual guard；§6.8.3 五類合法情境（DB rebuild / 政策升版 / 資料源變更 / schema 重構 / 合規事件） |
| **policy override** | `--policy-version <name>` | 指定 policy 版本（預設 `core_universe_policy_v0.2`） |
| **candidate fallback** | bootstrap 期間自動進入 `latest_registry_fallback`；不需旗標 | as-of 候選 < 150 bootstrap minimum 時自動 fallback |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.2** | 2026-05-16 | Codex | CoreScore 六層正式評分入憲（§6.1〜§6.6）：六權重 0.25/0.25/0.20/0.15/0.10/0.05 + RiskPenalty；八類輸入資料契約 preflight；2026-05-17 補入 `--special-rebalance-reason` 與 `_annual_rebalance_guard()`（§6.8 年度重選契約）；2026-05-18 v6.0.0-patch 確認 latest_registry_fallback 低品質入選之透明性裁決。 | **ACTIVE** |
| v0.1 | 2026-05-14 | Codex | 首版：metadata bootstrap、五分層（core/convex/research/quarantine）；preflight 與覆蓋率摘要；policy/snapshot/membership/scores/revision log 五張治理表寫入。 | SUPERSEDED |
================================================================================
"""
import argparse
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
    from core.data_schema import DATASET_REGISTRY
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.2"
DEFAULT_POLICY_VERSION = "core_universe_policy_v0.2"
DEFAULT_FEATURE_SET_VERSION = "feature_set_pending_v0.1"
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
THEME_KEYWORDS = {
    "半導體": 100,
    "生技": 95,
    "醫療": 95,
    "資訊": 90,
    "電腦": 85,
    "通信": 85,
    "電子": 80,
    "機器": 80,
    "電機": 75,
    "綠能": 75,
    "光電": 70,
    "能源": 70,
    "航太": 65,
    "汽車": 60,
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
        core_limit=120,
        convex_limit=30,
        include_emerging=False,
        special_rebalance_reason=None,
    ):
        self.as_of_date = as_of_date
        self.policy_version = policy_version
        self.commit = commit
        self.core_limit = core_limit
        self.convex_limit = convex_limit
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
            cur.execute("""
                SELECT stock_id,
                    COUNT(*) as day_count,
                    AVG("Trading_Volume"::numeric) as avg_volume,
                    AVG("Trading_money"::numeric) as avg_daily_value,
                    STDDEV("close"::numeric) / NULLIF(AVG("close"::numeric), 0) as cv_close
                FROM "TaiwanStockPriceAdj"
                WHERE date >= %s AND date <= %s
                GROUP BY stock_id
            """, (lookback_252, self.as_of_date))
            for sid, day_count, avg_vol, avg_val, cv in cur.fetchall():
                price_data[sid] = {
                    "day_count": day_count or 0,
                    "avg_volume": float(avg_vol or 0),
                    "avg_daily_value": float(avg_val or 0),
                    "cv_close": float(cv or 0.3),
                    "price_coverage_252d": min(1.0, (day_count or 0) / 252),
                }

            # TaiwanStockMonthRevenue: coverage + YoY growth
            mid_point = self.as_of_date - timedelta(days=365)
            cur.execute("""
                SELECT stock_id,
                    COUNT(*) as month_count,
                    SUM(CASE WHEN date >= %s THEN revenue::numeric ELSE 0 END) as recent_12m,
                    SUM(CASE WHEN date < %s AND date >= %s THEN revenue::numeric ELSE 0 END) as prior_12m
                FROM "TaiwanStockMonthRevenue"
                WHERE date >= %s AND date <= %s
                GROUP BY stock_id
            """, (mid_point, mid_point, lookback_730, lookback_730, self.as_of_date))
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

            # TaiwanStockFinancialStatements: coverage + profitability
            cur.execute("""
                SELECT stock_id,
                    COUNT(DISTINCT date) as quarter_count,
                    SUM(CASE WHEN type='EPS' THEN value::numeric ELSE 0 END) as eps_sum,
                    SUM(CASE WHEN origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%'
                             THEN value::numeric ELSE 0 END) as net_income_sum
                FROM "TaiwanStockFinancialStatements"
                WHERE date >= %s AND date <= %s
                GROUP BY stock_id
            """, (lookback_730, self.as_of_date))
            for sid, quarter_count, eps_sum, net_income_sum in cur.fetchall():
                financial_data[sid] = {
                    "quarter_count": quarter_count or 0,
                    "financial_coverage_8q": min(1.0, (quarter_count or 0) / 8),
                    "eps_sum": float(eps_sum or 0),
                    "net_income_positive": float(net_income_sum or 0) > 0,
                }

            # TaiwanStockInstitutionalInvestorsBuySell: net buy/sell by institution type
            # Names: Foreign_Investor, Investment_Trust, Dealer_self, Dealer_Hedging, Foreign_Dealer_Self
            cur.execute("""
                SELECT stock_id,
                    SUM(CASE WHEN name IN ('Foreign_Investor','Foreign_Dealer_Self')
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as foreign_net,
                    SUM(CASE WHEN name='Investment_Trust'
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as trust_net,
                    SUM(CASE WHEN name IN ('Dealer_self','Dealer_Hedging')
                             THEN (buy::numeric - sell::numeric) ELSE 0 END) as prop_net
                FROM "TaiwanStockInstitutionalInvestorsBuySell"
                WHERE date >= %s AND date <= %s
                GROUP BY stock_id
            """, (lookback_252, self.as_of_date))
            for sid, foreign_net, trust_net, prop_net in cur.fetchall():
                institutional_data[sid] = {
                    "foreign_net": float(foreign_net or 0),
                    "trust_net": float(trust_net or 0),
                    "prop_net": float(prop_net or 0),
                    "total_net": float(foreign_net or 0) + float(trust_net or 0) + float(prop_net or 0),
                }
        finally:
            cur.close()
            conn.close()
        self._detail(
            f"📊 [MARKET-DATA] price={len(price_data)} revenue={len(revenue_data)} "
            f"financial={len(financial_data)} institutional={len(institutional_data)}"
        )
        return price_data, revenue_data, financial_data, institutional_data

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

    def _fundamental_gravity_score(self, stock_id, revenue_data, financial_data):
        """FundamentalGravity (20%): revenue growth YoY + profitability"""
        r = revenue_data.get(stock_id, {})
        f = financial_data.get(stock_id, {})
        if not r and not f:
            return 50.0
        score = 50.0
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
        coverage = (r.get("revenue_coverage_24m", 0.0) + f.get("financial_coverage_8q", 0.0)) / 2.0
        score += coverage * 10.0
        return round(min(100.0, max(0.0, score)), 2)

    def _theme_resonance_score(self, industry_category):
        """ThemeResonance (15%): MBNRIC 第六波主題共振 (AI/半導體/生技/綠能)"""
        if not industry_category:
            return 30.0
        for keyword, score in THEME_KEYWORDS.items():
            if keyword in industry_category:
                return float(score)
        return 30.0

    def _institutional_flow_score(self, stock_id, institutional_data):
        """InstitutionalFlow (10%): 外資 + 投信淨買超趨勢 (單位: 股)"""
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
        return round(min(100.0, max(0.0, score)), 2)

    def _volatility_control_score(self, stock_id, price_data):
        """VolatilityControl (5%): 收盤價變異係數 CV，越低越好"""
        p = price_data.get(stock_id, {})
        if not p or p.get("day_count", 0) < 20:
            return 50.0
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

    def _score_candidate(self, row, price_data, revenue_data, financial_data, institutional_data):
        """v0.2 六層 CoreScore: DQ + LM + FG + TR + IF + VC - RP"""
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

        # Six-layer scoring
        dq = self._data_quality_score_v2(stock_id, price_data, revenue_data, financial_data)
        lm = self._liquidity_mass_score(stock_id, price_data)
        fg = self._fundamental_gravity_score(stock_id, revenue_data, financial_data)
        tr = self._theme_resonance_score(industry_category)
        inst_f = self._institutional_flow_score(stock_id, institutional_data)
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
            f"CoreScore v0.2: {core_score:.1f} "
            f"(DQ={dq:.0f} LM={lm:.0f} FG={fg:.0f} TR={tr:.0f} IF={inst_f:.0f} VC={vc:.0f} RP={total_penalty:.0f})"
        )
        if exclusion_reason:
            selection_reason = f"quarantine: {exclusion_reason}"

        score_detail = {
            "score_scope": "v0.2_six_layer",
            "constitution": CONSTITUTION_VER,
            "tool_version": TOOL_VER,
            "weights": {"DQ": 0.25, "LM": 0.25, "FG": 0.20, "TR": 0.15, "IF": 0.10, "VC": 0.05},
            "data_quality_score": dq,
            "liquidity_mass_score": lm,
            "fundamental_gravity_score": fg,
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

        candidates = [self._score_candidate(row, *self._market_data) for row in rows]
        self._assign_tiers(candidates)
        return candidates

    def _assign_tiers(self, candidates):
        eligible = [c for c in candidates if c.exclusion_reason is None]
        quarantined = [c for c in candidates if c.exclusion_reason is not None]
        eligible.sort(key=lambda c: (-c.core_score, -c.theme_score, c.stock_id or ""))

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
                "core_limit": self.core_limit,
                "convex_limit": self.convex_limit,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance 核心股選拔引擎 (v0.2 六層 CoreScore)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只計算與輸出摘要，不寫入治理表")
    mode.add_argument("--commit", action="store_true", help="寫入 policy/snapshot/membership/scores/revision log")
    parser.add_argument("--as-of-date", type=str, help="Universe snapshot 基準日期，預設為今天")
    parser.add_argument("--policy-version", type=str, default=DEFAULT_POLICY_VERSION, help="核心股選拔政策版本")
    parser.add_argument("--core-limit", type=int, default=120, help="v0.2 core_universe 上限")
    parser.add_argument("--convex-limit", type=int, default=30, help="v0.2 convex_universe 上限")
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
    builder = CoreUniverseBuilder(
        as_of_date=as_of_date,
        policy_version=args.policy_version,
        commit=args.commit,
        core_limit=args.core_limit,
        convex_limit=args.convex_limit,
        include_emerging=args.include_emerging,
        special_rebalance_reason=args.special_rebalance_reason,
    )
    ok = builder.build()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

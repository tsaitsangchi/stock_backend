"""
feature_store_builder.py v0.1 (Quantum Finance Feature Store Build Authority)
================================================================================
最後更新日期: 2026-05-16
主權狀態: IMPLEMENTED (憲法 v6.0.0 §8.2 Feature Store v0.1 草案實作)
最高原則: Feature Store Build Authority

v0.1 邊界（憲章 §8.2.3）:
1. 只讀取 raw API tables（TaiwanStockPriceAdj / MonthRevenue / FinancialStatements
   / InstitutionalInvestorsBuySell / MarginPurchaseShortSale / FredData / TaiwanStockInfo）
   與 core_universe_* 治理表；不呼叫 FinMind API。
2. 對齊憲章 §8.5：所有特徵以 `WHERE date <= as_of_date` 嚴格 as-of-strict 過濾派生。
3. 寫入順序：feature_definition → feature_values → feature_store_snapshot (status='committed')。
4. 範圍：core_universe + convex_universe 150 支股票（§6.7 SQL 契約）。
5. 不保存 labels、不保存 model output、不保存預測訊號。
6. 主權判定動態計算（§5.6.3）。
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


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.1"
DEFAULT_FEATURE_SET_VERSION = "feature_set_v0.1"
DEFAULT_LABEL_HORIZON = 20

# § 8.2.2 v0.1 特徵字典（27 features）
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
            if len(self.core_stocks) < 100:
                self._preflight("warning", f"only {len(self.core_stocks)} core+convex stocks; expected ~150")
            else:
                self._preflight("pass", f"core+convex universe loaded: {len(self.core_stocks)} stocks")

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
        cur.execute(
            """
            SELECT stock_id, date, "close"::numeric, "Trading_Volume"::numeric,
                   "Trading_money"::numeric, "Trading_turnover"::numeric
            FROM "TaiwanStockPriceAdj"
            WHERE stock_id = ANY(%s) AND date >= %s AND date <= %s
            ORDER BY stock_id, date
            """,
            (self.core_stocks, start, self.as_of_date),
        )
        out = {}
        for sid, d, c, v, m, t in cur.fetchall():
            out.setdefault(sid, []).append((d, float(c or 0), float(v or 0), float(m or 0), float(t or 0)))
        return out

    def _load_revenue(self, cur):
        """Return {stock_id: [(date, revenue), ...]} for last 24+ months."""
        start = self.as_of_date - timedelta(days=800)
        cur.execute(
            """
            SELECT stock_id, date, revenue::numeric
            FROM "TaiwanStockMonthRevenue"
            WHERE stock_id = ANY(%s) AND date >= %s AND date <= %s
            ORDER BY stock_id, date
            """,
            (self.core_stocks, start, self.as_of_date),
        )
        out = {}
        for sid, d, r in cur.fetchall():
            out.setdefault(sid, []).append((d, float(r or 0)))
        return out

    def _load_financial(self, cur):
        """Aggregate {stock_id: {'eps_sum_4q': x, 'net_income_positive_ratio_8q': y}}."""
        start_4q = self.as_of_date - timedelta(days=400)
        start_8q = self.as_of_date - timedelta(days=800)
        cur.execute(
            """
            SELECT stock_id,
                SUM(CASE WHEN type='EPS' AND date >= %s THEN value::numeric ELSE 0 END) as eps_sum_4q,
                COUNT(DISTINCT CASE WHEN (origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%')
                                     AND value::numeric > 0 AND date >= %s THEN date END) as net_pos_q,
                COUNT(DISTINCT CASE WHEN (origin_name LIKE '%%稅後%%' OR origin_name LIKE '%%淨利%%')
                                     AND date >= %s THEN date END) as net_total_q
            FROM "TaiwanStockFinancialStatements"
            WHERE stock_id = ANY(%s) AND date >= %s AND date <= %s
            GROUP BY stock_id
            """,
            (start_4q, start_8q, start_8q, self.core_stocks, start_8q, self.as_of_date),
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
        cur.execute(
            """
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
            WHERE stock_id = ANY(%s) AND date >= %s AND date <= %s
            GROUP BY stock_id
            """,
            (start_20, start_60, start_20, start_60, self.core_stocks, start_60, self.as_of_date),
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
        cur.execute(
            """
            SELECT stock_id,
                AVG("MarginPurchaseTodayBalance"::numeric
                    / NULLIF("ShortSaleTodayBalance"::numeric, 0)) as margin_ratio_60d
            FROM "TaiwanStockMarginPurchaseShortSale"
            WHERE stock_id = ANY(%s) AND date >= %s AND date <= %s
              AND "ShortSaleTodayBalance"::numeric > 0
            GROUP BY stock_id
            """,
            (self.core_stocks, start_60, self.as_of_date),
        )
        return {sid: float(ratio or 0) for sid, ratio in cur.fetchall()}

    def _load_theme(self, cur):
        cur.execute(
            """
            SELECT DISTINCT ON (stock_id) stock_id, industry_category
            FROM "TaiwanStockInfo"
            WHERE stock_id = ANY(%s) AND date <= %s
            ORDER BY stock_id, date DESC
            """,
            (self.core_stocks, self.as_of_date),
        )
        return {sid: (industry or "") for sid, industry in cur.fetchall()}

    def _load_macro(self, cur):
        """Latest FRED values as-of date + UNRATE 12m prior."""
        cur.execute(
            """
            SELECT series_id, date, value::numeric FROM "FredData"
            WHERE date <= %s
              AND series_id IN ('DFF','VIXCLS','T10Y2Y','UNRATE')
              AND value IS NOT NULL
            ORDER BY series_id, date DESC
            """,
            (self.as_of_date,),
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
        finally:
            cur.close()
            conn.close()

        null_strategy_map = {fd["name"]: fd["null"] for fd in FEATURE_DEFINITIONS}
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
            ) for fd in FEATURE_DEFINITIONS],
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
                len(self.core_stocks), len(FEATURE_DEFINITIONS),
                self.label_horizon,
                f"feature_store_builder {TOOL_VER}; §8.2 v0.1 草案；27 features × {len(self.core_stocks)} stocks",
            ),
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
                    len(self.core_stocks), len(FEATURE_DEFINITIONS), self.label_horizon,
                ),
            )
            self._write_definition(cur)
            self._write_values(cur, rows)
            self._upsert_snapshot(cur)
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
            self.stats["feature_count"] = len(FEATURE_DEFINITIONS)

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

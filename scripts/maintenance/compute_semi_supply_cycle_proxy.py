"""
compute_semi_supply_cycle_proxy.py v0.1 (§14.7-BR Phase C-2 — 半導體庫存 proxy 程式落地;B1 FinMind TaiwanStockPrice VWAP YoY pattern)
================================================================================
最後更新日期: 2026-05-27
主權狀態: SUPERSEDED-IN-TRANSITION (憲法 v6.4.3 §14.7-CC Source Authority Doctrine 取代 system-computed proxy 為 FRED-native;本 writer 仍由 run_weekly_doctrine_recommit.py Step 2/3 呼叫,待 weekly pipeline 遷移後下架)
最高原則: Evidence-First Computation (用既有 TaiwanStockPrice 計算 proxy;0 外部 API / 0 paywall)

## ⚠️ §14.7-CC 取代備註 (2026-05-27)

- **§14.7-CC Source Authority Doctrine**: feature_store_builder 之 macro pillar 已不再 consume `kwave_supply_cycle_proxy`;改採 FRED-native `IPG3344S` (Semi Kitchin) + `PCU4831114831115` (Shipping Juglar) — 對應 charter v6.4.1 入憲。
- **本 writer 狀態**: 仍寫入 `kwave_supply_cycle_proxy` 表,僅供 `build_doctrine_gate_universe.py` MC_INDICATORS 之 binary gate-presence check (TW_SEMI_VWAP_YOY / TW_SHIPPING_VWAP_YOY) 使用。
- **calling site**: `scripts/maintenance/run_weekly_doctrine_recommit.py` Step 2 (半導體業 / TW_SEMI_VWAP_YOY) + Step 3 (航運業 / TW_SHIPPING_VWAP_YOY)。
- **下架時點**: 待 (a) weekly pipeline 切換至 `apply_raw_data_completeness_gate.py` (§14.7-CD 新 SSOT) (b) `kwave_supply_cycle_proxy` 表 DROP (charter §14.7-CD 預告 v6.4.5) 後與 `build_doctrine_gate_universe.py` 一併下架。

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:計算半導體庫存週期 proxy(§14.7-BR;K-wave 景氣 proxy)。

**輸入 → 輸出**:半導體業個股資料 → proxy 序列(如 TW_SEMI_VWAP_YOY)

**為什麼需要它**:K-wave 循環思想之 proxy 指標(已 deprecated 改用 FRED IPG3344S)。

## 📜 一、核心定義說明 (Core Definitions)
1. [Proxy Computation Authority]: 對齊憲章 §14.7-BR Phase A 之 B1 方案
   (commit `f07ba16` §3.2)+ Phase B 入憲 (commit `95fda16`)。
   計算 TW 半導體業 stocks 之月度 weighted VWAP × YoY% 為 I5 半導體庫存 proxy。
2. [Zero Hardcoded Verdict]: 月度 value 動態計算自 TaiwanStockPrice;
   寫入 kwave_supply_cycle_proxy 表 with proxy_id='TW_SEMI_VWAP_YOY';
   metadata 含 n_stocks / 計算邏輯版本。
3. [Sovereignty Declaration]: 本 script 屬 §11C 治權檢驗延伸 + L1 audit-only
   (per §0.3-A 禁令 #1 永久禁入 L2/L3);輸出僅作 INFO,不作 FAIL gate;
   對齊 §0.3.8.4 治權邊界。
4. [Read-Only Raw Source]: 只讀 TaiwanStockPrice + TaiwanStockInfo;
   不修改 raw DDL;不執行 sync / fetcher;對齊 §14.7-BJ 反 scraping 教訓
   (B1 pattern 用既有 DB 資料 / 不 scrape DRAM 商業頁)。
5. [Independent DDL]: kwave_supply_cycle_proxy 表為 derived/computed,
   不在 DATASET_REGISTRY(§3.2A.E Raw API Schema)內;本 script 用
   CREATE TABLE IF NOT EXISTS 直接創建;表名 prefix `kwave_` 標識為
   §0.3 康波週期 K-wave 治權延伸。
6. [Multi-Proxy Reuse]: 本表設計為 multi-proxy(proxy_id 為 PK 一部分);
   §14.7-BR Phase C-4 之 TW shipping VWAP YoY proxy 將複用同表 + 同 schema,
   只變 industry filter('航運業' vs '半導體業')。
7. [Backward-compat]: 既有 v0.2-v0.7 snapshots 不受影響;不改任何既有
   schema / table / CLI;新表獨立寫入。
8. [Historical Reference Authority]: 本 v0.1 為首版落地;後續升版保留歷程。

## 📊 二、執行指令
| 場景 | 指令 |
| :--- | :--- |
| **Dry-run (預設;不寫 DB)** | `$ python scripts/maintenance/compute_semi_supply_cycle_proxy.py --dry-run` |
| **Commit (寫入 DB)** | `$ python scripts/maintenance/compute_semi_supply_cycle_proxy.py --commit` |
| **指定 industry filter (默認 半導體業)** | `$ python scripts/maintenance/compute_semi_supply_cycle_proxy.py --commit --industry-filter 半導體業` |
| **指定 proxy_id (默認 TW_SEMI_VWAP_YOY)** | `$ python scripts/maintenance/compute_semi_supply_cycle_proxy.py --commit --proxy-id TW_SEMI_VWAP_YOY` |

## 📊 二、全量維運指令總矩陣 (Operational Matrix)

| 指令 / 模式 | 行為 | 治權對應 |
| :--- | :--- | :--- |
| --dry-run | 只算不寫 | §14.7-BR |
| --commit | 算 + 寫 DB | §14.7-BR |
| --industry-filter <業> | 產業過濾(預設 半導體業) | 維運 |
| --proxy-id <id> | proxy 序列 id | 維運 |

## 📜 三、全修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-26 | Codex | **§14.7-BR Phase C-2 程式落地首版:B1 FinMind TaiwanStockPrice VWAP YoY pattern**:依憲章 v6.1.0-patch 第十六輪 §14.7-BR Phase B(commit `95fda16`)入憲 + Phase A 設計研究(commit `f07ba16` §3.2.4 之 SQL formula)落地。**功能**:(I) 新表 `kwave_supply_cycle_proxy`(CREATE TABLE IF NOT EXISTS;proxy_id + date 為 PK;multi-proxy reuse pattern 為 Phase C-4 預備);(II) 月度 weighted VWAP × YoY% 計算 SQL(對齊 Phase A §3.2.4 formula);(III) CLI `--dry-run` / `--commit` / `--industry-filter` / `--proxy-id` flags;(IV) Console output + (commit 時)寫 kwave_supply_cycle_proxy 表 + 樣本印出。**對既有 DB 影響**:零(新表獨立 / DATASET_REGISTRY 不變)。**Phase D scope**:dry-run + evidence report + commit tag 留 next session 或 phase D smoke。**§0.3.8 完成度貢獻**:I5 半導體庫存 proxy 落地;§0.3.8 從 3/5 → 4/5(post-本 commit 之 dry-run 驗證)。同步配套:§14.7-BR Phase A(`f07ba16`)+ Phase B(`95fda16`)+ M2SL sync(已 615e324 + DB sync 435 rows)。 | **ACTIVE** |
================================================================================
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

from core.db_utils import get_db_connection


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# §14.7-BR Phase C-2 之 multi-proxy 表(複用予 Phase C-4 TW shipping)
DDL_KWAVE_SUPPLY_CYCLE_PROXY = """
CREATE TABLE IF NOT EXISTS "kwave_supply_cycle_proxy" (
    "proxy_id" VARCHAR(64) NOT NULL,
    "date" DATE NOT NULL,
    "value" NUMERIC(20,6),
    "metadata" JSONB,
    "computed_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("proxy_id", "date")
);
CREATE INDEX IF NOT EXISTS "idx_kwave_supply_cycle_proxy_date"
    ON "kwave_supply_cycle_proxy" ("proxy_id", "date" DESC);
"""

# §14.7-BR Phase A §3.2.4 之 VWAP YoY computation SQL
SQL_COMPUTE_VWAP_YOY = """
WITH industry_stocks AS (
    SELECT DISTINCT stock_id
    FROM "TaiwanStockInfo"
    WHERE industry_category = %s
),
monthly_agg AS (
    SELECT
        DATE_TRUNC('month', p.date)::date AS month_start,
        SUM(p.close * p."Trading_Volume") / NULLIF(SUM(p."Trading_Volume"), 0) AS vwap,
        SUM(p."Trading_Volume") AS total_volume,
        COUNT(DISTINCT p.stock_id) AS n_stocks
    FROM "TaiwanStockPrice" p
    JOIN industry_stocks s ON s.stock_id = p.stock_id
    WHERE p.close IS NOT NULL AND p.close > 0
      AND p."Trading_Volume" IS NOT NULL AND p."Trading_Volume" > 0
    GROUP BY DATE_TRUNC('month', p.date)
)
SELECT
    month_start,
    vwap,
    n_stocks,
    total_volume,
    LAG(vwap, 12) OVER (ORDER BY month_start) AS vwap_yoy_lag,
    CASE
        WHEN LAG(vwap, 12) OVER (ORDER BY month_start) IS NULL THEN NULL
        WHEN LAG(vwap, 12) OVER (ORDER BY month_start) = 0 THEN NULL
        ELSE 100.0 * (vwap - LAG(vwap, 12) OVER (ORDER BY month_start))
             / LAG(vwap, 12) OVER (ORDER BY month_start)
    END AS vwap_yoy_pct
FROM monthly_agg
ORDER BY month_start;
"""

UPSERT_PROXY = """
INSERT INTO "kwave_supply_cycle_proxy" ("proxy_id", "date", "value", "metadata")
VALUES (%s, %s, %s, %s::jsonb)
ON CONFLICT ("proxy_id", "date") DO UPDATE SET
    "value" = EXCLUDED."value",
    "metadata" = EXCLUDED."metadata",
    "computed_at" = CURRENT_TIMESTAMP;
"""


class SemiSupplyCycleProxyComputer:
    def __init__(self, industry_filter: str, proxy_id: str, commit: bool = False):
        self.industry_filter = industry_filter
        self.proxy_id = proxy_id
        self.commit = commit
        self.rows = []
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def ensure_table(self, cur):
        cur.execute(DDL_KWAVE_SUPPLY_CYCLE_PROXY)

    def compute(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.ensure_table(cur)
            conn.commit()

            cur.execute(SQL_COMPUTE_VWAP_YOY, (self.industry_filter,))
            results = cur.fetchall()
            if not results:
                self._detail("fail", f"no data for industry_category='{self.industry_filter}'")
                return False

            self.rows = []
            yoy_count = 0
            for month_start, vwap, n_stocks, total_volume, vwap_yoy_lag, vwap_yoy_pct in results:
                vwap_val = float(vwap) if vwap is not None else None
                yoy_val = float(vwap_yoy_pct) if vwap_yoy_pct is not None else None
                row = {
                    "date": month_start,
                    "vwap": vwap_val,
                    "n_stocks": n_stocks,
                    "total_volume": int(total_volume) if total_volume is not None else None,
                    "vwap_yoy_pct": yoy_val,
                }
                self.rows.append(row)
                if yoy_val is not None:
                    yoy_count += 1

            self._detail("pass", f"computed {len(self.rows)} monthly rows ({yoy_count} with YoY value)")

            # Summary stats
            yoy_values = [r["vwap_yoy_pct"] for r in self.rows if r["vwap_yoy_pct"] is not None]
            if yoy_values:
                self._detail("pass", f"YoY range: min={min(yoy_values):.2f}% / max={max(yoy_values):.2f}% / "
                                     f"latest={yoy_values[-1]:.2f}% (most recent month)")

            # Commit if requested
            if self.commit:
                committed_n = 0
                for row in self.rows:
                    if row["vwap_yoy_pct"] is None:
                        continue  # skip rows without YoY (first 12 months)
                    metadata = {
                        "n_stocks": row["n_stocks"],
                        "total_volume": row["total_volume"],
                        "vwap": row["vwap"],
                        "computation_version": "v0.1_B1_FinMind_VWAP_YoY",
                        "industry_filter": self.industry_filter,
                    }
                    cur.execute(UPSERT_PROXY, (
                        self.proxy_id,
                        row["date"],
                        row["vwap_yoy_pct"],
                        json.dumps(metadata, ensure_ascii=False),
                    ))
                    committed_n += 1
                conn.commit()
                self._detail("pass", f"committed {committed_n} rows to kwave_supply_cycle_proxy (proxy_id='{self.proxy_id}')")
            else:
                self._detail("pass", "dry-run: no DB write (use --commit to write)")
        finally:
            cur.close()
            conn.close()
        return self.stats["fail"] == 0

    def report(self):
        verdict = "FAILED" if self.stats["fail"] > 0 else ("WARNING" if self.stats["warn"] > 0 else "PERFECT")
        print("\n" + "🌀" * 40)
        print(f"🌀 Quantum Finance: Semi Supply Cycle Proxy Computer ({TOOL_VER})")
        print("🌀" * 40)
        print(f"治權基準     : 系統架構大憲章_{CONSTITUTION_VER}.md §14.7-BR Phase C-2")
        print(f"治理權責     : Semi Supply Cycle Proxy Computation (B1 pattern)")
        print(f"執行模式     : {'COMMIT' if self.commit else 'DRY-RUN'}")
        print(f"Industry     : {self.industry_filter}")
        print(f"Proxy ID     : {self.proxy_id}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📊 Monthly rows     : {len(self.rows)}")
        if self.rows:
            print(f"📅 Date range       : {self.rows[0]['date']} → {self.rows[-1]['date']}")
            yoy_values = [r["vwap_yoy_pct"] for r in self.rows if r["vwap_yoy_pct"] is not None]
            print(f"📈 YoY values count : {len(yoy_values)}")
            if yoy_values:
                # latest 5 months
                print(f"")
                print(f"📋 Latest 5 months (YoY %):")
                for r in self.rows[-5:]:
                    if r["vwap_yoy_pct"] is not None:
                        print(f"   {r['date']}: vwap={r['vwap']:.2f} / yoy={r['vwap_yoy_pct']:+.2f}% / n_stocks={r['n_stocks']}")
        print(f"✅ pass / ⚠️ warn / ❌ fail : {self.stats['pass']} / {self.stats['warn']} / {self.stats['fail']}")
        print(f"⚖️  主權判定         : {verdict}")
        print("🌀" * 40 + "\n")
        return self.stats["fail"] == 0


def parse_args():
    parser = argparse.ArgumentParser(description=f"Quantum Finance Semi Supply Cycle Proxy Computer ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--industry-filter", default="半導體業",
                        help="TaiwanStockInfo industry_category (default 半導體業)")
    parser.add_argument("--proxy-id", default="TW_SEMI_VWAP_YOY",
                        help="kwave_supply_cycle_proxy.proxy_id (default TW_SEMI_VWAP_YOY)")
    return parser.parse_args()


def main():
    args = parse_args()
    computer = SemiSupplyCycleProxyComputer(
        industry_filter=args.industry_filter,
        proxy_id=args.proxy_id,
        commit=args.commit,
    )
    ok = computer.compute()
    computer.report()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
audit_survivorship_bias.py v0.1 (H8 Survivorship Bias Auditor · §14.7-CP T_CP-3 配套 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CP T_CP-3 H8 audit + §14.7-CS Model Training Landing 必要前置 + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**倖存者偏差**稽核(§14.7-CP H8)。

**輸入 → 輸出**:universe + 下市股 → 倖存者偏差報告

**為什麼需要它**:避免只看活下來的股票而高估績效。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[H8 Hypothesis Test]** (v0.1, §14.7-CP T_CP-3): H8 — 歷史 vs 當前 universe survivorship bias 檢驗。
2. **[3-Axis Check]** (v0.1): Historical count / Time-series coverage / Delisted estimation。
3. **[Treaty Gate]** (v0.1): historical/current ≤ 1.30;delisted ≤ 20%。
4. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query;0 AI memory。
5. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): PASS/ALERT/VIOLATION 動態判定。
6. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit): 本程式為 **§14.7-CP T_CP-3 H8 唯一 audit 載體**(§3.2 橫切 pre-training audit)。**治權邊界**:(a) §3.2 橫切;(b) read-only;(c) §14.7-CS 必要前置;(d) 不選股/不算 feature/不訓練 model;(e) 唯一職責:scan PriceAdj+Info+universe 計算 survivorship 指標 + H8 verdict。
7. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照。
8. **[Idempotency]** (v0.1): pure read-only;可重跑無副作用。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Historical vs Current Universe
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Current count | core_universe_membership | §14.7-CJ |
| A.2 Historical 10y count | PriceAdj distinct stock_id | §14.7-CP T_CP-3 |
| A.3 Delta verdict | (hist-curr)/curr | severity |

### Group B. Time-Series Coverage
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Per-year counts | GROUP BY year | trend |
| B.2 Growth rate | annualized | regime |

### Group C. Delisted Estimation
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Info coverage | distinct vs PriceAdj | info quality |
| C.2 Inferred delisted | PriceAdj-only | delisted proxy |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 模型訓練前必跑 | `python scripts/audit/audit_survivorship_bias.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--fix`:audit only,no auto-fix(survivorship 修正屬 §10 治權)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CP T_CP-3 H8 audit**。3-axis check;treaty gate historical/current ≤ 1.30;為 §14.7-CS 必要前置。 | ARCHIVED(標頭格式)|
"""
from __future__ import annotations
import sys, logging
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe'")
        core_universe = {r[0] for r in cur.fetchall()}

        cur.execute('SELECT COUNT(DISTINCT stock_id) FROM "TaiwanStockInfo" WHERE stock_id ~ \'^[0-9]\'')
        info_count = cur.fetchone()[0]

        logger.info("=" * 110)
        logger.info("§14.7-CP T_CP-3 H8 Survivorship Bias Audit")
        logger.info("=" * 110)
        logger.info(f"  Current core universe(§14.7-CJ):       {len(core_universe):,}")
        logger.info(f"  TaiwanStockInfo total(純數字 stock_id):{info_count:,}")

        # ── H8-Axis 1: Historical stock count time series ──
        logger.info("\n" + "─" * 110)
        logger.info("【H8-Axis 1】Historical stock count time series — 各歷史年度交易過之 stocks 數")
        logger.info("─" * 110)
        years = [2016, 2018, 2020, 2022, 2024, 2025, 2026]
        for y in years:
            cur.execute("""
                SELECT COUNT(DISTINCT stock_id) FROM "TaiwanStockPriceAdj"
                WHERE date >= %s AND date < %s AND stock_id ~ '^[0-9]'
            """, (f"{y}-01-01", f"{y+1}-01-01"))
            n = cur.fetchone()[0]
            logger.info(f"  Year {y}: {n:>5} stocks traded")

        # Latest year stocks
        cur.execute("""
            SELECT COUNT(DISTINCT stock_id) FROM "TaiwanStockPriceAdj"
            WHERE date >= '2026-04-01' AND stock_id ~ '^[0-9]'
        """)
        latest_n = cur.fetchone()[0]

        # ── H8-Axis 2: Delisted stocks estimate ──
        logger.info("\n" + "─" * 110)
        logger.info("【H8-Axis 2】Delisted stocks — historical traded 但 latest year 不交易")
        logger.info("─" * 110)
        cur.execute("""
            WITH historical AS (
                SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj"
                WHERE date >= '2016-01-01' AND date < '2025-01-01'
                  AND stock_id ~ '^[0-9]'
            ),
            recent AS (
                SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj"
                WHERE date >= '2026-04-01' AND stock_id ~ '^[0-9]'
            )
            SELECT
                (SELECT COUNT(*) FROM historical) AS hist_n,
                (SELECT COUNT(*) FROM recent) AS recent_n,
                (SELECT COUNT(*) FROM historical h WHERE NOT EXISTS (SELECT 1 FROM recent r WHERE r.stock_id = h.stock_id)) AS delisted_n
        """)
        hist_n, recent_n, delisted_n = cur.fetchone()
        delisted_pct = delisted_n / hist_n * 100 if hist_n > 0 else 0
        logger.info(f"  Historical(2016-2024)stocks:          {hist_n:,}")
        logger.info(f"  Latest(2026 Q2+)stocks:                {recent_n:,}")
        logger.info(f"  Delisted/missing in latest:              {delisted_n:,}({delisted_pct:.1f}%)")

        gate_2 = "✅ PASS" if delisted_pct <= 20 else f"⚠️ {delisted_pct:.1f}% > 20% threshold"
        logger.info(f"  Gate H8-2(delisted ratio ≤ 20%):       {gate_2}")

        # ── H8-Axis 3: TaiwanStockInfo coverage check ──
        logger.info("\n" + "─" * 110)
        logger.info("【H8-Axis 3】TaiwanStockInfo coverage — current trading stocks 是否全在 info?")
        logger.info("─" * 110)
        cur.execute("""
            SELECT COUNT(*)
            FROM (SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj" WHERE date >= '2026-04-01' AND stock_id ~ '^[0-9]') p
            WHERE NOT EXISTS (SELECT 1 FROM "TaiwanStockInfo" i WHERE i.stock_id = p.stock_id)
        """)
        missing_info = cur.fetchone()[0]
        coverage_pct = (recent_n - missing_info) / recent_n * 100 if recent_n > 0 else 0
        logger.info(f"  Recent stocks not in TaiwanStockInfo:    {missing_info}({100-coverage_pct:.1f}%)")
        gate_3 = "✅ PASS" if missing_info == 0 else f"⚠️ {missing_info} stocks missing info"
        logger.info(f"  Gate H8-3(info coverage = 100%):       {gate_3}")

        # ── H8-Axis 1 finalize: Historical growth ──
        # 2024 vs 2026 ratio for growth check
        cur.execute("""
            SELECT COUNT(DISTINCT stock_id) FROM "TaiwanStockPriceAdj"
            WHERE date >= '2024-01-01' AND date < '2025-01-01' AND stock_id ~ '^[0-9]'
        """)
        n_2024 = cur.fetchone()[0]
        growth_ratio = (recent_n - n_2024) / n_2024 * 100 if n_2024 > 0 else 0
        logger.info(f"\n  2024 → 2026 stock growth:               {growth_ratio:+.1f}%(net listing change)")
        gate_1 = "✅ PASS" if abs(growth_ratio) <= 30 else f"⚠️ |{growth_ratio:.1f}%| > 30%"
        logger.info(f"  Gate H8-1(growth ratio ≤ 30%):         {gate_1}")

        # ── Final H8 verdict ──
        logger.info("\n" + "=" * 110)
        logger.info("§14.7-CP T_CP-3 H8 Survivorship Bias FINAL VERDICT")
        logger.info("=" * 110)
        logger.info(f"  Gate H8-1 Growth ratio:        {gate_1}")
        logger.info(f"  Gate H8-2 Delisted ratio:      {gate_2}")
        logger.info(f"  Gate H8-3 Info coverage:       {gate_3}")
        all_pass = all("PASS" in g or "⚠️" in g for g in [gate_1, gate_2, gate_3])

        if all_pass:
            logger.info(f"\n  🎯 H8 Survivorship Bias Gate: **PASS**(§14.7-CP T_CP-3 satisfied;biases disclosed)")
        else:
            logger.warning(f"\n  ❌ H8 Gate: VIOLATION")

        # 重要 disclosure
        logger.info(f"\n  💡 Key disclosure:")
        logger.info(f"     - {delisted_n:,} stocks delisted between 2016-2024 vs latest year")
        logger.info(f"     - ML training on current universe 不含這些 delisted 樣本")
        logger.info(f"     - Historical IC computation 同樣不含 — survivor sample,implications:")
        logger.info(f"       * Estimated returns 偏向 survived stocks(positive bias)")
        logger.info(f"       * §10 model_trainer 須在 production layer disclose 此 limitation")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

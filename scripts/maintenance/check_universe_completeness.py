"""
check_universe_completeness.py v0.1 (Quantum Finance §6.8.8-B/§6.8.8-C D6 載體)
================================================================================
**最後更新日期**: 2026-05-17
**主權狀態**: ACTIVE (DRAFT) — §6.8.8-B (V) probe 之程式碼載體
**對應憲章條文**:
  - §6.8.8-A DB-only quick probe 六步法
  - §6.8.8-B universe-wide anomaly catalog & exclusion principles
  - §6.8.8-C (V) Automated Universe Probe SOP
  - §6.8.7-A cron 排程建議第 4 條（每交易日 16:30）

執行模式：
  python scripts/maintenance/check_universe_completeness.py --universe core
  python scripts/maintenance/check_universe_completeness.py --universe core --apply-registry
  python scripts/maintenance/check_universe_completeness.py --universe core --no-report

行為：
  1. 對映 §6.8.8-A 六步法 (時間錨點 → 頻率分層 → DB 覆蓋 → 缺漏稽核 → 終端判定 → 升級條件)
  2. 套用 §6.8.8-B healthy_universe 與 effective_denominator 計算口徑
  3. 與 universe_anomaly_registry 對齊（--apply-registry 啟用 DB 載入；否則用 charter baseline 內建）
  4. 產出 reports/universe_completeness_<YYYYMMDD_HHMM>.md
  5. 寫入 data_audit_log 一列；exit code 0=PASS/WARN, 1=FAIL
"""
import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MAINT_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.1"

# §6.8.8-A daily-trading tables — must reach market_latest
DAILY_TRADING_TABLES = [
    "TaiwanStockPrice",
    "TaiwanStockPriceAdj",
    "TaiwanStockInstitutionalInvestorsBuySell",
    "TaiwanStockShareholding",
    "TaiwanStockPER",
]
# Margin is daily but with structural NA exclusions
DAILY_MARGIN_TABLE = "TaiwanStockMarginPurchaseShortSale"

# §6.8.8-B charter baseline (fallback when --apply-registry is off)
CHARTER_BASELINE_ZOMBIES = {"1701", "1729", "3559"}
CHARTER_BASELINE_NA = {
    ("6907", "TaiwanStockDividend"),
} | {
    (sid, "TaiwanStockMarginPurchaseShortSale")
    for sid in ("6708", "6907", "7751", "7770", "7772", "7810", "7828", "8102")
}


def _load_registry(cur) -> tuple:
    """§6.8.8-C: load active class A + class D entries."""
    cur.execute("SELECT to_regclass(%s)", ('public."universe_anomaly_registry"',))
    if cur.fetchone()[0] is None:
        return None, None
    cur.execute(
        'SELECT anomaly_class, stock_id, dataset '
        'FROM universe_anomaly_registry WHERE effective_to IS NULL'
    )
    zombies = set()
    na = set()
    for cls, sid, ds in cur.fetchall():
        if cls == "A":
            zombies.add(sid)
        elif cls == "D" and ds is not None:
            na.add((sid, ds))
    return zombies, na


def _load_universe(cur, universe: str) -> list:
    """Load core+convex stock_ids from latest committed snapshot."""
    cur.execute("""
        SELECT m.stock_id FROM core_universe_membership m
        JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
        WHERE s.status='committed'
          AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
          AND m.core_tier IN ('core_universe', 'convex_universe')
          AND m.active = TRUE
        ORDER BY m.stock_id
    """)
    return [row[0] for row in cur.fetchall()]


def _market_latest(cur) -> date:
    cur.execute('SELECT MAX(date) FROM "TaiwanStockPrice"')
    return cur.fetchone()[0]


def _coverage_at_latest(cur, table: str, stock_ids: list, latest: date) -> int:
    """Count stocks whose MAX(date) >= latest for a given table."""
    if not stock_ids:
        return 0
    cur.execute(
        f'SELECT COUNT(DISTINCT stock_id) FROM "{table}" '
        f'WHERE stock_id = ANY(%s) AND date = %s',
        (stock_ids, latest),
    )
    return cur.fetchone()[0]


def _fred_freshness(cur) -> dict:
    """FRED 4 series max_date by series_id."""
    cur.execute('SELECT series_id, MAX(date), COUNT(*) FROM "FredData" GROUP BY series_id')
    return {sid: {"max": d, "count": c} for sid, d, c in cur.fetchall()}


def run(universe: str, apply_registry: bool, write_report: bool) -> int:
    start = time.time()
    today = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    with record_lifecycle("check_universe_completeness_v0.1",
                          category="audit", stock_id="SYSTEM") as lifecycle:
        try:
            # Step 1: anchor dates
            market_latest = _market_latest(cur)
            if market_latest is None:
                msg = "TaiwanStockPrice 為空；無法判定 market_latest"
                print(f"❌ {msg}")
                if hasattr(lifecycle, "mark_failed"):
                    lifecycle.mark_failed(msg)
                return 1

            # Step 2 prep: load universe + registry
            full = _load_universe(cur, universe)
            if not full:
                msg = f"universe='{universe}' 無 active core+convex 成員"
                print(f"❌ {msg}")
                if hasattr(lifecycle, "mark_failed"):
                    lifecycle.mark_failed(msg)
                return 1

            registry_zombies, registry_na = None, None
            registry_source = "charter_baseline"
            if apply_registry:
                z, n = _load_registry(cur)
                if z is None:
                    print("⚠️  --apply-registry 但 universe_anomaly_registry 表不存在；fallback to charter baseline")
                else:
                    registry_zombies, registry_na = z, n
                    registry_source = "db_registry"
            if registry_zombies is None:
                registry_zombies = CHARTER_BASELINE_ZOMBIES
                registry_na = CHARTER_BASELINE_NA

            # §6.8.8-B (III).1: healthy_universe = full \ zombies
            healthy = [s for s in full if s not in registry_zombies]
            margin_na_set = {sid for sid, ds in registry_na if ds == DAILY_MARGIN_TABLE}
            healthy_for_margin = [s for s in healthy if s not in margin_na_set]

            # Step 3+5: coverage probe per frequency tier
            results = []
            ok_overall = True
            for tbl in DAILY_TRADING_TABLES:
                cov = _coverage_at_latest(cur, tbl, healthy, market_latest)
                ok = cov == len(healthy)
                if not ok:
                    ok_overall = False
                results.append({
                    "table": tbl, "tier": "daily",
                    "coverage": cov, "denominator": len(healthy),
                    "ok": ok,
                })
            cov_m = _coverage_at_latest(cur, DAILY_MARGIN_TABLE, healthy_for_margin, market_latest)
            ok_m = cov_m == len(healthy_for_margin)
            if not ok_m:
                ok_overall = False
            results.append({
                "table": DAILY_MARGIN_TABLE,
                "tier": "daily (effective)",
                "coverage": cov_m,
                "denominator": len(healthy_for_margin),
                "ok": ok_m,
            })

            fred = _fred_freshness(cur)

            # Final verdict
            verdict = "PASS" if ok_overall else "WARN"

            # Audit log
            try:
                write_data_audit_log(
                    "core_universe_membership", "SYSTEM",
                    today.strftime("%Y-%m-%d"),
                    f"UNIVERSE_COMPLETENESS_{verdict}",
                    len(healthy),
                )
            except Exception as exc:
                print(f"⚠️  data_audit_log write failed: {type(exc).__name__}: {exc}")

            # Report
            report_path = None
            if write_report:
                _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                report_path = _REPORTS_DIR / f"universe_completeness_{ts}.md"
                report_path.write_text(_render_report(
                    today=today, market_latest=market_latest, universe=universe,
                    full=full, healthy=healthy, healthy_for_margin=healthy_for_margin,
                    registry_zombies=registry_zombies, registry_na=registry_na,
                    registry_source=registry_source,
                    results=results, fred=fred, verdict=verdict,
                ), encoding="utf-8")

            # Console output
            print("🛡️" * 40)
            print(f"🚀 Quantum Finance: 核心股 universe-wide 完整性 probe ({TOOL_VER})")
            print("🛡️" * 40)
            print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §6.8.8-A/B/C")
            print(f"universe : {universe}")
            print(f"today    : {today}    market_latest : {market_latest}")
            print(f"|full|={len(full)} |zombies|={len(registry_zombies)} "
                  f"|healthy|={len(healthy)} |healthy_margin|={len(healthy_for_margin)}")
            print(f"registry source: {registry_source}")
            print("─" * 80)
            for r in results:
                tag = "OK" if r["ok"] else "MISS"
                print(f"  [{tag:>4s}] {r['table']:<48s} {r['coverage']}/{r['denominator']}")
            print("─" * 80)
            for sid, info in sorted(fred.items()):
                print(f"  FRED {sid:<8s} max={info['max']} count={info['count']}")
            print("─" * 80)
            if report_path:
                print(f"📄 報告 : {report_path.relative_to(_PROJECT_ROOT)}")
            print(f"🕒 耗時 : {(time.time() - start) * 1000:.2f} ms")
            print(f"⚖️  判定 : {verdict}")
            print("🛡️" * 40)

            if not ok_overall and hasattr(lifecycle, "mark_warning"):
                lifecycle.mark_warning(f"DB_COVERAGE_OK=False on universe={universe}")

            return 0 if verdict in ("PASS", "WARN") else 1
        except Exception as exc:
            msg = f"completeness probe failed: {type(exc).__name__}: {exc}"
            print(f"❌ {msg}")
            if hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(msg)
            return 1
        finally:
            cur.close()
            conn.close()


def _render_report(*, today, market_latest, universe, full, healthy, healthy_for_margin,
                   registry_zombies, registry_na, registry_source,
                   results, fred, verdict) -> str:
    lines = [
        f"# Universe Completeness Probe ({TOOL_VER})",
        "",
        f"- constitution: {CONSTITUTION_VER}",
        f"- universe: {universe}",
        f"- today: {today}",
        f"- market_latest: {market_latest}",
        f"- registry_source: {registry_source}",
        f"- |full|={len(full)} |zombies|={len(registry_zombies)} |healthy|={len(healthy)} |healthy_for_margin|={len(healthy_for_margin)}",
        f"- verdict: **{verdict}**",
        "",
        "## §6.8.8-B exclusions (active)",
        "",
        f"- class A (zombies): {sorted(registry_zombies)}",
        f"- class D (structural NA, {len(registry_na)} pairs):",
    ]
    for sid, ds in sorted(registry_na):
        lines.append(f"  - ({sid}, {ds})")
    lines += [
        "",
        "## Coverage at market_latest",
        "",
        "| table | tier | coverage | denominator | ok |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]
    for r in results:
        lines.append(f"| `{r['table']}` | {r['tier']} | {r['coverage']} | {r['denominator']} | {'OK' if r['ok'] else 'MISS'} |")
    lines += ["", "## FRED 4 series", "", "| series_id | max_date | count |", "| :--- | :--- | ---: |"]
    for sid, info in sorted(fred.items()):
        lines.append(f"| {sid} | {info['max']} | {info['count']} |")
    lines += ["", f"_Generated by check_universe_completeness.py {TOOL_VER}_", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Universe completeness probe ({TOOL_VER})")
    parser.add_argument("--universe", default="core",
                        help="universe to probe (default: core; only 'core' supported in v0.1)")
    parser.add_argument("--apply-registry", action="store_true",
                        help="load §6.8.8-C universe_anomaly_registry from DB (default: charter baseline)")
    parser.add_argument("--no-report", action="store_true",
                        help="skip writing reports/universe_completeness_*.md")
    args = parser.parse_args()
    sys.exit(run(args.universe, args.apply_registry, not args.no_report))

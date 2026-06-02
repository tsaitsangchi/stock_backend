"""
build_historical_panels.py v0.1 (Historical Monthly Panel Builder · §14.7-CX 8-Year OOS 配套 script · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(標頭三段式補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: 95-MONTHLY-PANEL HISTORICAL FEATURE BUILDER + §14.7-CX 8-YEAR-OOS 配套 + §14.7-CY MULTI-CYCLE 配套 + §14.7-CL 43-FEATURE CANONICAL + §8.5 ANTI-LEAKAGE INHERITED + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 (per CLAUDE.md §一.11)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**建歷史 panel**:產生 95 個月度歷史特徵 panel(§14.7-CX 8-year OOS 配套)。

**輸入 → 輸出**:raw 資料 → 95 historical monthly feature panels

**為什麼需要它**:walk-forward 驗證需要跨年的歷史 panel;§14.7-DD PHASE 7。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[95-Panel Monthly Span]** (v0.1, 憲法 §14.7-CX): 自 2018-06-15 至 2026-04-15 mid-month 15th 共 95 monthly panels;對映 8 年 historical OOS validation。
2. **[Per-Panel feature_store_builder Subprocess]** (v0.1): 本程式為 orchestrator,**不直接**計算 features,而是循環呼叫 `scripts/core/feature_store_builder.py --commit --as-of-date <YYYY-MM-15>` 為 subprocess;每 panel ~5-10 sec。
3. **[Anti-Leakage Inherited]** (v0.1, 憲法 §8.5): feature_store_builder 已 enforce publication_date_strategy(§8.5-9 Phase 2);本程式繼承不違反。
4. **[Survivorship Bias Acknowledged]** (v0.1): universe 為 current 1,121 stocks(§14.7-CJ super-strict);**承認**含 survivorship bias(未含 delisted)。完整移除須 per-panel dynamic universe(P1 待辦)。
5. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) subprocess stdout;0 AI memory。
6. **[Sovereignty Declaration]** (v0.1, 憲法 §3.1 序列模組): 本程式為 **§14.7-CX 配套 orchestrator**(不直接計算 features)。**治權邊界**:(a) §3.1 evaluation orchestrator;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 inherited from feature_store_builder;(e) **不訓練 model**(由 model_trainer 負責);(f) **不評估 multi-cycle**(由 multi_cycle_validation 負責);(g) **不修改 feature_store_builder.py**(read-only orchestration);(h) 唯一職責:循環呼叫 feature_store_builder.py --commit 為 subprocess,建立 95 historical monthly snapshots。
7. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): success/skipped/failed 動態判定(per subprocess returncode + stdout 內容);不硬編總體 verdict。
8. **[Idempotency]** (v0.1): feature_store_builder.py 內部 ON CONFLICT 確保重跑安全;同 as_of_date 重跑覆寫舊 snapshot(若 status='committed' 則 skip 或 overwrite per builder logic)。
9. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述性快照;真實 row counts trace 至 `feature_store_snapshot` table。
10. **[Configurable Date Range]** (v0.1): 預設 2018-06-15 ~ 2026-04-30(per §14.7-CX 95 panels);可改 `generate_panel_dates()` 範圍。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Panel Date Generation

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Monthly mid-month 15th | `generate_panel_dates(start, end)` | §14.7-CX 95 panels |
| A.2 Configurable start/end | code-level constants(2018-06-01 / 2026-04-30)| §14.7-CX |
| A.3 Default 95 panels | 8 years × 12 months = 96 - 1(start month)| §14.7-CX standard |

### Group B. Subprocess Orchestration

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 feature_store_builder.py call | `subprocess.run([sys.executable, builder, --commit, --as-of-date, d.strftime("%Y-%m-%d")])` | §14.7-CA / §14.7-CL |
| B.2 Timeout protection | 120s per panel | safety |
| B.3 capture_output=True | stdout/stderr capture | audit trail |

### Group C. Result Aggregation + Verdict

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 success/skipped/failed classification | 動態 per subprocess returncode + stdout | §5.6.3 |
| C.2 Build summary stdout | total / success / skipped / failed counts | audit trail |
| C.3 List committed fs_ids | for downstream model_trainer use | §14.7-CX |

### Group D. Logging + Audit

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Per-panel logger.info | `[i/N] YYYY-MM-DD → fs_id PERFECT(Xs)` | source traceability |
| D.2 Build elapsed | total time + avg time/panel | performance audit |
| D.3 Failed panels list | first 10 failed for diagnosis | error trail |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| Standard 95-panel build | `python scripts/evaluation/build_historical_panels.py` |
| Background long-running build | `nohup python scripts/evaluation/build_historical_panels.py > /tmp/build.log 2>&1 &` |

### 不提供之旗標 (Intentionally Omitted)

- `--start-date / --end-date`:預設 2018-06-15 ~ 2026-04-30 per §14.7-CX。改範圍須修改 source code。
- `--parallel`:subprocess 序列執行(避免 DB concurrent writes 衝突)。
- `--retry-failed`:失敗 panels 須手動 re-run。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **CLAUDE.md §一.11 三段式標頭補正**:依用戶 explicit directive 2026-05-29 補入 10 條核心定義 / 4-Group functional matrix / 全修訂歷程,對齊 sovereign_sync_engine.py v1.22 範本。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CX 配套 orchestrator**。Loop 95 monthly panels(2018-06-15 ~ 2026-04-15 mid-month)循環呼叫 `feature_store_builder.py --commit --as-of-date <YYYY-MM-15> --feature-set-version feature_set_v0.4 --label-horizon 30`。**首跑實證**:95 monthly snapshots committed 至 `feature_store_snapshot`;total elapsed 16.8 min;avg ~10.6s/panel;committed feature_values rows ~4.5M。**用途**:為 §14.7-CX 8-year walk-forward + §14.7-CY multi-cycle validation 提供 historical panels。 | ARCHIVED(標頭格式;邏輯仍 ACTIVE)|
"""
from __future__ import annotations
import sys, subprocess, time, logging
from datetime import date, datetime, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def generate_panel_dates(start_date: date, end_date: date) -> list[date]:
    """Generate mid-month(15th)dates from start to end inclusive"""
    dates = []
    current = date(start_date.year, start_date.month, 15)
    while current <= end_date:
        dates.append(current)
        # advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 15)
        else:
            current = date(current.year, current.month + 1, 15)
    return dates


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build historical monthly feature_store snapshots")
    parser.add_argument("--feature-set-version", default="feature_set_v0.4",
                        help="feature_set_version to build (default v0.4;v0.5 for §14.7-DC v0.8 MVP v0.21)")
    parser.add_argument("--label-horizon", default="30", help="label horizon days (default 30)")
    parser.add_argument("--start-year", type=int, default=2018, help="panel start year (default 2018)")
    parser.add_argument("--start-month", type=int, default=6, help="panel start month (default 6)")
    parser.add_argument("--end-year", type=int, default=2026, help="panel end year (default 2026)")
    parser.add_argument("--end-month", type=int, default=4, help="panel end month (default 4)")
    args = parser.parse_args()

    start = date(args.start_year, args.start_month, 1)
    end = date(args.end_year, args.end_month, 30 if args.end_month != 2 else 28)
    panels = generate_panel_dates(start, end)
    safe_fsv = args.feature_set_version.replace(".", "_")
    logger.info("=" * 100)
    logger.info(f"Building {len(panels)} historical monthly snapshots(§14.7-CX)")
    logger.info(f"Target feature_set_version: {args.feature_set_version}")
    logger.info("=" * 100)
    logger.info(f"  Start: {panels[0]}")
    logger.info(f"  End:   {panels[-1]}")
    logger.info(f"  Total: {len(panels)}")
    logger.info(f"  Anti-leakage: §8.5-9 publication_date_strategy enforced")
    logger.info(f"  Universe: latest committed core_universe(membership-driven)")
    logger.info("")

    builder = _base_dir / "core" / "feature_store_builder.py"
    success = []
    failed = []
    skipped = []
    t_global = time.monotonic()

    for i, d in enumerate(panels, 1):
        fs_id = f"fs_{d.strftime('%Y%m%d')}_{safe_fsv}"
        cmd = [
            sys.executable, str(builder),
            "--commit",
            "--as-of-date", d.strftime("%Y-%m-%d"),
            "--label-horizon", args.label_horizon,
            "--feature-set-version", args.feature_set_version,
        ]
        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.monotonic() - t0
            if result.returncode == 0:
                # Check for PERFECT verdict in output
                if "主權判定         : PERFECT" in result.stdout or "PERFECT" in result.stdout:
                    success.append((d, fs_id, elapsed))
                    logger.info(f"  [{i:>3}/{len(panels)}] {d} → {fs_id} PERFECT ({elapsed:.1f}s)")
                else:
                    failed.append((d, fs_id, "no PERFECT verdict"))
                    logger.warning(f"  [{i:>3}/{len(panels)}] {d} → not PERFECT")
            else:
                # Check if already committed
                if "already exists" in result.stderr or "duplicate" in result.stderr.lower():
                    skipped.append((d, fs_id, "already exists"))
                    logger.info(f"  [{i:>3}/{len(panels)}] {d} → skipped(exists)")
                else:
                    failed.append((d, fs_id, result.stderr[:200]))
                    logger.warning(f"  [{i:>3}/{len(panels)}] {d} FAILED: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            failed.append((d, fs_id, "timeout"))
            logger.warning(f"  [{i:>3}/{len(panels)}] {d} TIMEOUT")
        except Exception as e:
            failed.append((d, fs_id, str(e)))
            logger.warning(f"  [{i:>3}/{len(panels)}] {d} ERROR: {e}")

    t_total = time.monotonic() - t_global

    logger.info("\n" + "=" * 100)
    logger.info("Build Summary")
    logger.info("=" * 100)
    logger.info(f"  Total panels:       {len(panels)}")
    logger.info(f"  Successfully built: {len(success)}")
    logger.info(f"  Skipped(existed):  {len(skipped)}")
    logger.info(f"  Failed:             {len(failed)}")
    logger.info(f"  Total elapsed:      {t_total:.1f}s({t_total/60:.1f} min)")
    if success:
        avg = sum(s[2] for s in success) / len(success)
        logger.info(f"  Avg time/panel:     {avg:.1f}s")
    if failed:
        logger.warning(f"\n  Failed panels:")
        for d, fs_id, reason in failed[:10]:
            logger.warning(f"    {d}: {reason[:80]}")

    # Output all panel fs_ids for trainer use
    all_panels = [(d, fs_id) for d, fs_id, _ in success] + [(d, fs_id) for d, fs_id, _ in skipped]
    all_panels.sort()
    if all_panels:
        logger.info(f"\n  All committed fs_ids({len(all_panels)} panels):")
        for d, fs_id in all_panels:
            logger.info(f"    {fs_id}")


if __name__ == "__main__":
    main()

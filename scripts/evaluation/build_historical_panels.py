"""
build_historical_panels.py — Build 95 monthly historical feature_store snapshots
================================================================================
治權:§14.7-CX 8-year historical OOS backtest(v6.18.0 pending)
用戶 directive 2026-05-28:multi-period historical validation 為 institutional standard

Builds monthly feature_store snapshots from 2018-06-15 to 2026-04-15(95 panels)
for walk-forward LGBM training and regime-segmented OOS analysis.

Methodology disclosures:
  1. Universe: current 1,121 stocks(§14.7-CJ super-strict)— survivorship bias acknowledged
  2. Anti-leakage: feature_store_builder uses publication_date_strategy(§8.5-9 Phase 2)
  3. Snapshot frequency: monthly(mid-month 15th)
  4. Label horizon: 30 calendar days
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
    # 8-year span: 2018-06 to 2026-04 (BalanceSheet starts 2018-05)
    start = date(2018, 6, 1)
    end = date(2026, 4, 30)
    panels = generate_panel_dates(start, end)
    logger.info("=" * 100)
    logger.info(f"Building {len(panels)} historical monthly snapshots(§14.7-CX)")
    logger.info("=" * 100)
    logger.info(f"  Start: {panels[0]}")
    logger.info(f"  End:   {panels[-1]}")
    logger.info(f"  Total: {len(panels)}")
    logger.info(f"  Anti-leakage: §8.5-9 publication_date_strategy enforced")
    logger.info(f"  Universe: current 1,121 stocks(survivorship bias acknowledged)")
    logger.info("")

    builder = _base_dir / "core" / "feature_store_builder.py"
    success = []
    failed = []
    skipped = []
    t_global = time.monotonic()

    for i, d in enumerate(panels, 1):
        fs_id = f"fs_{d.strftime('%Y%m%d')}_feature_set_v0_4"
        cmd = [
            sys.executable, str(builder),
            "--commit",
            "--as-of-date", d.strftime("%Y-%m-%d"),
            "--label-horizon", "30",
            "--feature-set-version", "feature_set_v0.4"
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

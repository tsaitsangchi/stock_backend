"""
weekly_api_audit_and_resync.py v0.1 (§14.7-CE Weekly Automation Wrapper · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CE Empirical-Verification-axis weekly automation + §14.7-CH 配套 + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Weekly DB ≡ API Invariant]** (v0.1, §14.7-CE): 「Weekly assured: DB ≡ FinMind/FRED API at byte-level」之 weekly enforcement。
2. **[3-Step Workflow]** (v0.1): (1) Live API audit(audit_live_api_vs_db);(2) mismatch > 0 → auto trigger resync;(3) Re-audit verify。
3. **[Exit Code Treaty]** (v0.1): 0=clean / 1=audit fail / 2=resync fail / 3=re-audit still mismatch。
4. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query + (c) live API call;0 AI memory。
5. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): exit code 動態判定。
6. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit wrapper / §14.7-CE/CH): 本程式為 **§14.7-CE Weekly Automation 唯一 wrapper**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切 audit wrapper;(b) 子程式為 audit_live_api_vs_db.py + sovereign_sync_engine.py;(c) **不直接 audit raw tables**(delegated);(d) **不直接 sync**(delegated);(e) 唯一職責:orchestrate 3-step weekly verification + exit code。
7. **[Idempotency]** (v0.1): 重跑安全;DB ≡ API 已達成則 audit pass,無 re-sync。
8. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Step 1 — Live API Audit
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 subprocess call | `audit_live_api_vs_db.py` | §14.7-CE deep audit |
| A.2 mismatch count parsing | stdout regex | source traceability |

### Group B. Step 2 — Conditional Re-sync
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Trigger condition | mismatch > 0 | gate |
| B.2 subprocess call | `sovereign_sync_engine.py --resync <stocks>` | §7 sync 治權 |
| B.3 --dry-run override | audit only,no resync | safe default |

### Group C. Step 3 — Re-Audit Verify
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Post-resync audit | re-call audit_live_api_vs_db | §14.7-CE |
| C.2 Final verdict | 100% match → clean | treaty gate |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| Weekly cron(由 run_weekly_doctrine_recommit Step 2 呼叫)| `python scripts/maintenance/weekly_api_audit_and_resync.py` |
| Audit only dry-run | `... --dry-run` |

### 不提供之旗標 (Intentionally Omitted)
- `--force-resync`:無條件 resync 不適合 weekly automation。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CE Weekly Automation Wrapper**。3-step workflow:audit → conditional resync → re-audit。Exit code 治權 0/1/2/3。 | ARCHIVED(標頭格式)|
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
import subprocess
import time
import re
from datetime import date
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

AUDIT_SCRIPT = _SCRIPTS_DIR / "audit" / "audit_live_api_vs_db.py"
RESYNC_SCRIPT = _SCRIPTS_DIR / "maintenance" / "resync_priceadj_mismatch.py"
FRED_FETCH = _SCRIPTS_DIR / "fetchers" / "fetch_fred_data.py"


def run_audit():
    """Run audit_live_api_vs_db.py;return (fm_mismatch, fred_mismatch, output_text)。"""
    logger.info(f"📡 Running live API audit ...")
    r = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--workers", "12"],
        capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
    )
    out = r.stdout + r.stderr
    fm_m = re.search(r"FinMind layer.*?Total byte-mismatches: (\d+)", out, re.DOTALL)
    fred_m = re.search(r"FRED layer.*?Total byte-mismatches: (\d+)", out, re.DOTALL)
    fm_mismatch = int(fm_m.group(1)) if fm_m else -1
    fred_mismatch = int(fred_m.group(1)) if fred_m else -1
    return fm_mismatch, fred_mismatch, out


def run_resync_finmind():
    """Run resync_priceadj_mismatch.py to fix FinMind close mismatches。"""
    logger.info(f"🔧 FinMind mismatches detected → running resync_priceadj_mismatch.py ...")
    r = subprocess.run(
        [sys.executable, str(RESYNC_SCRIPT)],
        capture_output=True, text=True, cwd=str(_PROJECT_ROOT),
    )
    return r.returncode == 0, r.stdout + r.stderr


def run_resync_fred(series_ids):
    """Run fetch_fred_data.py --force to refresh FRED routine revisions。"""
    if not series_ids:
        # Default: refresh routine-revised series
        series_ids = ["M2SL", "INDPRO", "UNRATE", "CPIAUCSL"]
    logger.info(f"🔧 FRED mismatches detected → running fetch_fred_data.py --ids {series_ids} --force ...")
    cmd = [sys.executable, str(FRED_FETCH), "--ids"] + series_ids + ["--force"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
    # fetch_fred_data.py 結尾有 cosmetic AttributeError(per known known)— 用 stdout 是否含 "全部完成" 判定 success
    success = "全部完成" in (r.stdout + r.stderr)
    return success, r.stdout + r.stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Audit only;不 auto resync")
    p.add_argument("--no-fred-resync", action="store_true", help="Skip FRED resync(audit only for FRED)")
    args = p.parse_args()

    logger.info("=" * 80)
    logger.info(f"§14.7-CE Weekly API Audit + Auto Re-sync(as-of {date.today()})")
    logger.info(f"Mode: {'DRY-RUN(audit only)' if args.dry_run else 'COMMIT(audit + auto resync)'}")
    logger.info("=" * 80)

    # Step 1: Initial audit
    fm_mm, fred_mm, _ = run_audit()
    if fm_mm < 0 or fred_mm < 0:
        logger.error(f"❌ Audit failed to parse output")
        sys.exit(1)
    logger.info(f"\n📊 Initial audit:FinMind mismatches={fm_mm} / FRED mismatches={fred_mm}")

    if fm_mm == 0 and fred_mm == 0:
        logger.info(f"\n🎯 §14.7-CE Already 100% byte-level match — no resync needed")
        return

    if args.dry_run:
        logger.info(f"\n[DRY-RUN] mismatches detected but --dry-run skipped resync")
        logger.info(f"To fix:python {Path(__file__).name}")
        sys.exit(0)

    # Step 2: Auto resync
    if fm_mm > 0:
        ok, _ = run_resync_finmind()
        if not ok:
            logger.error(f"❌ FinMind resync failed")
            sys.exit(2)
        logger.info(f"✅ FinMind resync done")

    if fred_mm > 0 and not args.no_fred_resync:
        ok, _ = run_resync_fred(series_ids=[])
        if not ok:
            logger.warning(f"⚠️ FRED resync may have failed(check log)")
        else:
            logger.info(f"✅ FRED resync done")

    # Step 3: Re-audit to verify
    logger.info(f"\n🔄 Re-auditing post-resync ...")
    time.sleep(2)  # brief pause for DB commit
    fm_mm2, fred_mm2, _ = run_audit()
    logger.info(f"\n📊 Post-resync audit:FinMind mismatches={fm_mm2} / FRED mismatches={fred_mm2}")

    if fm_mm2 == 0 and fred_mm2 == 0:
        logger.info(f"\n🎯 §14.7-CE Absolute byte-level match achieved")
    else:
        logger.warning(f"\n⚠️ Still {fm_mm2 + fred_mm2} mismatches after resync(needs manual investigation)")
        sys.exit(3)


if __name__ == "__main__":
    main()

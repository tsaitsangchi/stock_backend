"""
run_weekly_doctrine_recommit.py v0.1 (§14.7-BX Phase C-3 — Weekly Doctrine-Driven Recommit Pipeline)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §14.7-BX Phase C-3 落地;Phase D-1 cron 啟動需 Phase C-2 之 M1/M2/M3 sub-option 治權選定後才可)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions)
1. [Weekly Recommit Orchestrator]: 本工具為 §14.7-BX inscribed 之 weekly doctrine recommit pipeline 之 orchestrator;依序執行 fetch FRED → compute K-wave proxies → build doctrine-gate universe(weekly mode)→ audit → drift report 5 步。
2. [Atomic Supersede]: 透過 build_doctrine_gate_universe.py --weekly-mode 確保 §6.7 SSOT 在過渡期間維持(任一時點 ≤ 1 committed snapshot)。
3. [Drift Report]: 每週生成 `reports/weekly_universe_recommit_<YYYYMMDD>.md`;對比前週 vs 當週 gate-pass set 差異(added/removed/stable)。
4. [Trading Day Check]: 預期執行時點為台股交易日收盤後(每週五 13:30 後);若提前執行則 WARN 但仍允許(--force-now 跳過 check)。
5. [Phase C-2 Pre-condition]: weekly cron 啟動前須先完成 Phase C-2(M1/M2/M3 model retrain 策略治權選定)+ Phase D-2(model_trainer / feature_store weekly mode);**否則 weekly recommit 將造成下游 model 不一致**(本工具不檢查此 prerequisite — orchestration 為治權者責任)。
6. [Zero Hardcoded Verdict]: 動態判定 verdict per §5.6.3。
7. [Sovereignty Declaration]: 本工具屬 §11C 治權檢驗延伸 + §14.7-BX Phase C orchestrator;不選股(委 builder)、不訓練模型、不預測、不分配資金;不涉 §0.1-A / §0.2-A / §0.3-A 五套禁令;不在 T1-T3 分層內。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 場景 | 指令 |
| :--- | :--- |
| Dry-run(顯示 pipeline 各步,不寫 DB)| `$ python scripts/maintenance/run_weekly_doctrine_recommit.py --dry-run` |
| Commit(執行 5 步 pipeline 並寫 DB)| `$ python scripts/maintenance/run_weekly_doctrine_recommit.py --commit` |
| 強制執行(非交易日 / 收盤前) | `$ python scripts/maintenance/run_weekly_doctrine_recommit.py --commit --force-now` |
| 跳過 FRED sync(若獨立 cron 已 sync)| `$ python scripts/maintenance/run_weekly_doctrine_recommit.py --commit --skip-fred-sync` |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.7-§一.11 | 2026-05-29 | Codex | **§一.11 三段式標頭規範對齊**:§二 標題「執行指令」→「全量維運指令總矩陣」+ §三 標題「修訂歷程」→「全修訂歷程 (Full Revision History)」對齊 CLAUDE.md §一.11 強制格式。原 v0.7 邏輯不變。 | **ACTIVE** |
| v0.7 | 2026-05-28 | Codex | **§14.7-CP H4/H5/H8 + §14.7-CT 整合(v6.14.1)**:Step 10 H4 Data Quality / Step 11 H5 Universe Selection / Step 12 H8 Survivorship audit 加入(per §14.7-CP T_CP-3 mandatory pre-check before §10 model_trainer);Step 13 prediction inference placeholder(per §14.7-CT manual trigger)。Weekly cron 完整涵蓋 9 + 3 + 1 = 13 steps(feature audits + model pre-checks + inference)。 | ACTIVE |
| v0.6 | 2026-05-28 | Codex | **§14.7-CN + §14.7-CO 整合(v6.11.1 patch)**:Step 8 插入 `audit_feature_necessity.py`(4-path necessity verdict);Step 9 插入 `audit_feature_sign_stability.py`(sign verdict + lit consistency)。Weekly cron 自動執行全 3 個 feature-layer audit(IC + necessity + sign)。 | SUPERSEDED |
| v0.5 | 2026-05-28 | Codex | **§14.7-CM 整合**:Step 7 插入 `audit_feature_ic_vs_future_return.py`(43 features × forward N-day return Spearman IC)。每週重算 IC 實證 model-training viability + treaty gate(Mean |IC|>0.03 + ≥30% sig);違反觸發 feature re-evaluation per T_CM-3。 | SUPERSEDED |
| v0.4 | 2026-05-28 | Codex | §14.7-CJ Step 4 升 super-strict `--with-reasonableness-gate`(v0.15 policy);觸發 outlier features 之 stocks 排除。 | SUPERSEDED |
| v0.3 | 2026-05-28 | Codex | §14.7-CI Step 4 升 `--with-feature-gate`(v0.14 strict)— 不符合 37/37 features 計算之 stocks 嚴格排除。 | SUPERSEDED |
| v0.2 | 2026-05-28 | Codex | **§14.7-CE P1 整合**:Step 3.5 插入 `weekly_api_audit_and_resync.py`(live API audit + auto resync)。確保 weekly recommit 之 Step 4 native gate build 前 DB 已 ≡ FinMind/FRED API byte-level;mismatch 自動 re-sync(per §14.7-CE absolute byte-level closure)。`--skip-api-audit` flag 加入。 | SUPERSEDED |
| v0.1 | 2026-05-26 | Codex | §14.7-BX Phase C-3 落地首版:5 步 pipeline orchestrator + atomic supersede via builder --weekly-mode + drift report 生成。**Phase C-2 + Phase D-2 之 cron 啟動前置條件 為治權者責任,本工具不檢查**(若直接 cron 跑時下游 model 未升 weekly mode → IC degradation 風險自負)。 | SUPERSEDED |
================================================================================
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.7"  # §14.7-CP H4/H5/H8 Steps 10/11/12 + §14.7-CT Step 13 added(v6.14.1 2026-05-28)


def check_trading_day_close():
    """Check 當前是否為台股交易日收盤後(週五 13:30 後)."""
    now = datetime.now()
    if now.weekday() == 4 and (now.hour > 13 or (now.hour == 13 and now.minute >= 30)):
        return True, "✅ Friday after 13:30"
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return False, f"⚠️ Current: {weekday_names[now.weekday()]} {now.hour:02d}:{now.minute:02d};expected Friday after 13:30"


def get_current_committed_snapshot(cur):
    """讀取當前 committed snapshot(預期 ≤ 1 per §6.7 SSOT)."""
    cur.execute("""
        SELECT snapshot_id, as_of_date, policy_version, core_count, convex_count
        FROM core_universe_snapshot WHERE status='committed'
        ORDER BY as_of_date DESC LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        return None
    return {
        'snapshot_id': row[0],
        'as_of_date': row[1],
        'policy_version': row[2],
        'core_count': row[3],
        'convex_count': row[4],
    }


def get_universe_stocks(cur, snapshot_id):
    """取 snapshot 之 core+convex universe stock_id set."""
    cur.execute("""
        SELECT stock_id, stock_name, core_tier, industry_category
        FROM core_universe_membership
        WHERE snapshot_id=%s AND core_tier IN ('core_universe', 'convex_universe')
    """, (snapshot_id,))
    return {r[0]: {'name': r[1], 'tier': r[2], 'industry': r[3]} for r in cur.fetchall()}


def compute_drift(prior_set, current_set):
    """Compute added / removed / stable stocks between two universe sets."""
    prior_ids = set(prior_set.keys())
    current_ids = set(current_set.keys())
    return {
        'added': sorted(current_ids - prior_ids),
        'removed': sorted(prior_ids - current_ids),
        'stable': sorted(current_ids & prior_ids),
        'prior_n': len(prior_ids),
        'current_n': len(current_ids),
        'churn_pct': round(100 * (len(current_ids ^ prior_ids) / max(len(prior_ids), 1)), 2),
    }


def write_drift_report(drift, prior_snapshot, current_snapshot, prior_set, current_set, as_of):
    """Write reports/weekly_universe_recommit_<YYYYMMDD>.md."""
    fp = _PROJECT_ROOT / "reports" / f"weekly_universe_recommit_{as_of}.md"
    lines = []
    lines.append(f"# Weekly Universe Recommit Drift Report — {as_of}\n")
    lines.append(f"**生成於**: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"**§14.7-BX Phase C-3 orchestrator**: `run_weekly_doctrine_recommit.py` v{TOOL_VER}\n")
    lines.append("---\n")
    lines.append("## 1. Snapshot 比對\n")
    lines.append("| 項目 | 前週 | 當週 |")
    lines.append("|---|---|---|")
    lines.append(f"| snapshot_id | {prior_snapshot['snapshot_id'] if prior_snapshot else '(無)'} | {current_snapshot['snapshot_id']} |")
    lines.append(f"| as_of_date | {prior_snapshot['as_of_date'] if prior_snapshot else '-'} | {current_snapshot['as_of_date']} |")
    lines.append(f"| policy | {prior_snapshot['policy_version'] if prior_snapshot else '-'} | {current_snapshot['policy_version']} |")
    lines.append(f"| N(core+convex)| {drift['prior_n']} | {drift['current_n']} |\n")
    lines.append("## 2. Drift summary\n")
    lines.append(f"- **Added**({len(drift['added'])}支)")
    lines.append(f"- **Removed**({len(drift['removed'])}支)")
    lines.append(f"- **Stable**({len(drift['stable'])}支)")
    lines.append(f"- **Churn rate**:{drift['churn_pct']}%(per §14.7-BX T_BX-3 stability check;若 > 5% 表 §0.1 source 或 §0.3 indicator 急劇變動)\n")
    if drift['added']:
        lines.append("### 2.1 Added(新進 doctrine pass-set)")
        for sid in drift['added']:
            info = current_set[sid]
            lines.append(f"- {sid} {info['name']} ({info['industry']})")
        lines.append("")
    if drift['removed']:
        lines.append("### 2.2 Removed(退出 doctrine pass-set)")
        for sid in drift['removed']:
            info = prior_set[sid]
            lines.append(f"- {sid} {info['name']} ({info['industry']})")
        lines.append("")
    lines.append("\n## 3. Stability verdict\n")
    if drift['churn_pct'] < 5:
        lines.append("✅ 穩定(per T_BX-3:churn < 5% threshold)")
    elif drift['churn_pct'] < 10:
        lines.append("🟡 中度 churn(5-10%);觀察 §0.1 source 或 §0.3 indicator 變動")
    else:
        lines.append("⚠️ 高 churn(> 10%);需治權者人工 review")
    lines.append("\n---\n")
    lines.append("**Status**: ✅ §14.7-BX Phase C-3 weekly drift evidence")
    fp.write_text("\n".join(lines))
    return fp


def run_step(name, cmd, dry_run, allow_fail=False):
    """Run a subprocess step;在 dry-run 模式下只 print。"""
    print(f"\n──── [{name}] ────")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        print(f"  (dry-run skip)")
        return 0
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        if allow_fail:
            print(f"  ⚠️ [{name}] returned {result.returncode} but allow_fail=True;continuing")
            return result.returncode
        print(f"  ❌ [{name}] failed with returncode {result.returncode}")
        sys.exit(result.returncode)
    print(f"  ✅ [{name}] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=f"§14.7-BX Phase C-3 Weekly Doctrine Recommit ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show pipeline steps without executing")
    mode.add_argument("--commit", action="store_true", help="Execute 5-step pipeline + commit new weekly snapshot")
    parser.add_argument("--force-now", action="store_true", help="Bypass trading-day-close check(allow non-Friday or pre-close)")
    parser.add_argument("--skip-fred-sync", action="store_true", help="Skip FRED sync(若獨立 cron 已 sync)")
    parser.add_argument("--skip-api-audit", action="store_true", help="Skip §14.7-CE live API audit + auto resync")
    args = parser.parse_args()

    as_of = date.today().isoformat()
    print("=" * 72)
    print(f"§14.7-BX Phase C-3 — Weekly Doctrine Recommit Pipeline ({TOOL_VER})")
    print(f"As-of: {as_of} / Mode: {'DRY-RUN' if args.dry_run else 'COMMIT'}")
    print("=" * 72)

    # Trading-day check
    is_close, msg = check_trading_day_close()
    print(f"\n[Pre-check] Trading day:{msg}")
    if not is_close and not args.force_now:
        print(f"  ❌ Not at trading-day close;use --force-now to bypass")
        sys.exit(1)

    # Snapshot before pipeline
    conn = get_db_connection()
    cur = conn.cursor()
    prior_snap = get_current_committed_snapshot(cur)
    prior_set = get_universe_stocks(cur, prior_snap['snapshot_id']) if prior_snap else {}
    conn.close()
    print(f"\n[Pre-pipeline] Prior committed:{prior_snap['snapshot_id'] if prior_snap else '(none)'} / N={len(prior_set)}")

    # ---- Step 1: Sync FRED ----
    if not args.skip_fred_sync:
        run_step("Step 1: FRED sync(M2SL/T10Y2Y/VIXCLS et al)",
                 [sys.executable, "scripts/fetchers/fetch_fred_data.py"],
                 args.dry_run, allow_fail=True)
    else:
        print("\n──── [Step 1: FRED sync] SKIPPED(--skip-fred-sync)────")

    # ---- Step 2/3 DEPRECATED per §14.7-CC FRED-native(IPG3344S + PCU4831114831115 取代 system-computed proxies)
    # 保留為註解 audit trail;v0.13 native gate 不需此 step
    # run_step("Step 2: TW_SEMI_VWAP_YOY proxy", ...)
    # run_step("Step 3: TW_SHIPPING_VWAP_YOY proxy", ...)

    # ---- Step 3.5(§14.7-CE P1): Live API audit + auto resync ----
    # 確認 DB ≡ FinMind/FRED API byte-level;mismatch 自動 re-sync
    # 必須在 Step 4 native gate build 前完成,確保 builder 讀的是最新 API 資料
    if not args.skip_api_audit:
        run_step("Step 3.5: §14.7-CE live API audit + auto resync",
                 [sys.executable, "scripts/maintenance/weekly_api_audit_and_resync.py"],
                 args.dry_run, allow_fail=True)
    else:
        print("\n──── [Step 3.5: API audit + resync] SKIPPED(--skip-api-audit)────")

    # ---- Step 4: Native gate v0.14 strict(§14.7-CI;production-mode strict)----
    # OLD:scripts/maintenance/build_doctrine_gate_universe.py(SUPERSEDED;標 DEPRECATED)
    # OLD:scripts/maintenance/apply_feature_completeness_gate.py + apply_raw_data_completeness_gate.py
    # OLD-v6.5.0:--mode doctrine-native(v0.13 standard / 寬鬆)
    # NEW-v6.5.1:--mode doctrine-native --with-feature-gate(v0.14 strict 嚴格)per §14.7-CI 用戶 directive
    #   「不符合條件就不入核心股」+ Stage 4-feature 37/37 enforcement
    run_step("Step 4: §14.7-CJ v0.15 super-strict native gate (Stage 1+2+3+4+4-feature+4-reasonable)",
             [sys.executable, "scripts/core/core_universe_builder.py",
              "--mode", "doctrine-native", "--with-feature-gate", "--with-reasonableness-gate", "--commit"],
             args.dry_run)

    # ---- Step 5: Audit ----
    run_step("Step 5: audit_universe_completeness.py",
             [sys.executable, "scripts/maintenance/audit_universe_completeness.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 6: Drift report ----
    print("\n──── [Step 6: Drift report] ────")
    if args.dry_run:
        print("  (dry-run skip drift report)")
    else:
        conn = get_db_connection()
        cur = conn.cursor()
        current_snap = get_current_committed_snapshot(cur)
        current_set = get_universe_stocks(cur, current_snap['snapshot_id']) if current_snap else {}
        conn.close()
        if current_snap is None:
            print("  ❌ no current committed snapshot after pipeline;builder may have failed")
            sys.exit(1)
        drift = compute_drift(prior_set, current_set)
        fp = write_drift_report(drift, prior_snap, current_snap, prior_set, current_set, as_of)
        print(f"  ✅ Drift report:{fp.relative_to(_PROJECT_ROOT)}")
        print(f"     N: {drift['prior_n']} → {drift['current_n']} (churn {drift['churn_pct']}%)")
        print(f"     Added: {len(drift['added'])} / Removed: {len(drift['removed'])} / Stable: {len(drift['stable'])}")

    # ---- Step 7(§14.7-CM): Empirical IC tracking ----
    # 每週重算 43 SPEC features 之 forward-N-day Spearman IC,實證 model-training viability
    # treaty gate:Mean |IC| > 0.03 + ≥30% features p<.05;違反觸發 feature re-evaluation
    run_step("Step 7: §14.7-CM Empirical IC audit(43 features × forward return)",
             [sys.executable, "scripts/audit/audit_feature_ic_vs_future_return.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 8(§14.7-CN): Feature Necessity audit ----
    # 4-path necessity verdict(literature + W1 + W2 + doctrine);0 NOT_NECESSARY required
    # treaty gate:0 NOT_NECESSARY + STRONG+NECESSARY ≥ 50%
    run_step("Step 8: §14.7-CN Feature Necessity audit(4-path verdict)",
             [sys.executable, "scripts/audit/audit_feature_necessity.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 9(§14.7-CO): Feature Sign Stability audit ----
    # Sign verdict(4-tier)+ literature sign consistency check
    # treaty gate:sign-stable ≥ 25% realistic + lit-mismatch ≤ 5 + REGIME-DEP disclosure
    run_step("Step 9: §14.7-CO Feature Sign Stability audit(sign verdict + lit consistency)",
             [sys.executable, "scripts/audit/audit_feature_sign_stability.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 10(§14.7-CP T_CP-3 H4): Data Quality Bias audit ----
    # Look-ahead / Imputation / Multicollinearity bias check
    # treaty gate per §14.7-CP T_CP-3 mandatory pre-check before §10 model_trainer
    run_step("Step 10: §14.7-CP H4 Data Quality Bias audit",
             [sys.executable, "scripts/audit/audit_feature_data_quality_bias.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 11(§14.7-CP T_CP-3 H5): Universe Selection Bias audit ----
    # Sector / Size / Volatility bias check between §14.7-CJ included vs excluded
    run_step("Step 11: §14.7-CP H5 Universe Selection Bias audit",
             [sys.executable, "scripts/audit/audit_universe_selection_bias.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 12(§14.7-CP T_CP-3 H8): Survivorship Bias audit ----
    # Historical growth / Delisted / Info coverage check
    run_step("Step 12: §14.7-CP H8 Survivorship Bias audit",
             [sys.executable, "scripts/audit/audit_survivorship_bias.py"],
             args.dry_run, allow_fail=True)

    # ---- Step 13(§14.7-CT): Production prediction inference(optional)----
    # 對最新 committed model 跑 inference,寫入 predictions table
    # 不啟用 by default(需 model_id 動態查詢);僅標註 placeholder
    # 啟用方式:在 main() 外手動 trigger production prediction(以避免 cron 自動寫 predictions)
    print("\n──── [Step 13: §14.7-CT Production prediction inference] PLACEHOLDER ────")
    print("  (per §14.7-CT,production inference 為 manual trigger;cron 不自動寫 predictions)")
    print("  (model retrain cadence per §14.7-CS T_CS-5,8 weeks rolling)")

    print("\n" + "=" * 72)
    print(f"🎯 Weekly recommit pipeline {'DRY-RUN' if args.dry_run else 'COMMITTED'}")
    print("=" * 72)


if __name__ == "__main__":
    main()

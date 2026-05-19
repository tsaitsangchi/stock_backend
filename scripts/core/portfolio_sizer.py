"""
portfolio_sizer.py v0.1 (Quantum Finance Portfolio Sizing Authority)
================================================================================
最後更新日期: 2026-05-19
主權狀態: IMPLEMENTED (憲法 v6.0.0 §0.2 槓鈴策略 + §9.2 配置層落地雛形 v0.1)
最高原則: Portfolio Sizing Authority — formal prediction → allocation proposal

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Sizing Authority]: 對齊憲章 §0.0-A.5 第五個轉換器裁決——本程式為「formal
   prediction → allocation proposal / position weights」的工程轉換器；不重選
   universe、不重訓 model、不重算 prediction。
2. [Read-Only Upstream]: 只讀唯一 committed `prediction_run` (status='committed')
   及其 `prediction_values`；source-of-truth 為 §8.4 prediction layer。
3. [Single Delivery Gate] (§8.8.8): 同 `prediction_policy_version` 下若
   committed prediction-backed run != 1，直接 FAIL；強制 exactly-one delivery。
4. [Universe Coverage Lock]: prediction_values rows 必須 = 150（即 core+convex
   全集）；否則 FAIL。
5. [Barbell Bucket Caps] (§0.2 槓鈴策略 v0.1):
     attack_total_weight_max  = 0.20  # 攻擊端總權重上限
     safety_total_weight_min  = 0.80  # 防禦端（cash sleeve）下限
     single_stock_weight_max  = 0.05  # 一般 core stock 單一上限
     convex_tier_weight_max   = 0.03  # convex tier 單一上限（更嚴防信仰）
     sector_weight_max        = 0.40  # sector 集中上限（§14.7-U semiconductor）
6. [Right-Tail Concentration]: 只配置 `signal_label='long'` 或 top rank bucket；
   bottom 20 / 'watch' 永不配置；中段不分散（§0.2 拒絕中段）。
7. [Cash As Safety]: 未配置權重保留為 CASH safety sleeve；不買 0050、不買 ETF。
8. [Read-Only Output v0.1]: 本版只輸出 markdown allocation proposal report；
   不寫入新治理表（portfolio_run / portfolio_weights 待 §9.2 強制契約升版後再建）。
9. [No T3 Element]: 永久禁用 §0.1-A 禁令 #2/#3 列出之 T3 元素（IFF Θ / SOC /
   重力井邊緣觸發）；本程式僅用 rank-based 加權，不使用任何物理隱喻公式。
10. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
    產出 reports/portfolio_allocation_proposal_<asof>.md。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 12-dry：配置 dry-run]** | `$ python scripts/core/portfolio_sizer.py --dry-run` | portfolio_sizer v0.1 |
| **2. [Step 12-report：產出 allocation proposal]** | `$ python scripts/core/portfolio_sizer.py --commit-report` | portfolio_sizer v0.1 |
| **3. [Step 12-asof：指定特定 as-of-date]** | `$ python scripts/core/portfolio_sizer.py --commit-report --as-of-date 2026-04-25` | portfolio_sizer v0.1 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 只在 stdout 輸出 allocation proposal；不寫 markdown 報告 |
| **commit-report** | `--commit-report` | 寫 markdown 報告至 reports/portfolio_allocation_proposal_<asof>.md |
| **as-of-date** | `--as-of-date YYYY-MM-DD` | 指定 prediction run as-of-date（預設使用最新 committed run） |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-19 | Codex | 首版：§9.2 配置層雛形；§0.0-A.5 五大轉換器裁決之第五個轉換器落地；對應 portfolio_sizer_barbell_allocation_research_20260519.md §7 sizing policy v0.1 十條規則；不建新治理表（待 §9.2 強制契約升版）。落地後解除 §0.0-B / §0.0-C / §0.0-D 三件套共同最後斷路點。 | **ACTIVE (DRAFT)** |
================================================================================
"""
import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.1"
DEFAULT_PREDICTION_POLICY_VERSION = "prediction_policy_v0.1"
DEFAULT_SIZING_POLICY_VERSION = "sizing_policy_v0.1"

# §0.2 槓鈴策略 v0.1 預設規則（依研究報告 §7）
DEFAULT_POLICY = {
    "attack_total_weight_max": 0.20,
    "safety_total_weight_min": 0.80,
    "single_stock_weight_max": 0.05,
    "convex_tier_weight_max": 0.03,
    "sector_weight_max": 0.40,
    "required_coverage": 150,
    "max_committed_runs": 1,
}


class PortfolioSizer:
    """§9.2 配置層 v0.1 雛形。

    輸入：
      - 唯一 committed prediction_run (status='committed', prediction_policy_version=v0.1)
      - 該 run 的 prediction_values
      - 對應 universe_snapshot_id 的 core_universe_membership

    輸出：
      - allocation proposal markdown report
      - dry-run stdout summary
    """

    def __init__(self, as_of_date=None, dry_run=True, commit_report=False, policy=None):
        self.as_of_date = as_of_date
        self.dry_run = dry_run
        self.commit_report = commit_report
        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self.run_meta = None
        self.predictions = []
        self.memberships = {}
        self.allocations = []
        self.cash_weight = 1.0
        self.sector_totals = defaultdict(float)
        self.tier_totals = defaultdict(float)
        self.risk_flags = []
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    # ── Input Gate ───────────────────────────────────────────────────────────

    def load_inputs(self):
        """載入唯一 committed prediction_run 與其 prediction_values + membership"""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Step 1: 找符合 single-delivery 規則之 committed run
            if self.as_of_date is None:
                cur.execute(
                    """
                    SELECT run_id, model_id, feature_set_id, as_of_date,
                           universe_snapshot_id, prediction_policy_version,
                           rows_written, notes
                    FROM "prediction_run"
                    WHERE status = 'committed'
                      AND prediction_policy_version = %s
                    ORDER BY as_of_date DESC, created_at DESC
                    """,
                    (DEFAULT_PREDICTION_POLICY_VERSION,),
                )
            else:
                cur.execute(
                    """
                    SELECT run_id, model_id, feature_set_id, as_of_date,
                           universe_snapshot_id, prediction_policy_version,
                           rows_written, notes
                    FROM "prediction_run"
                    WHERE status = 'committed'
                      AND prediction_policy_version = %s
                      AND as_of_date = %s
                    ORDER BY created_at DESC
                    """,
                    (DEFAULT_PREDICTION_POLICY_VERSION, self.as_of_date),
                )
            rows = cur.fetchall()

            if len(rows) == 0:
                self._detail("fail", "no committed prediction-backed run found "
                             f"(policy={DEFAULT_PREDICTION_POLICY_VERSION}, "
                             f"as_of_date={self.as_of_date})")
                return False
            if len(rows) > self.policy["max_committed_runs"]:
                self._detail("fail", f"§8.8.8 violation: found {len(rows)} committed "
                             f"runs, expected exactly {self.policy['max_committed_runs']}")
                return False

            keys = ["run_id", "model_id", "feature_set_id", "as_of_date",
                    "universe_snapshot_id", "prediction_policy_version",
                    "rows_written", "notes"]
            self.run_meta = dict(zip(keys, rows[0]))
            if self.as_of_date is None:
                self.as_of_date = self.run_meta["as_of_date"]
            self._detail("pass", f"committed prediction run located: "
                         f"{self.run_meta['run_id']} (model={self.run_meta['model_id']})")

            # Step 2: 載入 prediction_values
            cur.execute(
                """
                SELECT stock_id, prediction_value, prediction_rank,
                       signal_label, confidence
                FROM "prediction_values"
                WHERE run_id = %s
                ORDER BY prediction_rank ASC
                """,
                (self.run_meta["run_id"],),
            )
            for stock_id, value, rank, label, conf in cur.fetchall():
                self.predictions.append({
                    "stock_id": stock_id,
                    "prediction_value": float(value) if value is not None else 0.0,
                    "prediction_rank": int(rank) if rank is not None else 0,
                    "signal_label": label or "hold",
                    "confidence": float(conf) if conf is not None else 0.0,
                })
            if len(self.predictions) != self.policy["required_coverage"]:
                self._detail("fail", f"prediction coverage = {len(self.predictions)}, "
                             f"expected {self.policy['required_coverage']}")
                return False
            self._detail("pass", f"prediction_values loaded: rows={len(self.predictions)}")

            # Step 3: 載入 membership（tier + sector）
            cur.execute(
                """
                SELECT stock_id, core_tier, industry_category, stock_name
                FROM "core_universe_membership"
                WHERE snapshot_id = %s
                  AND core_tier IN ('core_universe', 'convex_universe')
                """,
                (self.run_meta["universe_snapshot_id"],),
            )
            for stock_id, tier, sector, name in cur.fetchall():
                self.memberships[stock_id] = {
                    "core_tier": tier,
                    "industry_category": sector or "UNKNOWN",
                    "stock_name": name or "",
                }
            if len(self.memberships) != self.policy["required_coverage"]:
                self._detail("warn", f"membership coverage = {len(self.memberships)}, "
                             f"expected {self.policy['required_coverage']}")
            else:
                self._detail("pass", f"membership loaded: rows={len(self.memberships)} "
                             f"(snapshot={self.run_meta['universe_snapshot_id']})")

            # Step 4: 確認 prediction_values ↔ membership 對映完整
            missing = [p["stock_id"] for p in self.predictions
                       if p["stock_id"] not in self.memberships]
            if missing:
                self._detail("fail", f"{len(missing)} prediction stocks not in "
                             f"membership; e.g. {missing[:3]}")
                return False
            self._detail("pass", "prediction × membership join complete (150/150)")
        finally:
            cur.close()
            conn.close()
        return self.stats["fail"] == 0

    # ── Sizing Policy v0.1 ─────────────────────────────────────────────────

    def apply_policy(self):
        """依 §0.2 槓鈴策略 + §14.7-U sector cap 計算 target_weight。

        演算法（保守版 v0.1）：
          1. Filter signal_label == 'long' （即 top 20 rank）
          2. 依 rank ascending 配置（最高分先得最高 cap）
          3. 每股 cap = convex_tier_weight_max（convex）或 single_stock_weight_max（core）
          4. 觀察 sector_total + proposed ≤ sector_weight_max；超出則 cap 到剩餘
          5. 觀察 attack_total + proposed ≤ attack_total_weight_max；超出則 cap 到剩餘
          6. 跳過 watch / hold；bottom 20 永不配置
          7. 剩餘權重 = CASH safety sleeve
        """
        # Step 1: 過濾 long 候選
        candidates = [p for p in self.predictions if p["signal_label"] == "long"]
        candidates.sort(key=lambda x: x["prediction_rank"])  # rank 越小越優先

        if not candidates:
            self._detail("fail", "no 'long' signal in prediction_values; "
                         "nothing to allocate")
            return False

        self._detail("pass", f"candidates filtered: {len(candidates)} long signals")

        attack_total = 0.0
        attack_cap = self.policy["attack_total_weight_max"]
        sector_cap = self.policy["sector_weight_max"]
        single_cap = self.policy["single_stock_weight_max"]
        convex_cap = self.policy["convex_tier_weight_max"]

        for cand in candidates:
            stock_id = cand["stock_id"]
            member = self.memberships.get(stock_id, {})
            tier = member.get("core_tier", "core_universe")
            sector = member.get("industry_category", "UNKNOWN")
            name = member.get("stock_name", "")

            # Step 2: 決定本股 cap
            stock_cap = convex_cap if tier == "convex_universe" else single_cap
            proposed = stock_cap
            reason_parts = [f"tier={tier}", f"cap={stock_cap:.0%}"]
            risk = []

            # Step 3: sector cap 檢查
            sector_used = self.sector_totals[sector]
            sector_available = sector_cap - sector_used
            if sector_available <= 0:
                self.allocations.append({
                    "stock_id": stock_id,
                    "stock_name": name,
                    "tier": tier,
                    "sector": sector,
                    "prediction_rank": cand["prediction_rank"],
                    "prediction_value": cand["prediction_value"],
                    "signal_label": cand["signal_label"],
                    "confidence": cand["confidence"],
                    "target_weight": 0.0,
                    "allocation_reason": "sector_cap_exceeded",
                    "risk_flags": [f"sector_{sector}_full"],
                })
                continue
            if proposed > sector_available:
                proposed = sector_available
                risk.append(f"sector_{sector}_partial")

            # Step 4: attack total cap 檢查
            attack_available = attack_cap - attack_total
            if attack_available <= 0:
                self.allocations.append({
                    "stock_id": stock_id,
                    "stock_name": name,
                    "tier": tier,
                    "sector": sector,
                    "prediction_rank": cand["prediction_rank"],
                    "prediction_value": cand["prediction_value"],
                    "signal_label": cand["signal_label"],
                    "confidence": cand["confidence"],
                    "target_weight": 0.0,
                    "allocation_reason": "attack_cap_exhausted",
                    "risk_flags": ["attack_budget_full"],
                })
                continue
            if proposed > attack_available:
                proposed = attack_available
                risk.append("attack_budget_partial")

            # Step 5: 配置
            if proposed <= 0.0001:  # 小於 1bp 視為 0
                continue
            self.allocations.append({
                "stock_id": stock_id,
                "stock_name": name,
                "tier": tier,
                "sector": sector,
                "prediction_rank": cand["prediction_rank"],
                "prediction_value": cand["prediction_value"],
                "signal_label": cand["signal_label"],
                "confidence": cand["confidence"],
                "target_weight": proposed,
                "allocation_reason": " | ".join(reason_parts),
                "risk_flags": risk,
            })
            attack_total += proposed
            self.sector_totals[sector] += proposed
            self.tier_totals[tier] += proposed

        # Step 6: CASH safety sleeve
        self.cash_weight = 1.0 - attack_total

        # Step 7: 治權驗證
        safety_min = self.policy["safety_total_weight_min"]
        if self.cash_weight < safety_min:
            self._detail("fail", f"cash weight={self.cash_weight:.4f} < "
                         f"safety_min={safety_min}; §0.2 防禦端違憲")
            return False
        if attack_total > attack_cap + 0.0001:
            self._detail("fail", f"attack total={attack_total:.4f} > "
                         f"cap={attack_cap}; §0.2 攻擊端違憲")
            return False
        for sec, total in self.sector_totals.items():
            if total > sector_cap + 0.0001:
                self._detail("fail", f"sector {sec} total={total:.4f} > "
                             f"cap={sector_cap}; §14.7-U sector 集中違憲")
                return False

        n_allocated = sum(1 for a in self.allocations if a["target_weight"] > 0)
        self._detail("pass", f"sizing policy v0.1 applied: "
                     f"attack_total={attack_total:.4f}, cash={self.cash_weight:.4f}, "
                     f"allocated_stocks={n_allocated}")
        return True

    # ── Output ────────────────────────────────────────────────────────────

    def _render_report(self):
        """產出 allocation proposal markdown 內容（純字串，不寫檔）"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Portfolio Allocation Proposal (v0.1)",
            "",
            f"- **generated_at**: {ts} Asia/Taipei",
            f"- **tool**: portfolio_sizer.py {TOOL_VER}",
            f"- **constitution**: 系統架構大憲章_{CONSTITUTION_VER}.md §9.2 / §0.2 / §0.0-A.5",
            f"- **prediction_run_id**: `{self.run_meta['run_id']}`",
            f"- **model_id**: `{self.run_meta['model_id']}`",
            f"- **as_of_date**: {self.as_of_date}",
            f"- **universe_snapshot_id**: `{self.run_meta['universe_snapshot_id']}`",
            f"- **prediction_policy**: {self.run_meta['prediction_policy_version']}",
            f"- **sizing_policy**: {DEFAULT_SIZING_POLICY_VERSION}",
            "",
            "## 1. 配置摘要 (Allocation Summary)",
            "",
            f"| 指標 | 值 |",
            f"|---|---:|",
            f"| 攻擊端總權重 | {sum(self.tier_totals.values()):.4f} ({sum(self.tier_totals.values())*100:.2f}%) |",
            f"| 防禦端 (CASH) | {self.cash_weight:.4f} ({self.cash_weight*100:.2f}%) |",
            f"| 配置股票數 | {sum(1 for a in self.allocations if a['target_weight'] > 0)} |",
            f"| 候選但未配置 | {sum(1 for a in self.allocations if a['target_weight'] == 0)} |",
            "",
            "### Tier 配置分布",
            "",
            f"| Tier | 總權重 | 股數 |",
            f"|---|---:|---:|",
        ]
        for tier in sorted(self.tier_totals.keys()):
            n = sum(1 for a in self.allocations
                    if a["tier"] == tier and a["target_weight"] > 0)
            lines.append(f"| {tier} | {self.tier_totals[tier]:.4f} | {n} |")
        lines.append(f"| **CASH (safety)** | **{self.cash_weight:.4f}** | n/a |")

        lines += [
            "",
            "### Sector 配置分布",
            "",
            f"| Sector | 總權重 | 股數 | cap (≤{self.policy['sector_weight_max']}) |",
            f"|---|---:|---:|---:|",
        ]
        for sec in sorted(self.sector_totals.keys()):
            if self.sector_totals[sec] > 0:
                n = sum(1 for a in self.allocations
                        if a["sector"] == sec and a["target_weight"] > 0)
                status = "✅" if self.sector_totals[sec] <= self.policy["sector_weight_max"] else "❌"
                lines.append(f"| {sec} | {self.sector_totals[sec]:.4f} | {n} | {status} |")

        lines += [
            "",
            "## 2. 配置明細 (Allocation Details)",
            "",
            "| Rank | Stock | Name | Tier | Sector | Pred Value | Signal | Conf | Weight | Reason | Risk Flags |",
            "|---:|---|---|---|---|---:|---|---:|---:|---|---|",
        ]
        # CASH row at top of details
        lines.append(
            f"| - | **CASH** | safety sleeve | safety | - | - | - | - | "
            f"**{self.cash_weight:.4f}** | §0.2 防禦端 (≥{self.policy['safety_total_weight_min']:.0%}) | - |"
        )
        for alloc in sorted(self.allocations, key=lambda a: a["prediction_rank"]):
            risk = ", ".join(alloc["risk_flags"]) if alloc["risk_flags"] else "-"
            lines.append(
                f"| {alloc['prediction_rank']} | `{alloc['stock_id']}` | "
                f"{alloc['stock_name']} | {alloc['tier']} | {alloc['sector']} | "
                f"{alloc['prediction_value']:.4f} | {alloc['signal_label']} | "
                f"{alloc['confidence']:.4f} | {alloc['target_weight']:.4f} | "
                f"{alloc['allocation_reason']} | {risk} |"
            )

        lines += [
            "",
            "## 3. Sizing Policy v0.1 規則",
            "",
            f"- attack_total_weight_max: {self.policy['attack_total_weight_max']}",
            f"- safety_total_weight_min: {self.policy['safety_total_weight_min']}",
            f"- single_stock_weight_max: {self.policy['single_stock_weight_max']}",
            f"- convex_tier_weight_max: {self.policy['convex_tier_weight_max']}",
            f"- sector_weight_max: {self.policy['sector_weight_max']}",
            f"- required_coverage: {self.policy['required_coverage']}",
            f"- max_committed_runs: {self.policy['max_committed_runs']}",
            "",
            "## 4. 治權邊界宣告",
            "",
            "- 本程式只讀 committed prediction_run / prediction_values / core_universe_membership；",
            "  **不**重選 universe、**不**重訓 model、**不**重算 prediction、**不**修改任何 raw 表。",
            "- 本配置為 dry-run / report proposal，**非**投資建議；signal_label='long' 為訊號標籤，",
            "  不等於買賣指令。",
            "- §0.1-A 永久禁令 #2/#3 守住：本程式未實作 IFF Θ / SOC / 重力井邊緣觸發。",
            "- §0.2 槓鈴策略：攻擊端 ≤20% + 防禦端 ≥80% + 單股 ≤5% (convex ≤3%) + sector ≤40%。",
            "- §0.0-G 憲章先行紀律對映：本程式為 §0.0-A.5 第五個轉換器；",
            "  治權邊界明文限定於「formal prediction → allocation proposal」。",
            "",
            "## 5. 執行紀錄",
            "",
        ]
        for detail in self.stats["details"]:
            lines.append(f"- {detail}")

        return "\n".join(lines) + "\n"

    def write_report(self):
        if not self.commit_report:
            return None
        content = self._render_report()
        report_path = (_PROJECT_ROOT / "reports" /
                       f"portfolio_allocation_proposal_{self.as_of_date}.md")
        report_path.write_text(content, encoding="utf-8")
        try:
            write_data_audit_log(
                "portfolio_allocation_proposal", str(self.as_of_date),
                self.as_of_date, "PORTFOLIO_SIZING_PROPOSAL",
                len(self.allocations),
            )
        except Exception as exc:
            self._detail("warn", f"data_audit_log failed: {type(exc).__name__}: {exc}")
        self._detail("pass", f"allocation proposal written: "
                     f"{report_path.relative_to(_PROJECT_ROOT)}")
        return report_path

    # ── Verdict / Report ──────────────────────────────────────────────────

    def verdict(self):
        if self.stats["fail"] > 0:
            return "FAILED"
        if self.stats["warn"] > 0:
            return "WARNING"
        return "PERFECT"

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle(
            f"portfolio_sizer_{TOOL_VER}", category="portfolio", stock_id="SYSTEM",
        ) if self.commit_report else None
        lifecycle = lifecycle_cm.__enter__() if lifecycle_cm else None
        try:
            print(f"💼 Portfolio Sizing Authority ({TOOL_VER} / 憲法 {CONSTITUTION_VER})")
            print("─" * 80)
            if not self.load_inputs():
                return self._finalize(start, lifecycle)
            if not self.apply_policy():
                return self._finalize(start, lifecycle)
            self.write_report()
            return self._finalize(start, lifecycle)
        except Exception as exc:
            self._detail("fail", f"{type(exc).__name__}: {exc}")
            if lifecycle:
                lifecycle.mark_failed(str(exc))
            return self._finalize(start, lifecycle)
        finally:
            if lifecycle_cm:
                lifecycle_cm.__exit__(None, None, None)

    def _finalize(self, start, lifecycle):
        verdict = self.verdict()
        if lifecycle and verdict == "FAILED":
            lifecycle.mark_failed("portfolio_sizer failed")
        elif lifecycle and verdict == "WARNING":
            lifecycle.mark_warning("portfolio_sizer warning")

        # Dry-run: stdout 簡報
        if self.dry_run and self.allocations:
            print("\n" + "─" * 80)
            print("📋 Allocation Proposal (dry-run summary)")
            print("─" * 80)
            print(f"  CASH (safety): {self.cash_weight:.4f}")
            for alloc in sorted(self.allocations, key=lambda a: a["prediction_rank"]):
                if alloc["target_weight"] > 0:
                    print(f"  rank {alloc['prediction_rank']:3d} | {alloc['stock_id']:6s} "
                          f"| {alloc['tier']:16s} | {alloc['sector']:16s} | "
                          f"w={alloc['target_weight']:.4f}")

        elapsed_ms = (time.time() - start) * 1000
        print("\n" + "🛡️" * 40)
        print(f"💼 Quantum Finance: Portfolio Sizer 執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準  : 系統架構大憲章_{CONSTITUTION_VER}.md §9.2 / §0.2 / §0.0-A.5")
        print(f"治理權責  : Portfolio Sizing Authority")
        print(f"執行模式  : {'COMMIT-REPORT' if self.commit_report else 'DRY-RUN'}")
        if self.run_meta:
            print(f"Source run: {self.run_meta['run_id']}")
            print(f"As-of-date: {self.as_of_date}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📊 PASS/WARN/FAIL : {self.stats['pass']}/{self.stats['warn']}/{self.stats['fail']}")
        if self.allocations:
            n = sum(1 for a in self.allocations if a["target_weight"] > 0)
            attack_total = sum(self.tier_totals.values())
            print(f"📈 配置股票數     : {n}")
            print(f"📈 攻擊端權重     : {attack_total:.4f} ({attack_total*100:.2f}%)")
            print(f"💰 CASH safety    : {self.cash_weight:.4f} ({self.cash_weight*100:.2f}%)")
        print(f"🕒 總計耗時       : {elapsed_ms:.2f} ms")
        print(f"⚖️  主權判定       : {verdict}")
        print("🛡️" * 40 + "\n")
        return self.stats["fail"] == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Finance Portfolio Sizer (v0.1) — §9.2 配置層雛形")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="只在 stdout 輸出 allocation proposal；不寫 markdown 報告")
    mode.add_argument("--commit-report", action="store_true",
                      help="寫 markdown 報告至 reports/portfolio_allocation_proposal_<asof>.md")
    parser.add_argument("--as-of-date", type=str, default=None,
                        help="指定 prediction run as-of-date（預設使用最新 committed run）")
    return parser.parse_args()


def main():
    args = parse_args()
    as_of_date = None
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    sizer = PortfolioSizer(
        as_of_date=as_of_date,
        dry_run=args.dry_run,
        commit_report=args.commit_report,
    )
    ok = sizer.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

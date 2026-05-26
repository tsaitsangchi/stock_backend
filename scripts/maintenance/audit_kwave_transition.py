"""
audit_kwave_transition.py v0.1 (§14.7-BR Phase C-3 — K-wave Transition Audit;讀 4 indicators 輸出 spring signal score INFO-only)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §0.3.8 5 leading indicators 之 §11C 治權檢驗延伸 / §0.3.8.4 INFO-only / §14.7-BR Phase C-3 落地)
最高原則: Audit-Only Authority (讀;不寫;不作 FAIL gate;對齊 §0.3-A 禁令 #1 / #6)

## 📜 一、核心定義說明 (Core Definitions)
1. [Audit-Only Authority]: 對齊憲章 §0.3.8.4 治權邊界 + §14.7-BR Phase C-3。
   讀 4 indicators 計算 spring signal composite;輸出僅作 **INFO** 不作 FAIL gate。
   永久禁止作為 §9.1 prediction / §9.2 sizing 之自動觸發條件(§0.3-A 禁令 #1)。
2. [Multi-Signal Consensus]: 對齊 §0.3.8.2 多訊號共振裁決:
   ≥ 4/5(或 ≥ 3/4 if BDI 缺)spring 訊號 → "spring_transition"(可考慮攻擊端窗口)
   ≤ 2/5(or ≤ 1/4)spring → "winter_continuing"(防護端維持 ≥ 90%)
   其他 → "transition_period"(維持現狀)
3. [4-of-5 vs 5-of-5]: §14.7-BR Phase C-2 為止支持 I1+I2+I4+I5 之 4 indicators;
   I3 BDI 待 Phase C-4 之 TW shipping proxy(複用 framework);
   本 audit v0.1 報 4-of-4 邏輯;Phase C-4 後可升 5-of-5。
4. [Spring Signal Definitions per §0.3.8.1]:
   I1 M2SL: 月度 YoY > 0(M2 由負/低 → 持續正成長)
   I2 T10Y2Y: latest >= 0(由倒掛 → 解除)
   I4 VIXCLS: latest < 15(由高波動 → 低波動穩定)
   I5 TW_SEMI_VWAP_YOY: latest 2 month avg > 0(由谷底 → 連續 2 季回升)
5. [Annual Review Hook]: 對齊 §0.3.8.4 「每年 12 月年度重選前須由治權者
   檢視 5 項 indicators 並寫入 `reports/kwave_transition_review_<YYYY>.md`」;
   本 audit 為**機械輔助工具**,人工檢視仍為治權主體。
6. [Read-Only Sovereignty]: 只讀 fred_series + kwave_supply_cycle_proxy;
   不寫 DB;不執行 sync / fetcher;對齊 §6.7 SSOT 讀取 pattern。
7. [Zero Hardcoded Verdict]: spring signal threshold 為 design choice
   (per §0.3.8.1 explicit table);本 v0.1 採 charter 明文 thresholds。
8. [Sovereignty Declaration]: 本 audit 屬 §11C 治權檢驗延伸 + L1 audit-only;
   不涉 §0.1-A / §0.2-A / §0.3-A 五套禁令;不在 T1/T2/T3 分層內;
   不處理 §8.5 anti-leakage;對映 §14.7-BR Phase C-3 design。
9. [Historical Reference Authority]: 本 v0.1 為首版落地;後續升版保留歷程。
10. [Hybrid Observability]: 維運觸發 `record_lifecycle`(若可用);
    主權判定動態計算(verdict 為 spring/transition/winter 之 enum)。

## 📊 二、執行指令
| 場景 | 指令 |
| :--- | :--- |
| **Default audit(latest data)** | `$ python scripts/maintenance/audit_kwave_transition.py` |
| **指定 as_of_date(historical 重算)** | `$ python scripts/maintenance/audit_kwave_transition.py --as-of-date 2026-05-21` |
| **Output JSON 之 machine-readable**| `$ python scripts/maintenance/audit_kwave_transition.py --output-format json` |
| **Annual review hook(2026 年度)** | `$ python scripts/maintenance/audit_kwave_transition.py --annual-review 2026` |

## 📜 三、全修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-26 | Codex | **§14.7-BR Phase C-3 落地首版:audit_kwave_transition.py**:依憲章 v6.1.0-patch 第十六輪 §14.7-BR Phase B(commit `95fda16`)入憲 + Phase A 設計研究(commit `f07ba16` §3.3.3 之 design pattern)+ §0.3.8.4 charter 預定(charter L2431 "v6.1.28 §14.7-BR Phase C-3 落地")落地。**功能**:(I) 讀 4 indicators(I1 M2SL via YoY% / I2 T10Y2Y latest / I4 VIXCLS latest / I5 TW_SEMI_VWAP_YOY latest 2-month avg);(II) per indicator 計算 spring signal (0 or 1);(III) composite verdict per §0.3.8.2 多訊號共振裁決;(IV) console + JSON output formats;(V) CLI flags(--as-of-date / --output-format / --annual-review);(VI) INFO-only per §0.3.8.4(不作 FAIL gate)。**對既有 DB 影響**:零(read-only;不寫任何表)。**為 §11C 治權檢驗延伸**:可加入 §11 audit chain 作為 INFO-only audit。**§0.3.8 完成度貢獻**:I5 半導體 proxy + audit tool 落地 → §0.3.8 4/5 + audit infrastructure ready;Phase C-4 後升 5/5(BDI proxy)。同步配套:§14.7-BR Phase A(`f07ba16`)+ Phase B(`95fda16`)+ Phase C-1 M2SL sync(`615e324`)+ Phase C-2 半導體 proxy(`f9a4ecc`)。 | **ACTIVE** |
================================================================================
"""
import argparse
import json
import sys
from datetime import datetime, date
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

# §0.3.8.1 spring signal thresholds(charter explicit values)
THRESHOLD_VIXCLS_LOW = 15.0     # I4 VIXCLS < 15 → low volatility / spring
THRESHOLD_T10Y2Y_NORMAL = 0.0   # I2 T10Y2Y >= 0 → curve not inverted / spring
THRESHOLD_M2SL_POSITIVE = 0.0   # I1 M2SL YoY > 0 → spring
THRESHOLD_SEMI_RECOVERY = 0.0   # I5 TW_SEMI_VWAP_YOY 2-month avg > 0 → spring


def check_i1_m2sl(conn, as_of_date):
    """I1 M2SL 月度 YoY > 0 → spring signal(1);else 0。

    Returns:
        dict: {'name', 'spring_signal' (0/1), 'value', 'threshold', 'context'}
    """
    cur = conn.cursor()
    try:
        # 取 latest 與 12 個月前之 M2SL value;計算 YoY %
        cur.execute("""
            WITH ordered AS (
                SELECT date, value, ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
                FROM "fred_series"
                WHERE series_id = 'M2SL' AND date <= %s
            )
            SELECT
                (SELECT value FROM ordered WHERE rn = 1) AS latest,
                (SELECT date FROM ordered WHERE rn = 1) AS latest_date,
                (SELECT value FROM ordered WHERE rn = 13) AS year_ago,
                (SELECT date FROM ordered WHERE rn = 13) AS year_ago_date
        """, (as_of_date,))
        row = cur.fetchone()
        if not row or row[0] is None or row[2] is None:
            return {
                'name': 'I1 M2SL', 'spring_signal': None,
                'value': None, 'threshold': THRESHOLD_M2SL_POSITIVE,
                'context': 'M2SL data missing or < 12 months history',
            }
        latest, latest_date, year_ago, year_ago_date = row
        latest_f = float(latest)
        year_ago_f = float(year_ago)
        yoy_pct = 100.0 * (latest_f - year_ago_f) / year_ago_f if year_ago_f != 0 else None
        spring = 1 if yoy_pct is not None and yoy_pct > THRESHOLD_M2SL_POSITIVE else 0
        return {
            'name': 'I1 M2SL',
            'spring_signal': spring,
            'value': round(yoy_pct, 2) if yoy_pct is not None else None,
            'threshold': THRESHOLD_M2SL_POSITIVE,
            'context': f'M2SL YoY {yoy_pct:+.2f}% ({latest_date} {latest_f:.2f} vs {year_ago_date} {year_ago_f:.2f})',
        }
    finally:
        cur.close()


def check_i2_t10y2y(conn, as_of_date):
    """I2 T10Y2Y latest >= 0 → spring signal(curve normalized;1)."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT date, value FROM "fred_series"
            WHERE series_id = 'T10Y2Y' AND date <= %s
            ORDER BY date DESC LIMIT 1
        """, (as_of_date,))
        row = cur.fetchone()
        if not row:
            return {'name': 'I2 T10Y2Y', 'spring_signal': None,
                    'value': None, 'threshold': THRESHOLD_T10Y2Y_NORMAL,
                    'context': 'T10Y2Y data missing'}
        d, v = row
        v_f = float(v)
        spring = 1 if v_f >= THRESHOLD_T10Y2Y_NORMAL else 0
        return {
            'name': 'I2 T10Y2Y',
            'spring_signal': spring,
            'value': round(v_f, 2),
            'threshold': THRESHOLD_T10Y2Y_NORMAL,
            'context': f'T10Y2Y latest {v_f:.2f} on {d}(curve {"normalized ≥ 0" if spring == 1 else "still inverted"})',
        }
    finally:
        cur.close()


def check_i4_vixcls(conn, as_of_date):
    """I4 VIXCLS latest < 15 → spring signal(low volatility;1)."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT date, value FROM "fred_series"
            WHERE series_id = 'VIXCLS' AND date <= %s AND value IS NOT NULL
            ORDER BY date DESC LIMIT 1
        """, (as_of_date,))
        row = cur.fetchone()
        if not row:
            return {'name': 'I4 VIXCLS', 'spring_signal': None,
                    'value': None, 'threshold': THRESHOLD_VIXCLS_LOW,
                    'context': 'VIXCLS data missing'}
        d, v = row
        v_f = float(v)
        spring = 1 if v_f < THRESHOLD_VIXCLS_LOW else 0
        return {
            'name': 'I4 VIXCLS',
            'spring_signal': spring,
            'value': round(v_f, 2),
            'threshold': THRESHOLD_VIXCLS_LOW,
            'context': f'VIXCLS latest {v_f:.2f} on {d}({"low vol < 15" if spring == 1 else "elevated vol ≥ 15"})',
        }
    finally:
        cur.close()


def check_i5_tw_semi_recovery(conn, as_of_date):
    """I5 TW_SEMI_VWAP_YOY 最近 2 月 avg > 0 → spring(由谷底回升;1)."""
    cur = conn.cursor()
    try:
        # 取最近 2 個月之 YoY 值(per §0.3.8.1 「連續 2 季回升」之 monthly approximation)
        cur.execute("""
            SELECT date, value FROM "kwave_supply_cycle_proxy"
            WHERE proxy_id = 'TW_SEMI_VWAP_YOY' AND date <= %s
            ORDER BY date DESC LIMIT 2
        """, (as_of_date,))
        rows = cur.fetchall()
        if not rows or len(rows) < 2:
            return {'name': 'I5 TW_SEMI_VWAP_YOY', 'spring_signal': None,
                    'value': None, 'threshold': THRESHOLD_SEMI_RECOVERY,
                    'context': f'TW_SEMI_VWAP_YOY data insufficient (need 2 months; got {len(rows)})'}
        latest, prev = rows
        latest_v = float(latest[1])
        prev_v = float(prev[1])
        avg_2m = (latest_v + prev_v) / 2.0
        spring = 1 if avg_2m > THRESHOLD_SEMI_RECOVERY else 0
        return {
            'name': 'I5 TW_SEMI_VWAP_YOY',
            'spring_signal': spring,
            'value': round(avg_2m, 2),
            'threshold': THRESHOLD_SEMI_RECOVERY,
            'context': f'TW_SEMI_VWAP_YOY 2-month avg {avg_2m:+.2f}% (latest {latest[0]} {latest_v:+.2f}% / prev {prev[0]} {prev_v:+.2f}%)',
        }
    finally:
        cur.close()


class AuditKwaveTransition:
    def __init__(self, as_of_date, output_format='console', annual_review=None):
        self.as_of_date = as_of_date
        self.output_format = output_format
        self.annual_review = annual_review
        self.indicator_results = []
        self.composite_score = None
        self.verdict = None
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def audit(self):
        conn = get_db_connection()
        try:
            self.indicator_results = [
                check_i1_m2sl(conn, self.as_of_date),
                check_i2_t10y2y(conn, self.as_of_date),
                check_i4_vixcls(conn, self.as_of_date),
                check_i5_tw_semi_recovery(conn, self.as_of_date),
            ]
        finally:
            conn.close()

        # Composite spring score(per §0.3.8.2)
        # 治權設計:fixed denominator(N_TOTAL=4),missing 計 0 而非縮減 denominator
        # (避免 partial data 之 inflated verdict;per §一 #8 報告誠實)
        N_TOTAL = 4  # I1 + I2 + I4 + I5(BDI 之 I3 留 Phase C-4)
        signals = [r['spring_signal'] for r in self.indicator_results if r['spring_signal'] is not None]
        n_valid = len(signals)
        n_missing = N_TOTAL - n_valid
        n_spring = sum(signals)
        self.composite_score = f'{n_spring}/{N_TOTAL}'

        # Partial data 警語(若 missing >= 1)
        partial_data_warning = (
            f'⚠️ partial data: {n_missing}/{N_TOTAL} indicators missing(verdict 之 confidence 降;'
            f'治權者須補 sync missing series 後重審)' if n_missing >= 1 else ''
        )

        # Verdict per §0.3.8.2(strict / honest):
        # spring_transition 要求 ≥ 3/N_TOTAL(non-missing)spring(missing 視為 0 / 非 spring)
        # winter_continuing 要求 ≤ 2/N_TOTAL spring(missing 視為 0)
        # 其他 = transition_period(含 partial data 之過渡狀態)
        if n_valid == 0:
            self.verdict = 'no_data'
            self._detail('fail', 'no indicators with valid data')
        elif n_spring >= 3:
            self.verdict = 'spring_transition'
            self._detail('pass', f'§0.3.8 spring signal: {self.composite_score}(spring_transition;可考慮攻擊端窗口 / 治權者人工裁決)')
        elif n_spring <= 1:
            self.verdict = 'winter_continuing'
            self._detail('warn', f'§0.3.8 spring signal: {self.composite_score}(winter_continuing;防護端維持 ≥ 90%)')
        else:
            self.verdict = 'transition_period'
            self._detail('pass', f'§0.3.8 spring signal: {self.composite_score}(transition_period;維持現狀)')
        if partial_data_warning:
            self._detail('warn', partial_data_warning)

        # Per-indicator detail
        for r in self.indicator_results:
            sig = r['spring_signal']
            if sig is None:
                self._detail('warn', f'{r["name"]}: {r["context"]}')
            elif sig == 1:
                self._detail('pass', f'{r["name"]}: ✅ spring | {r["context"]}')
            else:
                self._detail('warn', f'{r["name"]}: ❄️ winter | {r["context"]}')

        return self.stats['fail'] == 0

    def report(self):
        if self.output_format == 'json':
            payload = {
                'tool_ver': TOOL_VER,
                'constitution_ver': CONSTITUTION_VER,
                'as_of_date': str(self.as_of_date),
                'audit_timestamp': datetime.now().isoformat(),
                'indicators': self.indicator_results,
                'composite_score': self.composite_score,
                'verdict': self.verdict,
                'note': 'INFO-only per §0.3.8.4;不作 FAIL gate;不為 §9.1/§9.2 自動觸發條件(§0.3-A 禁令 #1)',
            }
            print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
            return

        # console format
        print("\n" + "🌸" * 40)
        print(f"🌸 Quantum Finance: K-wave Transition Audit ({TOOL_VER})")
        print("🌸" * 40)
        print(f"治權基準     : 系統架構大憲章_{CONSTITUTION_VER}.md §0.3.8 + §14.7-BR Phase C-3")
        print(f"治理權責     : §11C 治權檢驗延伸(INFO-only / 不作 FAIL gate)")
        print(f"執行模式     : AUDIT-ONLY(讀;不寫)")
        print(f"As-of date   : {self.as_of_date}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"🌱 Composite spring score : {self.composite_score}")
        print(f"⚖️  Verdict per §0.3.8.2  : {self.verdict}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"Per-indicator detail:")
        for r in self.indicator_results:
            sig = r['spring_signal']
            icon = '✅' if sig == 1 else ('❄️' if sig == 0 else '⚠️')
            print(f"  {icon} {r['name']}: {r['context']}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📋 §0.3.8.4 治權邊界:本 audit 為 INFO-only;治權者人工檢視仍為主體;")
        print(f"   永久禁止作為 §9.1 prediction / §9.2 sizing 之自動觸發條件(§0.3-A #1)")
        if self.annual_review:
            print(f"📅 Annual review {self.annual_review}:依 §0.3.8.4,治權者須於 12 月年度重選前")
            print(f"   檢視本 audit 結果並寫入 reports/kwave_transition_review_{self.annual_review}.md")
        print("🌸" * 40 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=f"Quantum Finance K-wave Transition Audit ({TOOL_VER}) — §14.7-BR Phase C-3")
    parser.add_argument("--as-of-date", type=str, default=None,
                        help="ISO date(默認今日);for historical 重算")
    parser.add_argument("--output-format", choices=['console', 'json'], default='console',
                        help="output format;json 為 machine-readable")
    parser.add_argument("--annual-review", type=str, default=None,
                        help="Annual review year(eg 2026)— 對映 §0.3.8.4 年度檢視 hook")
    args = parser.parse_args()
    if args.as_of_date:
        args.as_of_date = date.fromisoformat(args.as_of_date)
    else:
        args.as_of_date = date.today()
    return args


def main():
    args = parse_args()
    auditor = AuditKwaveTransition(
        as_of_date=args.as_of_date,
        output_format=args.output_format,
        annual_review=args.annual_review,
    )
    ok = auditor.audit()
    auditor.report()
    # Per §0.3.8.4 INFO-only:不 exit non-zero
    sys.exit(0)


if __name__ == "__main__":
    main()

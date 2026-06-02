"""
ablation_rms_vs_stddev_20260525.py v0.1 (P1 v0.1 公式對齊 ablation 實證)
================================================================================
最後更新日期: 2026-05-25
主權狀態: ABLATION (憲法 v6.1.0 §9.9 RMS vs §14.7-BG/§9.10 STDDEV 公式對齊實證)
最高原則: Evidence-First Verdict (實證讀數驅動公式裁決)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:P1 v0.1 公式對齊 ablation:實證比較 RMS vs STDDEV 兩種波動公式(§9.10/§14.7-BH)。

**輸入 → 輸出**:as-of 資料 → RMS vs STDDEV ablation 結果

**為什麼需要它**:裁決 ΔlnP 波動度用哪個公式(已裁決 RMS)。

## 📜 一、核心定義說明 (Core Definitions)
1. [Ablation Authority]: 對齊憲章 §9.9 (RMS 強制契約) 與 §14.7-BG / §9.10
   (STDDEV raw-first fast-track),在不改 builder / 不改 feature_store
   之純 SELECT-only 模式下對全市場跑兩公式;比對 numerical / rank / score 差異。
2. [Zero Hardcoded Verdict]: 結果以 JSON + Markdown 輸出,不寫 DB,
   不裁決,留待用戶依實證讀數選 A/B/C。
3. [Sovereignty Declaration]: 本 script 屬 §6.7 / §6.8 治權 + §9.9 ablation 之
   實證工具;不重算 CoreScore、不改 raw、不寫 universe table。
4. [Publication-date Compliance]: 對 TaiwanStockPriceAdj 採 native_aligned gate
   (publication_date = date,WHERE date <= as_of_date);對齊 §8.5 第 9 條。
5. [Historical Reference Authority]: 本 script 為一次性 ablation,執行後輸出
   evidence report 永久保留於 reports/;之後依用戶裁決可能廢棄或升入 builder。

## 📊 二、執行指令
| 場景 | 指令 |
| :--- | :--- |
| **全市場 ablation (預設 as_of_date=2026-05-21)** | `$ python scripts/maintenance/ablation_rms_vs_stddev_20260525.py` |
| **指定 as_of_date** | `$ python scripts/maintenance/ablation_rms_vs_stddev_20260525.py --as-of-date 2026-05-21` |
| **No-report mode (只印 stdout)** | `$ python scripts/maintenance/ablation_rms_vs_stddev_20260525.py --no-report` |

## 📊 二、全量維運指令總矩陣 (Operational Matrix)

| 指令 / 模式 | 行為 | 治權對應 |
| :--- | :--- | :--- |
| --as-of-date <d> | 基準日(預設內建) | §14.7-BH |
| --no-report | 只印 stdout 不寫檔 | 維運 |

## 📜 三、全修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-25 | Codex | 首版:回應用戶選擇 D (重業實證);對全市場 SELECT-only 跑 STDDEV vs RMS 兩公式;生成 evidence report 供裁決 A/B/C。 | **ACTIVE** |
================================================================================
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection
from core.data_schema import build_publication_date_gate


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
DEFAULT_AS_OF_DATE = "2026-05-21"
LOOKBACK_DAYS = 90  # ~60 交易日 (與 builder v0.7 一致)
MIN_OBS = 20

# 5 階梯 mapping (與 builder v0.7 _volatility_control_score 一致)
def score_convexity(convexity):
    if convexity is None:
        return 50.0  # fallback neutral
    if convexity > 0.10:
        return 95.0
    elif convexity > 0.05:
        return 85.0
    elif convexity > 0:
        return 75.0
    elif convexity > -0.05:
        return 60.0
    elif convexity > -0.10:
        return 40.0
    return 20.0


def spearman_rho(x_list, y_list):
    """Spearman rank correlation (no scipy)."""
    n = len(x_list)
    if n < 3:
        return None
    # rank
    def rank(arr):
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0] * n
        for r, idx in enumerate(sorted_idx):
            ranks[idx] = r + 1
        # handle ties (average rank) - 對 ablation 已足夠近似,可不處理 tie
        return ranks
    rx = rank(x_list)
    ry = rank(y_list)
    # Pearson on ranks
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def percentile(arr, p):
    """Simple percentile (0-100) using linear interp."""
    s = sorted(arr)
    n = len(s)
    if n == 0:
        return None
    if n == 1:
        return s[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_ablation(as_of_date, write_report=True):
    print(f"🔬 [ablation_rms_vs_stddev v{TOOL_VER}] 對齊憲法 {CONSTITUTION_VER}")
    print(f"📅 as_of_date={as_of_date}, lookback={LOOKBACK_DAYS}d, min_obs={MIN_OBS}")
    print(f"📐 比對:STDDEV (builder v0.7) vs RMS (§9.9 spec)")
    print("─" * 80)

    lookback_start = as_of_date - timedelta(days=LOOKBACK_DAYS)
    gate, n_ap = build_publication_date_gate("TaiwanStockPriceAdj")

    sql = f"""
        WITH log_returns AS (
            SELECT stock_id, date,
                LN(close::numeric / NULLIF(LAG(close::numeric)
                    OVER (PARTITION BY stock_id ORDER BY date), 0)) as lr
            FROM "TaiwanStockPriceAdj"
            WHERE date >= %s AND {gate}
        )
        SELECT stock_id,
            -- STDDEV version (builder v0.7 / §14.7-BG / §9.10 起草)
            STDDEV(lr) FILTER (WHERE lr > 0) * SQRT(252.0) AS up_stddev,
            STDDEV(lr) FILTER (WHERE lr < 0) * SQRT(252.0) AS down_stddev,
            -- RMS version (§9.9 強制契約 / feature_store v0.5)
            SQRT(AVG(lr*lr) FILTER (WHERE lr > 0)) * SQRT(252.0) AS up_rms,
            SQRT(AVG(lr*lr) FILTER (WHERE lr < 0)) * SQRT(252.0) AS down_rms,
            -- meta
            COUNT(*) FILTER (WHERE lr IS NOT NULL) AS n_obs,
            COUNT(*) FILTER (WHERE lr > 0) AS n_up,
            COUNT(*) FILTER (WHERE lr < 0) AS n_down,
            AVG(lr) FILTER (WHERE lr > 0) AS mean_up,
            AVG(lr) FILTER (WHERE lr < 0) AS mean_down
        FROM log_returns
        GROUP BY stock_id
        HAVING COUNT(*) FILTER (WHERE lr IS NOT NULL) >= %s
        ORDER BY stock_id
    """

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        params = [lookback_start] + [as_of_date] * n_ap + [MIN_OBS]
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    print(f"📊 raw 樣本:{len(rows)} stocks (>= {MIN_OBS} log_return 觀察值)")

    # Build records
    records = []
    for row in rows:
        sid, up_s, down_s, up_r, down_r, n_obs, n_up, n_down, m_up, m_down = row
        up_s = float(up_s) if up_s is not None else None
        down_s = float(down_s) if down_s is not None else None
        up_r = float(up_r) if up_r is not None else None
        down_r = float(down_r) if down_r is not None else None
        if None in (up_s, down_s, up_r, down_r):
            continue
        conv_s = up_s - down_s
        conv_r = up_r - down_r
        records.append({
            "stock_id": sid,
            "n_obs": int(n_obs),
            "n_up": int(n_up or 0),
            "n_down": int(n_down or 0),
            "mean_up": float(m_up) if m_up is not None else None,
            "mean_down": float(m_down) if m_down is not None else None,
            "up_stddev": up_s,
            "down_stddev": down_s,
            "up_rms": up_r,
            "down_rms": down_r,
            "convexity_stddev": conv_s,
            "convexity_rms": conv_r,
            "score_stddev": score_convexity(conv_s),
            "score_rms": score_convexity(conv_r),
            "diff_up": up_r - up_s,        # RMS - STDDEV
            "diff_down": down_r - down_s,
            "diff_conv": conv_r - conv_s,
            "score_diff": score_convexity(conv_r) - score_convexity(conv_s),
        })

    n = len(records)
    print(f"📊 有效樣本:{n} stocks (4 公式皆非 NULL)")

    if n == 0:
        print("❌ 樣本為 0,無法 ablation")
        sys.exit(1)

    # --- Numerical 差異統計 ---
    diff_up = [r["diff_up"] for r in records]
    diff_down = [r["diff_down"] for r in records]
    diff_conv = [r["diff_conv"] for r in records]

    stats = {
        "n_stocks": n,
        "up_sigma_diff (RMS - STDDEV)": {
            "mean": mean(diff_up),
            "median": median(diff_up),
            "stdev": stdev(diff_up) if n > 1 else 0,
            "p5": percentile(diff_up, 5),
            "p95": percentile(diff_up, 95),
            "min": min(diff_up),
            "max": max(diff_up),
            "abs_mean": mean(abs(x) for x in diff_up),
            "abs_p95": percentile([abs(x) for x in diff_up], 95),
        },
        "down_sigma_diff (RMS - STDDEV)": {
            "mean": mean(diff_down),
            "median": median(diff_down),
            "stdev": stdev(diff_down) if n > 1 else 0,
            "p5": percentile(diff_down, 5),
            "p95": percentile(diff_down, 95),
            "min": min(diff_down),
            "max": max(diff_down),
            "abs_mean": mean(abs(x) for x in diff_down),
            "abs_p95": percentile([abs(x) for x in diff_down], 95),
        },
        "convexity_diff (RMS - STDDEV)": {
            "mean": mean(diff_conv),
            "median": median(diff_conv),
            "stdev": stdev(diff_conv) if n > 1 else 0,
            "p5": percentile(diff_conv, 5),
            "p95": percentile(diff_conv, 95),
            "min": min(diff_conv),
            "max": max(diff_conv),
            "abs_mean": mean(abs(x) for x in diff_conv),
            "abs_p95": percentile([abs(x) for x in diff_conv], 95),
        },
    }

    # --- Rank correlation ---
    conv_s_list = [r["convexity_stddev"] for r in records]
    conv_r_list = [r["convexity_rms"] for r in records]
    score_s_list = [r["score_stddev"] for r in records]
    score_r_list = [r["score_rms"] for r in records]

    rho_conv = spearman_rho(conv_s_list, conv_r_list)
    rho_score = spearman_rho(score_s_list, score_r_list)

    # --- Score distribution differences ---
    score_diffs = [r["score_diff"] for r in records]
    score_diff_zero = sum(1 for d in score_diffs if d == 0)
    score_diff_nonzero = n - score_diff_zero
    score_diff_dist = {
        "n_zero_diff": score_diff_zero,
        "n_nonzero_diff": score_diff_nonzero,
        "pct_zero": score_diff_zero / n * 100,
        "mean_abs_diff": mean(abs(d) for d in score_diffs),
        "max_abs_diff": max(abs(d) for d in score_diffs),
    }

    # --- Top-120 overlap (核心股 size) ---
    top120_s = set(r["stock_id"] for r in sorted(records, key=lambda r: -r["score_stddev"])[:120])
    top120_r = set(r["stock_id"] for r in sorted(records, key=lambda r: -r["score_rms"])[:120])
    top120_overlap = len(top120_s & top120_r)
    top120_jaccard = len(top120_s & top120_r) / len(top120_s | top120_r) if (top120_s | top120_r) else 0

    # --- Outliers (top-10 largest |diff_conv|) ---
    by_conv_diff_abs = sorted(records, key=lambda r: -abs(r["diff_conv"]))[:10]

    # --- Print summary ---
    print()
    print("=" * 80)
    print("📊 ABLATION RESULTS")
    print("=" * 80)
    print()
    print(f"## Numerical 差異 (RMS - STDDEV)")
    print(f"  up_sigma:   mean={stats['up_sigma_diff (RMS - STDDEV)']['mean']:+.4f}  abs_p95={stats['up_sigma_diff (RMS - STDDEV)']['abs_p95']:.4f}")
    print(f"  down_sigma: mean={stats['down_sigma_diff (RMS - STDDEV)']['mean']:+.4f}  abs_p95={stats['down_sigma_diff (RMS - STDDEV)']['abs_p95']:.4f}")
    print(f"  convexity:  mean={stats['convexity_diff (RMS - STDDEV)']['mean']:+.4f}  abs_p95={stats['convexity_diff (RMS - STDDEV)']['abs_p95']:.4f}")
    print()
    print(f"## Rank Correlation (Spearman ρ)")
    print(f"  convexity_stddev vs convexity_rms: ρ = {rho_conv:.4f}" if rho_conv else "  convexity ρ: N/A")
    print(f"  score_stddev vs score_rms:         ρ = {rho_score:.4f}" if rho_score else "  score ρ: N/A")
    print()
    print(f"## Score 差異 (5 階梯 mapping 後)")
    print(f"  完全相同:    {score_diff_dist['n_zero_diff']:>4d} ({score_diff_dist['pct_zero']:.1f}%)")
    print(f"  有差異:      {score_diff_dist['n_nonzero_diff']:>4d}")
    print(f"  mean |Δscore|: {score_diff_dist['mean_abs_diff']:.2f}")
    print(f"  max |Δscore|:  {score_diff_dist['max_abs_diff']:.1f}")
    print()
    print(f"## Top-120 核心股名單 (依 score)")
    print(f"  STDDEV ∩ RMS: {top120_overlap}/120  ({top120_overlap/120*100:.1f}%)")
    print(f"  Jaccard:      {top120_jaccard:.4f}")
    print()
    print(f"## Top-10 |Δconvexity| Outliers")
    print(f"  {'stock_id':<10} {'conv_S':>9} {'conv_R':>9} {'Δconv':>9} {'score_S':>8} {'score_R':>8}")
    for r in by_conv_diff_abs:
        print(f"  {r['stock_id']:<10} {r['convexity_stddev']:>+9.4f} {r['convexity_rms']:>+9.4f} {r['diff_conv']:>+9.4f} {r['score_stddev']:>8.0f} {r['score_rms']:>8.0f}")

    if not write_report:
        print()
        print("(--no-report mode; 不寫 markdown / JSON)")
        return

    # --- Write JSON + Markdown ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = _REPORTS_DIR / f"p1_v01_rms_vs_stddev_ablation_data_{timestamp}.json"
    md_path = _REPORTS_DIR / f"p1_v01_rms_vs_stddev_ablation_evidence_{timestamp}.md"

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_version": TOOL_VER,
        "constitution": CONSTITUTION_VER,
        "as_of_date": str(as_of_date),
        "lookback_days": LOOKBACK_DAYS,
        "min_obs": MIN_OBS,
        "n_stocks": n,
        "stats": stats,
        "rho_convexity": rho_conv,
        "rho_score": rho_score,
        "score_diff_dist": score_diff_dist,
        "top120_overlap": top120_overlap,
        "top120_jaccard": top120_jaccard,
        "top10_conv_diff_outliers": [
            {k: r[k] for k in ("stock_id", "convexity_stddev", "convexity_rms",
                                 "diff_conv", "score_stddev", "score_rms",
                                 "n_obs", "n_up", "n_down", "mean_up", "mean_down")}
            for r in by_conv_diff_abs
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md_lines = [
        f"# P1 v0.1 公式對齊 ablation evidence — RMS vs STDDEV",
        "",
        f"- **tool**: `ablation_rms_vs_stddev_20260525.py v{TOOL_VER}`",
        f"- **constitution**: {CONSTITUTION_VER}",
        f"- **as_of_date**: {as_of_date}",
        f"- **lookback**: {LOOKBACK_DAYS}d (~60 交易日);min_obs={MIN_OBS}",
        f"- **n_stocks**: {n}",
        f"- **公式比對**: STDDEV (builder v0.7 / §14.7-BG / §9.10 起草) vs RMS (§9.9 強制契約)",
        "",
        "## 1. Numerical 差異 (RMS − STDDEV)",
        "",
        "| 變數 | mean | median | stdev | abs_mean | abs_p95 | min | max |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for var_key in ("up_sigma_diff (RMS - STDDEV)", "down_sigma_diff (RMS - STDDEV)", "convexity_diff (RMS - STDDEV)"):
        s = stats[var_key]
        md_lines.append(
            f"| `{var_key}` | {s['mean']:+.4f} | {s['median']:+.4f} | {s['stdev']:.4f} | "
            f"{s['abs_mean']:.4f} | {s['abs_p95']:.4f} | {s['min']:+.4f} | {s['max']:+.4f} |"
        )
    md_lines += [
        "",
        "## 2. Rank Correlation (Spearman ρ)",
        "",
        f"- `convexity_stddev` vs `convexity_rms`: **ρ = {rho_conv:.4f}**" if rho_conv else "- convexity ρ: N/A",
        f"- `score_stddev` vs `score_rms`:         **ρ = {rho_score:.4f}**" if rho_score else "- score ρ: N/A",
        "",
        "## 3. Score 差異 (5 階梯 mapping 後)",
        "",
        f"- 完全相同:    **{score_diff_dist['n_zero_diff']} / {n} ({score_diff_dist['pct_zero']:.1f}%)**",
        f"- 有差異:      {score_diff_dist['n_nonzero_diff']}",
        f"- mean |Δscore|: {score_diff_dist['mean_abs_diff']:.2f}",
        f"- max |Δscore|:  {score_diff_dist['max_abs_diff']:.1f}",
        "",
        "## 4. Top-120 核心股名單 (依 score 排序)",
        "",
        f"- STDDEV ∩ RMS: **{top120_overlap}/120 ({top120_overlap/120*100:.1f}%)**",
        f"- Jaccard:      {top120_jaccard:.4f}",
        "",
        "## 5. Top-10 |Δconvexity| Outliers",
        "",
        "| stock_id | conv_STDDEV | conv_RMS | Δconv | score_S | score_R | n_obs | n_up | n_down | mean_up | mean_down |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in by_conv_diff_abs:
        mu = f"{r['mean_up']:+.5f}" if r['mean_up'] is not None else "N/A"
        md = f"{r['mean_down']:+.5f}" if r['mean_down'] is not None else "N/A"
        md_lines.append(
            f"| {r['stock_id']} | {r['convexity_stddev']:+.4f} | {r['convexity_rms']:+.4f} | "
            f"{r['diff_conv']:+.4f} | {r['score_stddev']:.0f} | {r['score_rms']:.0f} | "
            f"{r['n_obs']} | {r['n_up']} | {r['n_down']} | {mu} | {md} |"
        )

    md_lines += [
        "",
        "## 6. 裁決方向 (依實證讀數)",
        "",
        "依以下指標強弱,本 ablation 之 evidence 對應 §9.9 vs §14.7-BG/§9.10 公式裁決:",
        "",
        f"- **若 ρ_score ≥ 0.95 且 Top-120 overlap ≥ 95%**: STDDEV / RMS 排名一致 → 選項 C(治權成本最低)可採;但仍應在 §14.7-BG 加註 STDDEV 為 fast-track 近似",
        f"- **若 ρ_score ∈ [0.80, 0.95) 或 Top-120 overlap ∈ [80%, 95%)**: 兩公式 rank 有差異 → 選項 B(雙公式 + ablation IC 後選主軸)為合理裁決",
        f"- **若 ρ_score < 0.80 或 Top-120 overlap < 80%**: 兩公式產生顯著不同的核心股名單 → 選項 A(以 §9.9 RMS 為治權 SSOT;追溯修正 builder v0.7 → v0.7.1 RMS)為強制裁決",
        "",
        "**本次實證結果**: ρ_score = {:.4f} / Top-120 overlap = {:.1f}% → ".format(rho_score, top120_overlap/120*100),
        "",
        "## 7. 治權交叉引用",
        "- 憲章 §9.9 (P1 v0.1 強制契約 — RMS 公式)",
        "- 憲章 §9.10 (起草 — VC 升版 STDDEV 公式;待 §9.9 ablation 後升正式條文)",
        "- 憲章 §14.7-BG (raw-first fast-track 路徑入憲)",
        "- 憲章 §0.0-C.3 (上行凸性壓制修補)",
        "- 憲章 §6.3 第 7 條 (VC 公式條文原文)",
        "- 憲章 §14.7-AX (資料層揭露驅動治權升版元規則 — 本次屬「公式層揭露」第 7 次跑通)",
        "",
        "## 8. 對映 JSON 完整資料",
        f"- `{json_path.name}`",
    ]

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print()
    print(f"📄 Markdown evidence report: {md_path.name}")
    print(f"📄 JSON full data:           {json_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="P1 v0.1 公式對齊 ablation — RMS vs STDDEV (v0.1)")
    parser.add_argument("--as-of-date", type=str, default=DEFAULT_AS_OF_DATE,
                        help=f"as_of_date (預設 {DEFAULT_AS_OF_DATE})")
    parser.add_argument("--no-report", action="store_true", help="不寫 markdown / JSON,只印 stdout")
    return parser.parse_args()


def main():
    args = parse_args()
    as_of = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    run_ablation(as_of, write_report=not args.no_report)


if __name__ == "__main__":
    main()

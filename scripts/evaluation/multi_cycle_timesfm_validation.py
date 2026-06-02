"""
multi_cycle_timesfm_validation.py v0.1 (TimesFM · Google Zero-Shot Time-Series Foundation Model · Multi-Cycle Stock-Price Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: TIMESFM(GOOGLE DECODER-ONLY TIME-SERIES FOUNDATION MODEL · ZERO-SHOT · transformers-native TimesFmModelForPrediction)4-HORIZON WALK-FORWARD VALIDATION + §14.7-CY HORIZON-DOCTRINE 第三族(foundation)+ 最新 committed SOURCE-PURE UNIVERSE(914)+ §一.10 SOURCE-TRACEABLE(全 DB)+ §一.10 #3 MULTI-RUN + 共同比較基準(COMMON COMPARISON BASELINE)第六實作 + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,非序列落地鏈模組。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1/§0.2/§0.3);(c) T1-T3 不分層;
   (d) §8.5 anti-leakage:context window 僅含 as_of 之前(含)之 weekly close,forecast 之 target weeks 全在 as_of 之後 → 結構性無洩漏;TimesFM per-series loc/scale 正規化僅用 context 自身統計量;
   (e) **不訓練 production model**(zero-shot,完全不更新權重,不寫 model_registry);(f) **read-only**(不改 feature_values / TaiwanStockPriceAdj / universe);
   (g) 唯一職責:TimesFM zero-shot 4-horizon walk-forward 預測 + 共同比較基準 metrics + JSON 持久化。
2. **[Common Comparison Baseline]** (v0.1, reports/model_comparison_baseline_spec_20260601.md §七): 本程式為共同比較基準之**第六實作**(前五:TFT / iTransformer / PatchTST / Stockformer / HIST)—
   最新 committed source-pure universe × 95 monthly panels × 真實 forward log returns(TaiwanStockPriceAdj)× 4 horizons(5/20/60/252d)×
   top-20 equal-weight long × 0.6% cost × {Sharpe, Win, Eff-t, T_CZ-6 gate}。與全 tree / TFT / iTransformer / PatchTST / Stockformer / HIST / Chronos 套用**完全相同** protocol +
   **完全相同** realized targets → 精準度(precision)/ 信任度(trust)比較 apples-to-apples。模型用各自 natural representation
   (tree=37 cross-sectional features;**TimesFM=每股 weekly close 序列 → 預訓練 decoder-only foundation model zero-shot 外推 → 隱含 horizon close → log-return score**),比較點在 OUTPUT 預測之品質,非 input。
3. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全數據 (b) DB query(TaiwanStockPriceAdj close + core_universe_*)+ (a) program output;
   **0 AI memory reuse**;weekly close 全為 source-pure DB raw close(calendar-week resample 取週內最後交易日 close);
   **無 imputed / 無 forward-fill 補值 / 無 hardcoded knowledge / 無 §一.13 第四類幻像值**;
   foundation model 權重為 **Google 預訓練之外部 general-purpose time-series 權重(非本系統 stock 資料訓練、非 AI 幻像、非本機 fabrication)**,以 zero-shot 套用 — 屬 (a) 程式輸出之 deterministic 外部模型推論。
4. **[Zero-Shot Foundation Model — 無訓練、無 refit]** (v0.1, §一.8 誠實標示): TimesFM 為 Google decoder-only 預訓練基礎模型(transformers-native `TimesFmModelForPrediction`),
   **完全 zero-shot**:不在本系統股價上 fine-tune、不更新任何權重、無 walk-forward refit(與 tree / Stockformer / HIST 之逐期 refit 本質不同)。
   每期僅以該股 as_of 前之 weekly close context 餵入 → 模型外推未來 close。**誠實揭露**:此為對「通用基礎模型能否 zero-shot 預測台股」之直接檢定;
   股價近隨機漫步 → 基礎模型可能僅外推近期趨勢(predicted return 趨近 0 / 區分力弱)為**合理且須誠實報告之結果**,非程式缺陷。
5. **[Runtime Dependency — Checkpoint Download]** (v0.1, §一.8): 首次執行需自 HuggingFace Hub 下載權重(default `google/timesfm-2.0-500m-pytorch`,~2GB,context=512 / horizon=128);
   下載後快取於 `~/.cache/huggingface`。**無網路 / 下載失敗 → 程式明確記錄錯誤並中止**(不偽造結果)。可 `--checkpoint` 覆寫為較小模型。
6. **[Point/Quantile Forecast]** (v0.1): TimesFM 輸出 `mean_predictions`(點預測)+ `full_predictions`(9 分位數)。
   本驗證以 mean_predictions 計 horizon close → log-return score 作排序;**calibration_p10_p90_coverage 暫設 None**(point-forecast 排序基準與其他模型對齊;quantile 校準為後續可選增強)。信任度 = Eff-t significance + 多 seed 穩定度(但 zero-shot 為 deterministic → 多 seed 結果相同,見 #9)。
7. **[Precision / Trust / Profitability 三分]** (v0.1): 精準度(rank-IC / directional accuracy / RMSE / MAE / R²)、信任度(Eff-t significance / calibration[N/A])、賺錢能力(net-of-cost Sharpe / Eff-t / Win / annualized net / T_CZ-6 gate)分開報告。
8. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): significance 動態(abs(eff_t) > 1.997);T_CZ-6 gate(4.20/2.40/0.79)為 charter-mandated reference threshold(Tier 3 transparent disclosure,非 feature data,非硬編 verdict)。
9. **[§一.10 #3 Multi-Run — Zero-Shot Determinism 例外揭露]** (v0.1): TimesFM zero-shot 推論為 **deterministic**(無隨機 init / 無 dropout-at-eval / 無 sgd shuffle)→ 不同 seed 結果相同;
   故 multi-seed 對 TimesFM 為 trivially identical(誠實揭露:此非 stochasticity-bearing metric,§一.10 #3 之多跑取統計對 deterministic zero-shot 不適用;single run 即 deterministic fact)。
10. **[Schema-Compatible Output]** (v0.1): per-horizon JSON keys 與 multi_cycle_validation.py / Chronos / Stockformer 對齊 → build_model_comparison_report.py 可直接納入 leaderboard。
11. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
12. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON);可重跑(zero-shot deterministic);context / freq / checkpoint 可配置。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Common Comparison Baseline(共同比較基準 — 與全模型一致,函式逐字相同)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Universe | get_universe() → 最新 committed snapshot core_tier='core_universe'(914)| 最新 committed |
| A.2 Panels | get_panel_dates() → 95 monthly mid-month(2018-06-15~2026-04-15)| 與 baseline 同 grid |
| A.3 Forward returns | load_forward_returns() → 真實 log return(TaiwanStockPriceAdj)| §一.10 (b) DB |
| A.4 Portfolio | top-20 equal-weight long(np.argsort[-20:])| 與 baseline 同 |
| A.5 Profitability | aggregate_horizon():sharpe / win / mdd / eff_t / annualized_net / T_CZ-6 | §14.7-CY / §14.7-CZ |

### Group B. TimesFM Zero-Shot Foundation Model

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Close matrix | load_close_matrix() → daily close → calendar-week → weekly close matrix[week × stock]| §一.10 / §一.13 source-pure |
| B.2 Model | load_timesfm() → transformers TimesFmModelForPrediction.from_pretrained(zero-shot 預訓練)| 外部預訓練權重 |
| B.3 Forecast | forecast_panel() → 每股 last-context weekly close → mean_predictions(horizon close)→ log-return score | §8.5 leakage-safe |
| B.4 No refit | zero-shot,逐期僅推論(無訓練、無 expanding-window refit)| §一.8 誠實標示 |

### Group C. Precision / Trust(標準化 block — 全模型共用定義)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Precision | rank_ic / directional_accuracy / rmse / mae / r2(pred vs 真實 realized)| 共同基準 |
| C.2 Trust | effective_t_stat / is_significant_p05 / calibration(None)| 共同基準 |
| C.3 Multi-seed | zero-shot deterministic → 多 seed identical(#9 揭露)| §一.10 #3 例外 |

### Group D. Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 JSON | --output reports/multi_cycle_timesfm_<ts>.json(schema-compatible + precision/trust)| §一.10 / §二.4 |
| D.2 stdout | cross-cycle comparison matrix + precision/trust 摘要 | §一.12 進度 |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| Smoke(plumbing 驗證)| `python scripts/evaluation/multi_cycle_timesfm_validation.py --smoke --output reports/_smoke_timesfm.json` |
| 完整單跑(zero-shot deterministic)| `... --output reports/multi_cycle_timesfm_<ts>.json` |
| 較小 checkpoint | `... --checkpoint google/timesfm-1.0-200m-pytorch` |

### 不提供之旗標 (Intentionally Omitted)

- `--commit`:本工具不寫 evaluation_log / model_registry(屬 model_trainer 治權;§3.2 橫切只讀)。
- `--epochs` / `--refit-every`:zero-shot 無訓練、無 refit(#4)。
- `--cost-per-rebal`:0.6% standard per §14.7-CY T_CY-5(與 baseline 對齊,不可變更否則破壞比較基準)。
- `--horizons`:固定 5/20/60/252(共同比較基準定義之一部分,不可變更)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-02 | Claude | **首版**:TimesFM(Google decoder-only 時序基礎模型,transformers-native `TimesFmModelForPrediction`,zero-shot)multi-cycle 股價預測驗證 + 共同比較基準第六實作。每股 weekly close 序列(last-context)→ 預訓練 foundation model zero-shot 外推 horizon close → log-return score → 4-horizon top-20。與 baseline 同 universe(最新 committed 914)/ panels(95)/ forward returns(TaiwanStockPriceAdj)/ portfolio(top-20)/ cost(0.6%)/ gate(T_CZ-6)。precision(rank-IC / dir-acc / RMSE / MAE / R²)+ trust(Eff-t;calibration N/A)。§一.10 全 DB source-traceable + 外部預訓練權重(非本系統訓練、非幻像);§一.8 誠實標示 zero-shot 無訓練無 refit + checkpoint 下載 runtime 依賴 + deterministic multi-seed 例外 + 「基礎模型可能弱區分力」之誠實預期;§8.5 leakage-safe by context/forecast split + per-series loc/scale context-only;§一.11 三段式合規。 | **ACTIVE** |
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys, argparse, math, json, logging, time
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import pandas as pd
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# ── Common Comparison Baseline constants (FIXED — 不可變更,否則破壞跨模型比較) ──
HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]
N_TOP = 20                 # top-k long portfolio (baseline 對齊)
COST_PER_REBAL = 0.006     # TW broker round-trip (§14.7-CY T_CY-5)
PANEL_SPACING = 30         # monthly grid (for overlap-corrected n_eff)
GATE = {"effective_t_stat": 4.20, "sharpe": 2.40, "win_rate": 0.79}  # §14.7-CZ T_CZ-6 annual
CRIT_T = 1.997             # p<0.05 large-df

# ── TimesFM / weekly constants ──
CONTEXT_WEEKS = 128        # zero-shot context (~2.5 yr; multiple of patch_length 32)
HORIZON_STEPS = {5: 1, 20: 4, 60: 12, 252: 50}  # trading days → weekly forecast step index
DEFAULT_CHECKPOINT = "google/timesfm-2.0-500m-pytorch"
DEFAULT_FREQ = 1           # TimesFM freq index: 0=high(≤daily) 1=medium(weekly/monthly) 2=low(quarterly+)


# ════════════════════════════════════════════════════════════════════════════
# Group A — Common Comparison Baseline DB loaders (與 baseline / Chronos / Stockformer 逐字相同)
# ════════════════════════════════════════════════════════════════════════════
def get_universe(cur):
    cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                   JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                   WHERE s.status='committed' AND m.core_tier='core_universe'
                   AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                      WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
    return sorted({r[0] for r in cur.fetchall()})


def get_panel_dates():
    """95 mid-month as_of dates 2018-06-15 .. 2026-04-15 (與 baseline 同 grid)."""
    dates = []
    current = date(2018, 6, 15)
    while current <= date(2026, 4, 30):
        dates.append(current)
        current = date(current.year + 1, 1, 15) if current.month == 12 else date(current.year, current.month + 1, 15)
    return dates


def load_forward_returns(cur, as_of, horizon_days):
    """真實 forward log return at horizon (per (b) DB query) — 與 baseline 完全相同."""
    cur.execute("""SELECT MIN(date) FROM "TaiwanStockPriceAdj"
                   WHERE date >= (%s::date + INTERVAL '%s days')
                     AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')""",
                (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return {}, None
    cur.execute("""WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
                        t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
                   SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
                   FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id""", (str(as_of), str(label_date)))
    return {sid: float(v) for sid, v in cur.fetchall()}, label_date


# ════════════════════════════════════════════════════════════════════════════
# Group B — weekly CLOSE matrix (TimesFM forecasts price level, then derive log-return)
# ════════════════════════════════════════════════════════════════════════════
def load_close_matrix(cur, universe, max_stocks=None):
    """每股最長 daily 歷史 → calendar-week resample(取週內最後交易日 close)→ close matrix[week × stock].
    缺週留 NaN(不偽造、不 forward-fill)→ context 取每股 as_of 前之最後 N 個 finite close = source-pure."""
    if max_stocks:
        universe = universe[:max_stocks]
    stock_weekly = {}
    all_keys = set()
    for sid in universe:
        cur.execute("""SELECT date, close FROM "TaiwanStockPriceAdj"
                       WHERE stock_id=%s AND close>0 ORDER BY date""", (sid,))
        rec = cur.fetchall()
        wk = {}
        for d, c in rec:
            iso = d.isocalendar()
            key = (iso[0], iso[1])
            if key not in wk or d > wk[key][0]:
                wk[key] = (d, float(c))
        if len(wk) < CONTEXT_WEEKS + 55:   # need context + ~annual forecast horizon room
            continue
        stock_weekly[sid] = wk
        all_keys |= set(wk.keys())
    cols = [s for s in universe if s in stock_weekly]
    keys_sorted = sorted(all_keys)
    kidx = {k: i for i, k in enumerate(keys_sorted)}
    W, N = len(keys_sorted), len(cols)
    close_mat = np.full((W, N), np.nan)
    week_date = [None] * W
    for ci, sid in enumerate(cols):
        for key, (d, close) in stock_weekly[sid].items():
            i = kidx[key]
            close_mat[i, ci] = close
            if week_date[i] is None or d > week_date[i]:
                week_date[i] = d
    return close_mat, cols, week_date


def load_timesfm(checkpoint):
    """Load Google TimesFM zero-shot foundation model via transformers-native API.
    First run downloads weights from HuggingFace Hub (~2GB for 500m) → cached locally."""
    import torch
    from transformers import TimesFmModelForPrediction
    logger.info(f"Loading TimesFM checkpoint '{checkpoint}' (first run downloads from HF Hub ~2GB)...")
    t0 = time.monotonic()
    model = TimesFmModelForPrediction.from_pretrained(checkpoint, torch_dtype=torch.float32)
    model.eval()
    hl = getattr(model.config, "horizon_length", None)
    cl = getattr(model.config, "context_length", None)
    logger.info(f"TimesFM loaded in {time.monotonic()-t0:.1f}s (context_length={cl} horizon_length={hl})")
    return model, hl


def forecast_panel(model, close_mat, week_cut, context, cols, freq_val, batch_size=256):
    """Zero-shot forecast: per-stock last `context` finite weekly closes → mean_predictions (horizon close)
    → log-return score per horizon. No training, no weight update."""
    import torch
    fwd = {h: {} for _, h in HORIZONS}
    sub = close_mat[:week_cut + 1]                                    # closes up to (incl) as_of
    series, sids, last_close = [], [], []
    for ci, sid in enumerate(cols):
        col = sub[:, ci]
        fin = col[np.isfinite(col)]
        if fin.size < context:
            continue
        series.append(fin[-context:].astype(np.float32))
        sids.append(sid)
        last_close.append(float(fin[-1]))
    if len(sids) < N_TOP + 5:
        return fwd
    arr = np.stack(series)                                           # [B, context]
    preds = []
    with torch.no_grad():
        for bs in range(0, arr.shape[0], batch_size):
            past = torch.tensor(arr[bs:bs + batch_size], dtype=torch.float32)
            freq = torch.full((past.shape[0],), freq_val, dtype=torch.long)
            out = model(past_values=past, freq=freq)
            preds.append(out.mean_predictions.cpu().numpy())        # [b, horizon]
    mean_pred = np.concatenate(preds, axis=0)                        # [B, horizon]
    H = mean_pred.shape[1]
    for j, sid in enumerate(sids):
        lc = last_close[j]
        if lc <= 0:
            continue
        for _, h in HORIZONS:
            step = HORIZON_STEPS[h]
            if step <= H:
                pc = float(mean_pred[j, step - 1])
                if pc > 0:
                    fwd[h][sid] = math.log(pc / lc)
    return fwd


# ════════════════════════════════════════════════════════════════════════════
# Group A/C — metrics (portfolio + precision + trust) — 與 baseline / Chronos / Stockformer 逐字相同
# ════════════════════════════════════════════════════════════════════════════
def spearman_ic(pred, y):
    pred = np.asarray(pred); y = np.asarray(y)
    if len(pred) < 3:
        return 0.0
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def aggregate_horizon(label, horizon_days, panel_top_rets, panel_univ_rets,
                      panel_ics, panel_diracc, all_pred, all_real, calib_cov):
    """共同比較基準 metric 計算(與 multi_cycle_validation.py 完全相同之 profitability 數學)+ precision/trust."""
    n = len(panel_top_rets)
    if n == 0:
        return None
    pr = np.array(panel_top_rets); ur = np.array(panel_univ_rets)
    mean_ret = float(np.mean(pr)); std_ret = float(np.std(pr, ddof=1)) if n > 1 else 0.0
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0.0
    win_rate = float(np.mean(pr > 0))
    alphas = pr - ur
    mean_alpha = float(np.mean(alphas)); std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0.0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0.0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0.0
    running = peak = mdd = 0.0
    for r in pr:
        running += r; peak = max(peak, running); mdd = max(mdd, peak - running)
    rebals_per_year = 252.0 / horizon_days
    ann_log_gross = mean_ret * rebals_per_year
    ann_cost = COST_PER_REBAL * rebals_per_year
    ann_log_net = ann_log_gross - ann_cost
    ann_simple_net = math.exp(ann_log_net) - 1
    net_rets = pr - COST_PER_REBAL
    net_sharpe = float(np.mean(net_rets)) / (np.std(net_rets, ddof=1) * math.sqrt(1)) * math.sqrt(12) \
        if n > 1 and np.std(net_rets, ddof=1) > 0 else 0.0
    if horizon_days <= PANEL_SPACING:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (PANEL_SPACING / horizon_days)
        overlap_pct = (horizon_days - PANEL_SPACING) / horizon_days * 100
    eff_t = t_stat * math.sqrt(n_eff / n) if n > 0 else 0.0
    ap = np.array(all_pred); ar = np.array(all_real)
    rmse = float(np.sqrt(np.mean((ap - ar) ** 2))) if len(ap) else None
    mae = float(np.mean(np.abs(ap - ar))) if len(ap) else None
    ss_res = float(np.sum((ar - ap) ** 2)); ss_tot = float(np.sum((ar - np.mean(ar)) ** 2)) if len(ar) else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct, "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "std_ret_per_panel": std_ret,
        "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t, "is_significant_p05": bool(abs(eff_t) > CRIT_T),
        "annualized_log_gross": ann_log_gross, "annualized_simple_net": ann_simple_net,
        "annual_cost_drag_log": ann_cost, "net_sharpe_per_panel": net_sharpe,
        "rank_ic_mean": float(np.mean(panel_ics)) if panel_ics else None,
        "rank_ic_std": float(np.std(panel_ics, ddof=1)) if len(panel_ics) > 1 else None,
        "rank_ic_ir": (float(np.mean(panel_ics)) / float(np.std(panel_ics, ddof=1)) * math.sqrt(12))
                      if len(panel_ics) > 1 and np.std(panel_ics, ddof=1) > 0 else None,
        "directional_accuracy": float(np.mean(panel_diracc)) if panel_diracc else None,
        "rmse": rmse, "mae": mae, "r2": r2,
        "calibration_p10_p90_coverage": calib_cov,  # None for TimesFM (mean-prediction ranking)
    }


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════
def run(args):
    conn = get_db_conn()
    cur = conn.cursor()
    universe = get_universe(cur)
    logger.info(f"Universe: {len(universe)} stocks (latest committed source-pure)")
    panels = get_panel_dates()
    if args.max_panels:
        panels = panels[:args.max_panels]
    logger.info(f"Panels: {len(panels)} monthly as_of dates")

    t0 = time.monotonic()
    close_mat, cols, week_date = load_close_matrix(cur, universe, max_stocks=args.max_stocks)
    logger.info(f"Close matrix: {len(cols)} stocks × {close_mat.shape[0]} calendar-weeks "
                f"(load {time.monotonic()-t0:.1f}s)")

    model, model_hl = load_timesfm(args.checkpoint)
    context = args.context_weeks
    week_ord = np.array([d.toordinal() if d else -1 for d in week_date])

    def week_cut_for(as_of):
        ao = as_of.toordinal()
        idx = np.where(week_ord <= ao)[0]
        return int(idx[-1]) if len(idx) else None

    logger.info(f"TimesFM zero-shot: context={context}wk freq={args.freq} checkpoint={args.checkpoint} "
                f"(NO training / NO refit — deterministic)")

    acc = {h: {"top": [], "univ": [], "ic": [], "diracc": [], "pred": [], "real": []} for _, h in HORIZONS}
    n_pred_panels = 0

    for pi, as_of in enumerate(panels):
        wc = week_cut_for(as_of)
        if wc is None or wc < context:
            continue
        fwd = forecast_panel(model, close_mat, wc, context, cols, args.freq, batch_size=args.batch_size)
        for label, h in HORIZONS:
            real, _ = load_forward_returns(cur, as_of, h)
            preds = fwd.get(h, {})
            common = [s for s in preds if s in real]
            if len(common) < N_TOP + 5:
                continue
            p = np.array([preds[s] for s in common]); y = np.array([real[s] for s in common])
            top_idx = np.argsort(p)[-N_TOP:]
            acc[h]["top"].append(float(np.mean(y[top_idx])))
            acc[h]["univ"].append(float(np.mean(y)))
            acc[h]["ic"].append(spearman_ic(p, y))
            acc[h]["diracc"].append(float(np.mean(np.sign(p) == np.sign(y))))
            acc[h]["pred"].extend(p.tolist()); acc[h]["real"].extend(y.tolist())
        n_pred_panels += 1
        if n_pred_panels % 6 == 0:
            logger.info(f"  forecast {n_pred_panels} panels (latest as_of={as_of}, elapsed {time.monotonic()-t0:.0f}s)")

    results = {}
    for label, h in HORIZONS:
        a = acc[h]
        r = aggregate_horizon(label, h, a["top"], a["univ"], a["ic"], a["diracc"], a["pred"], a["real"], None)
        if r:
            results[label] = r

    conn.close()
    return results, universe, panels, cols, close_mat, model_hl


def main():
    ap = argparse.ArgumentParser(description=f"Multi-Cycle TimesFM Zero-Shot Validation {TOOL_VER}")
    ap.add_argument("--seed", type=int, default=5422, help="(recorded only; zero-shot is deterministic — see core-def #9)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--context-weeks", type=int, default=CONTEXT_WEEKS)
    ap.add_argument("--freq", type=int, default=DEFAULT_FREQ, help="TimesFM freq index (0 high / 1 medium / 2 low)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-stocks", type=int, default=None, help="limit universe for smoke")
    ap.add_argument("--max-panels", type=int, default=None, help="limit #panels for smoke")
    ap.add_argument("--smoke", action="store_true", help="tiny config: 40 stocks, 6 panels")
    args = ap.parse_args()
    if args.smoke:
        args.max_stocks = args.max_stocks or 40
        args.max_panels = args.max_panels or 6
        args.batch_size = 64

    logger.info("=" * 100)
    logger.info(f"Multi-Cycle TimesFM Zero-Shot Validation {TOOL_VER} (Google foundation model · transformers-native)")
    logger.info(f"  COMMON COMPARISON BASELINE: universe latest-committed × 95 panels × 4 horizons × top-{N_TOP} × cost {COST_PER_REBAL}")
    logger.info(f"  checkpoint={args.checkpoint} context={args.context_weeks}wk smoke={args.smoke} max_stocks={args.max_stocks}")
    logger.info("=" * 100)

    t_global = time.monotonic()
    try:
        results, universe, panels, cols, close_mat, model_hl = run(args)
    except Exception as e:
        logger.error(f"TimesFM run FAILED ({type(e).__name__}): {e}")
        logger.error("  若為下載/網路錯誤 → 確認可連 HuggingFace Hub 或改 --checkpoint;不偽造結果(§一.8)")
        raise

    logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix (TimesFM zero-shot)\n{'='*100}")
    logger.info(f"  {'Horizon':10} {'N':>4} {'Eff t':>7} {'Sig':>4} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9} "
                f"{'rankIC':>7} {'DirAcc':>7}")
    for label, r in results.items():
        sig = "✅" if r["is_significant_p05"] else "❌"
        ic = r.get("rank_ic_mean"); da = r.get("directional_accuracy")
        logger.info(f"  {label:10} {r['n_panels']:>4} {r['effective_t_stat']:>+7.3f} {sig:>4} "
                    f"{r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}% "
                    f"{(ic if ic is not None else float('nan')):>+7.3f} "
                    f"{(da*100 if da is not None else float('nan')):>6.1f}%")
    ann = results.get("annual")
    if ann:
        verdict = all((ann.get(k) is not None and ann.get(k) >= thr) for k, thr in GATE.items())
        logger.info(f"\n  T_CZ-6 annual gate: {'✅ PASS' if verdict else '❌ FAIL'} — "
                    + " | ".join(f"{k}={ann.get(k)}(≥{thr})" for k, thr in GATE.items()))
    logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

    if args.output:
        out = {label: r for label, r in results.items()}
        out["_meta"] = {
            "tool": "multi_cycle_timesfm_validation.py", "tool_ver": TOOL_VER,
            "model": f"TimesFM zero-shot (Google decoder-only time-series foundation model, transformers-native TimesFmModelForPrediction, checkpoint={args.checkpoint})",
            "model_family": "foundation",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": args.seed, "horizons": [h for _, h in HORIZONS],
            "n_universe": len(universe), "n_stocks_with_series": len(cols),
            "n_weeks": int(close_mat.shape[0]), "n_panels_input": len(panels),
            "checkpoint": args.checkpoint, "context_weeks": args.context_weeks,
            "model_horizon_length": model_hl, "freq": args.freq, "smoke": args.smoke,
            "zero_shot": True, "training": "NONE (zero-shot; no fine-tune, no refit)",
            "deterministic": True,
            "variates": "stocks (per-stock weekly close series → zero-shot foundation forecast → log-return score)",
            "common_baseline": f"universe latest-committed × {len(panels)} panels × {HORIZONS} × top-{N_TOP} × cost {COST_PER_REBAL} × T_CZ-6",
            "source_traceability": "per CLAUDE.md §一.10 — series from (b) DB query TaiwanStockPriceAdj/core_universe_*; model = Google external pretrained foundation weights applied zero-shot (not system-trained, not hallucinated)",
        }
        op = Path(_base_dir).parent / args.output
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()

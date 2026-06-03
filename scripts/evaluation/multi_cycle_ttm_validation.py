"""
multi_cycle_ttm_validation.py v0.1 (TTM · TinyTimeMixer · IBM Granite 2024 時序基礎模型 Foundation Model · Ekambaram et al. "Tiny Time Mixers (TTMs)" · Multi-Cycle Stock-Price Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-03
**主權狀態**: TTM / TINYTIMEMIXER(IBM GRANITE 時序基礎模型 / 2024)ZERO-SHOT 4-HORIZON WALK-FORWARD VALIDATION + ⚠️ EXTERNAL-PRETRAINED PRIOR(模型權重非 DB-source-pure;與 Chronos 同類 external-prior 揭露)+ §14.7-DC v0.18 SOURCE-PURE *INPUT* UNIVERSE + §一.10 INPUT SOURCE-TRACEABLE(全 DB)+ 共同比較基準(COMMON COMPARISON BASELINE)實作 + §一.11 三段式合規 + §14.7-DE §0.0-I panel-date helper 單一引用源
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 TTM(TinyTimeMixer 時間序列基礎模型 foundation model) 序列模型,吃每支股票的「歷史週報酬序列」,預測未來報酬,評估「靠它選股能不能賺錢、準不準、可不可信」。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」,載入每支的歷史價格序列(序列模型看時間走勢,非橫斷面特徵)→ 換算成每週報酬。
2. 把 2013-05 ~ 2026-06 切成月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每點只用那之前的序列(週報酬)餵給模型,預測之後報酬,不偷看未來(防洩漏)。
4. 依預測挑最看好的股票做多,跟全市場平均比,算賺賠。
5. 在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
6. 算成績:報酬率、Sharpe、勝率,加上**排序 IC、淨 Sharpe**(此類序列模型用自有 `aggregate_horizon`;§14.7-DF 註明 torch/foundation 暫不套樹模型 metric helper,各模型 rework 時再對齊共同欄位後與樹模型並比)。
7. 判定這模型在哪個週期「真的能賺錢且可信」。

**輸入**:資料庫(股價序列)+ IBM Granite TTM 預訓練權重(由 HuggingFace 下載,非本 DB)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不改資料庫(純讀取評估;§3.1 evaluation 角色)。**不在本 DB fine-tune**(zero-shot 直接用預訓練先驗)。
**為什麼需要它**:序列/基礎模型路線的實證裁判,與樹模型並列比較(共同欄位對齊後)。⚠️ 注意此模型權重來自外部預訓練語料(非本 DB),為揭露之 caveat(見 §一.2)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,非序列落地鏈模組。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1/§0.2/§0.3);(c) T1-T3 不分層;
   (d) §8.5 anti-leakage:context 序列僅含 as_of 之前(含)之 weekly return,forecast target weeks 全在 as_of 之後 → 結構性無洩漏;
   (e) **不訓練 production model**(不寫 model_registry);(f) **read-only**(不改 feature_values / TaiwanStockPriceAdj / universe);
   (g) 唯一職責:TTM zero-shot 4-horizon walk-forward 預測 + 共同比較基準 metrics + JSON 持久化。
2. **[⚠️ External-Pretrained Prior — Source-Purity Caveat]** (v0.1, 同 Chronos external-prior 揭露慣例):
   ⚠️ **本模型與從頭訓練之 TFT / iTransformer / PatchTST / transformer_dedicated 在治權上有本質差異,必須醒目(PROMINENT)揭露**:
   - TTM(TinyTimeMixer)為 IBM Granite 之**預訓練時序基礎模型**(MLP-Mixer backbone,pretrained on 天量 EXTERNAL 真實時序語料,**NOT in this DB**);
   - **INPUT context 序列 100% DB-source-pure**(TaiwanStockPriceAdj weekly close → log return,§一.10 (b) DB query);
   - **但 MODEL WEIGHTS 編碼來自本 DB 之外之知識** → 此 predictive prior **非 §一.10 (a)(b)(c) 可 trace 至 DB / FinMind / FRED**,
     屬「外部預訓練先驗」之**新 source 類別**(與 from-scratch 模型「只從本 DB 學」**本質不同**);
   - 使用此 external prior 之**揭露條件 = 本 caveat 須於程式標頭 + 驗證報告醒目揭露**(本條 + report §三);
   - 本程式為**真正之 IBM Granite TinyTimeMixer**(非 from-scratch surrogate);為「時序基礎模型(foundation model)族」之
     另一代表實作(與 Chronos 同類 external-pretrained 路線,可並列比較不同 foundation model 之 OUTPUT 預測品質)。
3. **[Common Comparison Baseline]** (v0.1, reports/common_model_comparison_baseline_v1.md): 本程式為共同比較基準之實作 —
   universe v0.18(398 source-pure)× canonical monthly panels(§14.7-DE 資料驅動)× 真實 forward log returns(TaiwanStockPriceAdj)×
   4 horizons(5/20/60/252d)× top-20 equal-weight long × 0.6% cost × {Sharpe, Win, Eff-t, T_CZ-6 gate}。
   **評估協定與 realized targets 與全模型完全相同** → 精準度 / 信任度比較 apples-to-apples;
   模型用各自 natural representation(TTM = 每股 zero-shot 單變量 weekly return 序列),比較點在 OUTPUT 預測品質。
4. **[Input Source Traceability]** (v0.1, CLAUDE.md §一.10): **INPUT 全數據** (b) DB query(TaiwanStockPriceAdj close + core_universe_*)+
   (a) program output(本 JSON / log);weekly return 全為 close 之 source-pure mathematical transform(calendar-week resample → log return);
   **無 imputed / 無 forward-fill / 無 hardcoded knowledge**;calendar-week gap 不補值(跨 gap 之 return 留 NaN → 含 gap 之 context 自然排除)。
   ⚠️ **唯一 exception = model prior(見 §一.2 caveat)**。fixed-length context tensor 之 left-pad 位置以 `past_observed_mask=0` 明示為「未觀測」
   (非偽造 feature value;模型 scaling 與 attention 忽略 padded 位置)→ 不違反 source-purity。
5. **[Longest-Available Per-Stock History]** (v0.1, 用戶 2026-05-30 directive): 每股取其在 DB 之最長 daily 歷史
   (calendar-week resample → weekly return),context 取最近 --context-weeks 個**真實觀測週**(歷史愈長之股 → context 愈完整);
   不足 MIN_CONTEXT_WEEKS 真實觀測週之股直接排除(不補值)。
6. **[Real TTM — Zero-Shot Point Forecasting]** (v0.1, Ekambaram et al. 2024 "Tiny Time Mixers"): TTM 以 MLP-Mixer backbone
   對 patch 化之時序做 mixing;**zero-shot**(不在本 DB fine-tune),以每股 context(最近 context_length 真實觀測週 weekly return,
   left-pad 至模型 context_length + observed_mask)直接 forecast 未來 prediction_length 週之 weekly return point estimate
   (`.prediction_outputs`)。cumsum → 各 horizon 之 forward log return。score = cumulative forward log return[horizon step]。
7. **[Point-Forecast → Calibration N/A]** (v0.1): TTM `.prediction_outputs` 為 point forecast(config loss="mse")→
   **calibration_p10_p90_coverage = None**(同 iTransformer / PatchTST;unlike Chronos quantile 模型)。信任度 = Eff-t significance + 多 seed 穩定度。
8. **[Precision / Trust / Profitability 三分]** (v0.1): 精準度(rank-IC / dir-acc / RMSE / MAE / R²)、信任度(Eff-t significance /
   多 seed 穩定度 / calibration[N/A])、賺錢能力(net-of-cost Sharpe / Eff-t / Win / annualized net / T_CZ-6 gate)分開報告。
9. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): significance 動態(abs(eff_t) > 1.997);T_CZ-6 gate(4.20/2.40/0.79)
   為 charter-mandated reference threshold(Tier 3 transparent disclosure,非 feature data,非硬編 verdict)。
10. **[§一.10 #3 Multi-Run]** (v0.1): 跑 ≥3 seeds {5422,7331,1009} 以 protocol-consistency;
    ⚠️ **TTM zero-shot point forecast 為 ~deterministic(無 autoregressive sampling)→ 跨 seed spread ≈ 0**;
    此 determinism 本身為一 trust 觀察(無 seed 變異),報告須誠實揭露(非以 spread 假裝 stochastic 穩定度)。
11. **[Schema-Compatible Output]** (v0.1): per-horizon JSON keys 與 multi_cycle_validation.py / chronos / itransformer / patchtst 對齊 → _aggregate.py 可直接 roll up。
12. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
13. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON);checkpoint 快取於 .hf_cache;可重跑;model / context / horizon 全可配置。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Common Comparison Baseline(共同比較基準 — 與全模型一致,函式逐字相同)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Universe | get_universe() → 最新 committed snapshot core_tier='core_universe'(398 v0.18)| §14.7-DC v0.18 |
| A.2 Panels | get_canonical_panel_dates() → canonical monthly mid-month(§14.7-DE 資料驅動)| 與 baseline 同 grid |
| A.3 Forward returns | load_forward_returns() → 真實 log return(TaiwanStockPriceAdj)| §一.10 (b) DB |
| A.4 Portfolio | top-20 equal-weight long(np.argsort[-20:])| 與 baseline 同 |
| A.5 Profitability | aggregate_horizon():sharpe / win / mdd / eff_t / annualized_net / T_CZ-6 | §14.7-CY / §14.7-CZ |

### Group B. TTM Foundation Model(zero-shot 每股單變量 weekly return → point forecast)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Return matrix | load_return_matrix() → daily close → calendar-week → log-return matrix[week × stock]| §一.10 source-pure INPUT |
| B.2 Model | load_model() → TinyTimeMixerForPrediction.from_pretrained(ibm-granite/granite-timeseries-ttm-r2)(⚠️ external-pretrained,見 §一.2)| Ekambaram et al. 2024 |
| B.3 Predict | predict_panel() → 每股 context(最近 N 真實觀測週,left-pad + observed_mask)→ .prediction_outputs | §8.5 leakage-safe(context ≤ as_of)|
| B.4 Score | score = cumsum(predicted weekly returns)[horizon step]| 機率/序列預測契約 |
| B.5 Walk-forward | zero-shot → 無 refit / 無 training(基礎模型先驗固定)| 基礎模型特性 |

### Group C. Precision / Trust(標準化 block — 全模型共用定義)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Precision | rank_ic / directional_accuracy / rmse / mae / r2(pred vs 真實 realized)| 共同基準 |
| C.2 Trust | effective_t_stat / is_significant_p05 / calibration(None, point-forecast)| 共同基準 + §9.1 |
| C.3 Multi-seed | --seed {5422,7331,1009} → _aggregate.py(⚠️ TTM deterministic → spread≈0,誠實揭露)| §一.10 #3 |

### Group D. Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 JSON | --output reports/ttm_v0/<...>.json(schema-compatible + precision/trust + external_prior caveat in _meta)| §一.10 / §二.4 |
| D.2 stdout | cross-cycle comparison matrix + 進度回報 | §一.12 進度 |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令(本機 ./venv 已含 tsfm_public + torch)|
| :--- | :--- |
| Smoke(plumbing 驗證)| `./venv/bin/python scripts/evaluation/multi_cycle_ttm_validation.py --smoke --output reports/ttm_v0/_smoke.json` |
| 完整單 seed | `./venv/bin/python ... --seed 5422 --output reports/ttm_v0/ttm_s5422.json` |
| 3-run 教義 | 對 {5422,7331,1009} 各跑 → _aggregate.py(注意 TTM determinism)|

### 不提供之旗標 (Intentionally Omitted)

- `--commit`:本工具不寫 evaluation_log / model_registry(§3.2 橫切只讀)。
- `--cost-per-rebal`:0.6% standard per §14.7-CY T_CY-5(與 baseline 對齊,不可變更否則破壞比較基準)。
- `--horizons`:固定 5/20/60/252(共同比較基準定義之一部分,不可變更)。
- `--refit-every` / `--epochs`:TTM zero-shot 無訓練(基礎模型先驗固定),刻意省略。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-03 | Codex | **首版**:TTM(TinyTimeMixer / IBM Granite 時序基礎模型 / 2024 / Ekambaram et al.)zero-shot multi-cycle 股價預測驗證 + 共同比較基準實作。每股最長 weekly log-return 序列(取最近 context_length 真實觀測週,left-pad + past_observed_mask)→ TinyTimeMixerForPrediction.from_pretrained → .prediction_outputs(point forecast 未來 prediction_length 週 weekly return)→ cumsum → 4-horizon scores(HORIZON_STEPS {5:1,20:4,60:12,252:50})。⚠️ **external-pretrained prior**(模型權重非 DB-source-pure;同 Chronos external-prior 揭露慣例,醒目揭露於 §一.2)。INPUT 全 DB source-pure(TaiwanStockPriceAdj weekly close → log return,無 imputed/forward-fill;fixed-length context 之 left-pad 以 observed_mask=0 明示未觀測,非偽造值)。calibration N/A(point forecast,同 iTransformer/PatchTST)。與 baseline 同 universe(v0.18/398)/ panels(§14.7-DE 資料驅動)/ forward returns / portfolio(top-20)/ cost(0.6%)/ gate(T_CZ-6)。§一.10 #3 multi-seed(TTM deterministic → spread≈0,誠實揭露)。§8.5 leakage-safe(context ≤ as_of)。§一.11 三段式合規。 | **ACTIVE** |
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")  # NumPy log(NaN) + torch/transformers non-fatal warnings
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, argparse, math, json, logging, time
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # stock_backend/
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))  # contain checkpoint cache locally

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
from core.db_utils import get_db_conn, get_canonical_panel_dates

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

# ── TTM / weekly constants ──
CONTEXT_WEEKS = 512        # per-stock zero-shot context = TTM r2 native context_length (~10 yr of weeks)
MIN_CONTEXT_WEEKS = 60     # minimum real observed weeks to include a stock (rest left-padded + masked)
HORIZON_STEPS = {5: 1, 20: 4, 60: 12, 252: 50}  # trading days → weekly forecast step index (all ≤ TTM pred_len 96)
DEFAULT_MODEL = "ibm-granite/granite-timeseries-ttm-r2"


# ════════════════════════════════════════════════════════════════════════════
# Group A — Common Comparison Baseline DB loaders (與 baseline / chronos / itransformer 逐字相同)
# ════════════════════════════════════════════════════════════════════════════
def get_universe(cur):
    cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                   JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                   WHERE s.status='committed' AND m.core_tier='core_universe'
                   AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                      WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
    return sorted({r[0] for r in cur.fetchall()})


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
# Group B — TTM return-matrix construction (per-stock univariate weekly return) + zero-shot forecast
# ════════════════════════════════════════════════════════════════════════════
def load_return_matrix(cur, universe, max_stocks=None):
    """每股最長 daily 歷史 → calendar-week resample → 跨股 log-return matrix (與 itransformer load_return_matrix 同構).

    Returns:
        ret_mat: np.ndarray [W weeks × N stocks], weekly log return; NaN where stock 未交易該週
                 (或前一週缺 → 跨 gap 之 return 不偽造,留 NaN → 含 gap 之 context 自然排除 = source-pure)。
        cols:    list[str] stock_id 對應 matrix 欄序。
        week_date: list[date] 各 week_idx 之代表日(該週全股觀測之最大日期)→ 用於 as_of → week_idx 對映。"""
    if max_stocks:
        universe = universe[:max_stocks]
    stock_weekly = {}   # sid -> {(iso_year, iso_week): (week_end_date, close)}
    all_keys = set()
    for sid in universe:
        cur.execute("""SELECT date, close FROM "TaiwanStockPriceAdj"
                       WHERE stock_id=%s AND close>0 ORDER BY date""", (sid,))
        rec = cur.fetchall()
        wk = {}
        for d, c in rec:
            iso = d.isocalendar()
            key = (iso[0], iso[1])
            if key not in wk or d > wk[key][0]:   # last close in the calendar week
                wk[key] = (d, float(c))
        if len(wk) < MIN_CONTEXT_WEEKS + 5:
            continue
        stock_weekly[sid] = wk
        all_keys |= set(wk.keys())
    cols = [s for s in universe if s in stock_weekly]
    keys_sorted = sorted(all_keys)              # tuple sort = chronological
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
    ret_mat = np.full((W, N), np.nan)
    ret_mat[1:] = np.log(close_mat[1:]) - np.log(close_mat[:-1])  # NaN propagates over gaps/edges
    return ret_mat, cols, week_date


def load_model(model_id, seed):
    """Load REAL pretrained IBM Granite TinyTimeMixer (⚠️ external-pretrained prior — §一.2 caveat)."""
    import torch
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyTimeMixerForPrediction.from_pretrained(model_id)
    model.eval()
    ctx_len = int(model.config.context_length)
    pred_len = int(model.config.prediction_length)
    return model, ctx_len, pred_len


def predict_panel(model, ret_mat, week_cut, cols, model_ctx_len, pred_len,
                  context_weeks, batch_size):
    """Zero-shot forecast next `pred_len` weekly returns per stock from observed context ending ≤ week_cut.
    Returns fwd: per-horizon dict sid -> predicted forward LOG RETURN (cumsum of predicted weekly returns).
    Point forecast → no quantiles → weekly calibration N/A. §8.5 leakage-safe (context ≤ as_of)."""
    import torch
    fwd = {h: {} for _, h in HORIZONS}
    sub = ret_mat[1:week_cut + 1]                  # weekly returns with week_date ≤ as_of (leakage-safe)
    eff_ctx = min(context_weeks, model_ctx_len)    # real-week context cap (≤ model native context_length)
    ctxs, masks, sids = [], [], []
    for ci, sid in enumerate(cols):
        obs = sub[:, ci]
        obs = obs[np.isfinite(obs)]                # real observed weekly returns only (no imputation)
        if obs.size < MIN_CONTEXT_WEEKS:
            continue
        ctx = obs[-eff_ctx:]                        # most recent real observed weeks
        n_real = ctx.size
        padded = np.zeros(model_ctx_len, dtype=np.float32)   # left-pad to model context_length
        mask = np.zeros(model_ctx_len, dtype=np.float32)
        padded[-n_real:] = ctx                      # real values right-aligned
        mask[-n_real:] = 1.0                        # observed_mask=1 for real weeks, 0 for left-pad
        ctxs.append(padded)
        masks.append(mask)
        sids.append(sid)
    if len(sids) < N_TOP + 5:
        return fwd
    for b in range(0, len(sids), batch_size):
        cb = np.stack(ctxs[b:b + batch_size])[:, :, None]    # [B, context_length, 1 channel]
        mb = np.stack(masks[b:b + batch_size])[:, :, None]   # [B, context_length, 1]
        pv = torch.tensor(cb, dtype=torch.float32)
        om = torch.tensor(mb, dtype=torch.float32)
        with torch.no_grad():
            out = model(past_values=pv, past_observed_mask=om)
        pred = out.prediction_outputs.cpu().numpy()[:, :, 0]  # [B, pred_len] weekly return point forecast
        cum = np.cumsum(pred, axis=1)                         # cumulative forward log return
        for k in range(pred.shape[0]):
            sid = sids[b + k]
            for _, h in HORIZONS:
                step = HORIZON_STEPS[h]
                if step > cum.shape[1]:
                    continue
                fwd[h][sid] = float(cum[k, step - 1])
    return fwd


# ════════════════════════════════════════════════════════════════════════════
# Group A/C — metrics (portfolio + precision + trust) — 與 baseline / chronos / itransformer 逐字相同
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
    """共同比較基準 metric 計算(profitability 數學與 multi_cycle_validation.py 完全相同)+ precision/trust."""
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
        # ── standardized PRECISION block ──
        "rank_ic_mean": float(np.mean(panel_ics)) if panel_ics else None,
        "rank_ic_std": float(np.std(panel_ics, ddof=1)) if len(panel_ics) > 1 else None,
        "rank_ic_ir": (float(np.mean(panel_ics)) / float(np.std(panel_ics, ddof=1)) * math.sqrt(12))
                      if len(panel_ics) > 1 and np.std(panel_ics, ddof=1) > 0 else None,
        "directional_accuracy": float(np.mean(panel_diracc)) if panel_diracc else None,
        "rmse": rmse, "mae": mae, "r2": r2,
        # ── standardized TRUST block ──
        "calibration_p10_p90_coverage": calib_cov,  # None for TTM (point forecast, non-quantile)
    }


# ════════════════════════════════════════════════════════════════════════════
# Orchestration (zero-shot — no training / no refit)
# ════════════════════════════════════════════════════════════════════════════
def run(args):
    conn = get_db_conn()
    cur = conn.cursor()
    universe = get_universe(cur)
    logger.info(f"Universe: {len(universe)} stocks (v0.18 source-pure INPUT)")
    panels = [d for _fsid, d in get_canonical_panel_dates("feature_set_v0.5")]  # §14.7-DE / §0.0-I 單一引用源
    if args.max_panels:
        panels = panels[:args.max_panels]
    logger.info(f"Panels: {len(panels)} monthly as_of dates")

    t0 = time.monotonic()
    ret_mat, cols, week_date = load_return_matrix(cur, universe, max_stocks=args.max_stocks)
    logger.info(f"Return matrix: {len(cols)} stocks × {ret_mat.shape[0]} calendar-weeks (load {time.monotonic()-t0:.1f}s)")

    logger.info(f"Loading TTM model '{args.model_id}' (⚠️ EXTERNAL-PRETRAINED prior — §一.2 caveat; first run downloads checkpoint)...")
    tp = time.monotonic()
    model, model_ctx_len, pred_len = load_model(args.model_id, args.seed)
    logger.info(f"  model ready in {time.monotonic()-tp:.1f}s | native_context={model_ctx_len}wk pred={pred_len}wk "
                f"context_weeks={min(args.context_weeks, model_ctx_len)} batch={args.batch_size} seed={args.seed}")

    week_ord = np.array([d.toordinal() if d else -1 for d in week_date])

    def week_cut_for(as_of):
        ao = as_of.toordinal()
        idx = np.where(week_ord <= ao)[0]
        return int(idx[-1]) if len(idx) else None

    acc = {h: {"top": [], "univ": [], "ic": [], "diracc": [], "pred": [], "real": []} for _, h in HORIZONS}
    n_pred_panels = 0
    for pi, as_of in enumerate(panels):
        wc = week_cut_for(as_of)
        if wc is None or wc < MIN_CONTEXT_WEEKS:
            continue
        fwd = predict_panel(model, ret_mat, wc, cols, model_ctx_len, pred_len,
                            args.context_weeks, args.batch_size)
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
            logger.info(f"  predicted {n_pred_panels} panels (latest as_of={as_of}, elapsed {time.monotonic()-t0:.0f}s)")

    results = {}
    for label, h in HORIZONS:
        a = acc[h]
        r = aggregate_horizon(label, h, a["top"], a["univ"], a["ic"], a["diracc"], a["pred"], a["real"], None)
        if r:
            results[label] = r

    conn.close()
    return results, universe, panels, cols, ret_mat


def main():
    ap = argparse.ArgumentParser(description=f"Multi-Cycle TTM (IBM Granite foundation model) Validation {TOOL_VER}")
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL,
                    help="HF model id (ibm-granite/granite-timeseries-ttm-r2 or -r1)")
    ap.add_argument("--context-weeks", type=int, default=CONTEXT_WEEKS,
                    help="real-week context cap (capped to model native context_length)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-stocks", type=int, default=None, help="limit universe for smoke")
    ap.add_argument("--max-panels", type=int, default=None, help="limit #panels for smoke")
    ap.add_argument("--smoke", action="store_true", help="tiny config: 40 stocks, 6 panels")
    args = ap.parse_args()
    if args.smoke:
        args.max_stocks = args.max_stocks or 40
        args.max_panels = args.max_panels or 6
        args.batch_size = min(args.batch_size, 16)

    logger.info("=" * 100)
    logger.info(f"Multi-Cycle TTM Validation {TOOL_VER} (TinyTimeMixer · IBM Granite Time-Series Foundation Model / 2024)")
    logger.info("  ⚠️  EXTERNAL-PRETRAINED PRIOR — model weights NOT DB-source-pure (see header §一.2)")
    logger.info(f"  COMMON COMPARISON BASELINE: source-pure universe (data-driven §14.7-DE) × 4 horizons × top-{N_TOP} × cost {COST_PER_REBAL}")
    logger.info(f"  model={args.model_id} seed={args.seed} smoke={args.smoke} max_stocks={args.max_stocks}")
    logger.info("=" * 100)

    t_global = time.monotonic()
    results, universe, panels, cols, ret_mat = run(args)

    logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix (TTM · zero-shot)\n{'='*100}")
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
            "tool": "multi_cycle_ttm_validation.py", "tool_ver": TOOL_VER,
            "model": "TTM / TinyTimeMixer (IBM Granite, 2024, external pretrained, zero-shot)",
            "model_id": args.model_id,
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": args.seed, "horizons": [h for _, h in HORIZONS],
            "n_universe": len(universe), "n_stocks_with_series": len(cols),
            "n_weeks": int(ret_mat.shape[0]), "n_panels_input": len(panels),
            "context_weeks": args.context_weeks, "batch_size": args.batch_size,
            "variates": "per-stock univariate weekly log return (zero-shot; foundation model)",
            "is_foundation_model": True, "zero_shot": True, "calibration_available": False,
            "external_pretrained_prior": True,
            "source_purity_caveat": ("⚠️ INPUT series 100% DB-source-pure (TaiwanStockPriceAdj weekly close → log return, "
                                     "§一.10 (b)); BUT model weights are EXTERNAL-PRETRAINED on non-DB corpora — predictive "
                                     "prior NOT DB/FinMind/FRED-traceable. Fixed-length context left-pad uses past_observed_mask=0 "
                                     "(positions flagged unobserved, NOT fabricated values). TTM is IBM Granite TinyTimeMixer "
                                     "(Ekambaram et al. 2024), a foundation-model peer to Chronos."),
            "common_baseline": f"universe v0.18 × {len(panels)} panels × {HORIZONS} × top-{N_TOP} × cost {COST_PER_REBAL} × T_CZ-6",
            "source_traceability": "per CLAUDE.md §一.10 — INPUT from (b) DB query TaiwanStockPriceAdj/core_universe_*; model prior external (see caveat)",
        }
        op = PROJECT_ROOT / args.output
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()

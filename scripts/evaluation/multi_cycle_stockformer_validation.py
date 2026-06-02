"""
multi_cycle_stockformer_validation.py v0.1 (Stockformer · Stock-Specific Temporal Transformer w/ Series Decomposition · Multi-Cycle Stock-Price Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: STOCKFORMER(STOCK-SPECIFIC TEMPORAL TRANSFORMER · SERIES-DECOMPOSITION + FULL-RESOLUTION TEMPORAL SELF-ATTENTION)4-HORIZON WALK-FORWARD VALIDATION + §14.7-CY HORIZON-DOCTRINE 第四族(stock-specific neural)+ 最新 committed SOURCE-PURE UNIVERSE(914)+ §一.10 SOURCE-TRACEABLE(全 DB)+ §一.10 #3 MULTI-RUN + 共同比較基準(COMMON COMPARISON BASELINE)第四實作 + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,非序列落地鏈模組。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1/§0.2/§0.3);(c) T1-T3 不分層;
   (d) §8.5 anti-leakage:lookback window 僅含 as_of 之前(含)之 weekly bars,forecast 之 target weeks 全在 as_of 之後 → 結構性無洩漏;RevIN per-instance norm 僅用 lookback 自身統計量;
   (e) **不訓練 production model**(不寫 model_registry);(f) **read-only**(不改 feature_values / TaiwanStockPriceAdj / universe);
   (g) 唯一職責:Stockformer 4-horizon walk-forward 預測 + 共同比較基準 metrics + JSON 持久化。
2. **[Common Comparison Baseline]** (v0.1, reports/model_comparison_baseline_spec_20260601.md §七): 本程式為共同比較基準之**第四實作**(前三:TFT / iTransformer / PatchTST)—
   最新 committed source-pure universe × 95 monthly panels × 真實 forward log returns(TaiwanStockPriceAdj)× 4 horizons(5/20/60/252d)×
   top-20 equal-weight long × 0.6% cost × {Sharpe, Win, Eff-t, T_CZ-6 gate}。與全 tree / TFT / iTransformer / PatchTST 套用**完全相同** protocol +
   **完全相同** realized targets → 精準度(precision)/ 信任度(trust)比較 apples-to-apples。模型用各自 natural representation
   (tree=37 cross-sectional features;iTransformer=跨股 cross-variate attention;PatchTST=channel-independent patched;
   **Stockformer=每股全解析度 weekly 序列 → series decomposition(trend+殘差)→ 時間維 self-attention**),比較點在 OUTPUT 預測之品質,非 input。
3. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全數據 (b) DB query(TaiwanStockPriceAdj close + core_universe_*)+ (a) program output(本 JSON / log);
   **0 AI memory reuse**;weekly returns 全為 close 之 source-pure mathematical transform(calendar-week resample → log return),
   **無 imputed / 無 forward-fill / 無 hardcoded knowledge / 無 §一.13 第四類幻像值**;calendar-week gap 之 stock 於該 window 直接排除(不補值)。
4. **[Longest-Available Per-Stock History]** (v0.1, 用戶 2026-05-30 directive): 每股取其在 DB 之最長 daily 歷史(calendar-week resample → weekly return)。
   共同 weekly grid(全股 ISO-week union),各股缺週為 NaN;歷史愈長之股 → 進入愈多 training sample(自然偏好長歷史)。
5. **[Stockformer — Series Decomposition + Temporal Attention(誠實標示:from-scratch 改編,非特定論文逐字復現)]** (v0.1):
   本實作為 **stock-transformer 家族之 from-scratch torch.nn 改編**(非宣稱復現某特定 "Stockformer" 論文之完整架構;§一.8 誠實標示)。
   核心設計與 PatchTST / iTransformer 明確區隔 —
   (i) **Series Decomposition**:每股 lookback 經 moving-average 拆為 **trend**(低頻景氣/趨勢分量)+ **seasonal/residual**(高頻波動分量)兩通道
       — 對映 §0.3-A 「景氣 cycle + 股價變化 cycle」多尺度循環思想之**資料驅動分解**(無 hardcoded 週期長度,kernel 為標準 calendar 卷積窗,Tier 1);
   (ii) **Full-Resolution Temporal Self-Attention**:對 L 個 weekly timestep 逐點 token 化(非 patch、非跨股)→ 學習式 positional → nn.TransformerEncoder 跨時間步 attention → 時間維 mean-pool → forecast head;
   (iii) **RevIN per-instance normalization**(lookback-only mean/std → leakage-safe)。與 PatchTST(patch token、序列內局部語義)/ iTransformer(跨 variate attention)為三種不同 inductive bias。
6. **[Point-Forecast → Calibration N/A]** (v0.1): Stockformer 為 point forecast(MSE loss),非 quantile →
   **calibration_p10_p90_coverage = None**。信任度 = Eff-t significance + 多 seed 穩定度(§一.10 #3)。
7. **[Precision / Trust / Profitability 三分]** (v0.1): 精準度(rank-IC / directional accuracy / RMSE / MAE / R²)、
   信任度(Eff-t significance / 多 seed 穩定度 / calibration[N/A])、賺錢能力(net-of-cost Sharpe / Eff-t / Win / annualized net / T_CZ-6 gate)分開報告。
8. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): significance 動態(abs(eff_t) > 1.997);T_CZ-6 gate(4.20/2.40/0.79)
   為 charter-mandated reference threshold(Tier 3 transparent disclosure,非 feature data,非硬編 verdict)。
9. **[§一.10 #3 Multi-Run]** (v0.1): stochastic(torch init / dropout / sgd shuffle)→ 須 ≥3 seeds {5422,7331,1009};
   single-run 不得作為 deterministic charter fact;median 為 inscription central estimate。
10. **[Schema-Compatible Output]** (v0.1): per-horizon JSON keys 與 multi_cycle_validation.py / TFT / iTransformer / PatchTST 對齊 → build_model_comparison_report.py 可直接納入 leaderboard。
11. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
12. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON);可重跑;refit cadence / model size / epochs / lookback / kernel 全可配置。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Common Comparison Baseline(共同比較基準 — 與全模型一致,函式逐字相同)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Universe | get_universe() → 最新 committed snapshot core_tier='core_universe'(914)| 最新 committed |
| A.2 Panels | get_panel_dates() → 95 monthly mid-month(2018-06-15~2026-04-15)| 與 baseline 同 grid |
| A.3 Forward returns | load_forward_returns() → 真實 log return(TaiwanStockPriceAdj)| §一.10 (b) DB |
| A.4 Portfolio | top-20 equal-weight long(np.argsort[-20:])| 與 baseline 同 |
| A.5 Profitability | aggregate_horizon():sharpe / win / mdd / eff_t / annualized_net / T_CZ-6 | §14.7-CY / §14.7-CZ |

### Group B. Stockformer Model(series-decomposition + full-resolution temporal attention)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Return matrix | load_return_matrix() → daily close → calendar-week → log-return matrix[week × stock]| §一.10 / §一.13 source-pure |
| B.2 Model | Stockformer:RevIN + SeriesDecomp(trend/seasonal)+ per-step embed + pos + TransformerEncoder(跨時間步)+ mean-pool head(→S)| from-scratch torch.nn |
| B.3 Train | train_stockformer() → channel-independent (window×stock) samples + Adam + MSE + grad-clip(CPU)| per-stock 序列 |
| B.4 Predict | predict_forward() → 每股 lookback → forecast S weekly returns → cumsum → 4-horizon scores | §8.5 leakage-safe |
| B.5 Walk-forward refit | --refit-every N panels(default annual≈12)| expanding window |

### Group C. Precision / Trust(標準化 block — 全模型共用定義)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Precision | rank_ic / directional_accuracy / rmse / mae / r2(pred vs 真實 realized)| 共同基準 |
| C.2 Trust | effective_t_stat / is_significant_p05 / calibration(None, point-forecast)| 共同基準 |
| C.3 Multi-seed | --seed {5422,7331,1009} → 跨 seed 聚合 | §一.10 #3 |

### Group D. Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 JSON | --output reports/multi_cycle_stockformer_<ts>.json(schema-compatible + precision/trust)| §一.10 / §二.4 |
| D.2 stdout | cross-cycle comparison matrix + precision/trust 摘要 | §一.12 進度 |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| Smoke(plumbing 驗證)| `python scripts/evaluation/multi_cycle_stockformer_validation.py --smoke --output reports/_smoke_stockformer.json` |
| 完整單 seed | `... --seed 5422 --output reports/multi_cycle_stockformer_<ts>.json` |
| 3-run 教義全合規 | 對 {5422,7331,1009} 各跑一次 → 跨 seed 聚合 |

### 不提供之旗標 (Intentionally Omitted)

- `--commit`:本工具不寫 evaluation_log / model_registry(屬 model_trainer 治權;§3.2 橫切只讀)。
- `--cost-per-rebal`:0.6% standard per §14.7-CY T_CY-5(與 baseline 對齊,不可變更否則破壞比較基準)。
- `--horizons`:固定 5/20/60/252(共同比較基準定義之一部分,不可變更)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-02 | Claude | **首版**:Stockformer(stock-specific temporal transformer, from-scratch 改編)multi-cycle 股價預測驗證 + 共同比較基準第四實作。每股 weekly return 序列 → RevIN → series decomposition(moving-avg trend + seasonal/residual 兩通道,對映景氣+股價多尺度循環思想之資料驅動分解)→ 逐時間步 token + 學習式 positional → TransformerEncoder 跨時間步 self-attention → 時間維 mean-pool → forecast S weeks → 4-horizon cumsum scores。與 baseline 同 universe(最新 committed 914)/ panels(95)/ forward returns(TaiwanStockPriceAdj)/ portfolio(top-20)/ cost(0.6%)/ gate(T_CZ-6)。precision(rank-IC / dir-acc / RMSE / MAE / R²)+ trust(Eff-t / 多 seed;calibration N/A 因 point-forecast)。§一.10 全 DB source-traceable;§一.10 #3 multi-seed;§8.5 leakage-safe by lookback/forecast split + RevIN lookback-only;§一.8 誠實標示為 from-scratch 改編非特定論文逐字復現;§一.11 三段式合規。 | **ACTIVE** |
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

# ── Stockformer / weekly constants ──
LOOKBACK_WEEKS = 104       # per-stock lookback (~2 yr)
PRED_WEEKS = 52            # forecast horizon (covers annual 252d ≈ 50 wk)
DECOMP_KERNEL = 13         # moving-average kernel for trend/seasonal decomposition (~quarter, odd)
HORIZON_STEPS = {5: 1, 20: 4, 60: 12, 252: 50}  # trading days → weekly forecast step index


# ════════════════════════════════════════════════════════════════════════════
# Group A — Common Comparison Baseline DB loaders (與 baseline / PatchTST / iTransformer 逐字相同)
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
# Group B — Stockformer return-matrix construction (channel = stock) — 與 PatchTST 逐字相同
# ════════════════════════════════════════════════════════════════════════════
def load_return_matrix(cur, universe, max_stocks=None):
    """每股最長 daily 歷史 → calendar-week resample → log-return matrix[week × stock].
    含 calendar-week gap 之 return 不偽造,留 NaN → 含 gap 之 window 自然排除 = source-pure。"""
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
        if len(wk) < LOOKBACK_WEEKS + PRED_WEEKS + 5:
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
    ret_mat = np.full((W, N), np.nan)
    ret_mat[1:] = np.log(close_mat[1:]) - np.log(close_mat[:-1])
    return ret_mat, cols, week_date


# ── Stockformer architecture (from-scratch torch.nn; series-decomposition + temporal attention + RevIN) ──
def _build_model(L, S, d_model, n_heads, n_layers, kernel_size, dropout):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SeriesDecomp(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):                                    # x:[B,L]
            pad_l = (self.k - 1) // 2
            pad_r = self.k - 1 - pad_l
            xp = x.unsqueeze(1)                                  # [B,1,L]
            xp = F.pad(xp, (pad_l, pad_r), mode="replicate")
            trend = F.avg_pool1d(xp, self.k, stride=1).squeeze(1)  # [B,L] low-freq trend
            seasonal = x - trend                                 # [B,L] high-freq residual
            return seasonal, trend

    class Stockformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.decomp = SeriesDecomp(kernel_size)
            self.embed = nn.Linear(2, d_model)                   # [seasonal, trend] per timestep → token
            self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers)  # attention ACROSS timesteps (within a stock)
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, S)                    # temporal-pooled rep → future S weekly returns

        def forward(self, x):                                    # x:[B,L] one stock per row
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True) + 1e-5
            xn = (x - mu) / sigma                                # RevIN (lookback-only → leakage-safe)
            seasonal, trend = self.decomp(xn)
            tok = torch.stack([seasonal, trend], dim=-1)         # [B,L,2]
            h = self.embed(tok) + self.pos                       # [B,L,d_model]
            h = self.drop(h)
            h = self.encoder(h)                                  # cross-timestep self-attention
            h = self.norm(h)
            h = h.mean(dim=1)                                    # temporal mean-pool [B,d_model]
            out = self.head(h)                                   # [B,S] (normalized space)
            return out * sigma + mu                              # RevIN denormalize → raw weekly returns

    return Stockformer()


def train_stockformer(ret_mat, week_cut, L, S, seed, epochs, lr, d_model, n_heads, n_layers,
                      kernel_size, batch_size, dropout, max_train_samples, min_samples=50):
    """Train channel-independent Stockformer on all (window × stock) samples whose lookback+target ∈ weeks ≤ week_cut."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    we_list, ci_list = [], []
    for we in range(L, week_cut - S + 1):
        lb = ret_mat[we - L + 1:we + 1]
        tg = ret_mat[we + 1:we + S + 1]
        valid = np.where(np.isfinite(lb).all(axis=0) & np.isfinite(tg).all(axis=0))[0]
        if valid.size:
            we_list.append(np.full(valid.size, we, dtype=np.int64))
            ci_list.append(valid.astype(np.int64))
    if not we_list:
        return None
    we_arr = np.concatenate(we_list)
    ci_arr = np.concatenate(ci_list)
    n_samples = we_arr.size
    if n_samples < min_samples:
        return None
    if n_samples > max_train_samples:
        sel = np.random.choice(n_samples, max_train_samples, replace=False)
        we_arr, ci_arr = we_arr[sel], ci_arr[sel]
        n_samples = max_train_samples

    ar_L = np.arange(L)[None, :]
    ar_S = np.arange(1, S + 1)[None, :]

    model = _build_model(L, S, d_model, n_heads, n_layers, kernel_size, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    order = np.arange(n_samples)
    for _ep in range(epochs):
        np.random.shuffle(order)
        for bs in range(0, n_samples, batch_size):
            idx = order[bs:bs + batch_size]
            we_b = we_arr[idx][:, None]
            ci_b = ci_arr[idx][:, None]
            rows_x = we_b - L + 1 + ar_L
            rows_y = we_b + ar_S
            Xb = ret_mat[rows_x, ci_b]
            Yb = ret_mat[rows_y, ci_b]
            X = torch.tensor(Xb, dtype=torch.float32)
            Y = torch.tensor(Yb, dtype=torch.float32)
            pred = model(X)
            loss = lossf(pred, Y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def predict_forward(model, ret_mat, week_cut, L, S, cols):
    """Forecast next S weekly returns per stock from lookback ending at week_cut → cumsum → 4-horizon scores."""
    import torch
    fwd = {h: {} for _, h in HORIZONS}
    if week_cut - L + 1 < 1:
        return fwd, {}
    X = ret_mat[week_cut - L + 1:week_cut + 1].T               # [N,L]
    lbv = np.isfinite(X).all(axis=1)
    valid = np.where(lbv)[0]
    if valid.size < N_TOP + 5:
        return fwd, {}
    Xv = torch.tensor(X[valid], dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xv).numpy()                               # [n_valid,S] raw weekly returns
    cum = np.cumsum(pred, axis=1)
    for j, ci in enumerate(valid):
        sid = cols[ci]
        for _, h in HORIZONS:
            step = HORIZON_STEPS[h]
            if step <= cum.shape[1]:
                fwd[h][sid] = float(cum[j, step - 1])
    return fwd, {}


# ════════════════════════════════════════════════════════════════════════════
# Group A/C — metrics (portfolio + precision + trust) — 與 baseline / PatchTST / iTransformer 逐字相同
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
        "calibration_p10_p90_coverage": calib_cov,  # None for Stockformer (point forecast, non-quantile)
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
    ret_mat, cols, week_date = load_return_matrix(cur, universe, max_stocks=args.max_stocks)
    logger.info(f"Return matrix: {len(cols)} stocks × {ret_mat.shape[0]} calendar-weeks "
                f"(load {time.monotonic()-t0:.1f}s)")

    L, S = args.lookback, args.horizon_weeks
    week_ord = np.array([d.toordinal() if d else -1 for d in week_date])

    def week_cut_for(as_of):
        ao = as_of.toordinal()
        idx = np.where(week_ord <= ao)[0]
        return int(idx[-1]) if len(idx) else None

    refit_every = args.refit_every
    refit_points = list(range(len(panels)))[::refit_every] if refit_every > 0 else [0]
    logger.info(f"Stockformer refit at panel indices {refit_points} (every {refit_every}); "
                f"L={L}wk S={S}wk kernel={args.kernel} d_model={args.d_model} "
                f"heads={args.n_heads} layers={args.n_layers} epochs={args.epochs} seed={args.seed}")

    acc = {h: {"top": [], "univ": [], "ic": [], "diracc": [], "pred": [], "real": []} for _, h in HORIZONS}
    model = None
    n_pred_panels = 0

    for pi, as_of in enumerate(panels):
        wc = week_cut_for(as_of)
        if wc is None or wc < L + S:
            continue
        if pi in refit_points or model is None:
            tr0 = time.monotonic()
            model = train_stockformer(ret_mat, wc, L, S, args.seed, args.epochs, args.lr,
                                      args.d_model, args.n_heads, args.n_layers, args.kernel,
                                      args.batch_size, args.dropout, args.max_train_samples)
            if model is None:
                logger.warning(f"  panel {pi} {as_of}: insufficient training samples, skip refit")
                continue
            logger.info(f"  [refit @ panel {pi} {as_of} week_cut={wc}] trained in {time.monotonic()-tr0:.1f}s")
        if model is None:
            continue
        fwd, _ = predict_forward(model, ret_mat, wc, L, S, cols)
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
    ap = argparse.ArgumentParser(description=f"Multi-Cycle Stockformer Validation {TOOL_VER}")
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--refit-every", type=int, default=12, help="refit every N monthly panels (default annual≈12)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lookback", type=int, default=LOOKBACK_WEEKS)
    ap.add_argument("--horizon-weeks", type=int, default=PRED_WEEKS)
    ap.add_argument("--kernel", type=int, default=DECOMP_KERNEL, help="moving-avg kernel for series decomposition (odd)")
    ap.add_argument("--max-train-samples", type=int, default=40000, help="compute-budget cap on (window×stock) samples per refit")
    ap.add_argument("--max-stocks", type=int, default=None, help="limit universe for smoke")
    ap.add_argument("--max-panels", type=int, default=None, help="limit #panels for smoke")
    ap.add_argument("--smoke", action="store_true", help="tiny config: 40 stocks, 6 panels, 1 refit, 2 epochs")
    args = ap.parse_args()
    if args.smoke:
        args.max_stocks = args.max_stocks or 40
        args.max_panels = args.max_panels or 6
        args.epochs = 2
        args.refit_every = 0
        args.batch_size = 128
        args.d_model = 32
        args.n_heads = 2
        args.n_layers = 1
        args.max_train_samples = 4000

    logger.info("=" * 100)
    logger.info(f"Multi-Cycle Stockformer Validation {TOOL_VER} (stock-specific temporal transformer · series-decomp)")
    logger.info(f"  COMMON COMPARISON BASELINE: universe latest-committed × 95 panels × 4 horizons × top-{N_TOP} × cost {COST_PER_REBAL}")
    logger.info(f"  seed={args.seed} smoke={args.smoke} max_stocks={args.max_stocks}")
    logger.info("=" * 100)

    t_global = time.monotonic()
    results, universe, panels, cols, ret_mat = run(args)

    logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix (Stockformer)\n{'='*100}")
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
            "tool": "multi_cycle_stockformer_validation.py", "tool_ver": TOOL_VER,
            "model": "Stockformer (stock-specific temporal transformer, series-decomposition + full-resolution temporal self-attention, from-scratch torch.nn adaptation)",
            "model_family": "stock-specific",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": args.seed, "horizons": [h for _, h in HORIZONS],
            "n_universe": len(universe), "n_stocks_with_series": len(cols),
            "n_weeks": int(ret_mat.shape[0]), "n_panels_input": len(panels),
            "lookback_weeks": args.lookback, "pred_weeks": args.horizon_weeks,
            "decomp_kernel": args.kernel,
            "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
            "refit_every": args.refit_every, "epochs": args.epochs,
            "max_train_samples": args.max_train_samples, "smoke": args.smoke,
            "variates": "stocks (per-stock temporal sequence; series-decomposed full-resolution temporal attention)",
            "common_baseline": f"universe latest-committed × {len(panels)} panels × {HORIZONS} × top-{N_TOP} × cost {COST_PER_REBAL} × T_CZ-6",
            "source_traceability": "per CLAUDE.md §一.10 — all data from (b) DB query TaiwanStockPriceAdj/core_universe_*; model from-scratch (no external weights)",
        }
        op = Path(_base_dir).parent / args.output
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()

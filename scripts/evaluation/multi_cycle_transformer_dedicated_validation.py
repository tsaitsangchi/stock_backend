"""
multi_cycle_transformer_dedicated_validation.py v0.1 (Multi-Cycle FT-Transformer Validation + Precision/Reliability Analysis · §14.7-CY 第 10 實作 dedicated · 首次非 tree 跨架構 · per Canonical Comparison Framework · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: MULTI-CYCLE 4-HORIZON FT-TRANSFORMER VALIDATION + PRECISION/RELIABILITY + CANONICAL COMPARISON FRAMEWORK + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS + 10-MODEL 第 10 實作 dedicated(首次非 tree)+ §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 + §一.12 5-MIN-REPORTING-AWARE
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, §14.7-CY T_CY-2): 4 horizons(weekly 5d / monthly 20d / quarterly 60d / annual 252d)完全對齊其他 multi-cycle validators。
2. **[FT-Transformer Deep Learning]** (v0.1): d_model=64 / n_heads=4 / n_layers=2 / ffn_dim=128 / dropout=0.3 / seed=5422;**multi-cycle 用 epochs=15**(降自 trainer 之 30 以控制 multi-cycle compute,per §一.10 #3 honest disclosure)。
3. **[Canonical Comparison Framework]** (v0.1, per RF 建立): metrics 與 9-tree CCF validators 完全 standardized,確保 10-model cross-architecture comparison reliable。
4. **[Overlap-Corrected n_effective]** (v0.1, §14.7-CY T_CY-3): n_eff = n × (30/horizon),長 horizon 之 overlap penalty;effective t-stat = t × sqrt(n_eff/n)。
5. **[Honest Annualization]** (v0.1, §14.7-CY T_CY-4): mean × (252/horizon),非 √N 高估。
6. **[Cost-Drag Per Horizon]** (v0.1, §14.7-CY T_CY-5): 0.6%/rebal × rebals_per_year。
7. **[Precision Analysis]** (v0.1, new layer): Directional Hit Rate / Top-20 Actual Overlap / RMSE / MAE。
8. **[Reliability Analysis]** (v0.1, new layer): IC Stability CoV / Significance Robustness。
9. **[System Script Mandatory]** (v0.1, §14.7-CY T_CY-1): system 永久 script。
10. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory reuse。
11. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): significance + precision tier 動態判定。
12. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§14.7-CY 第 10 evaluation 實作 dedicated(首次非 tree)**(9-tree CCF 為前九)。**治權邊界**:(a) §3.1 evaluation;(b) read-only;(c) 不訓練 production model;(d) 不修改 DB;(e) 唯一職責:FT-Transformer 4-horizon walk-forward + precision/reliability + JSON 持久化。
13. **[Historical Reference Authority]** (v0.1): 9-tree CCF multi-cycle 結果為 reference;本 Transformer dedicated v0.1 為 cross-architecture 首次 multi-cycle 驗證。
14. **[Idempotency]** (v0.1): pure read-only;JSON output 含 timestamp。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Horizon-Specific Walk-Forward — `--horizons <days_csv>`
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1-4 weekly/monthly/quarterly/annual | `evaluate_horizon()` 4 calls | §14.7-CY T_CY-2 |

### Group B. Walk-Forward FT-Transformer Training (per horizon)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 FT-Transformer | d_model=64 / heads=4 / layers=2 / epochs=15 / early_stopping=3 / batch=512 | 學界 SOTA |
| B.3 Adam optimizer | lr=1e-3 / weight_decay=1e-4 | standard |
| B.4 Spearman IC | rank correlation | §14.7-CM |

### Group C. Overlap Correction + Honest Annualization
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 n_effective | n × (30/horizon) | §14.7-CY T_CY-3 |
| C.2 Effective t-stat | t × sqrt(n_eff/n) | §14.7-CY T_CY-3 |
| C.3 Annualization | mean × (252/horizon) | §14.7-CY T_CY-4 |
| C.4 Cost-drag | 0.006 × rebals_per_year | §14.7-CY T_CY-5 |

### Group D. Precision Analysis(新層)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Directional Hit Rate | sign(pred) == sign(actual) | precision |
| D.2 Top-20 Actual Overlap | predicted top-20 ∩ actual top-20 / 20 | precision |
| D.3 RMSE / MAE | magnitude error | regression |

### Group E. Reliability Analysis(新層)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 IC Stability CoV | std(IC) / |mean(IC)| | reliability |
| E.2 Significance Robust | abs(eff_t) > 1.997 | §14.7-CY T_CY-3 |

### Group F. JSON Persistence
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| F.1 Cross-cycle matrix stdout | per-horizon row | §14.7-CY T_CY-6 |
| F.2 JSON output | `reports/multi_cycle_transformer_dedicated_<ts>.json` | §一.10 |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/evaluation/multi_cycle_transformer_dedicated_validation.py --dry-run` |

### 不提供之旗標 (Intentionally Omitted)
- `--seed`:固定 5422 per §14.7-CW T_CW-4。
- `--epochs`:固定 15(multi-cycle compute control,trainer 為 30)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CY 第 10 實作 dedicated(FT-Transformer)** under Canonical Comparison Framework — **首次非 tree 跨架構 multi-cycle**。 (1) 4-horizon walk-forward,FT-Transformer;(2) Precision/Reliability 新層延續其他 multi-cycle validators;(3) **compute control:epochs=15**(trainer 為 30 / per §一.10 honest disclosure);(4) §一.12 5-min reporting aware(預計 multi-cycle ≥ 5 min)。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422

TF_PARAMS = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 15,
    "early_stopping_patience": 3,
    "batch_size": 512,
    "val_fraction": 0.15,
    "seed": SEED,
}

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "amihud_illiquidity_60d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_concentration_60d", "barbell_balance_60d", "preferential_attachment_60d",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
    # §14.7-DC v0.3 strict: theme_is_semiconductor + fitness_signal_60d + theme_strength all removed (hardcoded knowledge / transitively tainted = AI hallucination)
]


# === FT-Transformer (same as trainer) ===
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1/math.sqrt(d_model), 1/math.sqrt(d_model))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, ffn_dim, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, 1))

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        return self.head(out[:, 0, :]).squeeze(-1)


def get_panel_dates():
    dates = []
    current = date(2018, 6, 15)
    while current <= date(2026, 4, 30):
        dates.append((f"fs_{current.strftime('%Y%m%d')}_feature_set_v0_4", current))
        if current.month == 12: current = date(current.year+1, 1, 15)
        else: current = date(current.year, current.month+1, 15)
    return dates


def load_features(cur, fs_id, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)", (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    X, sids = [], []
    for sid in universe:
        if sid in feat_data:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43]); sids.append(sid)
    return X, sids


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute("SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')", (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return {}, None
    cur.execute("WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id", (str(as_of), str(label_date)))
    return {sid: float(r) for sid, r in cur.fetchall()}, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def train_transformer_fold(X_train, y_train, params):
    torch.manual_seed(params["seed"]); np.random.seed(params["seed"])
    n = len(X_train); val_n = max(int(n * params["val_fraction"]), 50)
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0) + 1e-6
    X_tr_norm = (X_tr - mu) / sigma; X_val_norm = (X_val - mu) / sigma

    Xt = torch.tensor(X_tr_norm, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val_norm, dtype=torch.float32); yv = torch.tensor(y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=params["batch_size"], shuffle=True)

    model = FTTransformer(n_features=Xt.size(1), d_model=params["d_model"], n_heads=params["n_heads"], n_layers=params["n_layers"], ffn_dim=params["ffn_dim"], dropout=params["dropout"])
    opt = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad_epochs = 0
    for epoch in range(params["epochs"]):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad(); pred = model(Xb); loss = loss_fn(pred, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= params["early_stopping_patience"]: break

    if best_state: model.load_state_dict(best_state)
    return model, mu, sigma


def predict_transformer(model, X, mu, sigma):
    X_norm = (X - mu) / sigma
    Xt = torch.tensor(X_norm, dtype=torch.float32)
    model.eval()
    with torch.no_grad(): return model(Xt).cpu().numpy()


def evaluate_horizon(cur, panels, horizon_days, universe, label):
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d)\n{'='*100}")
    panel_data = {}
    t0 = time.monotonic()
    for fs_id, as_of in panels:
        X, sids = load_features(cur, fs_id, universe)
        if not X: continue
        returns, label_date = load_forward_returns(cur, as_of, horizon_days)
        if not returns: continue
        XX, yy, sids_matched = [], [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                XX.append(X[i]); yy.append(returns[sid]); sids_matched.append(sid)
        if XX: panel_data[as_of] = (XX, yy, sids_matched, label_date)
    logger.info(f"  Loaded {len(panel_data)} panels(load: {time.monotonic()-t0:.1f}s)")

    panel_keys = sorted(panel_data.keys())
    panel_ics, panel_top20_rets, panel_univ_rets = [], [], []
    panel_hit_rates, panel_overlaps, panel_rmses, panel_maes = [], [], [], []

    t_fold_start = time.monotonic()
    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, _, _ = panel_data[test_key]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X, dtype=np.float32); y_tr = winsorize(np.array(train_y, dtype=np.float32)).astype(np.float32)
        if len(X_tr) < 100: continue

        fold_model, mu, sigma = train_transformer_fold(X_tr, y_tr, TF_PARAMS)
        X_te = np.array(X_test, dtype=np.float32)
        pred_te = predict_transformer(fold_model, X_te, mu, sigma)
        ic = spearman_ic(pred_te, y_test)

        n_top = min(20, len(pred_te))
        top_idx = np.argsort(pred_te)[-n_top:]
        actual_top_idx = np.argsort(y_test)[-n_top:]
        top20_ret = float(np.mean([y_test[k] for k in top_idx]))
        univ_ret = float(np.mean(y_test))

        y_arr = np.array(y_test); pred_arr = np.array(pred_te)
        hit_rate = float(np.mean(np.sign(pred_arr) == np.sign(y_arr)))
        overlap = len(set(top_idx.tolist()) & set(actual_top_idx.tolist())) / n_top
        rmse = float(np.sqrt(np.mean((pred_arr - y_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - y_arr)))

        panel_ics.append(ic); panel_top20_rets.append(top20_ret); panel_univ_rets.append(univ_ret)
        panel_hit_rates.append(hit_rate); panel_overlaps.append(overlap); panel_rmses.append(rmse); panel_maes.append(mae)

        if i % 10 == 0:
            elapsed = time.monotonic() - t_fold_start
            logger.info(f"    [{label}] {i}/{len(panel_keys)-1} folds done — elapsed {elapsed:.1f}s — last IC={ic:+.4f}")

    if not panel_top20_rets: return None

    n = len(panel_top20_rets)
    mean_ret = float(np.mean(panel_top20_rets))
    std_ret = float(np.std(panel_top20_rets, ddof=1)) if n > 1 else 0
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
    win_rate = sum(1 for r in panel_top20_rets if r > 0) / n
    alphas = [t - u for t, u in zip(panel_top20_rets, panel_univ_rets)]
    mean_alpha = float(np.mean(alphas))
    std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0
    running = 0; peak = 0; mdd = 0
    for r in panel_top20_rets:
        running += r
        if running > peak: peak = running
        if peak - running > mdd: mdd = peak - running

    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1
    cost_per_rebal = 0.006
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_simple_net = math.exp(annualized_log_gross - annual_cost_drag) - 1

    panel_spacing = 30
    if horizon_days <= panel_spacing:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997

    mean_hit = float(np.mean(panel_hit_rates))
    mean_overlap = float(np.mean(panel_overlaps))
    mean_rmse = float(np.mean(panel_rmses))
    mean_mae = float(np.mean(panel_maes))
    ic_cov = float(np.std(panel_ics, ddof=1) / abs(np.mean(panel_ics))) if np.mean(panel_ics) != 0 else float('inf')

    result = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct, "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)),
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_simple_net": annualized_simple_net,
        "precision_directional_hit_rate": mean_hit,
        "precision_top20_actual_overlap": mean_overlap,
        "precision_rmse": mean_rmse,
        "precision_mae": mean_mae,
        "reliability_ic_stability_cov": ic_cov,
    }

    logger.info(f"\n  Results({label}, {horizon_days}d):")
    logger.info(f"    OOS panels: {n} | n_eff: {n_eff:.1f} | total elapsed: {time.monotonic()-t_fold_start:.1f}s")
    logger.info(f"    Sharpe: {sharpe:+.4f} | Win: {win_rate*100:.1f}% | α: {mean_alpha*100:+.4f}% | IR: {ir:+.4f}")
    logger.info(f"    Eff t-stat: {eff_t_stat:+.3f} | Sig p<0.05: {'✅' if is_significant else '❌'}")
    logger.info(f"    Annualized NET: {annualized_simple_net*100:+.2f}%/yr | Mean IC: {result['mean_ic']:+.4f}")
    logger.info(f"    --- Precision ---")
    logger.info(f"    Directional hit rate: {mean_hit*100:.1f}% | Top-20 actual overlap: {mean_overlap*100:.1f}%")
    logger.info(f"    RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}")
    logger.info(f"    --- Reliability ---")
    logger.info(f"    IC stability(CoV): {ic_cov:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle FT-Transformer Validation {TOOL_VER}(Canonical Comparison Framework)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle FT-Transformer Validation {TOOL_VER}(per §14.7-CY / Canonical Comparison Framework)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  PyTorch version: {torch.__version__} threads={torch.get_num_threads()}")
    logger.info(f"  Mode: {'COMMIT' if args.commit else 'DRY-RUN'}")
    logger.info(f"  epochs(multi-cycle compute control)= {TF_PARAMS['epochs']}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe' AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)")
        universe = list({r[0] for r in cur.fetchall()})
        logger.info(f"  Universe: {len(universe)} stocks")

        panels = get_panel_dates()
        logger.info(f"  Panels: {len(panels)}(2018-06-15 ~ 2026-04-15 monthly)")

        results = {}
        t_global = time.monotonic()
        for label, days in horizon_labels:
            r = evaluate_horizon(cur, panels, days, universe, label)
            if r: results[label] = r
            logger.info(f"\n  [Cumulative elapsed: {time.monotonic()-t_global:.1f}s after {label}]")

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(FT-Transformer dedicated)+ Precision/Reliability\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'NetAnn':>9} {'HitRate':>9} {'Overlap':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['annualized_simple_net']*100:>+8.2f}% {r['precision_directional_hit_rate']*100:>8.1f}% {r['precision_top20_actual_overlap']*100:>8.1f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_transformer_dedicated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: r for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_transformer_dedicated_validation.py", "tool_ver": TOOL_VER,
                "model_family": "transformer", "pytorch_version": torch.__version__,
                "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER, "seed": SEED,
                "horizons": horizon_days_list, "n_universe": len(universe), "n_panels_input": len(panels),
                "source_traceability": "per CLAUDE.md §一.10 — all (b) DB query",
                "compute_control_note": "epochs=15 (trainer 30) per §一.10 honest disclosure",
            }
            with open(output_full, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            logger.info(f"  Results persisted: {output_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

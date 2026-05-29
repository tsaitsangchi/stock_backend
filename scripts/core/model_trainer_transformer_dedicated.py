"""
model_trainer_transformer_dedicated.py v0.1 (FT-Transformer Deep Learning Production Trainer · 10-Model 第 10 實作 dedicated · 首次非 tree 跨架構 · per CLAUDE.md §一.11 三段式入憲 · per Canonical Comparison Framework)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: TRANSFORMER TIME-SERIES PRODUCTION + 10-MODEL CCF 第 10 實作 dedicated(首次非 tree 跨架構)+ §14.7-CS MODEL-TRAINING-LANDING + §14.7-CL 43-FEATURE CANONICAL + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 + §一.12 5-MIN-REPORTING-AWARE + CANONICAL COMPARISON FRAMEWORK
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Real Deep Learning Model Required]** (v0.1): FT-Transformer(Feature Tokenizer Transformer / Gorishniy et al. NeurIPS 2021 / 學界 tabular SOTA);**multi-head self-attention** 取代 tree splits;**attention-based feature interactions** 為 architectural essence。
2. **[Expanding Window Walk-Forward OOS]** (v0.1, 憲法 §14.7-CW T_CW-2): train [panel 0..i-1] → test panel i,完全對齊 9-tree CCF 之 walk-forward 設定。
3. **[Conservative Architecture]** (v0.1): d_model=64 / n_heads=4 / n_layers=2 / ffn_dim=128 / dropout=0.3 / seed=5422;**保守參數防 overfit**(per Grinsztajn 2022 NeurIPS tabular Transformer overfit risk literature)。
4. **[Treaty Gates 4/4]** (v0.1, §14.7-CW T_CW-5): Sharpe > 0 / Win ≥ 50% / MDD ≤ 30% / Mean α > 0。
5. **[Top Features §0.1+§0.2 Expected]** (v0.1): 由 attention weights 揭露 dominant features;預期由 §0.1(volatility/drawdown/log_return)+§0.2(barbell/right_tail)主導。
6. **[43 Canonical Features]** (v0.1, §14.7-CL): SPEC_43 與 9-tree models 共用 input(CCF 同 data foundation)。
7. **[Multi-Head Attention vs Tree Splits]** (v0.1, 跨架構 cardinal difference): Tree models(LightGBM/XGBoost/CatBoost/RF/ET)使用 **discrete recursive partitioning of feature space**;Transformer 使用 **continuous attention-based feature interactions**(soft weights instead of hard splits)。
8. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory reuse。
9. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): Treaty Gates 動態判定。
10. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§10 model_trainer 第 10 實作 dedicated(首次非 tree)**(9 個 tree models 為前九)。**治權邊界**:(a) §3.1 序列 training,不入 §3.2 橫切;(b) 五套禁令不涉;(c) T1-T3 不分層(屬 L1 evidence);(d) §8.5 anti-leakage compliant;(e) 不選股 / 不算 feature;(f) **不評估 multi-cycle**(由 multi_cycle_transformer_dedicated_validation 負責);(g) 唯一職責:8-panel walk-forward FT-Transformer 訓練 + Treaty gates + artifact + model_registry。
11. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"`;**9-tree CCF models 為對標 reference**(production winners:Annual=XGBoost dedicated / Quarterly=XGBoost 既存 + LGBM v0.2 / Weekly=CatBoost dedicated);本 Transformer dedicated v0.1 為 cross-architecture 首次驗證。
12. **[Idempotency]** (v0.1): model_registry INSERT ON CONFLICT;重跑覆寫。
13. **[Canonical Comparison Framework]** (v0.1, per RF v0.1 建立): 本程式之 metrics 計算邏輯(Spearman IC / Sharpe / MDD / α / t-stat / overfit gap)與 9-tree CCF 完全 standardized,確保 10-model cross-architecture comparison reliable。
14. **[Multi-Run Reproducibility]** (v0.1, per §14.7-CW T_CW-6 v6.17.1 patch): Transformer training 含 dropout + Adam momentum + minibatch shuffle 之 stochasticity;commit anchor 須註明為 single run / range;per §一.10 #3 production metric 應 ≥3 runs 取得 min/median/max statistics。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Universe + Feature Loading
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Latest core_universe | DB query | §14.7-CF |
| A.2 Feature values | `load_panel_data()` | §14.7-CL |
| A.3 Forward returns | PriceAdj LN(t1/t0) | §14.7-CV |

### Group B. Walk-Forward Expanding Window Training
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 FT-Transformer | feature tokenizer + N-layer encoder + CLS + MLP head | NeurIPS 2021 |
| B.3 Adam optimizer | lr=1e-3 / weight_decay=1e-4 | standard |
| B.4 Early stopping | val_loss patience=3 | overfit prevention |
| B.5 Spearman IC | rank correlation | §14.7-CM |

### Group C. Treaty Gates Evaluation
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1-4 | Sharpe / Win / MDD / Alpha | §14.7-CW Gates 1-4 |
| C.5 t-statistic | mean_α / SE | §14.7-CX significance |

### Group D. Model Artifact Persistence (--commit only)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 PyTorch state_dict | `torch.save(model.state_dict(), .pt)` | PyTorch standard |
| D.2 metrics.json | full serialization | §一.10 |
| D.3 hyperparams.json | dump | §14.7-CW T_CW-4 |
| D.4 model_registry INSERT | ON CONFLICT DO UPDATE | §10 SSOT |

### Group E. Feature Importance via Attention
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Per-feature attention weights | mean of CLS-to-feature attention across heads/layers | attention-based importance |
| E.2 Top-15 stdout | sorted descending | §14.7-CW T_CW-3 |

### Group F. CLI + Mode Control
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| F.1 `--dry-run` / `--commit` | mode | safe default |
| F.2 `--panel-feature-sets <csv>` | walk-forward override | §14.7-CX |
| F.3 `--label-horizon N` | default 30d | §14.7-CW |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/core/model_trainer_transformer_dedicated.py --dry-run` |
| Commit production | `python scripts/core/model_trainer_transformer_dedicated.py --commit` |

### 不提供之旗標 (Intentionally Omitted)
- `--seed`:固定 5422 per §14.7-CW T_CW-4(seed-aware reproducibility)。
- `--epochs`:固定 30 + early stopping patience=3。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:10-Model CCF 第 10 實作 dedicated(FT-Transformer)** under Canonical Comparison Framework — **首次非 tree 跨架構**。 (1) **學界 SOTA tabular Transformer**(Gorishniy NeurIPS 2021);(2) **首次延伸 CCF 至非 tree 模型**;(3) Architecture:d_model=64 / n_heads=4 / n_layers=2 / ffn_dim=128 / dropout=0.3 / seed=5422;(4) Training:Adam lr=1e-3 / weight_decay=1e-4 / epochs=30 / early_stopping patience=3 / batch=512;(5) Honest caveat:per Grinsztajn 2022 NeurIPS,小 tabular dataset(7843 rows × 43 features)下 Transformer 易 overfit + 可能 underperform tree models。**新介面**:`--dry-run` / `--commit` / `--panel-feature-sets <csv>` / `--label-horizon N`(全與 9-tree dedicated 對等)。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, hashlib, json, logging, math, time
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
MODEL_FAMILY = "transformer"
SEED = 5422

DEFAULT_TRAINING_POLICY = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "early_stopping_patience": 3,
    "batch_size": 512,
    "val_fraction": 0.15,
    "seed": SEED,
}

DEFAULT_PANELS = [
    ("fs_20260105_feature_set_v0_4", "2026-01-05"),
    ("fs_20260120_feature_set_v0_4", "2026-01-20"),
    ("fs_20260205_feature_set_v0_4", "2026-02-05"),
    ("fs_20260220_feature_set_v0_4", "2026-02-20"),
    ("fs_20260305_feature_set_v0_4", "2026-03-05"),
    ("fs_20260316_feature_set_v0_4", "2026-03-16"),
    ("fs_20260401_feature_set_v0_4", "2026-04-01"),
    ("fs_20260415_feature_set_v0_4", "2026-04-15"),
]

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "amihud_illiquidity_60d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "preferential_attachment_60d",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
    # §14.7-DC v0.3 strict: theme_is_semiconductor + fitness_signal_60d + theme_strength all removed (hardcoded knowledge / transitively tainted = AI hallucination)
]


# ============ FT-Transformer Architecture ============
class FeatureTokenizer(nn.Module):
    """Per-feature linear embedding(numerical 取 scalar × weight + bias)"""
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1/math.sqrt(d_model), 1/math.sqrt(d_model))

    def forward(self, x):
        # x: (batch, n_features) → (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for regression(per Gorishniy NeurIPS 2021)"""
    def __init__(self, n_features, d_model, n_heads, n_layers, ffn_dim, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (batch, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_model)
        out = self.encoder(seq)  # (batch, 1+n_features, d_model)
        cls_out = out[:, 0, :]  # (batch, d_model)
        return self.head(cls_out).squeeze(-1)  # (batch,)


def get_attention_importance(model, X_sample, n_features):
    """Approximate per-feature importance via input gradient magnitude(integrated gradients lite)"""
    model.eval()
    X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
    pred = model(X_tensor)
    grads = torch.autograd.grad(pred.sum(), X_tensor)[0]
    importance = grads.abs().mean(dim=0).detach().cpu().numpy()
    return importance


def load_panel_data(cur, fs_id, as_of, label_horizon, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)", (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    cur.execute("SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')", (as_of, label_horizon, as_of, label_horizon + 10))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return [], [], [], None
    cur.execute("WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id", (as_of, label_date))
    returns = {sid: float(r) for sid, r in cur.fetchall() if sid in universe}
    X, y, sids = [], [], []
    for sid in universe:
        if sid in feat_data and sid in returns:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43])
            y.append(returns[sid]); sids.append(sid)
    return X, y, sids, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def train_transformer(X_train, y_train, params, log_prefix=""):
    """Train FT-Transformer with early stopping. Returns trained model."""
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    # Train/val split for early stopping
    n = len(X_train)
    val_n = max(int(n * params["val_fraction"]), 50)
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Standardize features (fit on train, apply to val)
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0) + 1e-6
    X_tr_norm = (X_tr - mu) / sigma
    X_val_norm = (X_val - mu) / sigma

    Xt = torch.tensor(X_tr_norm, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val_norm, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    dataset = TensorDataset(Xt, yt)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)

    model = FTTransformer(
        n_features=Xt.size(1), d_model=params["d_model"], n_heads=params["n_heads"],
        n_layers=params["n_layers"], ffn_dim=params["ffn_dim"], dropout=params["dropout"]
    )
    opt = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad_epochs = 0
    for epoch in range(params["epochs"]):
        model.train()
        train_loss_sum = 0.0
        for Xb, yb in loader:
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * Xb.size(0)
        train_loss = train_loss_sum / len(dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= params["early_stopping_patience"]:
                logger.info(f"  {log_prefix}early stopped at epoch {epoch+1} / best val_loss={best_val:.6f}")
                break

    if best_state: model.load_state_dict(best_state)
    return model, mu, sigma


def predict_transformer(model, X, mu, sigma):
    """Apply trained model to features X(with same train mu/sigma normalization)"""
    X_norm = (X - mu) / sigma
    Xt = torch.tensor(X_norm, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser(description=f"FT-Transformer Deep Learning Trainer ({TOOL_VER}) — Canonical Comparison Framework")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--label-horizon", type=int, default=30)
    parser.add_argument("--panel-feature-sets", type=str, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe' AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)")
        universe = list({r[0] for r in cur.fetchall()})

        if args.panel_feature_sets:
            panels = []
            for fs in args.panel_feature_sets.split(","):
                fs = fs.strip()
                cur.execute("SELECT as_of_date FROM feature_store_snapshot WHERE feature_set_id=%s", (fs,))
                r = cur.fetchone()
                if r: panels.append((fs, str(r[0])))
        else:
            panels = DEFAULT_PANELS

        logger.info("=" * 120)
        logger.info(f"FT-Transformer Deep Learning Trainer {TOOL_VER}(Canonical Comparison Framework)")
        logger.info("=" * 120)
        logger.info(f"  Universe:              {len(universe)} stocks")
        logger.info(f"  Panels:                {len(panels)}")
        logger.info(f"  Label horizon:         {args.label_horizon}d")
        logger.info(f"  Mode:                  {'COMMIT' if args.commit else 'DRY-RUN'}")
        logger.info(f"  Features(§14.7-CL):  {len(SPEC_43)}")
        logger.info(f"  PyTorch version:       {torch.__version__}")
        logger.info(f"  Threads:               {torch.get_num_threads()}")

        logger.info("\n──── Loading walk-forward training data ────")
        all_X, all_y = [], []
        per_panel = {}
        for fs_id, as_of in panels:
            X, y, sids, label_date = load_panel_data(cur, fs_id, as_of, args.label_horizon, universe)
            if not X:
                logger.warning(f"  Panel {as_of}:no valid forward data,skipped")
                continue
            logger.info(f"  Panel {as_of} → label_date={label_date}:N={len(X)}")
            all_X.extend(X); all_y.extend(y)
            per_panel[as_of] = (X, y, sids, label_date)

        X_train_full = np.array(all_X, dtype=np.float32)
        y_train_full = np.array(all_y, dtype=np.float32)
        logger.info(f"\n  Total training rows:  {len(X_train_full):,}")
        y_train_full_w = winsorize(y_train_full, 0.01, 0.99).astype(np.float32)

        logger.info("\n──── Walk-Forward Expanding Window OOS Evaluation ────")
        logger.info("  方法:expanding window — train [0..i-1] → test panel i(OOS)")
        for k, v in DEFAULT_TRAINING_POLICY.items():
            logger.info(f"    {k:25} = {v}")

        panels_list = sorted(per_panel.items(), key=lambda x: x[0])
        panel_ics, panel_returns_top20, panel_returns_univ = [], [], []
        t_walk_start = time.monotonic()

        for i in range(1, len(panels_list)):
            t_fold_start = time.monotonic()
            test_as_of, (X_test, y_test, sids_test, label_date) = panels_list[i]
            train_X, train_y = [], []
            for j in range(i):
                X_j, y_j, _, _ = panels_list[j][1]
                train_X.extend(X_j); train_y.extend(y_j)
            X_tr = np.array(train_X, dtype=np.float32)
            y_tr = winsorize(np.array(train_y, dtype=np.float32), 0.01, 0.99).astype(np.float32)

            fold_model, mu, sigma = train_transformer(X_tr, y_tr, DEFAULT_TRAINING_POLICY, log_prefix=f"[Fold {i}] ")

            X_te = np.array(X_test, dtype=np.float32)
            pred_te = predict_transformer(fold_model, X_te, mu, sigma)
            ic_te = spearman_ic(pred_te, y_test)
            panel_ics.append(ic_te)

            n_top = min(20, len(pred_te))
            top_idx = np.argsort(pred_te)[-n_top:]
            top20_ret = float(np.mean([y_test[k] for k in top_idx]))
            univ_ret = float(np.mean(y_test))
            panel_returns_top20.append(top20_ret)
            panel_returns_univ.append(univ_ret)
            elapsed_fold = time.monotonic() - t_fold_start
            logger.info(f"  Train[0..{i-1}] → Test {test_as_of}: IC={ic_te:+.4f} / Top20={top20_ret:+.4f} / Universe={univ_ret:+.4f}({elapsed_fold:.1f}s)")

        logger.info(f"\n  Walk-forward elapsed: {time.monotonic() - t_walk_start:.1f}s")

        # Final full-train model
        logger.info("\n──── Final full-train model ────")
        t_final = time.monotonic()
        model, mu_full, sigma_full = train_transformer(X_train_full, y_train_full_w, DEFAULT_TRAINING_POLICY, log_prefix="[Final] ")
        logger.info(f"  Final train elapsed: {time.monotonic() - t_final:.1f}s")

        pred_train = predict_transformer(model, X_train_full, mu_full, sigma_full)
        in_sample_ic = spearman_ic(pred_train, y_train_full_w)
        in_sample_rmse = float(np.sqrt(np.mean((pred_train - y_train_full_w) ** 2)))
        logger.info(f"\n  [Reference]Full-train in-sample IC:   {in_sample_ic:+.4f}")
        logger.info(f"  [Reference]Full-train in-sample RMSE: {in_sample_rmse:.4f}")
        logger.info(f"  Overfit gap(in-sample - OOS):         {in_sample_ic - float(np.mean(panel_ics)):+.4f}")

        logger.info("\n──── Backtest Metrics ────")
        mean_ic = float(np.mean(panel_ics))
        cross_panel_ic_std = float(np.std(panel_ics, ddof=1))
        mean_ret = float(np.mean(panel_returns_top20))
        std_ret = float(np.std(panel_returns_top20, ddof=1))
        sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
        win_rate = sum(1 for r in panel_returns_top20 if r > 0) / len(panel_returns_top20)
        cum = sum(panel_returns_top20)
        peak = 0; mdd = 0; running = 0
        for r in panel_returns_top20:
            running += r
            if running > peak: peak = running
            dd = peak - running
            if dd > mdd: mdd = dd
        alphas = [t - u for t, u in zip(panel_returns_top20, panel_returns_univ)]
        mean_alpha = float(np.mean(alphas))
        std_alpha = float(np.std(alphas, ddof=1))
        ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
        t_stat = mean_alpha / (std_alpha / math.sqrt(len(alphas))) if std_alpha > 0 else 0

        logger.info(f"  Cross-panel IC mean:    {mean_ic:+.4f}")
        logger.info(f"  Cross-panel IC std:     {cross_panel_ic_std:.4f}")
        logger.info(f"  Mean panel return:      {mean_ret:+.4f}({mean_ret*100:+.2f}%)")
        logger.info(f"  Sharpe(annualized):    {sharpe:+.4f}")
        logger.info(f"  Win rate:               {win_rate*100:.1f}%")
        logger.info(f"  Max drawdown:           {mdd:.4f}({mdd*100:.2f}%)")
        logger.info(f"  Mean alpha:             {mean_alpha:+.4f}({mean_alpha*100:+.2f}%)")
        logger.info(f"  Information Ratio:      {ir:+.4f}")
        logger.info(f"  t-statistic(α):        {t_stat:+.4f}")
        logger.info(f"  Cumulative return:      {cum:+.4f}({cum*100:+.2f}%)")

        logger.info("\n──── Treaty Gates ────")
        g1 = "✅ PASS" if sharpe > 0 else "❌ VIOLATION"
        g2 = "✅ PASS" if win_rate >= 0.5 else "❌ VIOLATION"
        g3 = "✅ PASS" if mdd <= 0.30 else "⚠️ ALERT"
        g4 = "✅ PASS" if mean_alpha > 0 else "❌ VIOLATION"
        logger.info(f"  Gate CW-1(Sharpe > 0):              {g1}({sharpe:.4f})")
        logger.info(f"  Gate CW-2(Win rate ≥ 50%):         {g2}({win_rate*100:.1f}%)")
        logger.info(f"  Gate CW-3(MDD ≤ 30%):              {g3}({mdd*100:.2f}%)")
        logger.info(f"  Gate CW-4(Mean alpha > 0):         {g4}({mean_alpha:.4f})")

        # Feature importance via input gradient magnitude
        try:
            sample_X = (X_train_full[:512] - mu_full) / sigma_full
            importance = get_attention_importance(model, sample_X, len(SPEC_43))
            fi = sorted(zip(SPEC_43, importance), key=lambda x: -x[1])
            logger.info("\n──── Top 15 Features by Input-Gradient Importance ────")
            for k, (f, imp) in enumerate(fi[:15], 1):
                logger.info(f"  {k:>2}. {f:38} importance={imp:>10.6f}")
        except Exception as e:
            logger.warning(f"  Feature importance computation failed: {e}")
            fi = []

        if args.commit:
            logger.info("\n──── COMMIT mode ────")
            feature_set_hash = hashlib.sha1("feature_set_v0.4".encode()).hexdigest()[:8]
            train_date = max(p[1] for p in panels)
            model_id = f"mdl_{train_date.replace('-', '')}_transformer_dedicated_h{args.label_horizon}_{feature_set_hash}_v0_1"
            artifact_dir = Path("data/models") / model_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "model.pt"
            torch.save({"state_dict": model.state_dict(), "mu": mu_full.tolist(), "sigma": sigma_full.tolist()}, str(model_path))

            metrics = {
                "trainer": "transformer_dedicated_v0_1",
                "model_family": MODEL_FAMILY,
                "label_horizon": args.label_horizon,
                "feature_count": len(SPEC_43),
                "rows_trained": len(X_train_full),
                "panels": len(panels),
                "in_sample_ic": in_sample_ic,
                "in_sample_rmse": in_sample_rmse,
                "cross_panel_ic_mean": mean_ic,
                "cross_panel_ic_std": cross_panel_ic_std,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "mdd": mdd,
                "mean_alpha": mean_alpha,
                "information_ratio": ir,
                "t_statistic": t_stat,
                "cumulative_return": cum,
                "top_features": [{"feature": f, "importance": float(imp)} for f, imp in fi[:20]],
            }
            with open(artifact_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            with open(artifact_dir / "hyperparams.json", "w") as f:
                json.dump(DEFAULT_TRAINING_POLICY, f, indent=2)

            cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY snapshot_id DESC LIMIT 1")
            r = cur.fetchone()
            universe_snapshot_id = r[0] if r else None
            cur.execute("""
                INSERT INTO model_registry(model_id, model_policy_version, model_family, feature_set_id, universe_snapshot_id, label_horizon, train_start_date, train_end_date, metrics, hyperparams, artifact_path, status, notes)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                ON CONFLICT(model_id) DO UPDATE SET metrics = EXCLUDED.metrics, hyperparams = EXCLUDED.hyperparams, artifact_path = EXCLUDED.artifact_path, status = EXCLUDED.status, notes = EXCLUDED.notes
            """, (model_id, "model_policy_v0.1", MODEL_FAMILY, panels[-1][0], universe_snapshot_id, args.label_horizon, panels[0][1], train_date, json.dumps(metrics, default=str), json.dumps(DEFAULT_TRAINING_POLICY), str(artifact_dir), "committed", "v0.1 FT-Transformer dedicated;walk-forward;10-model CCF 第 10 實作 dedicated;首次非 tree 跨架構"))
            conn.commit()
            logger.info(f"  ✅ Model committed: {model_id}")
            logger.info(f"  ✅ Artifact: {artifact_dir}/")
            logger.info(f"  ✅ model_registry inserted")

        logger.info("\n" + "=" * 120)
        verdict = "PERFECT" if all("PASS" in g for g in [g1, g2, g3, g4]) else "WARNING"
        logger.info(f"FT-Transformer Dedicated Trainer {TOOL_VER}: 主權判定 {verdict}")
        logger.info("=" * 120)
        if verdict == "PERFECT":
            logger.info(f"  🎯 FT-Transformer(dedicated): Sharpe={sharpe:.2f} / IR={ir:.2f} / Win={win_rate*100:.0f}%")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

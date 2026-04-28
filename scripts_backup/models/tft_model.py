"""
models/tft_model.py — Temporal Fusion Transformer 骨幹模型
使用 pytorch-forecasting 的官方 TFT 實作。

設計原則：
  - 輸入：過去 LOOKBACK 天的時間序列 + 靜態特徵 + 已知未來事件
  - 輸出：未來 HORIZON 天的多分位預測（0.1, 0.25, 0.5, 0.75, 0.9）
  - 可解釋性：attention 權重 + variable importance（SHAP 替代）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TFTPredictor", "HAS_TFT"]

# ── 選擇性匯入（自動對齊 pytorch-forecasting 使用的 Lightning 版本）──
#
# 問題根源：pytorch-forecasting 的 TFT 繼承自某個 LightningModule，
# Trainer 必須來自「完全相同的套件」，否則 isinstance 檢查失敗。
# 偵測策略：載入 TFT 後，讀取其 MRO 找出實際的 LightningModule 來源。
#
try:
    import torch
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    from pytorch_forecasting.metrics import QuantileLoss

    # ── 偵測 TFT 實際繼承哪個 LightningModule ─────────────────
    import inspect as _inspect
    _lightning_pkg = None
    for _base in TemporalFusionTransformer.__mro__:
        _mod = _inspect.getmodule(_base)
        _mod_name = getattr(_mod, "__name__", "") or ""
        if "lightning" in _mod_name and "module" in _mod_name.lower():
            _lightning_pkg = _mod_name.split(".")[0]   # "lightning" 或 "pytorch_lightning"
            break

    if _lightning_pkg == "lightning":
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import (
            EarlyStopping, ModelCheckpoint, LearningRateMonitor
        )
        logger.info("TFT Lightning backend: lightning.pytorch")
    else:
        # fallback → pytorch_lightning（最常見的相容情況）
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            EarlyStopping, ModelCheckpoint, LearningRateMonitor
        )
        logger.info("TFT Lightning backend: pytorch_lightning")

    HAS_TFT = True

except BaseException as _tft_import_err:  # catch C-extension failures too
    HAS_TFT = False
    logger.warning(
        f"TFT 不可用：{_tft_import_err}\n"
        "  若無 GPU：pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  套件對齊：pip install 'pytorch-forecasting>=1.1' 'lightning>=2.1,<2.5'"
    )


# ─────────────────────────────────────────────
# 特徵分組（對應 TFT 的不同輸入類型）
# ─────────────────────────────────────────────

# 靜態特徵（不隨時間變動）
STATIC_CATEGORICALS = ["stock_id"]
STATIC_REALS: list[str] = []

# 時間上已知的未來特徵（例如：已公告的除息日程）
TIME_VARYING_KNOWN_CATEGORICALS: list[str] = []
TIME_VARYING_KNOWN_REALS = [
    "days_to_next_ex_dividend",   # 已知的未來除息日距離
    "dividend_ex_dummy",
    "days_since_last_earnings",
]

# 時間上未知的特徵（需要預測的 covariates，模型只在 encoder 端看到）
TIME_VARYING_UNKNOWN_REALS = [
    # 技術動能
    "log_return_1d", "log_return_5d", "log_return_10d", "log_return_20d",
    "realized_vol_10d", "realized_vol_20d",
    "ma_cross_20_50", "ma_cross_50_120",
    "price_to_ma20", "price_to_ma50",
    "rsi_14", "macd", "macd_hist", "bb_pct", "atr_14",
    "volume_ratio_20", "price_volume_corr_20",
    "high_low_spread", "open_close_spread",
    # 資金流
    "foreign_net_vol_ratio", "trust_net_vol_ratio",
    "foreign_holding_chg_5d", "foreign_net_ma5",
    "margin_balance_chg", "short_balance_chg", "margin_short_ratio",
    "retail_vs_inst",
    # 基本面
    "revenue_yoy", "revenue_mom",
    "gross_margin", "gross_margin_chg_qoq",
    "eps_ttm", "eps_qoq",
    "roe_ttm", "debt_ratio",
    # 估值
    "per", "pbr", "dividend_yield",
    "per_pct_rank_252", "per_deviation_from_ma",
    # 宏觀
    "fed_rate", "fed_rate_chg_30d",
    "usd_twd_chg_10d",
    "taiex_ret_5d", "taiex_ret_20d",
    "taiex_rel_strength",
    # 滾動統計
    "skew_20d", "skew_60d",
    "sharpe_20d", "sharpe_60d",
]

TARGET_COL = "target_30d"


class TFTPredictor:
    """
    TFT 訓練 / 推論封裝。

    使用方式：
        predictor = TFTPredictor(params)
        predictor.fit(train_df, val_df)
        result = predictor.predict(test_df)
        predictor.save("checkpoints/tft_best.ckpt")
    """

    def __init__(self, params: dict):
        if not HAS_TFT:
            raise RuntimeError("pytorch-forecasting 未安裝，無法使用 TFT")
        self.params = params
        self.model: Optional[TemporalFusionTransformer] = None
        self.dataset: Optional[TimeSeriesDataSet] = None
        # 保存訓練集尾端，供 val/predict 做 encoder context 補齊
        self._context_df: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────
    # 資料集準備
    # ─────────────────────────────────────────

    def _prepare_dataset(self, df: pd.DataFrame,
                          is_train: bool = True) -> TimeSeriesDataSet:
        """
        將 feature_df 轉為 pytorch-forecasting 所需的 TimeSeriesDataSet。
        非訓練模式（val/predict）時，自動在前方補上 encoder context，
        確保行數 >= max_encoder_length + max_prediction_length。
        """
        max_enc  = self.params["max_encoder_length"]
        max_pred = self.params["max_prediction_length"]
        min_rows = max_enc + max_pred

        df = df.copy()

        # ── Val / Predict 模式：補 encoder context ──────────────
        if not is_train and self._context_df is not None:
            if len(df) < min_rows:
                need = min_rows - len(df)
                ctx  = self._context_df.iloc[-need:].copy()
                # 清除 context 中的目標欄，避免洩漏（設為 0）
                if TARGET_COL in ctx.columns:
                    ctx[TARGET_COL] = 0.0
                df = pd.concat([ctx, df], ignore_index=True)
                logger.debug(f"  [TFT] 補充 {need} 行 encoder context，合計 {len(df)} 行")

        # ── Predict 模式效能優化：截取最後 min_rows 行 ────────────
        # 原理：TFT 的 DataSet/DataLoader 建立時間與列數呈線性關係。
        #       預測只需要最後 max_encoder_length + max_prediction_length 行
        #       （encoder context + decoder placeholder），多餘的歷史資料
        #       完全不影響輸出，但會讓 DataSet 建立慢 25~30 倍。
        # 效果：~7000 rows → 282 rows，DataSet 建立從 ~5s 降至 ~0.2s/call。
        if not is_train and len(df) > min_rows:
            df = df.iloc[-min_rows:].copy()
            logger.debug(f"  [TFT] 截取最後 {min_rows} 行（predict mode 效能優化）")

        # TFT 需要整數時間索引
        df["time_idx"] = np.arange(len(df))
        df["stock_id"] = "2330"    # 靜態類別特徵

        # 過濾 encoder/decoder 所需特徵
        all_reals = (
            TIME_VARYING_KNOWN_REALS
            + TIME_VARYING_UNKNOWN_REALS
            + [TARGET_COL]
        )
        available_reals = [c for c in all_reals if c in df.columns]

        # 以 0 填補缺失特徵（不中斷訓練）
        for c in all_reals:
            if c not in df.columns:
                df[c] = 0.0
                logger.debug(f"  [TFT] 特徵 {c} 不存在，填 0")

        # 填補 NaN
        df[all_reals] = df[all_reals].fillna(0)

        max_enc = self.params["max_encoder_length"]
        max_pred = self.params["max_prediction_length"]

        if is_train:
            self.dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target=TARGET_COL,
                group_ids=["stock_id"],
                max_encoder_length=max_enc,
                max_prediction_length=max_pred,
                static_categoricals=STATIC_CATEGORICALS,
                static_reals=STATIC_REALS,
                time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
                time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
                time_varying_unknown_reals=[
                    c for c in TIME_VARYING_UNKNOWN_REALS + [TARGET_COL]
                    if c in df.columns
                ],
                target_normalizer="auto",
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            return self.dataset
        else:
            # Validation / prediction dataset（從 train dataset 繼承 normalizer）
            if self.dataset is not None:
                return TimeSeriesDataSet.from_dataset(self.dataset, df, predict=True)
            elif self.model is not None and hasattr(self.model, "dataset_parameters"):
                return TimeSeriesDataSet.from_parameters(self.model.dataset_parameters, df, predict=True)
            else:
                raise RuntimeError("無套用預測的 dataset 或 model.dataset_parameters")

    # ─────────────────────────────────────────
    # 訓練
    # ─────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            checkpoint_dir: str = "checkpoints"):
        """Walk-Forward 的一個 fold 訓練入口。"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 儲存訓練集尾端作為 encoder context（val/predict 補齊用）
        max_enc = self.params["max_encoder_length"]
        self._context_df = train_df.iloc[-max_enc:].copy()

        train_ds = self._prepare_dataset(train_df, is_train=True)
        val_ds   = self._prepare_dataset(val_df, is_train=False)

        batch_size = self.params["batch_size"]

        # ── CPU vs GPU DataLoader 策略 ───────────────────────────
        # CPU 模式：num_workers MUST be 0
        #   原因：num_workers > 0 會 fork N 個子進程，每個各持一份完整資料副本。
        #   12 核 × 8 workers → RAM 暴增 8 倍 → OOM killer 強制結束進程。
        #   CPU 訓練本身已是 single-threaded，worker 增加只有 overhead 無收益。
        # GPU 模式：num_workers > 0 才有意義（非同步預取減少 GPU 等待）
        _is_gpu = torch.cuda.is_available()
        _num_workers = min(4, max(1, mp.cpu_count() - 1)) if _is_gpu else 0
        _persist = _is_gpu and _num_workers > 0
        logger.info(
            f"  TFT DataLoader: num_workers={_num_workers}, "
            f"persistent={_persist}  "
            f"({'GPU' if _is_gpu else 'CPU 模式，workers=0 避免 OOM'})"
        )
        train_loader = train_ds.to_dataloader(
            train=True, batch_size=batch_size,
            num_workers=_num_workers,
            persistent_workers=_persist,
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=batch_size * 2,
            num_workers=_num_workers,
            persistent_workers=_persist,
        )

        # ── CPU 快速模式：縮減模型規模以讓訓練在合理時間完成 ───────
        # GPU hidden_size=128 / 2 LSTM / patience=15 需數天（CPU ~100s/batch）
        # CPU 快速模式：hidden_size=32 / 1 LSTM / patience=5 → ~15min/fold
        _run_params = self.params.copy()
        if not torch.cuda.is_available():
            _run_params["hidden_size"]        = min(_run_params["hidden_size"], 32)
            _run_params["lstm_layers"]        = 1
            _run_params["patience"]           = min(_run_params["patience"], 5)
            _run_params["max_epochs"]         = min(_run_params["max_epochs"], 30)
            _run_params["attention_head_size"] = 1
            logger.info(
                f"  [CPU 模式] TFT 縮減："
                f"hidden={_run_params['hidden_size']}, "
                f"lstm={_run_params['lstm_layers']}, "
                f"patience={_run_params['patience']}, "
                f"epochs={_run_params['max_epochs']}"
            )

        # 建構 TFT 模型
        self.model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=_run_params["learning_rate"],
            hidden_size=_run_params["hidden_size"],
            attention_head_size=_run_params["attention_head_size"],
            dropout=_run_params["dropout"],
            hidden_continuous_size=_run_params["hidden_size"] // 2,
            output_size=len(_run_params["quantiles"]),
            loss=QuantileLoss(_run_params["quantiles"]),
            lstm_layers=_run_params["lstm_layers"],
            reduce_on_plateau_patience=3,
            log_interval=-1,
        )

        logger.info(
            f"  TFT 參數量：{sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=_run_params["patience"],
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
            LearningRateMonitor("epoch"),
        ]

        # 訓練器：自動偵測 GPU；若 CUDA 不可用則退回 CPU
        _accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        _devices     = 1
        logger.info(f"  TFT 訓練裝置：{_accelerator.upper()}")
        # log_every_n_steps must be <= number of training batches per epoch;
        # with small datasets this can be <50 (Lightning default), so we clamp it.
        _n_batches = max(1, len(train_loader))
        _log_steps = min(10, _n_batches)
        trainer = pl.Trainer(
            max_epochs=_run_params["max_epochs"],
            accelerator=_accelerator,
            devices=_devices,
            gradient_clip_val=_run_params["gradient_clip_val"],
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=True,
            log_every_n_steps=_log_steps,
        )
        # Suppress sklearn's 'X does not have valid feature names' warning that
        # originates inside pytorch_forecasting's EncoderNormalizer (third-party
        # code fits StandardScaler on pandas Series then transforms numpy arrays).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            trainer.fit(self.model, train_loader, val_loader)

        # 載入最佳模型
        best_path = trainer.checkpoint_callback.best_model_path
        self.model = TemporalFusionTransformer.load_from_checkpoint(best_path)
        logger.info(f"  最佳 checkpoint：{best_path}")

        return self

    # ─────────────────────────────────────────
    # 推論
    # ─────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        回傳：{
            'point': float,            # 中位數預測（q=0.5）
            'quantiles': np.ndarray,   # [q10, q25, q50, q75, q90]
            'prob_up': float,          # 上漲機率（q50>0 的模糊估計）
            'attention': dict,         # variable importance
        }
        df 應至少包含 max_encoder_length + max_prediction_length 行；
        若不足則由 _prepare_dataset 自動補 context。
        """
        if self.model is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")

        pred_ds = self._prepare_dataset(df, is_train=False)
        loader  = pred_ds.to_dataloader(
            train=False, batch_size=256, num_workers=0
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            raw_pred = self.model.predict(
                loader,
                mode="quantiles",
                return_x=False,
            )
        # raw_pred shape: [N, horizon, n_quantiles]
        # ── 取最後一個 sample 的完整 horizon 軌跡 ──────────────
        daily_quantiles_raw = raw_pred[-1, :, :]  # shape: [horizon, n_quantiles]
        daily_quantiles = (
            daily_quantiles_raw.numpy()
            if hasattr(daily_quantiles_raw, "numpy")
            else np.array(daily_quantiles_raw)
        )

        last_q = daily_quantiles[-1, :]           # horizon 末端作為總結分位數
        quantiles = last_q
        point = quantiles[2]           # q=0.5 的點預測

        # CDF 線性插值平滑上漲機率 (Smooth Probabilities)
        q_probs = self.params.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
        p_down = None
        if 0.0 <= quantiles[0]:
            p_down = q_probs[0] / 2.0
        elif 0.0 >= quantiles[-1]:
            p_down = 1.0 - (1.0 - q_probs[-1]) / 2.0
        else:
            for i in range(len(quantiles) - 1):
                if quantiles[i] <= 0.0 <= quantiles[i+1]:
                    spread = quantiles[i+1] - quantiles[i]
                    if spread < 1e-6:
                        p_down = q_probs[i]
                    else:
                        weight = (0.0 - quantiles[i]) / spread
                        p_down = q_probs[i] + weight * (q_probs[i+1] - q_probs[i])
                    break

        if p_down is None:
            p_down = float((quantiles <= 0).mean())

        raw_prob_up = 1.0 - float(p_down)
        prob_up = float(np.clip(raw_prob_up, 0.15, 0.85))
        if raw_prob_up != prob_up:
            logger.debug(f"  [TFT] prob_up clipped: {raw_prob_up:.4f} -> {prob_up:.4f}")


        # Variable importance（attention 權重）
        try:
            interp = self.model.interpret_output(
                self.model.predict(loader, mode="raw", return_x=True)[1]
            )
            attention = {
                "encoder_importance": interp["encoder_variables"].numpy().tolist(),
                "decoder_importance": interp["decoder_variables"].numpy().tolist(),
            }
        except Exception:
            attention = {}

        return {
            "point":          float(point),
            "quantiles":      quantiles.tolist(),
            "daily_quantiles": daily_quantiles.tolist(),  # [horizon, n_quantiles] 每日完整軌跡
            "prob_up":        prob_up,
            "prob_up_raw":    raw_prob_up,
            "attention":      attention,
        }

    def predict_batch(
        self,
        contexts: dict[str, pd.DataFrame],  # {label -> df_context}
    ) -> dict[str, dict]:
        """
        批量推論：一次 forward pass 處理多個預測點。

        原理：
          pytorch-forecasting 的 TimeSeriesDataSet 支援多 group_id，
          每個 group 代表一個獨立的時間序列。
          將每個預測日期視為一個獨立 group，
          把所有 context windows 拼成一個大 DataFrame → 一次 predict()
          → 比逐筆呼叫快 10x~50x（省去 DataLoader 重建開銷）。

        Args:
            contexts: {label -> df_feat_up_to_predict_date}
                label 可以是日期字串、整數等唯一識別子

        Returns:
            {label -> predict() 格式的 dict}
        """
        if self.model is None:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()")
        if not contexts:
            return {}

        max_enc  = self.params["max_encoder_length"]
        max_pred = self.params["max_prediction_length"]
        min_rows = max_enc + max_pred
        q_probs  = self.params.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])

        all_reals = (
            TIME_VARYING_KNOWN_REALS
            + TIME_VARYING_UNKNOWN_REALS
            + [TARGET_COL]
        )

        # ── Step 1：整備每個 context，拼接為大 DataFrame ────────
        big_chunks = []
        labels_ordered = []  # 保持順序，與 TFT 輸出對齊

        for label, df_ctx in contexts.items():
            df_ctx = df_ctx.copy()
            # 補 encoder context（若不足 min_rows）
            if len(df_ctx) < min_rows and self._context_df is not None:
                need = min_rows - len(df_ctx)
                ctx  = self._context_df.iloc[-need:].copy()
                if TARGET_COL in ctx.columns:
                    ctx[TARGET_COL] = 0.0
                df_ctx = pd.concat([ctx, df_ctx], ignore_index=True)
            elif len(df_ctx) < min_rows:
                logger.warning(f"  [{label}] context 太短（{len(df_ctx)} < {min_rows}），跳過")
                continue

            # 只取最後 min_rows 行（確保不同 group 長度一致，也避免爆記憶體）
            df_ctx = df_ctx.iloc[-min_rows:].copy()

            # 補缺失特徵
            for c in all_reals:
                if c not in df_ctx.columns:
                    df_ctx[c] = 0.0
            df_ctx[all_reals] = df_ctx[all_reals].fillna(0)

            df_ctx["time_idx"] = np.arange(len(df_ctx))
            df_ctx["stock_id"] = str(label)    # 用 label 作為 group_id
            big_chunks.append(df_ctx)
            labels_ordered.append(str(label))

        if not big_chunks:
            return {}

        # 拼接時確保全局 time_idx 唯一：每個 group 在前一個 group 結束後接續
        # pytorch-forecasting 要求每個 (group_id, time_idx) 對唯一
        offset = 0
        for chunk in big_chunks:
            chunk["time_idx"] = np.arange(offset, offset + len(chunk))
            offset += len(chunk)

        big_df = pd.concat(big_chunks, ignore_index=True)

        # ── Step 2：建立批次 Dataset ─────────────────────────────
        try:
            if self.model is not None and hasattr(self.model, "dataset_parameters"):
                pred_ds = TimeSeriesDataSet.from_parameters(
                    self.model.dataset_parameters, big_df, predict=True
                )
            elif self.dataset is not None:
                pred_ds = TimeSeriesDataSet.from_dataset(
                    self.dataset, big_df, predict=True
                )
            else:
                raise RuntimeError("無法取得 dataset_parameters")
        except Exception as e:
            logger.error(f"批次 Dataset 建立失敗：{e}，回退逐筆推論")
            return {}

        loader = pred_ds.to_dataloader(
            train=False, batch_size=min(64, len(labels_ordered)), num_workers=0
        )

        # ── Step 3：批次 predict ─────────────────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names",
                                    category=UserWarning)
            raw_pred = self.model.predict(
                loader, mode="quantiles", return_x=False
            )
        # raw_pred shape: [N_groups, horizon, n_quantiles]

        # ── Step 4：拆解結果，逐 group 組裝 ─────────────────────
        results = {}
        n = raw_pred.shape[0]
        if n != len(labels_ordered):
            logger.warning(
                f"輸出筆數 {n} ≠ 輸入筆數 {len(labels_ordered)}，嘗試對齊"
            )

        for i, label in enumerate(labels_ordered):
            if i >= n:
                break
            daily_q_raw   = raw_pred[i, :, :]   # [horizon, n_quantiles]
            daily_quantiles = (
                daily_q_raw.numpy()
                if hasattr(daily_q_raw, "numpy")
                else np.array(daily_q_raw)
            )

            last_q     = daily_quantiles[-1, :]   # 第 30 天分位數
            quantiles  = last_q.tolist()
            point      = float(last_q[2])         # q50

            # 計算 prob_up（同 predict() 邏輯）
            q_arr  = np.array(quantiles)
            p_down = None
            if 0.0 <= q_arr[0]:
                p_down = q_probs[0] / 2.0
            elif 0.0 >= q_arr[-1]:
                p_down = 1.0 - (1.0 - q_probs[-1]) / 2.0
            else:
                for j in range(len(q_arr) - 1):
                    if q_arr[j] <= 0.0 <= q_arr[j+1]:
                        spread = q_arr[j+1] - q_arr[j]
                        if spread < 1e-6:
                            p_down = q_probs[j]
                        else:
                            weight = (0.0 - q_arr[j]) / spread
                            p_down = q_probs[j] + weight * (q_probs[j+1] - q_probs[j])
                        break
            if p_down is None:
                p_down = float((q_arr <= 0).mean())

            raw_prob_up = 1.0 - float(p_down)
            prob_up = float(np.clip(raw_prob_up, 0.15, 0.85))

            results[label] = {
                "point":           point,
                "quantiles":       quantiles,
                "daily_quantiles": daily_quantiles.tolist(),
                "prob_up":         prob_up,
                "prob_up_raw":     raw_prob_up,
                "attention":       {},
            }

        logger.info(f"  [TFT batch] 完成 {len(results)}/{len(labels_ordered)} 筆推論")
        return results

    # ─────────────────────────────────────────
    # 儲存 / 載入
    # ─────────────────────────────────────────

    def save(self, path: str):
        """相容各 lightning 版本的穩健儲存方法。"""
        if not self.model:
            logger.warning("TFT 模型尚未訓練，無法儲存")
            return
        try:
            # 方法一：官方 save_checkpoint（部分 lightning 版本可用）
            self.model.save_checkpoint(path)
            logger.info(f"TFT 已儲存（save_checkpoint）：{path}")
        except AttributeError:
            # 方法二：lightning Trainer.save_checkpoint（load_from_checkpoint 後
            # save_checkpoint 有時不會正確綁定，改用 torch.save state_dict）
            try:
                import torch
                save_data = {
                    "state_dict":   self.model.state_dict(),
                    "hyper_parameters": getattr(self.model, "hparams", {}),
                }
                torch.save(save_data, path)
                logger.warning(
                    f"TFT 以 state_dict 備份儲存：{path}\n"
                    "  （save_checkpoint 不可用，load 時請用 TFTPredictor.load_state_dict）"
                )
            except Exception as e2:
                logger.error(f"TFT 儲存失敗（兩種方法均失敗）：{e2}")
        except Exception as e:
            logger.error(f"TFT 儲存失敗：{e}")

    @classmethod
    def load(cls, path: str, params: dict) -> "TFTPredictor":
        """從 checkpoint 載入（相容 save_checkpoint 與 state_dict 兩種格式）。"""
        try:
            import torch
            import numpy as np
            from lightning.fabric.utilities.data import AttributeDict
            from pytorch_forecasting.data.encoders import EncoderNormalizer
            torch.serialization.add_safe_globals([AttributeDict, EncoderNormalizer, np.core.multiarray.scalar, np.dtype])
        except Exception:
            pass

        predictor = cls(params)
        try:
            # 優先嘗試官方 load_from_checkpoint
            predictor.model = TemporalFusionTransformer.load_from_checkpoint(path)
        except Exception as e:
            # 備用：state_dict 格式（由 save() 備用路徑儲存）
            logger.warning(f"load_from_checkpoint 失敗（{e}），嘗試 state_dict 載入")
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "state_dict" in data:
                # 需要先建立一個空模型再載入權重（須有 dataset，推論前必須先 fit 一次）
                raise RuntimeError(
                    "state_dict 格式需要先呼叫 fit() 建立模型結構，"
                    "才能用 load_state_dict() 載入權重。\n"
                    f"  原始錯誤：{e}"
                )
            raise
        return predictor


# ── 安全防護：確保 TFTPredictor 永遠可被匯入 ─────────────────────────────────
# 若上方類別定義因任何原因未執行（例如檔案截斷、Python 版本問題），
# 此 stub 確保 `from models.tft_model import TFTPredictor` 不拋出 ImportError。
if "TFTPredictor" not in dir():
    class TFTPredictor:  # type: ignore[no-redef]
        """Stub：TFT 依賴套件不可用時的佔位類別。"""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "TFT 不可用：pytorch-forecasting / lightning 安裝失敗。\n"
                "請執行：pip install 'pytorch-forecasting>=1.1' 'lightning>=2.1,<2.5'"
            )
        def fit(self, *a, **kw):     raise RuntimeError("TFT 不可用")
        def predict(self, *a, **kw): raise RuntimeError("TFT 不可用")
        def save(self, *a, **kw):    raise RuntimeError("TFT 不可用")
        @classmethod
        def load(cls, *a, **kw):     raise RuntimeError("TFT 不可用")

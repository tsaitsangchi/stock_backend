# System Architecture Optimization Plan

Based on the `系統架構全面審查報告.md`, we are implementing critical fixes to address fundamental data issues, model drift monitoring, and training leaks.

## User Review Required

> [!IMPORTANT]
> The retraining logic in `parallel_train.py` will now be integrated with `model_health_check.py`. This means a model will only be considered "deployable" if it passes the PSI and drift checks.

## Proposed Changes

### [Core] Data Pipeline & Features
#### [MODIFY] [feature_engineering.py](file:///home/hugo/project/stock_backend/scripts/feature_engineering.py)
- Added `add_staleness_features` to explicitly measure the age of fundamental data (revenue/financial statements) to prevent look-ahead bias and handle `ffill` staleness.
- Fixed `price_volume_corr_20` calculation stability.

#### [MODIFY] [config.py](file:///home/hugo/project/stock_backend/scripts/config.py)
- Enabled `physics_signals` (gravity, entropy) in `FEATURE_GROUPS` to improve model capture of momentum physics.
- Deduplicated redundant feature selections.

---

### [Model] Training & Calibration
#### [MODIFY] [train_evaluate.py](file:///home/hugo/project/stock_backend/scripts/train_evaluate.py)
- **Stock-specific OOF Saving**: Changed OOF filenames to include `stock_id` to prevent overwriting during parallel training.
- **Reference Distribution Export**: Exporting OOF distributions as `.npy` files for high-speed PSI calculation.
- **[TODO] Calibration Refinement**: Implement `TimeSeriesSplit` for the Isotonic Regression phase to prevent overfitting to historical environments.

#### [MODIFY] [ensemble_model.py](file:///home/hugo/project/stock_backend/scripts/models/ensemble_model.py)
- **[TODO] Calibrator Update**: Refactor `XGBPredictor.calibrate` to use `CalibratedClassifierCV` style splitting if necessary, or implement a rolling window for calibration.

---

### [Monitoring] Model Health Check
#### [MODIFY] [model_health_check.py](file:///home/hugo/project/stock_backend/scripts/model_health_check.py)
- **PSI Source Fix**: Aligned reference distribution loading with the new stock-specific OOF paths.
- **Improved Priority**: Now uses `.npy` -> `CSV` -> `Historical Live` -> `Beta Prior` as fallback chain.

---

### [Automation] Parallel Training
#### [MODIFY] [parallel_train.py](file:///home/hugo/project/stock_backend/scripts/parallel_train.py)
- Fixed relative pathing issues for log and output directories.
- **[TODO] Health Check Integration**: Add a post-training step that runs `model_health_check.py` on the newly trained OOF and only updates the "Production" model if PSI < 0.25.

## Verification Plan

### Automated Tests
1. Run `scripts/parallel_train.py` for a small subset of stocks (e.g., 2330, 2317).
2. Verify that `scripts/outputs/oof_predictions_with_dates_{stock_id}.csv` and `scripts/models/oof_ref_dist_{stock_id}.npy` are created.
3. Run `scripts/model_health_check.py` and confirm it picks up the "oof" source for PSI.

### Manual Verification
- Check logs for "採用 OOF 參考分佈 (.npy)" message.
- Verify that fundamental features in `FeatureStore` now include `rev_staleness` and `fin_staleness`.

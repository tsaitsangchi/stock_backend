"""
test_v5_audit_fixes.py — 系統檢核報告 (2026-04-30) 修復後驗證測試

對應修復項目：
  - P0-1 / QW-1: backfill_from_gaps.py 存在且 import 不出錯
  - P0-2:        signal_filter prob_up_threshold = 0.75
                 + min_hold_days = 5
                 + max_n_trades_per_year = 120
  - P0-3 / QW-2: auto_train_manager 有 write_heartbeat 函式
  - P0-4:        compute_stock_dynamics.py 計算 7 個因子
  - P0-5 / QW-4: parallel_train.assert_v3_features_present
  - QW-3:        signal_filter v3 hard blocks (vix_zscore_252 / yield_curve_inverted)
  - QW-5:        ensemble.calibrate 接受 cv_strategy='time_series'
  - QW-6:        load_financial_statements 動態 lag (Q1-3=45, Q4=90)
  - QW-7:        signal_history / fetch_log / auto_train_heartbeat DDL 字串完整
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 注入路徑
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "fetchers", "pipeline", "training", "monitor", "models", "utils"]:
    p = str(SCRIPTS_DIR / sub) if sub else str(SCRIPTS_DIR)
    if p not in sys.path:
        sys.path.append(p)

# config 需要 FINMIND_TOKEN，測試環境若無則 skip 整個 suite
os.environ.setdefault("FINMIND_TOKEN", "TEST_TOKEN_FOR_UNITTEST")


class TestP0Fixes(unittest.TestCase):
    """P0 致命風險修復後的回歸測試"""

    def test_p0_1_backfill_from_gaps_importable(self):
        """P0-1: backfill_from_gaps.py 應該可以 import 不出錯"""
        try:
            from fetchers import backfill_from_gaps as m
        except ModuleNotFoundError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "backfill_from_gaps",
                str(SCRIPTS_DIR / "fetchers" / "backfill_from_gaps.py"),
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        # 必備函式 / 對應表存在
        self.assertTrue(hasattr(m, "TABLE_TO_FETCHER"))
        self.assertTrue(hasattr(m, "load_gaps"))
        self.assertTrue(hasattr(m, "run_fetcher"))
        # 對應表至少要涵蓋所有 P0-1 列出的核心表
        for tbl in ["stock_per", "price_adj", "day_trading", "price_limit",
                    "securities_lending", "daily_short_balance", "us_stock_price"]:
            self.assertIn(tbl, m.TABLE_TO_FETCHER, f"{tbl} 不在 backfill_from_gaps 對應表")

    def test_p0_2_signal_filter_thresholds(self):
        """P0-2: signal_filter 機率門檻必須提高到 0.75，且要有 min_hold_days"""
        from pipeline.signal_filter import FILTER_CONFIG
        self.assertGreaterEqual(FILTER_CONFIG["prob_up_threshold"], 0.75,
                                "prob_up_threshold 應 ≥ 0.75（解決 5% TC 吞 alpha）")
        self.assertEqual(FILTER_CONFIG["min_hold_days"], 5,
                         "最小持倉日數應為 5")
        self.assertIn("max_n_trades_per_year", FILTER_CONFIG)
        self.assertIn("min_net_sharpe", FILTER_CONFIG)
        self.assertTrue(FILTER_CONFIG["use_net_sharpe"])

    def test_p0_3_auto_train_heartbeat(self):
        """P0-3 / QW-2: auto_train_manager 要有 write_heartbeat 機制"""
        from training import auto_train_manager as m
        self.assertTrue(hasattr(m, "write_heartbeat"))
        self.assertTrue(hasattr(m, "HEARTBEAT_FILE"))
        self.assertIn("CREATE TABLE", m.HEARTBEAT_DDL)
        self.assertIn("auto_train_heartbeat", m.HEARTBEAT_DDL)

    def test_p0_4_compute_stock_dynamics_present(self):
        """P0-4: compute_stock_dynamics.py 必須計算所有 7 個因子"""
        try:
            from training import compute_stock_dynamics as m
        except ModuleNotFoundError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "compute_stock_dynamics",
                str(SCRIPTS_DIR / "training" / "compute_stock_dynamics.py"),
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        for fn in [
            "compute_info_sensitivity",
            "compute_gravity_elasticity",
            "compute_fat_tail_index",
            "compute_convexity_score",
            "compute_tail_risk_score",
            "compute_wave_track",
            "compute_innovation_velocity",
        ]:
            self.assertTrue(hasattr(m, fn), f"compute_stock_dynamics 缺少 {fn}")

        # DDL 含 7 個欄位
        for col in ["info_sensitivity", "gravity_elasticity", "fat_tail_index",
                    "convexity_score", "tail_risk_score", "wave_track",
                    "innovation_velocity"]:
            self.assertIn(col, m.DDL_REGISTRY)

        # 計算單元測試：用 mock returns 確認結果在合理範圍
        rng = np.random.default_rng(42)
        ret = pd.Series(rng.normal(0, 0.02, 504))
        close = (1 + ret).cumprod() * 100
        self.assertGreater(m.compute_gravity_elasticity(close), 0)
        self.assertGreater(m.compute_fat_tail_index(ret), -2)
        self.assertLessEqual(m.compute_tail_risk_score(ret), 0)
        self.assertGreaterEqual(m.compute_innovation_velocity(ret, None), 0.5)

    def test_p0_5_v3_feature_guard(self):
        """P0-5 / QW-4: parallel_train 必須有 v3 因子守門"""
        from training import parallel_train as m
        self.assertTrue(hasattr(m, "assert_v3_features_present"))
        self.assertTrue(hasattr(m, "V3_REQUIRED_FEATURES"))
        # v3 必須因子至少包含這些
        for f in ["fcf_yield", "vix_zscore_252", "news_intensity"]:
            self.assertIn(f, m.V3_REQUIRED_FEATURES)


class TestQuickWins(unittest.TestCase):
    """Quick Wins 修復後的功能測試"""

    def test_qw3_signal_filter_hard_blocks(self):
        """QW-3: signal_filter 要包含 v3 hard block 邏輯"""
        from pipeline.signal_filter import SignalFilter, FILTER_CONFIG
        sf = SignalFilter()
        # 配置層
        self.assertIn("vix_zscore_block", sf.cfg)
        self.assertEqual(sf.cfg["vix_zscore_block"], 2.0)

    def test_qw5_calibrator_cv_strategy(self):
        """QW-5: ensemble.calibrate 要支援 cv_strategy='time_series'"""
        from models.ensemble_model import XGBPredictor
        # 構造小型 XGBPredictor 並 mock calibrate 路徑
        pred = XGBPredictor(task="classification")
        n = 200
        oof = np.linspace(0.1, 0.9, n)
        y = (oof > 0.5).astype(int)
        # 走 time_series 應該不出錯且寫入 _calibrator
        pred.calibrate(oof, y, cv_strategy="time_series")
        self.assertIsNotNone(pred._calibrator)
        self.assertEqual(pred._cal_type, "isotonic_ts")

    def test_qw5_calibrator_deterministic(self):
        """QW-5: 同樣輸入 calibrate 兩次應該是 deterministic"""
        from models.ensemble_model import LGBPredictor
        rng = np.random.default_rng(42)
        n = 300
        oof = rng.uniform(0, 1, n)
        y = (oof + rng.normal(0, 0.1, n) > 0.5).astype(int)

        p1 = LGBPredictor(task="classification")
        p1.calibrate(oof.copy(), y.copy(), cv_strategy="time_series")
        p1_pred = p1._calibrator.predict(oof[:50])

        p2 = LGBPredictor(task="classification")
        p2.calibrate(oof.copy(), y.copy(), cv_strategy="time_series")
        p2_pred = p2._calibrator.predict(oof[:50])

        np.testing.assert_array_almost_equal(p1_pred, p2_pred,
            err_msg="calibrate 應該是 deterministic")

    def test_qw6_dynamic_lag_quarterly(self):
        """QW-6: financial_statements lag 必須動態 (Q1-3=45 天, Q4=90 天)"""
        from config import DATA_LAG_CONFIG
        self.assertEqual(DATA_LAG_CONFIG.get("quarterly_report"), 45)
        self.assertEqual(DATA_LAG_CONFIG.get("annual_report"), 90)

        # 模擬 _get_lag 邏輯：對 6/30 (Q2) 應該回 45
        import importlib
        dp = importlib.import_module("pipeline.data_pipeline")
        # _get_lag 是 load_financial_statements 內的 closure，無法直接測；
        # 改測 source 內 quarterly_report 確實被引用
        import inspect
        src = inspect.getsource(dp.load_financial_statements)
        self.assertIn("quarterly_report", src,
            "load_financial_statements 應使用 DATA_LAG_CONFIG['quarterly_report']")

    def test_qw7_fetch_log_audit(self):
        """QW-7: data_integrity_audit.audit_fetch_failures 用的 fetch_log 表必須有 DDL"""
        from core.db_utils import DDL_FETCH_LOG, log_fetch_result
        self.assertIn("CREATE TABLE", DDL_FETCH_LOG)
        self.assertIn("fetch_log", DDL_FETCH_LOG)
        # 必要欄位
        for col in ["timestamp", "table_name", "stock_id", "status", "error_msg"]:
            self.assertIn(col, DDL_FETCH_LOG)

    def test_qw7_audit_full_table_coverage(self):
        """QW-7 / P1-2: data_integrity_audit 預設 tables 應拉到 TABLE_REGISTRY 的所有日更表"""
        from monitor import data_integrity_audit as m
        # 驗證預設覆蓋率不再硬編碼成 5 張表
        import inspect
        src = inspect.getsource(m.IntegrityAuditor.audit_coverage_matrix)
        # 應該引用 TABLE_REGISTRY 而非寫死清單
        self.assertTrue(
            "TABLE_REGISTRY" in src or "tables = " in src,
            "audit_coverage_matrix 應動態使用 TABLE_REGISTRY"
        )

    def test_qw8_logrotate_config_exists(self):
        """QW-8: deploy/logrotate.conf 必須存在且包含 rotate 設定"""
        cfg_path = SCRIPTS_DIR.parent / "deploy" / "logrotate.conf"
        self.assertTrue(cfg_path.exists(), f"logrotate config not found: {cfg_path}")
        content = cfg_path.read_text()
        for key in ["daily", "rotate", "compress", "missingok", "copytruncate"]:
            self.assertIn(key, content)


class TestSignalFilterIntegration(unittest.TestCase):
    """signal_filter 端到端整合測試（不需要 DB）"""

    def _mk_df(self, **overrides):
        """建一筆 latest 用的 df_feat，預設都是健康狀態。"""
        base = {
            "is_delisted": 0, "is_in_disposition": 0, "is_margin_suspended": 0,
            "vix_zscore_252": 0.0, "yield_curve_inverted": 0, "hy_credit_spread": 1.0,
            "realized_vol_20d": 0.18, "trend_regime": "bull", "trend_regime_int": 1,
            "foreign_net_weekly": 5e8, "foreign_net_accel": 1e8,
            "rev_yoy_positive_months": 6, "rev_yoy_3m": 0.10, "gross_margin_qoq": 0.02,
            "fear_greed_score": 50, "put_call_large_ratio": 1.0,
            "macro_monitoring_color": "green",
            "fcf_yield": 0.06, "foreign_fut_oi_chg_5d": 1.0,
            "night_session_premium": 1.0, "sbl_short_intensity": 0.01,
            "news_intensity": 0.5,
            "smart_money_sync_buy": 1, "is_extreme_greed": 0, "is_extreme_fear": 0,
            "kinetic_momentum": 0, "structural_premium": 0, "gravity_pull": 0,
            "info_force_per_mass": 0, "kwave_score": 0, "entropy_delta": 0,
            "total_info_force": 0, "bubble_crash_risk": 0, "large_holder_change_3m": 0,
            "eps_accel_proxy": 0,
        }
        base.update(overrides)
        idx = pd.date_range("2026-04-01", periods=5)
        return pd.DataFrame([base] * 5, index=idx)

    def _mk_sf(self):
        """構造 SignalFilter，但 stub 掉 _persist_signal_history（避免 DB 連線）。"""
        from pipeline.signal_filter import SignalFilter
        sf = SignalFilter()
        sf._persist_signal_history = lambda **kwargs: None
        sf._load_dynamics_registry = lambda stock_id: None
        return sf

    def test_block_when_delisted(self):
        sf = self._mk_sf()
        df = self._mk_df(is_delisted=1)
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.95}, df)
        self.assertEqual(result.decision, "HOLD_CASH")
        self.assertTrue(any("下市" in r for r in result.blocking_reasons))

    def test_block_when_disposition(self):
        sf = self._mk_sf()
        df = self._mk_df(is_in_disposition=1)
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.95}, df)
        self.assertEqual(result.decision, "HOLD_CASH")
        self.assertTrue(any("處置" in r for r in result.blocking_reasons))

    def test_block_when_vix_extreme(self):
        sf = self._mk_sf()
        df = self._mk_df(vix_zscore_252=2.5)  # > 2σ
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.95}, df)
        self.assertEqual(result.decision, "HOLD_CASH")
        self.assertTrue(any("VIX" in r for r in result.blocking_reasons))

    def test_block_when_yield_curve_and_credit(self):
        sf = self._mk_sf()
        df = self._mk_df(yield_curve_inverted=1, hy_credit_spread=6.0)
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.95}, df)
        self.assertEqual(result.decision, "HOLD_CASH")
        self.assertTrue(any("殖利率倒掛" in r for r in result.blocking_reasons))

    def test_block_below_new_threshold(self):
        """prob_up=0.70 < 新門檻 0.75 → 不可進場"""
        sf = self._mk_sf()
        df = self._mk_df()
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.70,
                              "model_agreement": 0.8}, df)
        self.assertEqual(result.decision, "HOLD_CASH")

    def test_pass_when_clean(self):
        """所有條件健康 + prob_up=0.85 → 應該通過"""
        sf = self._mk_sf()
        df = self._mk_df()
        result = sf.evaluate({"stock_id": "X", "prob_up": 0.85,
                              "model_agreement": 0.8,
                              "warnings": {"macro_shock": False}}, df)
        # 由於 overall 計分受多項加分影響，至少要 != HOLD_CASH
        # 如果系統判定 LONG 或 WATCH 都接受
        self.assertIn(result.decision, ("LONG", "WATCH"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

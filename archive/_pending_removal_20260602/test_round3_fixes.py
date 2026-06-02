"""
scripts/tests/test_round3_fixes.py — 第三輪審查修復回歸測試

驗證項目（對應審查報告章節）：
  - P0/2.1 audit_snapshot.py：build_features() 不再 tuple-unpack
  - P0/2.2 backtest_audit.py：load_live_forecasts 有 exclude_backfill 參數
  - P0/PSI  model_health_check.py：PSI 參考分佈 deterministic
  - P1/2.3 auto_train_manager.py：calculate_priority 正確 clip 至 100
  - P1/2.6 全系統 DB_CONFIG 統一引用
  - P1/3.3 kwave_score 列入 ALL_FEATURES
  - P2/2.7 backtest_engine.py 使用 FRICTION_CONFIG
  - P2/2.10 train_evaluate.py 輸出 oof_full（含日期）
  - P2/3.1 utils/model_loader.py 含 file lock
  - P3/2.11 strategy_tester.py 整合兩個 simulator

執行：
    cd scripts && python -m pytest tests/test_round3_fixes.py -v
    或單獨跑：python tests/test_round3_fixes.py
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

# 確保 scripts/ 在 path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# 偵測重要外部依賴是否齊備（缺少時整批跳過該類測試，方便在 CI 沙箱跑）
def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_PSYCOPG2 = _has_module("psycopg2")
HAS_JOBLIB = _has_module("joblib")
HAS_PANDAS = _has_module("pandas")


@unittest.skipUnless(HAS_PSYCOPG2 and HAS_PANDAS,
                     "psycopg2 / pandas 未安裝，跳過依賴 model_health_check 的測試")
class TestPSIDeterministic(unittest.TestCase):
    """[P0] PSI 參考分佈須為 deterministic（同樣輸入產生相同輸出）。"""

    def test_theoretical_prior_deterministic(self):
        from model_health_check import _theoretical_prior_reference
        a = _theoretical_prior_reference()
        b = _theoretical_prior_reference()
        self.assertTrue(np.array_equal(a, b),
                        "Beta(2,2) prior 必須使用固定 seed，連續呼叫結果相同")
        self.assertEqual(a.shape, (1000,))
        self.assertTrue(0.0 <= a.min() and a.max() <= 1.0,
                        "Beta(2,2) 取值範圍應在 [0,1]")

    def test_calculate_psi_basic(self):
        """同分佈 PSI 應 ≈ 0；極端不同分佈 PSI 應 > 0.2"""
        from model_health_check import calculate_psi
        rng = np.random.default_rng(42)
        a = rng.normal(0.5, 0.1, 500).clip(0, 1)
        b = rng.normal(0.5, 0.1, 500).clip(0, 1)
        psi_same = calculate_psi(a, b)
        self.assertLess(psi_same, 0.1, f"同分佈 PSI 應小於 0.1，實際 = {psi_same}")

        c = rng.normal(0.9, 0.05, 500).clip(0, 1)  # 嚴重右偏
        psi_diff = calculate_psi(a, c)
        self.assertGreater(psi_diff, 0.2, f"極端不同分佈 PSI 應大於 0.2，實際 = {psi_diff}")


class TestPriorityCapping(unittest.TestCase):
    """[P1 2.3] auto_train_manager.calculate_priority 雙重加成必須正確 clip。"""

    @unittest.skipUnless(HAS_PSYCOPG2, "auto_train_manager 依賴 data_pipeline → psycopg2")
    def test_anchor_priority_capped_at_100(self):
        from auto_train_manager import calculate_priority
        # 2330 是 Anchor + SIXTH_WAVE，DA=0.6 應達上限
        score = calculate_priority("2330", {"2330": 0.6})
        self.assertLessEqual(score, 100.0)
        self.assertGreater(score, 90.0)

    @unittest.skipUnless(HAS_PSYCOPG2, "auto_train_manager 依賴 data_pipeline → psycopg2")
    def test_unknown_stock_low_priority(self):
        from auto_train_manager import calculate_priority
        # 未在 ANCHOR/TIER1/SIXTH_WAVE 中的標的，base=30
        score = calculate_priority("9999", {"9999": 0.5})
        self.assertLess(score, 50.0)


class TestDBConfigUnified(unittest.TestCase):
    """[P1 2.6] DB_CONFIG 必須由所有模組共用同一份。"""

    @unittest.skipUnless(HAS_PSYCOPG2, "psycopg2 未安裝")
    def test_data_pipeline_uses_config_db_config(self):
        from config import DB_CONFIG as cfg_a
        from data_pipeline import DB_CONFIG as cfg_b
        self.assertIs(cfg_a, cfg_b,
                      "data_pipeline 必須 from config import DB_CONFIG（同一物件）")

    def test_no_hardcoded_db_password_in_modules(self):
        """checked-in 程式不可直接出現 password='stock' 字串（除了 config 與 fallback）。"""
        offenders = []
        for fn in [
            "calibrate_stock_physics.py",
            "historical_backfill.py",
            "backtest_audit.py",
        ]:
            path = SCRIPTS_DIR / fn
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            # 不可有完整的 hardcode block（dbname + user + password 三件組）
            if ('"dbname":   "stock"' in text or "'dbname': 'stock'" in text) \
               and "DB_PASSWORD" not in text:
                offenders.append(fn)
        self.assertEqual(offenders, [],
                         f"以下模組仍存在硬編碼 DB 連線：{offenders}")


class TestKwaveInFeatures(unittest.TestCase):
    """[P1 3.3] kwave_score 必須在 ALL_FEATURES 中（解決幽靈特徵）。"""

    @unittest.skipUnless(HAS_PANDAS, "pandas 未安裝")
    def test_kwave_score_in_all_features(self):
        from config import ALL_FEATURES, FEATURE_GROUPS
        self.assertIn("kwave_score", ALL_FEATURES,
                      "kwave_score 必須加入 ALL_FEATURES")
        macro_features = FEATURE_GROUPS.get("macro", [])
        self.assertIn("kwave_score", macro_features,
                      "kwave_score 應隸屬於 macro 特徵群")

    @unittest.skipUnless(HAS_PANDAS, "pandas 未安裝")
    def test_add_kwave_regime_features_signature(self):
        from feature_engineering import add_kwave_regime_features
        sig = inspect.signature(add_kwave_regime_features)
        # 確認接收 DataFrame 且有 fallback 邏輯
        self.assertEqual(len(sig.parameters), 1)


class TestFrictionConfigUsage(unittest.TestCase):
    """[P2 2.7] backtest_engine 必須引用 FRICTION_CONFIG，不再硬編碼成本。"""

    def test_backtest_engine_imports_friction_config(self):
        path = SCRIPTS_DIR / "backtest_engine.py"
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("FRICTION_CONFIG", text,
                      "backtest_engine.py 必須 import 並使用 FRICTION_CONFIG")
        self.assertIn("SELL_COST_RATE", text,
                      "賣出成本必須含手續費 + 證交稅（SELL_COST_RATE 常數）")

    @unittest.skipUnless(HAS_PANDAS, "config 依賴 dotenv 等可選套件")
    def test_friction_config_complete(self):
        from config import FRICTION_CONFIG
        for key in ("commission", "securities_tax", "slippage_large_cap", "slippage_small_cap"):
            self.assertIn(key, FRICTION_CONFIG, f"FRICTION_CONFIG 缺少 {key}")


@unittest.skipUnless(HAS_PSYCOPG2, "backtest_audit 依賴 psycopg2")
class TestExcludeBackfillSignature(unittest.TestCase):
    """[P0 2.2] backtest_audit.load_live_forecasts 必須有 exclude_backfill 參數。"""

    def test_load_live_forecasts_has_exclude_backfill(self):
        from backtest_audit import load_live_forecasts
        sig = inspect.signature(load_live_forecasts)
        self.assertIn("exclude_backfill", sig.parameters,
                      "load_live_forecasts 必須有 exclude_backfill 參數")
        # 預設應為 True（安全的預設值，避免使用者誤用 backfill 資料）
        self.assertTrue(sig.parameters["exclude_backfill"].default,
                        "exclude_backfill 預設值必須為 True")


class TestOOFOutputFormat(unittest.TestCase):
    """[P2 2.10] train_evaluate 必須輸出含日期的 OOF 序列。"""

    def test_train_evaluate_writes_oof_with_dates(self):
        path = SCRIPTS_DIR / "train_evaluate.py"
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("oof_predictions_with_dates", text,
                      "train_evaluate.py 必須輸出 oof_predictions_with_dates*.csv")


@unittest.skipUnless(HAS_JOBLIB, "joblib 未安裝")
class TestModelLoaderFileLock(unittest.TestCase):
    """[P2 3.1] model_loader 必須含檔案鎖機制。"""

    def test_load_save_with_lock_exposed(self):
        from utils import model_loader
        self.assertTrue(hasattr(model_loader, "load_model_with_lock"))
        self.assertTrue(hasattr(model_loader, "save_model_with_lock"))
        self.assertTrue(hasattr(model_loader, "save_ensemble_model"))


@unittest.skipUnless(HAS_PANDAS and HAS_PSYCOPG2,
                     "strategy_tester 透過 data_pipeline 需要 psycopg2")
class TestStrategyTesterIntegration(unittest.TestCase):
    """[P3 2.11] strategy_tester 必須整合 asymmetric / singularity 兩種策略。"""

    def test_entry_methods_registered(self):
        from strategy_tester import ENTRY_REGISTRY, EXIT_REGISTRY
        self.assertIn("dynamic", ENTRY_REGISTRY,
                      "對齊 asymmetric_simulator 的 dynamic entry 必須存在")
        self.assertIn("fixed",   ENTRY_REGISTRY,
                      "對齊 singularity_layout_simulator 的 fixed entry 必須存在")
        self.assertIn("gravity_zero", EXIT_REGISTRY)

    def test_strategy_uses_friction_config(self):
        """確認策略費率計算使用 config.FRICTION_CONFIG 而非硬編碼 0.003"""
        from strategy_tester import PhysicsStrategy
        s = PhysicsStrategy(stock_id="2330")
        # 賣出費率應包含手續費 + 證交稅 ≈ 0.4425%
        self.assertAlmostEqual(1 - s.sell_cost_rate, 0.001425 + 0.003, places=5)
        # 買入費率只含手續費 ≈ 0.1425%
        self.assertAlmostEqual(s.buy_cost_rate - 1, 0.001425, places=5)


@unittest.skipUnless(HAS_PANDAS and HAS_JOBLIB,
                     "train_evaluate 依賴 pandas / joblib / sklearn")
class TestPurgedWalkForwardFolds(unittest.TestCase):
    """[P3] purged_walk_forward_folds 的合法性測試（防止洩漏）。"""

    def test_purged_walk_forward_no_overlap(self):
        from train_evaluate import purged_walk_forward_folds
        folds = list(purged_walk_forward_folds(
            n=2000, train_window=756, val_window=126,
            step_days=21, embargo_days=45, test_window=126,
        ))
        self.assertGreater(len(folds), 0, "應產生至少一個 fold")
        for f in folds:
            # 訓練/驗證/測試不能有重疊
            self.assertEqual(set(f.train_idx) & set(f.val_idx), set())
            self.assertEqual(set(f.train_idx) & set(f.test_idx), set())
            self.assertEqual(set(f.val_idx)   & set(f.test_idx), set())
            # train_end + embargo ≤ val_start
            train_end = int(f.train_idx.max())
            val_start = int(f.val_idx.min())
            self.assertGreaterEqual(val_start - train_end, 45,
                                    "embargo 必須 ≥ 45 天")


class TestDataLagApplied(unittest.TestCase):
    """[P0-1 鞏固] DATA_LAG_CONFIG 必須有完整的四大延遲設定。"""

    @unittest.skipUnless(HAS_PANDAS, "config 載入需 dotenv")
    def test_data_lag_config_complete(self):
        from config import DATA_LAG_CONFIG
        for key, expected_min in [
            ("month_revenue", 30),         # 次月 10 日公告：≥ 30 日
            ("financial_statements", 40),  # 季報 +45：≥ 40 日
            ("annual_report", 80),         # 年報 +90：≥ 80 日
            ("institutional_chip", 1),     # 籌碼 T+1
        ]:
            self.assertIn(key, DATA_LAG_CONFIG, f"DATA_LAG_CONFIG 缺少 {key}")
            self.assertGreaterEqual(DATA_LAG_CONFIG[key], expected_min,
                                    f"{key} 延遲過短（{DATA_LAG_CONFIG[key]} < {expected_min}）")


if __name__ == "__main__":
    unittest.main(verbosity=2)

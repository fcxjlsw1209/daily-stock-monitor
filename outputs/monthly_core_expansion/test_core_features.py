"""Independent formula, causality and data-coverage checks; no returns testing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("core_features_for_tests", Path(__file__).with_name("core_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def panel(rows=780):
    rng = np.random.default_rng(15021)
    market = rng.normal(.0005, .01, rows - 1)
    stock = .0007 + 1.3 * market + rng.normal(0., .012, rows - 1)
    etf = -.0001 + .8 * market + rng.normal(0., .007, rows - 1)
    r = np.column_stack((market, stock, etf))
    p = np.vstack((np.ones(3), np.cumprod(1. + r, axis=0))) * [100., 217., 51.]
    return {"adj": pd.DataFrame(p, index=pd.bdate_range("2010-01-04", periods=rows), columns=["SPY", "A", "ETF"])}


def calculate(p, t=-1, symbols=("SPY", "A", "ETF")):
    return features.build_features(p, [(p["adj"].index[t], symbols)])


class TestCoreFeatures(unittest.TestCase):
    def test_ordinary_and_hac_match_independent_exact_formula(self):
        p = panel()
        result, audit = calculate(p)
        vals = p["adj"]["A"].pct_change(fill_method=None).iloc[-252:-42].to_numpy()
        z = vals - vals.mean()
        long_variance = (z * z).sum() / 210.
        for lag in range(1, 6):
            long_variance += 2. * (1. - lag / 6.) * (z[lag:] * z[:-lag]).sum() / 210.
        self.assertAlmostEqual(result["ordinary_sharpe"][0, 1], vals.mean() / vals.std(ddof=1), places=14)
        self.assertAlmostEqual(result["hac_sharpe"][0, 1], vals.mean() / np.sqrt(long_variance), places=14)
        self.assertEqual(audit["last_used_offset"], -42)
        self.assertEqual(audit["first_required_residual_price_offset"], -504)
        self.assertFalse(audit["trading_eligibility_modified"])

    def test_residual_beta_uses_prior252_not_same_day_and_retains_intercept(self):
        p = panel()
        result, _ = calculate(p)
        r = p["adj"].pct_change(fill_method=None)
        t = len(r) - 1
        residual = []
        for u in range(t - 251, t - 41):
            past = r.iloc[u - 252:u]
            beta = np.cov(past["A"], past["SPY"], ddof=1)[0, 1] / past["SPY"].var(ddof=1)
            residual.append(r["A"].iloc[u] - beta * r["SPY"].iloc[u])
        expected = np.mean(residual) / np.std(residual, ddof=1)
        self.assertAlmostEqual(result["market_residual_sharpe"][0, 1], expected, places=12)
        beta, _ = features._market_beta(r)
        changed = r.copy()
        changed.loc[r.index[500], "A"] = .7
        changed.loc[r.index[500], "SPY"] = -.4
        changed_beta, _ = features._market_beta(changed)
        self.assertEqual(beta["A"].iloc[500], changed_beta["A"].iloc[500])
        self.assertNotEqual(beta["A"].iloc[501], changed_beta["A"].iloc[501])

    def test_prefix_future_and_skipped42_sessions_cannot_change_scores(self):
        p = panel()
        expected, _ = calculate(p, 650)
        prefix = {"adj": p["adj"].iloc[:651].copy()}
        prefixed, _ = calculate(prefix)
        p["adj"].iloc[609:] = np.nan
        perturbed, _ = calculate(p, 650)
        for key in expected:
            np.testing.assert_array_equal(expected[key], prefixed[key])
            np.testing.assert_array_equal(expected[key], perturbed[key])
        p["adj"].iloc[608, 1] = np.nan
        endpoint, _ = calculate(p, 650)
        self.assertTrue(np.isnan(endpoint["ordinary_sharpe"][0, 1]))

    def test_scale_invariance_and_input_unchanged(self):
        p = panel()
        saved = p["adj"].copy(deep=True)
        expected, _ = calculate(p)
        pd.testing.assert_frame_equal(p["adj"], saved)
        scaled = {"adj": saved * [37.2, .071, 1e4]}
        actual, _ = calculate(scaled)
        for key in features.FEATURE_NAMES:
            np.testing.assert_allclose(expected[key], actual[key], atol=2e-12, rtol=2e-12)
            np.testing.assert_array_equal(expected[key + "_valid"], actual[key + "_valid"])

    def test_spy_zero_and_etf_uses_real_residual_not_manual_neutral(self):
        result, audit = calculate(panel())
        self.assertEqual(result["market_residual_sharpe"][0, 0], 0.)
        self.assertTrue(result["market_residual_sharpe_valid"][0, 0])
        self.assertFalse(result["market_residual_sharpe_fallback"][0, 0])
        self.assertNotEqual(result["market_residual_sharpe"][0, 2], 0.)
        self.assertEqual(audit["degenerate_counts"]["zero_market_residual"], 1)

    def test_tiny_residual_is_zero_but_nonzero_constant_residual_falls_back(self):
        p = panel()
        market = p["adj"]["SPY"].pct_change(fill_method=None).iloc[1:].to_numpy()
        # This predictable intercept must remain in the residual; subtracting
        # it would incorrectly report zero rather than this documented fallback.
        nonzero_constant = 1.3 * market + .001
        tiny = 1.3 * market + 1e-14 * np.sin(np.arange(len(market)))
        p["adj"]["A"] = 100. * np.r_[1., np.cumprod(1. + nonzero_constant)]
        p["adj"]["ETF"] = 100. * np.r_[1., np.cumprod(1. + tiny)]
        result, audit = calculate(p)
        self.assertEqual(result["market_residual_sharpe"][0, 1], result["ordinary_sharpe"][0, 1])
        self.assertTrue(result["market_residual_sharpe_fallback"][0, 1])
        self.assertEqual(audit["fallback_reasons"]["market_residual_sharpe"]["nonzero_constant_market_residual"], 1)
        self.assertEqual(result["market_residual_sharpe"][0, 2], 0.)
        self.assertTrue(result["market_residual_sharpe_valid"][0, 2])
        self.assertFalse(result["market_residual_sharpe_fallback"][0, 2])

    def test_autocorrelation_changes_hac_in_expected_direction(self):
        p = panel()
        n = len(p["adj"]) - 1
        persistent = .002 + .009 * np.sin(np.arange(n) / 10.)
        alternating = .002 + .009 * ((-1.) ** np.arange(n))
        p["adj"]["A"] = np.r_[1., np.cumprod(1. + persistent)] * 100.
        p["adj"]["ETF"] = np.r_[1., np.cumprod(1. + alternating)] * 100.
        result, _ = calculate(p)
        self.assertLess(result["hac_sharpe"][0, 1], result["ordinary_sharpe"][0, 1])
        self.assertGreater(result["hac_sharpe"][0, 2], result["ordinary_sharpe"][0, 2])

    def test_missing_asset_beta_history_falls_back_without_eligibility_filter(self):
        p = panel()
        # Before the formation window, but inside the first beta's 252 returns.
        p["adj"].iloc[-450, 1] = np.nan
        result, audit = calculate(p)
        self.assertEqual(result["market_residual_sharpe"][0, 1], result["ordinary_sharpe"][0, 1])
        self.assertTrue(result["market_residual_sharpe_fallback"][0, 1])
        self.assertTrue(result["market_residual_sharpe_valid"][0, 1])
        self.assertEqual(audit["fallback_reasons"]["market_residual_sharpe"]["invalid_asset_beta_history"], 1)

    def test_missing_or_tiny_market_variance_fallback_causes(self):
        for mode, reason in (("missing", "missing_market_symbol"),
                             ("history", "invalid_market_beta_history"),
                             ("constant", "zero_or_tiny_market_variance")):
            with self.subTest(mode=mode):
                p = panel()
                if mode == "missing":
                    p["adj"] = p["adj"].drop(columns="SPY")
                elif mode == "history":
                    p["adj"].iloc[-450, 0] = np.nan
                else:
                    p["adj"]["SPY"] = 100. * (1. + np.arange(len(p["adj"])) * 1e-14)
                result, audit = calculate(p, symbols=("A", "ETF"))
                np.testing.assert_array_equal(result["market_residual_sharpe"], result["ordinary_sharpe"])
                self.assertEqual(audit["fallback_reasons"]["market_residual_sharpe"][reason], 2)

    def test_missing_price_window_stays_missing_not_zero_or_fallback(self):
        for bad in (np.nan, np.inf, 0., -1.):
            with self.subTest(bad=bad):
                p = panel()
                p["adj"].iloc[-90, 1] = bad
                result, _ = calculate(p)
                for key in features.FEATURE_NAMES:
                    self.assertTrue(np.isnan(result[key][0, 1]))
                    self.assertFalse(result[key + "_valid"][0, 1])
                    self.assertFalse(result[key + "_fallback"][0, 1])

    def test_warmup_boundary_and_unknown_columns_preserve_shape(self):
        p = panel()
        result, audit = calculate(p, 252, ("A", "UNKNOWN"))
        self.assertEqual(result["ordinary_sharpe"].shape, (1, 2))
        self.assertTrue(result["ordinary_sharpe_valid"][0, 0])
        self.assertTrue(result["market_residual_sharpe_fallback"][0, 0])
        self.assertEqual(audit["fallback_reasons"]["market_residual_sharpe"]["insufficient_beta_history"], 1)
        result, _ = calculate(p, 503)
        self.assertTrue(result["market_residual_sharpe_fallback"].all())
        result, _ = calculate(p, 504)
        self.assertFalse(result["market_residual_sharpe_fallback"].any())
        empty, _ = features.build_features(p, [])
        self.assertTrue(all(v.shape == (0, 0) for v in empty.values()))

    def test_degenerate_zero_variance_stays_finite_and_audited(self):
        p = panel()
        p["adj"]["A"] = 125.
        result, audit = calculate(p)
        for key in features.FEATURE_NAMES:
            self.assertEqual(result[key][0, 1], 0.)
            self.assertTrue(result[key + "_valid"][0, 1])
        self.assertTrue(result["hac_sharpe_fallback"][0, 1])
        self.assertGreater(audit["degenerate_counts"]["ordinary_zero_standard_deviation"], 0)

    def test_input_shape_and_event_invariants(self):
        p = panel()
        with self.assertRaisesRegex(ValueError, "same number"):
            features.build_features(p, [(p["adj"].index[-1], ["A"]), (p["adj"].index[-2], ["A", "ETF"])])
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            calculate(p, symbols=("A", "A"))
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            features.build_features(p, [(pd.Timestamp("1900-01-01"), ["A"])])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            calculate({"adj": p["adj"].iloc[::-1]})
        with self.assertRaisesRegex(ValueError, "columns must be unique"):
            calculate({"adj": p["adj"].rename(columns={"ETF": "A"})})

    def test_candidate_order_and_irrelevant_columns_do_not_change_values(self):
        p = panel()
        expected, _ = calculate(p)
        altered = {"adj": p["adj"][["ETF", "SPY", "A"]].assign(UNUSED=np.nan)}
        rows = [(p["adj"].index[-1], ["ETF", "A"]), (p["adj"].index[-1], ["A", "ETF"])]
        actual, _ = features.build_features(altered, rows)
        for key in expected:
            np.testing.assert_array_equal(actual[key][0], expected[key][0, [2, 1]])
            np.testing.assert_array_equal(actual[key][1], expected[key][0, [1, 2]])


if __name__ == "__main__":
    unittest.main()

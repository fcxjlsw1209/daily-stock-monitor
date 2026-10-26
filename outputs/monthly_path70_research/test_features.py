"""Formula and causal-data tests only; no strategy return evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


_spec = importlib.util.spec_from_file_location("path70_features_for_tests", Path(__file__).with_name("features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def panel(rows=440):
    rng = np.random.default_rng(7021042)
    returns = rng.normal([.0008, .0001, -.0009], [.014, .009, .012], (rows - 1, 3))
    prices = np.vstack((np.ones(3), np.exp(np.cumsum(returns, axis=0)))) * [100., 219., 77.]
    return pd.DataFrame(prices, index=pd.bdate_range("2018-01-02", periods=rows), columns=["A", "B", "ETF"])


def from_formation(log_returns):
    formation = 100. * np.exp(np.r_[0., np.cumsum(log_returns)])
    return pd.DataFrame({"A": np.r_[formation, np.full(42, np.nan)]},
                        index=pd.bdate_range("2020-01-02", periods=253))


class TestPathFeatures(unittest.TestCase):
    def test_formulas_match_independent_endpoints_population_std_rms_and_mad(self):
        adj = panel()
        result, audit = features.build_features(adj)
        t = len(adj) - 1
        p = adj["A"].to_numpy()
        boundaries = [t - 252, t - 182, t - 112, t - 42]
        growth = np.array([np.log(p[b] / p[a]) for a, b in zip(boundaries[:-1], boundaries[1:])])
        downside = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            daily = np.log(p[a + 1:b + 1] / p[a:b])
            downside.append(np.sqrt(np.sum(np.minimum(daily, 0.) ** 2) / 70.))
        simple = p[t - 251:t - 41] / p[t - 252:t - 42] - 1.
        mad = np.sum(np.abs(simple - np.sum(simple) / 210.)) / 210.
        expected = {
            "weakest_segment": min(growth),
            "balanced_growth": np.mean(growth) - np.sqrt(np.mean((growth - np.mean(growth)) ** 2)),
            "worst_segment_downside": np.sum(growth) / max(downside),
            "centered_mad_momentum": (p[t - 42] / p[t - 252] - 1.) / mad,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(result[key].iloc[t]["A"], value, places=11)
        self.assertEqual(audit["segment_price_offsets_inclusive"], [[-252, -182], [-182, -112], [-112, -42]])
        self.assertEqual(audit["required_positive_finite_formation_prices"], 211)

    def test_exact_seventy_return_segments_and_downside_mean_includes_up_days(self):
        logs = np.r_[[-.07], np.full(69, .001), np.full(70, .002), [-.03], np.full(69, .003)]
        adj = from_formation(logs)
        result, _ = features.build_features(adj)
        growth = np.array([-.07 + 69 * .001, 70 * .002, -.03 + 69 * .003])
        self.assertAlmostEqual(result["weakest_segment"].iloc[-1, 0], min(growth), places=13)
        self.assertAlmostEqual(result["balanced_growth"].iloc[-1, 0], growth.mean() - growth.std(ddof=0), places=13)
        self.assertAlmostEqual(result["worst_segment_downside"].iloc[-1, 0], growth.sum() / (.07 / np.sqrt(70)), places=10)

    def test_prefix_and_future_perturbation_cannot_change_any_past_feature(self):
        adj = panel()
        expected, _ = features.build_features(adj)
        cutoff = 350
        prefix, _ = features.build_features(adj.iloc[:cutoff + 1])
        changed = adj.copy()
        changed.iloc[cutoff + 1:] = np.nan
        future, _ = features.build_features(changed)
        for name in features.FEATURE_NAMES:
            pd.testing.assert_frame_equal(prefix[name], expected[name].iloc[:cutoff + 1])
            pd.testing.assert_frame_equal(future[name].iloc[:cutoff + 1], expected[name].iloc[:cutoff + 1])

    def test_latest42_never_used_even_for_price_availability(self):
        adj = panel()
        expected, _ = features.build_features(adj)
        changed = adj.copy()
        t = len(adj) - 1
        changed.iloc[t - 41:] = np.nan
        changed.iloc[t - 41, 0] = np.inf
        changed.iloc[t - 40, 0] = -100.
        changed.iloc[t - 39, 0] = 0.
        result, audit = features.build_features(changed)
        for name in features.FEATURE_NAMES:
            pd.testing.assert_series_equal(result[name].iloc[t], expected[name].iloc[t])
        self.assertFalse(audit["skipped_recent_sessions_used_for_values_or_availability"])

    def test_missing_nonpositive_or_nonfinite_formation_price_is_not_filled(self):
        adj = panel()
        expected, _ = features.build_features(adj)
        for offset in (-252, -182, -112, -42):
            for bad in (np.nan, np.inf, 0., -1.):
                with self.subTest(offset=offset, bad=bad):
                    changed = adj.copy()
                    changed.iloc[len(adj) - 1 + offset, 0] = bad
                    result, audit = features.build_features(changed)
                    for name in features.FEATURE_NAMES:
                        self.assertTrue(np.isnan(result[name].iloc[-1, 0]))
                        pd.testing.assert_series_equal(result[name].iloc[-1, 1:], expected[name].iloc[-1, 1:])
                    self.assertFalse(audit["trading_eligibility_modified"])

    def test_zero_denominators_stay_nan_without_replacing_growth_scores(self):
        adj = from_formation(np.zeros(210))
        result, audit = features.build_features(adj)
        self.assertEqual(result["weakest_segment"].iloc[-1, 0], 0.)
        self.assertEqual(result["balanced_growth"].iloc[-1, 0], 0.)
        self.assertTrue(np.isnan(result["worst_segment_downside"].iloc[-1, 0]))
        self.assertTrue(np.isnan(result["centered_mad_momentum"].iloc[-1, 0]))
        self.assertEqual(audit["coverage"]["zero_downside_denominator_cells"], 1)
        self.assertEqual(audit["coverage"]["zero_mad_denominator_cells"], 1)

    def test_downside_and_mad_missingness_are_independent(self):
        adj = from_formation(np.tile([.001, .004], 105))
        result, _ = features.build_features(adj)
        self.assertTrue(np.isnan(result["worst_segment_downside"].iloc[-1, 0]))
        self.assertTrue(np.isfinite(result["centered_mad_momentum"].iloc[-1, 0]))
        # Powers of two provide an exactly constant, nonzero simple return.
        adj.iloc[:211, 0] = np.exp2(np.arange(211, dtype=float))
        constant, _ = features.build_features(adj)
        self.assertTrue(np.isnan(constant["centered_mad_momentum"].iloc[-1, 0]))
        self.assertTrue(np.isfinite(constant["weakest_segment"].iloc[-1, 0]))

    def test_negative_growth_is_retained_without_an_eligibility_gate(self):
        adj = from_formation(np.tile([-.001, -.003], 105))
        result, _ = features.build_features(adj)
        for name in features.FEATURE_NAMES:
            self.assertTrue(np.isfinite(result[name].iloc[-1, 0]))
            self.assertLess(result[name].iloc[-1, 0], 0.)

    def test_exact_warmup_position_empty_shapes_and_input_unchanged(self):
        adj = panel()
        saved = adj.copy(deep=True)
        result, _ = features.build_features(adj)
        pd.testing.assert_frame_equal(adj, saved)
        for name in features.FEATURE_NAMES:
            self.assertTrue(result[name].iloc[:252].isna().all().all())
            self.assertTrue(np.isfinite(result[name].iloc[252]).all())
            self.assertTrue(result[name].index.equals(adj.index))
            self.assertTrue(result[name].columns.equals(adj.columns))
        for empty in (adj.iloc[:0], adj.iloc[:, :0]):
            frames, _ = features.build_features(empty)
            self.assertTrue(all(frame.shape == empty.shape for frame in frames.values()))

    def test_scale_column_order_and_unrelated_assets_do_not_change_raw_features(self):
        adj = panel()
        expected, _ = features.build_features(adj)
        other, _ = features.build_features((adj * [37., .13, 220.])[['ETF', 'A', 'B']].assign(UNKNOWN=np.nan))
        for name in features.FEATURE_NAMES:
            np.testing.assert_allclose(expected[name], other[name][adj.columns], atol=2e-11, rtol=2e-11, equal_nan=True)
            self.assertTrue(other[name]["UNKNOWN"].isna().all())

    def test_validation_and_audit_source_provenance(self):
        adj = panel()
        for invalid in (adj.iloc[::-1], pd.concat([adj.iloc[:1], adj]), adj.rename(columns={"B": "A"})):
            with self.assertRaises(ValueError):
                features.build_features(invalid)
        with self.assertRaises(TypeError):
            features.build_features(adj.reset_index(drop=True))
        with self.assertRaises(TypeError):
            features.build_features(adj.to_numpy())
        _, audit = features.build_features(adj)
        json.dumps(audit, allow_nan=False)
        source = Path(features.__file__).resolve()
        self.assertEqual(audit["source_hashes"][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertFalse(audit["cross_sectional_ranking_applied"])
        self.assertEqual(set(audit["finite_feature_cells"]), set(features.FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()

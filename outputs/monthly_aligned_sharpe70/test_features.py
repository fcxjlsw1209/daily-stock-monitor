"""Synthetic raw-factor mathematics and timing, without any strategy returns."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import unittest

import numpy as np
import pandas as pd


PATH = Path(__file__).with_name("features.py")
SPEC = importlib.util.spec_from_file_location("aligned_sharpe70_features", PATH)
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def fixture(width=3, periods=420):
    index = pd.bdate_range(end="2024-05-31", periods=periods)
    symbols = [f"S{i}" for i in range(width)]
    rng = np.random.default_rng(1729)
    prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001, .018, (periods, width)), axis=0)),
                          index=index, columns=symbols)
    meta = dict(dates=[str(index[-1].date())], symbols=[symbols])
    return prices, meta


class AlignedSharpeTests(unittest.TestCase):
    def compare(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def test_formulas_use_same_252_returns_and_sample_std(self):
        px, meta = fixture()
        raw, audit = f.build_features(px, meta)
        p = px.iloc[-295:-42].to_numpy()
        self.assertEqual(len(p), 253)
        r = p[1:]/p[:-1]-1
        log = np.log(p[1:]/p[:-1])
        np.testing.assert_allclose(raw["raw_arithmetic"][0], r.mean(axis=0)/r.std(axis=0, ddof=1)*np.sqrt(252))
        np.testing.assert_allclose(raw["raw_geometric"][0], (p[-1]/p[0]-1)/r.std(axis=0, ddof=1))
        np.testing.assert_allclose(raw["raw_log"][0], log.mean(axis=0)/log.std(axis=0, ddof=1)*np.sqrt(252))
        self.assertFalse(np.allclose(raw["raw_arithmetic"][0], r.mean(axis=0)/r.std(axis=0, ddof=0)*np.sqrt(252)))
        self.assertEqual(audit["window_returns"], 252)
        self.assertEqual(audit["std_ddof"], 1)

    def test_old_294_formula_controls_match_without_ranking(self):
        px, meta = fixture()
        raw, _ = f.build_features(px, meta)
        r = px.pct_change(fill_method=None)
        vol = r.rolling(252, min_periods=252).std(ddof=1).shift(42)
        old_mean = r.rolling(252, min_periods=252).mean().shift(42)/vol
        old_geo = (px.shift(42)/px.shift(294)-1)/vol
        np.testing.assert_allclose(raw["raw_arithmetic"][0], old_mean.iloc[-1]*np.sqrt(252), rtol=1e-12)
        np.testing.assert_allclose(raw["raw_geometric"][0], old_geo.iloc[-1], rtol=1e-12)

    def test_log_uses_log_std_not_old_simple_std(self):
        px, meta = fixture(width=1)
        r = np.tile([-.25, .35, -.07, .11], 63)
        p = np.r_[100., 100.*np.cumprod(1+r)]
        px.iloc[-295:-42, 0] = p
        raw, _ = f.build_features(px, meta)
        log = np.log1p(r)
        expected = log.mean()/log.std(ddof=1)*np.sqrt(252)
        wrong_denominator = log.mean()/r.std(ddof=1)*np.sqrt(252)
        self.assertAlmostEqual(raw["raw_log"][0, 0], expected)
        self.assertGreater(abs(raw["raw_log"][0, 0]-wrong_denominator), .01)
        self.assertGreater(abs(raw["raw_log"][0, 0]-raw["raw_arithmetic"][0, 0]), .01)

    def test_formation_endpoints_and_excluded_prices(self):
        px, meta = fixture()
        raw, audit = f.build_features(px, meta)
        record = audit["records"][0]
        self.assertEqual(record["formation_start"], str(px.index[-295].date()))
        self.assertEqual(record["first_return_date"], str(px.index[-294].date()))
        self.assertEqual(record["formation_end"], str(px.index[-43].date()))
        outside = px.copy()
        outside.iloc[-296] = np.nan
        outside.iloc[-42:] = np.nan
        other, _ = f.build_features(outside, meta)
        self.compare(raw, other)
        for position in (-295, -43):
            inside = px.copy()
            inside.iloc[position, 0] = np.nan
            changed, _ = f.build_features(inside, meta)
            self.assertTrue(all(np.isnan(value[0, 0]) for value in changed.values()))

    def test_all_recent42_values_and_availability_are_ignored(self):
        px, meta = fixture()
        raw, _ = f.build_features(px, meta)
        changed = px.copy()
        changed.iloc[-42:, 0] = np.nan
        changed.iloc[-42:, 1] = np.inf
        changed.iloc[-42:, 2] = 0.
        other, _ = f.build_features(changed, meta)
        self.compare(raw, other)

    def test_missing_or_nonpositive_entire_path_cannot_be_filled(self):
        px, meta = fixture(width=5)
        px.iloc[-150] = [np.nan, 0., -1., np.inf, -np.inf]
        raw, audit = f.build_features(px, meta)
        self.assertTrue(all(np.isnan(value).all() for value in raw.values()))
        self.assertEqual(audit["records"][0]["invalid_formation_price_cells"], 5)

    def test_flat_and_constant_nonzero_daily_returns_have_zero_denominator(self):
        px, meta = fixture(width=2)
        px.iloc[-295:-42, 0] = 100.
        px.iloc[-295:-42, 1] = np.exp2(np.arange(253, dtype=float))
        raw, audit = f.build_features(px, meta)
        self.assertTrue(all(np.isnan(value).all() for value in raw.values()))
        self.assertEqual(audit["records"][0]["zero_simple_std_cells"], 2)
        self.assertEqual(audit["records"][0]["zero_log_std_cells"], 2)

    def test_negative_momentum_is_not_a_new_gate(self):
        px, meta = fixture(width=1)
        r = np.tile([-.02, .001], 126)
        px.iloc[-295:-42, 0] = np.r_[100., 100*np.cumprod(1+r)]
        raw, audit = f.build_features(px, meta)
        self.assertTrue(all(np.isfinite(value).all() and (value < 0).all() for value in raw.values()))
        self.assertFalse(audit["trading_eligibility_modified"])

    def test_exact_warmup_294_index_and_unknown_symbols(self):
        px, meta = fixture(periods=295)
        full, _ = f.build_features(px, meta)
        self.assertTrue(all(np.isfinite(value).all() for value in full.values()))
        meta["dates"] = [str(px.index[-2].date())]
        short, audit = f.build_features(px.iloc[:-1], meta)
        self.assertTrue(all(np.isnan(value).all() for value in short.values()))
        self.assertEqual(audit["records"][0]["insufficient_history_cells"], 3)
        meta["dates"] = [str(px.index[-1].date())]
        meta["symbols"][0][-1] = "UNKNOWN"
        unknown, audit = f.build_features(px, meta)
        self.assertTrue(all(np.isnan(value[0, -1]) for value in unknown.values()))
        self.assertEqual(audit["records"][0]["unknown_symbol_cells"], 1)

    def test_prefix_and_future_perturbations(self):
        px, meta = fixture()
        positions = [294, 330, 380, 419]
        meta["dates"] = [str(px.index[t].date()) for t in positions]
        meta["symbols"] = [meta["symbols"][0].copy() for _ in positions]
        full, audit = f.build_features(px, meta)
        for count, t in enumerate(positions, 1):
            prefix_meta = dict(dates=meta["dates"][:count], symbols=meta["symbols"][:count])
            truncated, short_audit = f.build_features(px.iloc[:t+1], prefix_meta)
            self.compare(truncated, {name: value[:count] for name, value in full.items()})
            self.assertEqual(short_audit["records"], audit["records"][:count])
            corrupted = px.copy()
            corrupted.iloc[t+1:] = 1e200
            changed, _ = f.build_features(corrupted, prefix_meta)
            self.compare(truncated, changed)

    def test_all49_axes_no_eligibility_ranking_or_input_mutation(self):
        px, meta = fixture(width=49)
        before, old_meta = px.copy(), copy.deepcopy(meta)
        raw, _ = f.build_features(px, meta)
        self.assertTrue(all(value.shape == (1, 49) for value in raw.values()))
        meta["eligible"] = np.zeros((1, 49), bool)
        other, _ = f.build_features(px, meta)
        self.compare(raw, other)
        pd.testing.assert_frame_equal(px, before)
        self.assertEqual(meta["dates"], old_meta["dates"])
        self.assertEqual(meta["symbols"], old_meta["symbols"])
        reverse_meta = dict(dates=meta["dates"], symbols=[list(reversed(meta["symbols"][0]))])
        reversed_raw, _ = f.build_features(px, reverse_meta)
        self.compare(reversed_raw, {name: value[:, ::-1] for name, value in raw.items()})

    def test_audit_source_hash_json_and_definition_disclosure(self):
        px, meta = fixture()
        _, audit = f.build_features(px, meta)
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit["source_hashes"][str(PATH.resolve())], hashlib.sha256(PATH.read_bytes()).hexdigest())
        self.assertIn("matched control", audit["definition_status"]["raw_arithmetic"])
        self.assertIn("matched control", audit["definition_status"]["raw_geometric"])
        self.assertIn("210", audit["interpretation"])
        self.assertEqual(audit["last_used_offset"], -42)

    def test_validation(self):
        px, meta = fixture()
        for changed in (dict(meta, dates=["2099-01-01"]), dict(meta, symbols=[["S0", "S0"]]),
                        dict(meta, dates=[meta["dates"][0]]*2), dict(meta, dates=[None])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                f.build_features(px, changed)
        with self.assertRaises(ValueError):
            f.build_features(px.iloc[::-1], meta)
        with self.assertRaises(ValueError):
            f.build_features(px.set_axis(px.index.tz_localize("UTC")), meta)


if __name__ == "__main__":
    unittest.main()

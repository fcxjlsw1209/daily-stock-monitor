"""Synthetic maturity, identity, weighting, missing and prefix tests only."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import unittest

import numpy as np


PATH = Path(__file__).with_name("features.py")
SPEC = importlib.util.spec_from_file_location("company_zero70_features", PATH)
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def fixture():
    dates = ["2019-01-31", "2019-02-28", "2019-03-29", "2019-04-30"]
    ends = ["2019-02-28", "2019-03-29", "2019-04-30", "2019-05-31"]
    return dict(dates=dates, ends=ends, symbols=np.array(["A"]*4),
        y=np.array([.20, -.10, .30, -.05]), good=np.ones(4, bool),
        shape=np.zeros((4, 8)), state=np.zeros((4, 3)), signal=np.ones(4, int),
        event_dates=["2020-01-31"], event_ends=["2020-02-28"],
        event_symbols=np.array([["A", "B"]]), event_shape=np.zeros((1, 2, 8)),
        event_state=np.zeros((1, 2, 3)), event_signal=np.ones((1, 2), int),
        event_good=np.ones((1, 2), bool))


def decay(query, dates):
    age = (np.datetime64(query, "D")-np.asarray(dates, dtype="datetime64[D]")).astype(float)/30.4375
    return 2.**(-age/24.)


class OwnCompanyTests(unittest.TestCase):
    def compare(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def record(self, audit, model="company_mean_k6", symbol="A"):
        return next(r for r in audit["records"] if r["model"] == model and r["symbol"] == symbol)

    def test_exact_four_formulas_zero_prior_and_neff_audit(self):
        data = fixture()
        pred, audit = f.annual_forecasts(**data)
        self.assertEqual(set(pred), set(f.MODEL_NAMES) | {"available_"+name for name in f.MODEL_NAMES})
        w = decay(data["event_dates"][0], data["dates"])
        for kind in ("company_mean", "company_signal_context"):
            for k in (6, 12):
                name = kind+"_k"+str(k)
                self.assertAlmostEqual(pred[name][0, 0], w @ data["y"]/(w.sum()+k))
                self.assertTrue(pred["available_"+name][0, 0])
                self.assertEqual(pred[name][0, 1], 0.)
                self.assertFalse(pred["available_"+name][0, 1])
        record = self.record(audit)
        self.assertAlmostEqual(record["n_eff"], w.sum()**2/(w@w))
        self.assertAlmostEqual(record["own_weight"], w.sum()/(w.sum()+6))
        self.assertFalse(audit["uses_shared_ridge"])

    def test_one_sample_uses_mass_shrink_without_new_sample_threshold(self):
        data = fixture()
        data["good"][1:] = False
        pred, audit = f.annual_forecasts(**data)
        w = decay(data["event_dates"][0], data["dates"])[0]
        self.assertAlmostEqual(pred["company_mean_k6"][0, 0], w*.20/(w+6))
        self.assertNotAlmostEqual(pred["company_mean_k6"][0, 0], .20/7)
        self.assertTrue(pred["available_company_mean_k6"][0, 0])
        self.assertEqual(self.record(audit)["n_eff"], 1.)
        self.assertLess(abs(pred["company_mean_k12"][0, 0]), abs(pred["company_mean_k6"][0, 0]))

    def test_signal_and_context_exact_kernel(self):
        data = fixture()
        data["signal"] = np.array([1, 2, 1, 3])
        data["shape"][2] = .25
        data["state"][2] = .5
        pred, audit = f.annual_forecasts(**data)
        w = decay(data["event_dates"][0], data["dates"])[[0, 2]]
        w[1] *= np.exp(-.5*.25**2/.5**2-.5*.5**2)
        self.assertAlmostEqual(pred["company_signal_context_k6"][0, 0], w @ data["y"][[0, 2]]/(w.sum()+6))
        self.assertEqual(self.record(audit, "company_signal_context_k6")["observations"], 2)
        changed = copy.deepcopy(data)
        changed["y"][[1, 3]] = 999.
        other, _ = f.annual_forecasts(**changed)
        np.testing.assert_array_equal(pred["company_signal_context_k6"], other["company_signal_context_k6"])

    def test_annual_cutoff_is_holding_end_year_and_five_calendar_years(self):
        data = fixture()
        data.update(dates=["2014-12-30", "2014-12-31", "2019-11-29", "2019-12-31"],
            ends=["2015-01-30", "2015-01-30", "2019-12-31", "2020-01-31"],
            event_dates=["2019-12-31"], event_ends=["2020-01-31"])
        pred, audit = f.annual_forecasts(**data)
        annual = audit["annual"][0]
        self.assertEqual(annual["holding_end_year"], 2020)
        self.assertEqual(annual["cutoff"], "2019-12-31")
        self.assertEqual(annual["lookback_start"], "2014-12-31")
        self.assertEqual(annual["mature_eligible_rows"], 2)
        self.assertEqual(self.record(audit)["latest_label_end"], "2019-12-31")
        w = decay("2019-12-31", ["2014-12-31", "2019-11-29"])
        self.assertAlmostEqual(pred["company_mean_k6"][0, 0], w @ data["y"][[1, 2]]/(w.sum()+6))
        # Alter out-of-window and the query's own unmatured outcome.
        data["y"][[0, 3]] = 999.
        changed, _ = f.annual_forecasts(**data)
        self.compare(pred, changed)

    def test_first_signal_caps_annual_cutoff_before_december31(self):
        data = fixture()
        data.update(dates=["2016-09-30", "2016-10-31", "2016-11-30", "2016-12-30"],
            ends=["2016-10-31", "2016-11-30", "2016-12-31", "2017-01-31"],
            event_dates=["2016-12-30"], event_ends=["2017-01-31"])
        _, audit = f.annual_forecasts(**data)
        self.assertEqual(audit["annual"][0]["cutoff"], "2016-12-30")
        self.assertEqual(audit["annual"][0]["mature_eligible_rows"], 2)
        self.assertEqual(self.record(audit)["latest_label_end"], "2016-11-30")

    def test_same_security_only_and_history_good_required(self):
        data = fixture()
        data["symbols"] = np.array(["A", "B", "A", "A"])
        data["good"][3] = False
        pred, audit = f.annual_forecasts(**data)
        self.assertEqual(self.record(audit)["observations"], 2)
        altered = copy.deepcopy(data)
        altered["y"][[1, 3]] = -999.
        changed, _ = f.annual_forecasts(**altered)
        for name in f.MODEL_NAMES:
            self.assertEqual(pred[name][0, 0], changed[name][0, 0])
        self.assertNotEqual(pred["company_mean_k6"][0, 1], changed["company_mean_k6"][0, 1])

    def test_zero_and_missing_evidence_are_distinct(self):
        data = fixture()
        data["y"][:] = 0.
        pred, _ = f.annual_forecasts(**data)
        for name in f.MODEL_NAMES:
            self.assertEqual(pred[name][0, 0], 0.)
            self.assertTrue(pred["available_"+name][0, 0])
            self.assertEqual(pred[name][0, 1], 0.)
            self.assertFalse(pred["available_"+name][0, 1])
        data["good"][:] = False
        pred, _ = f.annual_forecasts(**data)
        self.assertTrue(all(not value.any() for value in pred.values()))

    def test_missing_signal_context_and_query_eligibility(self):
        data = fixture()
        for signal in (0, -1):
            data["event_signal"][0, 0] = signal
            pred, audit = f.annual_forecasts(**data)
            self.assertFalse(pred["available_company_signal_context_k6"][0, 0])
            self.assertTrue(pred["available_company_mean_k6"][0, 0])
            self.assertEqual(self.record(audit, "company_signal_context_k6")["unavailable_reason"], "no_current_signal")
        data["event_signal"][0, 0] = 1
        data["event_state"][0, 0, 0] = np.nan
        pred, _ = f.annual_forecasts(**data)
        self.assertFalse(pred["available_company_signal_context_k6"][0, 0])
        self.assertTrue(pred["available_company_mean_k6"][0, 0])
        data["event_good"][0, 0] = False
        pred, _ = f.annual_forecasts(**data)
        self.assertTrue(all(not value.any() for value in pred.values()))

    def test_missing_history_context_does_not_block_mean_or_impute_features(self):
        data = fixture()
        data["shape"][:] = np.nan
        pred, _ = f.annual_forecasts(**data)
        self.assertTrue(pred["available_company_mean_k6"][0, 0])
        self.assertFalse(pred["available_company_signal_context_k6"][0, 0])
        data["y"][:] = np.nan
        pred, _ = f.annual_forecasts(**data)
        self.assertTrue(all(not value.any() for value in pred.values()))

    def test_future_training_and_future_event_prefix_independence(self):
        data = fixture()
        data.update(dates=["2019-01-31", "2019-02-28", "2020-03-31", "2020-12-31"],
            ends=["2019-02-28", "2019-03-29", "2020-04-30", "2021-01-29"])
        pred, audit = f.annual_forecasts(**data)
        future = copy.deepcopy(data)
        future["y"][2:] = 999.
        future["shape"][2:] = 999.
        future["state"][2:] = 999.
        future["signal"][2:] = 3
        changed, changed_audit = f.annual_forecasts(**future)
        self.compare(pred, changed)
        self.assertEqual(audit, changed_audit)
        for key in ("event_symbols", "event_shape", "event_state", "event_signal", "event_good"):
            future[key] = np.concatenate([future[key], future[key]], axis=0)
        future["event_dates"] += ["2021-01-29"]
        future["event_ends"] += ["2021-02-26"]
        full, _ = f.annual_forecasts(**future)
        self.compare(pred, {name: value[:1] for name, value in full.items()})

    def test_far_context_zero_mass_is_unavailable_not_a_perfect_zero(self):
        data = fixture()
        data["event_shape"][0, 0] = 1e200
        pred, audit = f.annual_forecasts(**data)
        self.assertFalse(pred["available_company_signal_context_k6"][0, 0])
        self.assertEqual(pred["company_signal_context_k6"][0, 0], 0.)
        self.assertEqual(self.record(audit, "company_signal_context_k6")["unavailable_reason"], "zero_weight_matching_history")

    def test_inputs_unchanged_audit_json_source_and_no_target_reclipping(self):
        data = fixture()
        data["y"][:] = 2.  # Contract is inherited input: this module never clips it.
        before = copy.deepcopy(data)
        pred, audit = f.annual_forecasts(**data)
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        w = decay(data["event_dates"][0], data["dates"])
        self.assertAlmostEqual(pred["company_mean_k6"][0, 0], 2*w.sum()/(w.sum()+6))
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit["source_hashes"][str(PATH.resolve())], hashlib.sha256(PATH.read_bytes()).hexdigest())

    def test_reject_wrong_shapes_duplicate_identity_and_invalid_types(self):
        changes = [dict(shape=np.zeros((4, 7))), dict(event_good=np.ones((1, 2))),
            dict(signal=np.ones(4)), dict(signal=np.full(4, 4)),
            dict(event_symbols=np.array([["A", "A"]])),
            dict(ends=["2019-01-31"]*4),
            dict(dates=["2019-01-31"]*4)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                f.annual_forecasts(**dict(fixture(), **change))


if __name__ == "__main__":
    unittest.main()

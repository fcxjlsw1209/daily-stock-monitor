"""Synthetic122-month accounting and binding tests; no real data are loaded."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_financial_coverage150_outcome_review as m


def order(day, end, names):
    return dict(date=str(pd.Timestamp(day).date()), signal_date=str(pd.Timestamp(day).date()),
                end=str(pd.Timestamp(end).date()), symbols=list(names), scores=[.2]*len(names))


def synthetic_outputs():
    days = pd.bdate_range("2016-01-04", periods=245)
    x = np.arange(len(days))
    specs = [("AAPL", 10, .001, 4), ("AMZN", 20, .0007, 7), ("CSCO", 30, .0013, 6)]
    close = pd.DataFrame({s: base*np.exp(rate*x+.007*np.sin(x/freq))
                          for s, base, rate, freq in specs}, index=days)
    opening = close*np.array([.999, 1.001, 1.0002])
    names = ["AAPL", "AMZN", "CSCO"]+["S"+str(i) for i in range(29)]+["ETF"+str(i) for i in range(17)]
    meta = dict(dates=[str(d.date()) for d in days[:-1:2]], ends=[str(d.date()) for d in days[2::2]],
                symbols=[names.copy() for _ in range(122)], etfs=names[-17:])
    orders = {name: [order(day, end, ["AAPL", "CSCO"] if k and i % (k+2) == 0 else ["AAPL", "AMZN"])
                     for i, (day, end) in enumerate(zip(meta["dates"], meta["ends"]))]
              for k, name in enumerate(m.NAMES)}
    data, results, navs, open_navs = {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta["dates"][0], meta["ends"][52]),
                  original2021_2026=(meta["dates"][53], meta["ends"][-1]),
                  training2016_2023=(meta["dates"][0], meta["ends"][87]),
                  later34=(meta["dates"][88], meta["ends"][-1]))
    for name in m.NAMES:
        results[name] = {}
        for execution, cost, key in m.PATHS:
            met, nav, trades, holdings = m.replay(close, opening, orders[name], days, execution, cost)
            prefix = name+"_"+key
            results[name][key] = dict(metrics=met, slices={
                k: m.h.slice_stat(days, nav, *span) for k, span in bounds.items()})
            data[prefix+"_DAILY.json"] = [dict(date=str(d.date()), equity=float(v)) for d, v in zip(days, nav)]
            data[prefix+"_TRADES.json"], data[prefix+"_HOLDINGS.json"] = trades, holdings
            if key == "same_close":
                navs[name] = nav
            if key == "next_open_cost":
                open_navs[name] = nav
    rows = [dict(name=name, **m.h.measures(days, navs[name], meta)) for name in m.NAMES]
    opened = [dict(name=name, **m.h.measures(days, open_navs[name], meta)) for name in m.NAMES]
    selection = m.select(rows, opened)
    arrays = dict(earnings_original=np.full((122, 49), np.nan),
                  earnings_expanded=np.full((122, 49), np.nan),
                  original_available=np.zeros((122, 49), bool),
                  expanded_available=np.zeros((122, 49), bool))
    arrays["earnings_original"][:, 0] = .2
    arrays["earnings_expanded"][:, :2] = [.2, -.3]
    arrays["original_available"][:, 0] = True
    arrays["expanded_available"][:, :2] = True
    cells = [dict(signal_index=i, pool_index=j, signal_date=day, symbol=symbol,
                  available=j == 1, reason="available" if j == 1 else "not_added",
                  earnings_bounded=-.3 if j == 1 else None)
             for i, (day, pool) in enumerate(zip(meta["dates"], meta["symbols"]))
             for j, symbol in enumerate(pool)]
    feature = dict(cells=cells, original_cells=122, expanded_cells=244, added_cells=122,
                   original_input_unchanged=True, new_model_fits=0, returns_computed=False,
                   reasons=dict(Counter(c["reason"] for c in cells)))
    data.update({
        "RESULTS.json": results, "TRAIN_TRIALS.json": rows, "TRAIN_OPEN_TRIALS.json": opened,
        "FROZEN.json": dict(selection=selection),
        "WALKFORWARD.json": m.walkforward(days, navs, open_navs, meta), "FEATURE_AUDIT.json": feature,
        "VALIDATION.json": dict(selection=selection, new_model_fits=0, no_observer_writes=True,
                               no_intramonth_trades=True, independent_outcome_review_pending=True,
                               pristine_oos=False)})
    return close, opening, orders, meta, data, arrays


class SelectionTests(unittest.TestCase):
    def groups(self):
        rows = [dict(name=name, training=dict(cagr=.5, sharpe=2., max_drawdown=-.2),
                     early=dict(cagr=.4), recent=dict(cagr=.6)) for name in m.NAMES]
        return rows, deepcopy(rows)

    def winner(self):
        rows, opened = self.groups()
        rows[2]["training"]["cagr"] = opened[2]["training"]["cagr"] = .6
        return rows, opened

    def test_ties_retain_incumbent(self):
        self.assertEqual(m.select(*self.groups())["primary"], m.NAMES[0])

    def test_original_control_cannot_be_promoted(self):
        rows, opened = self.groups()
        rows[1]["training"]["cagr"] = opened[1]["training"]["cagr"] = 9.
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_primary_wins_both_controls_both_paths(self):
        self.assertTrue(m.select(*self.winner())["admitted"])

    def test_open_incumbent_noninferiority(self):
        rows, opened = self.winner()
        opened[0]["training"]["cagr"] = .6
        self.assertTrue(m.select(rows, opened)["admitted"])

    def test_open_original_tie_rejected(self):
        rows, opened = self.winner()
        opened[1]["training"]["cagr"] = .6
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_both_closed_ties_rejected(self):
        for i in range(2):
            rows, opened = self.winner()
            rows[i]["training"]["cagr"] = .6
            self.assertFalse(m.select(rows, opened)["admitted"])

    def test_each_risk_guard_each_path_each_control(self):
        for mode in (0, 1):
            for i in (0, 1):
                for section, key, tol in (("training", "sharpe", .05),
                                          ("training", "max_drawdown", .03),
                                          ("early", "cagr", .025), ("recent", "cagr", .025)):
                    pair = self.winner()
                    pair[mode][i][section][key] = pair[mode][2][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)["admitted"])

    def test_exact_risk_boundaries_inclusive(self):
        rows, opened = self.winner()
        for group in (rows, opened):
            for section, key, tol in (("training", "sharpe", .05),
                                     ("training", "max_drawdown", .03),
                                     ("early", "cagr", .025), ("recent", "cagr", .025)):
                group[2][section][key] = group[0][section][key]-tol
        self.assertTrue(m.select(rows, opened)["admitted"])

    def test_tiny_strict_advantage_rejected(self):
        rows, opened = self.winner()
        rows[2]["training"]["cagr"] = .5+5e-13
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_invalid_metrics_fail(self):
        for value in (None, True, "2", float("nan"), float("inf")):
            rows, opened = self.winner()
            opened[2]["training"]["sharpe"] = value
            self.assertFalse(m.select(rows, opened)["admitted"])
        rows, opened = self.winner()
        del rows[2]["early"]["cagr"]
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_later34_does_not_select(self):
        rows, opened = self.groups()
        rows[2]["later34"] = {"cagr": 999}
        self.assertFalse(m.select(rows, opened)["admitted"])

    def test_exact_arm_order(self):
        rows, opened = self.winner()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError):
            m.select(rows, opened)


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_outputs()

    def setUp(self):
        self.px, self.op, self.orders, self.meta, self.data, self.arrays = deepcopy(self.fixture)

    def check(self):
        return m.check_outcomes(self.px, self.op, self.orders, self.meta, self.data.__getitem__)

    def matrix(self):
        return m.verify_feature_matrices(self.arrays, self.data["FEATURE_AUDIT.json"], self.meta)

    def test_nine_paths_and_366_orders(self):
        out = self.check()
        self.assertTrue(out["passed"])
        self.assertEqual(out["execution_paths_verified"], 9)
        self.assertEqual(out["total_monthly_orders_verified"], 366)
        self.assertEqual(len(out["independent_monthly_comparison"]), 122)

    def test_nav_tamper(self):
        self.data[m.NAMES[2]+"_next_open_cost_DAILY.json"][4]["equity"] += 10
        with self.assertRaises(AssertionError):
            self.check()

    def test_trade_cost_tamper(self):
        self.data[m.NAMES[0]+"_next_open_cost_TRADES.json"][0]["cost"] += 1
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_holdings_tamper(self):
        self.data[m.NAMES[1]+"_same_close_HOLDINGS.json"][0]["symbols"] = []
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_wrong_open_execution_date(self):
        self.data[m.NAMES[0]+"_next_open_zero_TRADES.json"][0]["date"] = self.meta["dates"][0]
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_single_name_order_rejected(self):
        self.orders[m.NAMES[0]][0].update(symbols=["AAPL"], scores=[.2])
        with self.assertRaises(ValueError):
            m.validate_orders(self.px, self.op, self.orders, self.meta)

    def test_frozen_selection_tamper(self):
        self.data["FROZEN.json"]["selection"]["primary"] = "forged"
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_training_open_metric_tamper(self):
        self.data["TRAIN_OPEN_TRIALS.json"][2]["training"]["sharpe"] += .01
        with self.assertRaises(ValueError):
            self.check()

    def test_walkforward_selection_tamper(self):
        entry = self.data["WALKFORWARD.json"]["folds"][0]["selection"]
        entry["admitted"] = not entry["admitted"]
        with self.assertRaises(ValueError):
            self.check()

    def test_new_model_or_noncausal_diagnostic_scope(self):
        for key, value in (("new_model_fits", 1), ("new_model_fits", False),
                           ("pristine_oos", True), ("no_intramonth_trades", False)):
            original = deepcopy(self.data["VALIDATION.json"])
            self.data["VALIDATION.json"][key] = value
            with self.assertRaises(ValueError):
                self.check()
            self.data["VALIDATION.json"] = original

    def test_distinct_masks_valid_and_four_arrays_checked(self):
        self.assertEqual(self.matrix(), 4)
        self.assertFalse(np.array_equal(self.arrays["original_available"],
                                       self.arrays["expanded_available"]))

    def test_old_earnings_cannot_change(self):
        self.arrays["earnings_expanded"][0, 0] += .01
        with self.assertRaises(ValueError):
            self.matrix()

    def test_old_coverage_cannot_disappear(self):
        self.arrays["expanded_available"][0, 0] = False
        self.arrays["earnings_expanded"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.matrix()

    def test_no_unrelated_new_coverage(self):
        self.arrays["expanded_available"][0, 3] = True
        self.arrays["earnings_expanded"][0, 3] = .1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_value_matches_cell(self):
        self.arrays["earnings_expanded"][0, 1] += .01
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_value_must_be_bounded(self):
        self.arrays["earnings_expanded"][0, 1] = 1.1
        self.data["FEATURE_AUDIT.json"]["cells"][1]["earnings_bounded"] = 1.1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_new_issuer_not_preexisting(self):
        self.arrays["earnings_original"][0, 1] = -.3
        self.arrays["original_available"][0, 1] = True
        with self.assertRaises(ValueError):
            self.matrix()

    def test_unavailable_pseudo_value_rejected(self):
        self.data["FEATURE_AUDIT.json"]["cells"][2]["earnings_bounded"] = .1
        with self.assertRaises(ValueError):
            self.matrix()

    def test_etf_cannot_have_either_financial_hand(self):
        self.arrays["earnings_original"][0, 32] = self.arrays["earnings_expanded"][0, 32] = .1
        self.arrays["original_available"][0, 32] = self.arrays["expanded_available"][0, 32] = True
        with self.assertRaises(ValueError):
            self.matrix()

    def test_exact_cell_axes_and_population(self):
        for mutate in ("drop", "reorder"):
            feature = deepcopy(self.data["FEATURE_AUDIT.json"])
            if mutate == "drop":
                feature["cells"].pop()
            else:
                feature["cells"][0], feature["cells"][1] = feature["cells"][1], feature["cells"][0]
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(self.arrays, feature, self.meta)

    def test_coverage_counts_and_reasons(self):
        for key in ("original_cells", "expanded_cells", "added_cells"):
            feature = deepcopy(self.data["FEATURE_AUDIT.json"])
            feature[key] += 1
            with self.assertRaises(ValueError):
                m.verify_feature_matrices(self.arrays, feature, self.meta)
        self.data["FEATURE_AUDIT.json"]["reasons"] = {}
        with self.assertRaises(ValueError):
            self.matrix()

    def test_no_producer_modules_imported(self):
        for name in ("monthly_financial_coverage150_runner", "monthly_financial_coverage150_factor",
                     "monthly_financial_coverage150_selection"):
            self.assertNotIn(name, sys.modules)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root/m.STUDY
        self.folder.mkdir(parents=True)
        self.audit_time = datetime(2026, 10, 20, tzinfo=timezone.utc)
        self.plan_sha = "a"*64

    def put(self, path, value):
        p = self.root/path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def marker(self, **changes):
        data = dict(plan_sha256=self.plan_sha, started_utc="2026-10-19T00:00:00+00:00")
        data.update(changes)
        self.put(m.STUDY+"/DIAGNOSE_STARTED.json", data)

    def chain(self, source_change=None, missing_ref=None):
        source = self.put("synthetic_review.py", {})
        legacy_orders = self.put("legacy/PRETRAIN_ORDERS.json", dict(earnings_control=[]))
        legacy_cache = self.put("legacy/CACHE.json", {"PRETRAIN_ORDERS.json": self.sha(legacy_orders)})
        legacy_refs = dict(cache=dict(path=str(legacy_cache), sha256=self.sha(legacy_cache)))
        legacy_files = [legacy_orders, legacy_cache]
        for key in ("plan", "completion", "outcome"):
            path = self.put("legacy/"+key+".json", {})
            legacy_refs[key] = dict(path=str(path), sha256=self.sha(path))
            legacy_files.append(path)
        dep = self.put("source/dep.json", {})
        source_plan = self.put("source/PLAN.json", {})
        policy = self.put("source/POLICY.json", {})
        review_data = dict(passed=True, plan_sha256=self.sha(source_plan))
        if source_change and source_change[0] == "review":
            review_data[source_change[1]] = source_change[2]
        review_path = self.put("source/REVIEW.json", review_data)
        source_inputs = {str(p): self.sha(p) for p in (dep, source_plan, policy, review_path)}
        links = dict(policy_sha256=self.sha(policy), source_plan_sha256=self.sha(source_plan),
                     independent_review_sha256=self.sha(review_path))
        source_objects = dict(dataset=deepcopy(links),
                              admission=dict(passed=True, input_hashes=source_inputs, **links))
        if source_change and source_change[0] in source_objects:
            source_objects[source_change[0]][source_change[1]] = source_change[2]
        pins = {str(p): self.sha(p) for p in (source, *legacy_files, dep, source_plan)}
        source_files = dict(dataset=self.put("source/DATASET.json", source_objects["dataset"]),
                            coverage=self.put("source/COVERAGE.json", {}),
                            admission=self.put("source/ADMISSION.json", source_objects["admission"]),
                            policy=policy, review=review_path)
        source_files["manifest"] = self.put("source/MANIFEST.json", dict(
            input_hashes=source_inputs,
            outputs={name+".json": self.sha(source_files[key]) for name, key in
                     (("DATASET", "dataset"), ("COVERAGE", "coverage"), ("ADMISSION", "admission"))}))
        source_refs = {k: dict(path=str(p), sha256=self.sha(p)) for k, p in source_files.items()}
        if missing_ref:
            (source_refs if missing_ref[0] == "coverage" else legacy_refs).pop(missing_ref[1])
        inputs = {k: self.put("input/"+k+".json", {}) for k in m.INPUTS}
        inputs["prices"] = self.put(m.PRICE, {})
        inputs["original_orders"] = self.put("input/orders.json", [])
        self.proposal = dict(round=150, arms=list(m.NAMES), primary_challenger=m.NAMES[2], dose=.01,
                             sources=dict(legacy149=legacy_refs, coverage_source=source_refs),
                             selection_contract=dict(later34_for_selection=False),
                             measurement=dict(invalid_latest_blocks_fallback=True, intramonth_trades=False))
        inputs["proposal"] = self.put("input/proposal.json", self.proposal)
        for path in list(inputs.values())+list(source_files.values()):
            pins[str(path)] = self.sha(path)
        self.plan = dict(round=150, status="REGISTERED", arms=list(m.NAMES),
                         primary_challenger=m.NAMES[2], output_directory=str(self.folder), dose=.01,
                         runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                         inputs={k: dict(path=str(p), sha256=self.sha(p)) for k, p in inputs.items()},
                         hashes=pins, raw_source_hashes={str(inputs["prices"]): self.sha(inputs["prices"])},
                         independent_prepare_source_hashes={str(source): self.sha(source)},
                         legacy149=legacy_refs, coverage_source=source_refs,
                         actual_feature_calculations_before_registration=0,
                         actual_returns_before_registration=0, actual_model_fits_before_registration=0)
        self.put(m.STUDY+"/PLAN.json", self.plan)
        self.plan_sha = self.sha(self.folder/"PLAN.json")
        for name in m.CACHE_FILES:
            self.put(m.STUDY+"/"+name, {"synthetic": name})
        self.put(m.STUDY+"/PRETRAIN_ORDERS.json", {name: [] for name in m.NAMES})
        self.put(m.STUDY+"/CACHE.json", {name: self.sha(self.folder/name) for name in m.CACHE_FILES})
        self.prep = dict(passed=True, stage="prepare", **dict.fromkeys(m.PREPARE_CHECKS, True),
                         orders_checked=366, source_cells_checked=5978, stock_cells_checked=3904,
                         monthly_label_models_fit=0, producer_modules_imported=False,
                         returns_or_nav_computed=False, plan_sha256=self.plan_sha,
                         cache_sha256=self.sha(self.folder/"CACHE.json"),
                         orders_sha256=self.sha(self.folder/"PRETRAIN_ORDERS.json"),
                         source_hashes={str(source): self.sha(source)},
                         artifact_hashes={str(self.folder/name): self.sha(self.folder/name)
                                          for name in (*m.CACHE_FILES, "PLAN.json", "CACHE.json")})
        self.put(m.STUDY+"/INDEPENDENT_PREPARE.json", self.prep)
        for name in ("TRAIN_TRIALS.json", "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json"):
            self.put(m.STUDY+"/"+name, {})
        self.put(m.STUDY+"/FROZEN.json", dict(
            hashes={name: self.sha(self.folder/name) for name in m.FROZEN_FILES},
            pristine_oos=False, all_diagnostic_orders_frozen=True))
        self.marker()
        mock = patch.object(m, "source_files", return_value=[source])
        mock.start()
        self.addCleanup(mock.stop)

    def check(self):
        return m.bindings(m.Reader(self.root), self.plan_sha, self.audit_time)

    def test_complete_binding_chain(self):
        self.chain()
        self.assertEqual(self.check(), self.folder)

    def test_source_refs_must_be_complete(self):
        self.chain(missing_ref=("coverage", "review"))
        with self.assertRaisesRegex(ValueError, "Exact legacy"):
            self.check()

    def test_legacy_refs_must_include_completion(self):
        self.chain(missing_ref=("legacy", "completion"))
        with self.assertRaisesRegex(ValueError, "Exact legacy"):
            self.check()

    def test_source_review_must_pass_even_with_valid_file_hash(self):
        self.chain(source_change=("review", "passed", False))
        with self.assertRaisesRegex(ValueError, "Independent source"):
            self.check()

    def test_dataset_policy_link_must_match_ref(self):
        self.chain(source_change=("dataset", "policy_sha256", "b"*64))
        with self.assertRaisesRegex(ValueError, "Source policy"):
            self.check()

    def test_admission_review_link_must_match_ref(self):
        self.chain(source_change=("admission", "independent_review_sha256", "b"*64))
        with self.assertRaisesRegex(ValueError, "Independent source"):
            self.check()

    def test_source_plan_link_consistency(self):
        self.chain(source_change=("admission", "source_plan_sha256", "b"*64))
        with self.assertRaisesRegex(ValueError, "Source policy"):
            self.check()

    def test_plan_digest_syntax_before_read(self):
        for value in ("", None, "A"*64, "a"*63, "x"*64):
            with self.assertRaises(ValueError):
                m.bindings(m.Reader(self.root), value)

    def test_diagnostic_marker_missing(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            m.verify_diagnostic_start(m.Reader(self.root), self.folder, self.plan_sha, self.audit_time)

    def test_diagnostic_marker_time_and_binding(self):
        self.marker()
        m.verify_diagnostic_start(m.Reader(self.root), self.folder, self.plan_sha, self.audit_time)
        for change in (dict(plan_sha256="b"*64), dict(started_utc="2999-01-01T00:00:00Z"),
                       dict(started_utc="2026-10-19T00:00:00"), dict(extra=True)):
            self.marker(**change)
            with self.assertRaises(ValueError):
                m.verify_diagnostic_start(m.Reader(self.root), self.folder, self.plan_sha, self.audit_time)

    def test_reader_rejects_mutation(self):
        self.marker()
        reader = m.Reader(self.root)
        path = self.folder/"DIAGNOSE_STARTED.json"
        reader.sha(path)
        self.marker(plan_sha256="b"*64)
        with self.assertRaises(ValueError):
            reader.sha(path)

    def test_new_prepare_guarantees_required(self):
        self.chain()
        del self.prep["latest_catalog_blocking_independently_verified"]
        self.put(m.STUDY+"/INDEPENDENT_PREPARE.json", self.prep)
        with self.assertRaisesRegex(ValueError, "Independent prepare"):
            self.check()

    def test_exact_prepare_counts(self):
        self.chain()
        for key, value in (("orders_checked", 488), ("source_cells_checked", 1),
                           ("stock_cells_checked", 0), ("monthly_label_models_fit", False)):
            bad = deepcopy(self.prep)
            bad[key] = value
            self.put(m.STUDY+"/INDEPENDENT_PREPARE.json", bad)
            with self.assertRaisesRegex(ValueError, "Exact prepare count"):
                self.check()

    def test_prepare_artifact_omission(self):
        self.chain()
        del self.prep["artifact_hashes"][str(self.folder/"PRICE_PARITY.json")]
        self.put(m.STUDY+"/INDEPENDENT_PREPARE.json", self.prep)
        with self.assertRaisesRegex(ValueError, "Complete prepare artifacts"):
            self.check()

    def test_cache_tamper(self):
        self.chain()
        self.put(m.STUDY+"/SCORES.npz", {"tampered": True})
        with self.assertRaises(ValueError):
            self.check()

    def test_source_reference_tamper(self):
        self.chain()
        self.put("source/DATASET.json", {"tampered": True})
        with self.assertRaises(ValueError):
            self.check()

    def test_source_dependency_tamper(self):
        self.chain()
        self.put("source/dep.json", {"tampered": True})
        with self.assertRaises(ValueError):
            self.check()

    def test_legacy_control_order_tamper(self):
        self.chain()
        self.put("legacy/PRETRAIN_ORDERS.json", {"earnings_control": ["tampered"]})
        with self.assertRaises(ValueError):
            self.check()

    def test_open_training_frozen_binding_required(self):
        self.chain()
        p = self.folder/"FROZEN.json"
        frozen = json.loads(p.read_text())
        del frozen["hashes"]["TRAIN_OPEN_TRIALS.json"]
        self.put(m.STUDY+"/FROZEN.json", frozen)
        with self.assertRaisesRegex(ValueError, "Exact frozen"):
            self.check()

    def test_source_files_exclude_producer(self):
        paths = m.source_files(self.root)
        self.assertIn(self.root/"work/test_monthly_financial_coverage150_outcome_review.py", paths)
        self.assertFalse(any("150_runner" in str(p) or "150_factor" in str(p) for p in paths))


if __name__ == "__main__":
    write_review = "--write-review" in sys.argv
    if write_review:
        sys.argv.remove("--write-review")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log = stream.getvalue()
    print(log)
    if write_review:
        if not result.wasSuccessful():
            raise SystemExit("No receipt for failing review")
        directory = m.ROOT/m.STUDY
        if (directory/"PLAN.json").exists():
            raise SystemExit("Cannot write prospective review after registration")
        directory.mkdir(parents=True, exist_ok=True)
        logpath = directory/"OUTCOME_REVIEW_TESTS.log"
        receiptpath = m.ROOT/"work/monthly_financial_coverage150_outcome_prereview.json"
        if logpath.exists() or receiptpath.exists():
            raise SystemExit("Preserve existing prospective evidence")
        receipt = m.synthetic_receipt(result)
        proposal_path = m.ROOT/"work/monthly_financial_coverage150_proposal.json"
        proposal_sha = hashlib.sha256(proposal_path.read_bytes()).hexdigest()
        if proposal_sha != "32d3e2d02adfd47b07e8c2bf09dd16e4499220d5392278ba6ad6342a92dc11c4":
            raise SystemExit("Final prospective proposal changed")
        receipt["round"] = 150
        receipt["proposal_sha256"] = proposal_sha
        receipt["reviewed_contract_hashes"] = {
            str(m.ROOT/"work"/name): hashlib.sha256((m.ROOT/"work"/name).read_bytes()).hexdigest()
            for name in ("monthly_financial_coverage150_runner.py",
                         "monthly_financial_coverage150_factor.py",
                         "monthly_financial_coverage150_selection.py",
                         "monthly_financial_coverage150_proposal.json")}
        with logpath.open("x") as stream:
            stream.write(log)
        receipt["test_log"] = dict(path=str(logpath), sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
        with receiptpath.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)

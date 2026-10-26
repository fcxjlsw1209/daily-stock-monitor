import unittest
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import numpy as np
import monthly_stock_comp149_runner as r


class Adapter:
    def __init__(self):
        self.calls = 0

    def score_and_choose(self, *args, **kwargs):
        self.calls += 1
        assert args[0] == "incumbent66" and args[1] == 0.
        assert kwargs["boundary_selector"] == "original"
        return [], kwargs["pre_score"][:args[5]].copy()


def fixture():
    adapter = Adapter()
    arrays = dict(baseline=np.zeros((2, 49)), eligible=np.ones((2, 49), bool),
                  effects=np.zeros((2, 49)))
    names = ["STOCK"+str(i) for i in range(32)] + ["ETF"+str(i) for i in range(17)]
    meta = dict(symbols=[names, names], etfs=names[32:])
    return arrays, meta, SimpleNamespace(adapter=adapter, selector="original")


def row(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


def source_fixture():
    from test_monthly_stock_comp149_factor import candidate
    from test_monthly_cash_quality146_factor import event
    events = []
    for i in range(44):
        e = event(identifier="synthetic:"+str(i))
        e["cash_quality_admission"].update(admitted=i < 31, reasons=[] if i < 31 else ["old_hold"])
        e["inherited_rejection_codes"] = [] if i < 31 else ["original_rejection"]
        events.append(e)
    original = dict(schema_version=1, method=r.factor.METHOD, events=events)
    original_receipt = dict(events=44, admitted_events=31, held_events=13,
        admitted_event_ids=[e["event_id"] for e in events[:31]],
        holds=[dict(event_id=e["event_id"], **e["cash_quality_admission"]) for e in events[31:]])
    new = [candidate(e) for e in events]
    for i,e in enumerate(new):
        e["stock_compensation_admission"].update(admitted=i < 30,
            reasons=[] if i < 30 else ["new_or_inherited_hold"])
    data = dict(schema_version=1, method=r.factor.MEASUREMENT, events=new)
    refs = {key: dict(path=path, sha256=letter*64)
            for (key,path),letter in zip(r.SBC_REFS.items(), "abcd")}
    receipt = dict(passed=True, events=44, admitted_events=30, held_events=14,
        model_fits=0, factor_calculations=0, return_calculations=0,
        admitted_event_ids=[e["event_id"] for e in new[:30]],
        holds=[dict(event_id=e["event_id"], **e["stock_compensation_admission"]) for e in new[30:]],
        data_sha256=refs["sbc_dataset"]["sha256"], policy_sha256=refs["sbc_policy"]["sha256"])
    policy = dict(purpose=r.factor.MEASUREMENT, definition=r.factor.DEFINITION)
    for key in ("same_current_report","latest_event_before_validity",
                "invalid_latest_blocks_older_fallback","preserve_all_13_original_holds"):
        policy[key] = True
    for key in ("first_public_version_verified","complete_correction_history_verified",
                "certified_daily_security_master","financial_ratios_permitted",
                "strategy_registration_permitted"):
        policy[key] = False
    review = dict(passed=True, admitted_events=30, held_events=14)
    return data, receipt, policy, review, original, original_receipt, refs


class RunnerTests(unittest.TestCase):
    def test_proposal_binds_controls_and_evaluation(self):
        p = dict(round=149, arms=list(r.NAMES), primary_challenger=r.NAMES[3],
                 dose=r.DOSE,
                 selection_contract=r.selection_contract(), measurement=r.measurement_contract(),
                 status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
                 actual_new_factors=0, actual_new_orders=0, actual_new_returns=0)
        r.validate_proposal(p)
        for key, value in (("dose", .02), ("primary_challenger", r.NAMES[1]),
                           ("selection_contract", {})):
            bad = dict(p, **{key: value})
            with self.assertRaises(ValueError):
                r.validate_proposal(bad)

    def test_exact_continuous_adjustment_and_one_episode(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, :3] = [-.8, 0., .7]
        a["eligible"][0, 2] = False
        original = hand.copy()
        _, score = r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})
        self.assertEqual(score[0, 0], -.008)
        self.assertEqual(score[0, 1], 0.)
        self.assertEqual(score[0, 2], 0.)
        self.assertEqual(b.adapter.calls, 1)
        np.testing.assert_array_equal(hand, original)

    def test_missing_exact_fallback(self):
        a, m, b = fixture()
        pre = np.arange(98, dtype=float).reshape(2, 49) / 100
        _, score = r.corrected(pre, np.full((2, 49), np.nan), a, m, 2, b, {})
        self.assertEqual(score.tobytes(), pre.tobytes())

    def test_zero_prefix(self):
        a, m, b = fixture()
        _, score = r.corrected(np.zeros((0, 49)), np.full((2, 49), np.nan), a, m, 0, b, {})
        self.assertEqual(score.shape, (0, 49))

    def test_infinity_or_unbounded_is_rejected(self):
        for value in (np.inf, -np.inf, 1.001, -1.001, 2.):
            a, m, b = fixture()
            h = np.full((2, 49), np.nan)
            h[0, 0] = value
            with self.assertRaises(ValueError):
                r.corrected(np.zeros((2, 49)), h, a, m, 2, b, {})

    def test_rounded_unit_endpoint_within_fixed_cap(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, :2] = [-1., 1.]
        _, score = r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})
        self.assertEqual(list(score[0, :2]), [-.01, .01])

    def test_future_values_do_not_affect_prefix(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[1] = np.inf
        _, score = r.corrected(np.zeros((1, 49)), hand, a, m, 1, b, {})
        np.testing.assert_array_equal(score, np.zeros((1, 49)))

    def test_etf_feature_rejected(self):
        a, m, b = fixture()
        hand = np.full((2, 49), np.nan)
        hand[0, 32] = 0.
        with self.assertRaises(ValueError):
            r.corrected(np.zeros((2, 49)), hand, a, m, 2, b, {})

    def test_all_guards_required(self):
        close = [row(n, cagr=.4 + i*.01) for i, n in enumerate(r.NAMES)]
        opened = [row(n, cagr=.4 + i*.01) for i, n in enumerate(r.NAMES)]
        self.assertEqual(r.select(close, opened)["primary"], r.NAMES[3])
        opened[2] = row(r.NAMES[2], cagr=.5)
        self.assertEqual(r.select(close, opened)["primary"], r.NAMES[0])

    def test_no_portfolio_change_cannot_validate_mechanism(self):
        rows = [row(n) for n in r.NAMES]
        self.assertFalse(r.select(rows, rows)["admitted"])

    def test_risk_and_segment_regression_rejected(self):
        for field, value in (("sharpe", 1.44), ("dd", -.281),
                             ("early", .274), ("recent", .474)):
            rows = [row(n) for n in r.NAMES]
            rows[-1] = row(r.NAMES[-1], cagr=.9, **{field: value})
            self.assertFalse(r.select(rows, rows)["admitted"])

    def test_arm_order_guard(self):
        rows = [row(n) for n in reversed(r.NAMES)]
        with self.assertRaises(ValueError):
            r.select(rows, rows)

    def test_pre_slices_every_prefix_and_preserves_full_bytes(self):
        full = np.arange(122*49, dtype=float).reshape(122, 49)/100
        before = full.tobytes()
        backend = object.__new__(r.Backend)
        arrays = {"synthetic": True}
        with patch.object(r.previous.Backend, "pre", autospec=True, return_value=full) as prior:
            for n in range(123):
                prefix = backend.pre(arrays, n)
                self.assertEqual(prefix.shape, (n, 49))
                self.assertEqual(prefix.tobytes(), full[:n].tobytes())
                self.assertFalse(np.shares_memory(prefix, full))
                prior.assert_called_with(backend, arrays, n)
                if n:
                    prefix[0, 0] = -999.
                self.assertEqual(full.tobytes(), before)
            self.assertEqual(prior.call_count, 123)

    def test_pre_rejects_nonfull_or_nonfloat_parent_contract(self):
        backend = object.__new__(r.Backend)
        for value in (np.zeros((121, 49)), np.zeros((122, 48)),
                      np.zeros((122, 49), dtype=int), np.zeros((0, 49))):
            with patch.object(r.previous.Backend, "pre", return_value=value):
                with self.assertRaises(ValueError):
                    backend.pre({}, 1)

    def test_pre_rejects_invalid_prefix_before_parent_call(self):
        backend = object.__new__(r.Backend)
        with patch.object(r.previous.Backend, "pre") as prior:
            for n in (-1, 123, True, 1., "1", None):
                with self.assertRaises(ValueError):
                    backend.pre({}, n)
            prior.assert_not_called()

    def test_baseline_is_inherited_and_unmodified(self):
        self.assertNotIn("baseline", r.Backend.__dict__)
        self.assertIs(r.Backend.baseline, r.previous.Backend.baseline)
        backend = object.__new__(r.Backend)
        with patch.object(r.previous.Backend, "baseline", return_value=("original", "scores")) as base:
            self.assertEqual(r.choose(r.NAMES[0], {}, {}, 122, backend, {}), ("original", "scores"))
            base.assert_called_once_with({}, {}, 122, {})

    def test_primary_full_bounded_range_and_future_isolation(self):
        a, m, b = fixture()
        a[r.HANDS[2]] = np.full((2, 49), np.nan)
        a[r.HANDS[2]][0, :2] = [-1., 1.]
        b.pre = lambda arrays, n: np.zeros((n, 49))
        _, score = r.choose(r.NAMES[3], a, m, 2, b, {})
        self.assertEqual(list(score[0, :2]), [-.01, .01])
        a[r.HANDS[2]][1, 0] = 1.001
        r.choose(r.NAMES[3], a, m, 1, b, {})
        with self.assertRaises(ValueError):
            r.choose(r.NAMES[3], a, m, 2, b, {})

    def test_all_three_features_pass_every_prefix_via_actual_backend_override(self):
        a, m, _ = fixture()
        a = {key: np.repeat(value[:1], 122, axis=0) for key, value in a.items()}
        m["symbols"] = [m["symbols"][0]]*122
        full = np.arange(122*49, dtype=float).reshape(122, 49)/100
        backend = object.__new__(r.Backend)
        backend.adapter, backend.selector = Adapter(), "original"
        for key, value in zip(r.HANDS, (.8, -.9, .4)):
            a[key] = np.full((122, 49), np.nan)
            a[key][:, 0] = value
        before = {key: value.tobytes() for key, value in a.items()}
        with patch.object(r.previous.Backend, "pre", return_value=full):
            for name in r.NAMES[1:]:
                _, all_scores = r.choose(name, a, m, 122, backend, {})
                for n in range(123):
                    _, scores = r.choose(name, a, m, n, backend, {})
                    self.assertEqual(scores.tobytes(), all_scores[:n].tobytes())
        self.assertEqual(before, {key: value.tobytes() for key, value in a.items()})
        self.assertEqual(backend.adapter.calls, 3*124)

    def test_common_mask_mismatch_rejected_before_backend_use(self):
        arrays = {key: np.full((122, 49), np.nan) for key in r.HANDS}
        arrays["common_available"] = np.zeros((122, 49), bool)
        arrays[r.HANDS[0]][0, 0] = .1
        with self.assertRaises(ValueError):
            r.panels(arrays, {}, None, {})

    def test_new_source_gate_accepts_only_narrower_population(self):
        inputs = source_fixture()
        before = deepcopy(inputs)
        admitted, held = r.validate_sbc_admission(*inputs)
        self.assertEqual((len(admitted),len(held)), (30,14))
        self.assertEqual(inputs,before)

    def test_source_gate_rejects_missing_reordered_or_changed_metadata(self):
        for operation in ("drop","reverse","source","period","old_gate"):
            inputs = source_fixture()
            data = inputs[0]
            if operation=="drop": data["events"].pop()
            elif operation=="reverse": data["events"].reverse()
            elif operation=="source": data["events"][0]["source"]["sha256"]="e"*64
            elif operation=="period": data["events"][0]["periods"]["current"]["period_end"]="2020-12-31"
            else: data["events"][31]["cash_quality_admission"]["admitted"]=True
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                r.validate_sbc_admission(*inputs)

    def test_source_gate_cannot_release_old_hold_or_accept_raw_gaps(self):
        for operation in ("old_hold","gaps","missing","definition","bool"):
            inputs = source_fixture()
            data = inputs[0]
            if operation=="old_hold":
                data["events"][31]["stock_compensation_admission"].update(admitted=True,reasons=[])
            elif operation=="gaps": data["events"][0]["stock_compensation"]["gaps"]=[{"reason":"ambiguous"}]
            elif operation=="missing": data["events"][0]["stock_compensation"]["value"]=None
            elif operation=="definition": data["events"][0]["stock_compensation_admission"]["definition"]="wrong"
            else: data["events"][0]["stock_compensation_admission"]["admitted"]=1
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                r.validate_sbc_admission(*inputs)

    def test_source_counts_bindings_and_review_must_match(self):
        for index,key,value in ((1,"data_sha256","wrong"),(1,"admitted_events",31),
                                (1,"holds",[]),(3,"passed",False),(3,"held_events",13),
                                (1,"model_fits",True),(2,"financial_ratios_permitted",True),
                                (2,"purpose","wrong"),
                                (2,"first_public_version_verified",True)):
            inputs = source_fixture()
            inputs[index][key]=value
            with self.subTest(index=index,key=key), self.assertRaises(ValueError):
                r.validate_sbc_admission(*inputs)

    def test_cell_source_period_availability_and_sha_must_match(self):
        for key,value in (("source",{}),("period",{}),("available_from","2026-02-03"),
                          ("source_sha256","e"*64)):
            inputs=source_fixture()
            inputs[0]["events"][0]["stock_compensation"]["value"][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_sbc_admission(*inputs)

    def test_no_concept_or_source_placeholder_is_an_actual_registration(self):
        self.assertEqual(len(r.INPUTS),9)
        self.assertEqual(len(r.SBC_REFS),4)
        self.assertEqual(len(r.CACHE_FILES),6)
        self.assertIn("TRAIN_OPEN_TRIALS.json",r.FROZEN_FILES)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = object.__new__(r.Runner)
        self.runner.root = self.root

    def save(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return dict(path=name, sha256=r.sha(path))

    def prepare(self, omission=None):
        data, receipt, policy, review, original, original_receipt, _ = source_fixture()
        original_ref = self.save("synthetic/original_admission.json", original_receipt)
        source = self.save("synthetic/source_and_checker.json", {"synthetic": True})
        refs = {key: self.save(r.SBC_REFS[key], obj)
                for key, obj in (("sbc_dataset", data), ("sbc_policy", policy))}
        receipt.update(data_sha256=refs["sbc_dataset"]["sha256"],
                       policy_sha256=refs["sbc_policy"]["sha256"],
                       input_hashes={source["path"]: source["sha256"]})
        refs["sbc_admission"] = self.save(r.SBC_REFS["sbc_admission"], receipt)
        review["input_hashes"] = {ref["path"]: ref["sha256"]
                                  for ref in (*refs.values(), source)}
        if omission == "review_source":
            review["input_hashes"].pop(source["path"])
        refs["sbc_review"] = self.save(r.SBC_REFS["sbc_review"], review)
        proposal = self.save("synthetic/proposal.json", {"sources": refs})
        self.runner.plan = deepcopy(refs)
        self.runner.inputs = dict(proposal=self.root/proposal["path"],
                                  source_admission=self.root/original_ref["path"])
        self.runner.pins = {str(self.root/ref["path"]): ref["sha256"]
                            for ref in (*refs.values(), source, proposal, original_ref)}
        if omission == "unbound_source":
            self.runner.pins.pop(str(self.root/source["path"]))
        return original, refs

    def test_full_bound_source_review_passes(self):
        original, _ = self.prepare()
        self.assertEqual(len(self.runner.require_sbc_admission(original)["events"]), 44)

    def test_review_must_cover_source_admission_closure(self):
        original, _ = self.prepare("review_source")
        with self.assertRaises(ValueError):
            self.runner.require_sbc_admission(original)

    def test_plan_must_pin_every_review_and_admission_dependency(self):
        original, _ = self.prepare("unbound_source")
        with self.assertRaises(ValueError):
            self.runner.require_sbc_admission(original)

    def test_changed_package_bytes_rejected(self):
        original, refs = self.prepare()
        with (self.root/refs["sbc_dataset"]["path"]).open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            self.runner.require_sbc_admission(original)

    def test_changed_plan_reference_rejected(self):
        original, _ = self.prepare()
        self.runner.plan["sbc_review"]["sha256"] = "f"*64
        with self.assertRaises(ValueError):
            self.runner.require_sbc_admission(original)


if __name__ == "__main__":
    unittest.main(verbosity=2)

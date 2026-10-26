import unittest
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import numpy as np
import monthly_asset_growth147_runner as r


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
    events = []
    for i in range(44):
        gate = dict(admitted=i < 31, reasons=[] if i < 31 else ["synthetic_hold"],
                    definition=r.factor.old.DEFINITION)
        events.append(dict(event_id="synthetic:"+str(i), cash_quality_admission=gate,
                           inherited_rejection_codes=[] if i < 31 else ["synthetic_legacy"]))
    dataset = dict(method=r.factor.METHOD, events=events)
    receipt = dict(events=44, admitted_events=31, held_events=13,
                   admitted_event_ids=[e["event_id"] for e in events[:31]],
                   holds=[dict(event_id=e["event_id"], **e["cash_quality_admission"])
                          for e in events[31:]])
    policy = dict(schema_version=1, round=147, stage="prospective_source_use_policy",
                  purpose=r.factor.MEASUREMENT, dataset_method=r.factor.METHOD,
                  required_endpoints=["current.total_assets", "comparison.total_assets"],
                  annual_days=[330, 400], max_age_days=550, new_events_admitted=0,
                  inputs={key: {} for key in r.ASSET_INPUTS})
    for key in ("same_current_report", "current_release_next_pinned_session",
                "contiguous_periods", "latest_event_before_validity",
                "invalid_latest_blocks_older_fallback", "same_report_currency_literal_assumption",
                "old_source_policy_unchanged"):
        policy[key] = True
    for key in ("factor_calculations_permitted_in_source_review", "first_public_version_verified",
                "complete_correction_history_verified", "certified_daily_security_master",
                "iso_currency_newly_certified", "organic_investment_or_CMA_replication"):
        policy[key] = False
    review = dict(round=147, stage="independent_asset_growth_source_use_review", passed=True,
                  conditional_source_use_only=True, policy_sha256="policy", dataset_sha256="dataset",
                  **receipt, checks={key: True for key in r.ASSET_CHECKS},
                  findings=[dict(event_id=e["event_id"], old_admission=deepcopy(e["cash_quality_admission"]),
                                 conditional_asset_use=e["cash_quality_admission"]["admitted"],
                                 inherited_rejection_codes=e["inherited_rejection_codes"])
                            for e in events])
    for key in ("financial_ratios", "asset_growth_values_computed", "factor_calculations",
                "model_fits", "portfolio_calculations", "return_calculations", "newly_admitted_events"):
        review[key] = 0
    for key in ("first_public_version_verified", "complete_correction_history_verified",
                "certified_daily_security_master"):
        review[key] = False
    return policy, review, dataset, receipt


class RunnerTests(unittest.TestCase):
    def test_proposal_binds_controls_and_evaluation(self):
        p = dict(round=147, arms=list(r.NAMES), primary_challenger=r.NAMES[3],
                 dose=r.DOSE, concept=dict(path="work/monthly_asset_growth147_concept.json",
                                          sha256=r.CONCEPT_SHA),
                 selection_contract=r.selection_contract())
        r.validate_proposal(p)
        for key, value in (("dose", .02), ("primary_challenger", r.NAMES[1]),
                           ("selection_contract", {}), ("concept", {})):
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

    def test_primary_asymmetric_range_and_future_isolation(self):
        a, m, b = fixture()
        a[r.HANDS[2]] = np.full((2, 49), np.nan)
        a[r.HANDS[2]][0, :2] = [-1., .5]
        b.pre = lambda arrays, n: np.zeros((n, 49))
        _, score = r.choose(r.NAMES[3], a, m, 2, b, {})
        self.assertEqual(list(score[0, :2]), [-.01, .005])
        a[r.HANDS[2]][1, 0] = .501
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

    def test_valid_source_purpose_and_unchanged_population(self):
        policy, review, dataset, receipt = source_fixture()
        before = deepcopy((policy, review, dataset, receipt))
        r.validate_asset_use(policy, review, dataset, receipt, "policy", "dataset")
        self.assertEqual((policy, review, dataset, receipt), before)

    def test_wrong_source_purpose_timing_and_claims_rejected(self):
        policy, review, dataset, receipt = source_fixture()
        for key, value in (("round", 146), ("purpose", "different"),
                           ("dataset_method", "different"), ("annual_days", [0, 400]),
                           ("max_age_days", 551), ("invalid_latest_blocks_older_fallback", False),
                           ("new_events_admitted", 1), ("same_current_report", False),
                           ("old_source_policy_unchanged", False),
                           ("first_public_version_verified", True),
                           ("organic_investment_or_CMA_replication", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_asset_use(dict(policy, **{key: value}), review, dataset, receipt,
                                     "policy", "dataset")

    def test_review_must_cover_exact_gate_and_every_check(self):
        policy, review, dataset, receipt = source_fixture()
        for key, value in (("policy_sha256", "other"), ("dataset_sha256", "other"),
                           ("admitted_event_ids", review["admitted_event_ids"][::-1]),
                           ("holds", []), ("findings", review["findings"][:-1]),
                           ("checks", {}), ("factor_calculations", 1), ("financial_ratios", False),
                           ("passed", False), ("conditional_source_use_only", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_asset_use(policy, dict(review, **{key: value}), dataset, receipt,
                                     "policy", "dataset")

    def test_false_check_and_changed_inherited_finding_rejected(self):
        policy, review, dataset, receipt = source_fixture()
        for change in ("check", "gate", "legacy"):
            bad = deepcopy(review)
            if change == "check":
                bad["checks"][next(iter(r.ASSET_CHECKS))] = False
            elif change == "gate":
                bad["findings"][31]["conditional_asset_use"] = True
            else:
                bad["findings"][31]["inherited_rejection_codes"] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                r.validate_asset_use(policy, bad, dataset, receipt, "policy", "dataset")

    def test_population_cannot_release_hold_drop_event_or_redefine_gate(self):
        _, _, dataset, receipt = source_fixture()
        for change in ("release", "drop", "duplicate", "definition", "legacy", "bool", "empty_hold"):
            bad = deepcopy(dataset)
            if change == "release":
                bad["events"][31]["cash_quality_admission"].update(admitted=True, reasons=[])
            elif change == "drop":
                bad["events"].pop()
            elif change == "duplicate":
                bad["events"][1]["event_id"] = bad["events"][0]["event_id"]
            elif change == "definition":
                bad["events"][0]["cash_quality_admission"]["definition"] = "new"
            elif change == "legacy":
                bad["events"][0]["inherited_rejection_codes"] = ["retained_hold"]
            elif change == "bool":
                bad["events"][0]["cash_quality_admission"]["admitted"] = 1
            else:
                bad["events"][31]["cash_quality_admission"]["reasons"] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                r.source_population(bad, receipt)


class SyntheticBindingTests(unittest.TestCase):
    """Only generated metadata in a temporary directory; never use source amounts."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.policy, self.review, self.dataset, self.receipt = source_fixture()
        self.runner = object.__new__(r.Runner)
        self.runner.root = self.root

    def save(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return dict(path=name, sha256=r.sha(path))

    def setup_bindings(self, omission=None):
        refs = {}
        for key in sorted(r.ASSET_INPUTS):
            content = self.dataset if key == "dataset" else self.receipt if key == "admission" else {}
            refs[key] = self.save("synthetic/"+key+".json", content)
        code = self.save("synthetic/source_code.json", {"generated": True})
        self.policy["inputs"] = refs
        self.policy["source_code_hashes"] = {code["path"]: code["sha256"]}
        use = self.save(r.ASSET_REFS["asset_use_policy"], self.policy)
        self.review.update(policy_sha256=use["sha256"], dataset_sha256=refs["dataset"]["sha256"])
        hashes = {ref["path"]: ref["sha256"] for ref in (*refs.values(), code, use)}
        self.review["input_hashes"] = dict(hashes)
        if omission == "review_code":
            del self.review["input_hashes"][code["path"]]
        checked = self.save(r.ASSET_REFS["asset_use_review"], self.review)
        source_refs = dict(asset_use_policy=use, asset_use_review=checked)
        proposal = self.save("synthetic/proposal.json", {"sources": source_refs})
        hashes.update({checked["path"]: checked["sha256"], proposal["path"]: proposal["sha256"]})
        self.runner.pins = {str(self.root/path): digest for path, digest in hashes.items()}
        self.runner.plan = dict(source_refs, source_policy=refs["prior_policy"])
        self.runner.inputs = {
            "proposal": self.root/proposal["path"], "admitted_dataset": self.root/refs["dataset"]["path"],
            "source_admission": self.root/refs["admission"]["path"],
            "sessions": self.root/refs["sessions"]["path"]}
        if omission == "plan_code":
            del self.runner.pins[str(self.root/code["path"])]

    def test_complete_synthetic_bindings_pass(self):
        self.setup_bindings()
        self.runner.require_asset_use(self.dataset, self.receipt)

    def test_review_must_bind_source_code(self):
        self.setup_bindings("review_code")
        with self.assertRaises(ValueError):
            self.runner.require_asset_use(self.dataset, self.receipt)

    def test_plan_must_bind_every_review_dependency(self):
        self.setup_bindings("plan_code")
        with self.assertRaises(ValueError):
            self.runner.require_asset_use(self.dataset, self.receipt)

    def test_proposal_cannot_reference_different_source_review(self):
        self.setup_bindings()
        self.runner.plan["asset_use_review"] = dict(self.runner.plan["asset_use_review"], sha256="0"*64)
        with self.assertRaises(ValueError):
            self.runner.require_asset_use(self.dataset, self.receipt)

    def test_bound_source_input_bytes_cannot_change(self):
        self.setup_bindings()
        self.runner.inputs["sessions"].write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            self.runner.require_asset_use(self.dataset, self.receipt)


if __name__ == "__main__":
    unittest.main()

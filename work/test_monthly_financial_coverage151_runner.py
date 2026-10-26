"""Synthetic runner/closure tests. Never load real financial or market data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import monthly_financial_coverage151_runner as r
from test_monthly_financial_coverage151_selection import fixture as scoring_fixture


def proposal():
    return dict(round=151, arms=list(r.NAMES), primary_challenger=r.NAMES[3], dose=.01,
                status="FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
                selection_contract=r.selection_contract(), measurement=r.measurement_contract(),
                actual_new_factors=0, actual_new_orders=0, actual_new_returns=0)


def source_fixture():
    refs = {key: dict(path=value, sha256=str(i)*64)
            for i, (key, value) in enumerate(r.COVERAGE_REFS["batch1"].items(), 1)}
    events, raw, reviews = [], [], []
    for i in range(26):
        symbol = "AMZN" if i < 10 else "CSCO"
        fields = {name: dict(synthetic_only=True) for name in r.factor.FIELDS}
        event = dict(event_id=symbol+":"+str(i), symbol=symbol, fiscal_year=2000+i,
                     source=dict(path="synthetic.html", sha256="a"*64), fields=fields,
                     fact_admitted=False, model_admitted=False, strategy_admitted=False)
        # Make the source definition independent of hypothetical calendar years.
        event["fiscal_year"] = 2018
        raw.append(deepcopy(event))
        event.update(ni_assets_admission=dict(admitted=i < 20, reasons=[] if i < 20 else ["hold"],
                                             definition=r.factor.DEFINITION),
                     identity=dict(synthetic=True), source_review=dict(synthetic=True),
                     clock_review=dict(synthetic=True))
        events.append(event)
        reviews.append(dict(event_id=event["event_id"], passed=True, metadata_verified=True,
                            fields={name: dict(is_null=False, verified=True, source_value_present=True)
                                    for name in r.factor.FIELDS}))
    catalog = [dict(event_id=e["event_id"], symbol=e["symbol"]) for e in events]
    catalog += [dict(event_id="AMZN:unprocessed"+str(i), symbol="AMZN") for i in range(6)]
    old_events = [dict(event_id="old"+str(i),
                       stock_compensation_admission=dict(admitted=i < 31))
                  for i in range(44)]
    old_data = dict(events=old_events)
    original = dict(events=44, admitted=31, held=13, preserved_without_modification=True,
                    records_canonical_sha256=r.canonical_hash(old_events))
    data = dict(schema_version=1, method=r.factor.METHOD, events=events, latest_event_catalog=catalog,
                original_four_issuer_reference=original, source_plan_sha256="e"*64,
                policy_sha256=refs["policy"]["sha256"],
                independent_review_sha256=refs["review"]["sha256"])
    for key in ("first_public_version_verified", "complete_correction_history_verified",
                "certified_daily_security_master"):
        data[key] = False
    for key in ("scores_computed", "ratios_computed", "returns_computed", "new_strategy_trials"):
        data[key] = 0
    policy = dict(stage="prospective_source_policy_not_strategy_registration",
                  required_fields=list(r.factor.FIELDS),
                  source_scope=dict(symbols=["AMZN", "CSCO"], no_automatic_admission=True),
                  known_time=dict(maximum_age_days=550),
                  preservation={k: False for k in (
                      "first_public_version_verified", "complete_correction_history_verified",
                      "certified_daily_security_master")})
    admission = dict(passed=True, stage="conditional_source_admission_not_strategy_registration",
                     input_hashes={"synthetic": "a"*64}, source_plan_sha256="e"*64,
                     policy_sha256=refs["policy"]["sha256"],
                     independent_review_sha256=refs["review"]["sha256"],
                     actual_factors_calculated=0, actual_returns_calculated=0, new_strategy_trials=0,
                     counts=dict(events=26, admitted=20, held=6, annual_catalog_events=32,
                                 unprocessed_annual_catalog_events=6, original_events=44,
                                 original_admitted=31, original_held=13))
    review = dict(passed=True, plan_sha256="e"*64, event_reviews=reviews)
    manifest = dict(status="conditional_source_package_no_strategy_registration",
                    outputs={"DATASET.json": refs["dataset"]["sha256"],
                             "ADMISSION.json": refs["admission"]["sha256"],
                             "COVERAGE.json": refs["coverage"]["sha256"]},
                    input_hashes=deepcopy(admission["input_hashes"]))
    coverage = dict(counts=dict(months=122, stocks=73, stock_slots=3904))
    extraction = dict(plan_sha256="e"*64, events=raw)
    source_plan = dict(records=[dict(event_id=e["event_id"]) for e in events])
    return [data, admission, policy, review, manifest, coverage, refs, old_data, extraction, source_plan]


def feature_fixture():
    original = np.full((3,49),np.nan);original[:,0]=.1
    batch1=original.copy();batch1[:,1]=.2
    expanded=batch1.copy();expanded[:,2]=-.3
    names=["AAPL","AMZN","INTC"]+["X"+str(i) for i in range(46)]
    meta=dict(dates=["2020-01-31","2020-02-28","2020-03-31"],symbols=[names.copy() for _ in range(3)])
    mask=np.isfinite(original);bmask=np.isfinite(batch1)
    result=dict(arrays=dict(earnings_original=original.copy(),earnings_batch1=batch1.copy(),
        earnings_expanded=expanded,original_available=mask.copy(),batch1_available=bmask.copy(),
        expanded_available=np.isfinite(expanded)))
    return result,original,mask,batch1,bmask,meta


def source2_fixture():
    args=source_fixture()
    data,adm,policy,review,manifest,cov,refs,olddata,ext,plan=args
    selected=list(range(3))+list(range(10,26))
    for obj,key in ((data,"events"),(review,"event_reviews"),(ext,"events"),(plan,"records")):
        obj[key]=[obj[key][i] for i in selected]
    for obj,key in ((data,"events"),(ext,"events")):
        for i,event in enumerate(obj[key]):
            event["symbol"]="INTC" if i<3 else "KLAC"
            event["context_holds"]=["boundary"] if i<3 else []
    for i,event in enumerate(data["events"]):
        event["ni_assets_admission"].update(admitted=i<14,reasons=[] if i<14 else ["hold"])
        rr=review["event_reviews"][i]
        rr.update(identity_verified=True,source=event["source"],retained_context_holds=event["context_holds"])
        if i<3:
            rr.update(boundary_assumption_verified=True,assumption_id="INTC_consecutive_annual_boundaries_v1",
                      start_is_source_quoted=False)
    data["latest_event_catalog"]=[dict(event_id=e["event_id"],symbol=e["symbol"]) for e in data["events"]]
    data["latest_event_catalog"] += [dict(event_id="INTC_missing"+str(i),symbol="INTC") for i in range(13)]
    policy["source_scope"]["symbols"]=["INTC","KLAC"]
    adm["counts"].update(events=19,admitted=14,held=5,unprocessed_annual_catalog_events=13,
                        prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9)
    return args


class RunnerTests(unittest.TestCase):
    def test_proposal_exact_contract(self):
        r.validate_proposal(proposal())
        for key, value in (("dose", .02), ("round", 149), ("arms", list(r.NAMES[::-1])),
                           ("primary_challenger", r.NAMES[1]), ("measurement", {}),
                           ("selection_contract", {}), ("actual_new_factors", True)):
            bad = proposal()
            bad[key] = value
            with self.assertRaises(ValueError):
                r.validate_proposal(bad)

    def test_source_complete_fixture(self):
        self.assertEqual(r.validate_source(*source_fixture()),
                         dict(events=26, admitted=20, held=6, catalog_events=32))

    def test_manifest_output_or_input_swap_rejected(self):
        for part in ("outputs", "input_hashes"):
            args = source_fixture()
            args[4][part] = {}
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_policy_or_review_links_rejected(self):
        for key in ("policy_sha256", "source_plan_sha256", "independent_review_sha256"):
            args = source_fixture()
            args[0][key] = "f"*64
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_missing_event_or_catalog_sentinel_rejected(self):
        for key in ("events", "latest_event_catalog"):
            args = source_fixture()
            args[0][key].pop()
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_duplicate_event_id_rejected(self):
        args = source_fixture()
        args[0]["events"][-1]["event_id"] = args[0]["events"][0]["event_id"]
        with self.assertRaises(ValueError):
            r.validate_source(*args)

    def test_extraction_value_or_metadata_mutation_rejected(self):
        for key, value in (("fields", {}), ("fact_admitted", True), ("source", {})):
            args = source_fixture()
            args[0]["events"][0][key] = value
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_old_gate_or_record_digest_change_rejected(self):
        for change in ("gate", "hash"):
            args = source_fixture()
            if change == "gate":
                args[7]["events"][40]["stock_compensation_admission"]["admitted"] = True
            else:
                args[0]["original_four_issuer_reference"]["records_canonical_sha256"] = "0"*64
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_original_13_holds_not_reopened(self):
        args = source_fixture()
        args[0]["original_four_issuer_reference"]["held"] = 12
        with self.assertRaises(ValueError):
            r.validate_source(*args)

    def test_new_gate_counts_and_reasons_rejected(self):
        for change in ("counts", "reasons", "definition"):
            args = source_fixture()
            if change == "counts":
                args[1]["counts"]["admitted"] = 21
            elif change == "reasons":
                args[0]["events"][0]["ni_assets_admission"]["reasons"] = ["hold"]
            else:
                args[0]["events"][0]["ni_assets_admission"]["definition"] = "cashflow"
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_2019_accounting_hold_cannot_reopen(self):
        args = source_fixture()
        args[0]["events"][0]["fiscal_year"] = args[8]["events"][0]["fiscal_year"] = 2019
        with self.assertRaises(ValueError):
            r.validate_source(*args)

    def test_unreviewed_nonnull_or_null_rejected(self):
        for change in ("present", "null"):
            args = source_fixture()
            field = args[3]["event_reviews"][0]["fields"]["annual_net_income"]
            field["source_value_present" if change == "present" else "is_null"] = False if change == "present" else True
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_certification_or_real_calculation_rejected(self):
        for key in ("first_public_version_verified", "scores_computed"):
            args = source_fixture()
            args[0][key] = True
            with self.assertRaises(ValueError):
                r.validate_source(*args)

    def test_distinct_masks_with_exact_old_cells_valid(self):
        r.validate_features(*feature_fixture())

    def test_original_value_or_mask_change_rejected(self):
        for key in ("earnings_original", "original_available"):
            args = feature_fixture()
            args[0]["arrays"][key][0, 0] = 0
            with self.assertRaises(ValueError):
                r.validate_features(*args)

    def test_expanded_cannot_change_old_or_nonbatch_cells(self):
        for column in (0, 3):
            args = feature_fixture()
            args[0]["arrays"]["earnings_expanded"][0, column] = .9
            args[0]["arrays"]["expanded_available"][0, column] = True
            with self.assertRaises(ValueError):
                r.validate_features(*args)

    def test_wrong_mask_dtype_unbounded_or_infinite_rejected(self):
        for change in ("dtype", "bound", "inf"):
            args = feature_fixture()
            if change == "dtype":
                args[0]["arrays"]["expanded_available"] = args[0]["arrays"]["expanded_available"].astype(int)
            else:
                args[0]["arrays"]["earnings_expanded"][0, 1] = 1.1 if change == "bound" else np.inf
            with self.assertRaises(ValueError):
                r.validate_features(*args)

    def test_inherits_prefix_fixed_backend_without_rebuilding_old_factor(self):
        self.assertIs(r.Backend.pre, r.old.Backend.pre)
        self.assertIs(r.Backend.baseline, r.old.Backend.baseline)
        self.assertIs(r.Backend.load, r.old.Backend.load)

    def test_actual_scoring_adapter_with_synthetic_prefixes(self):
        arrays, meta, backend = scoring_fixture()
        backend.make_orders = lambda choices, scores, meta: scores.tolist()
        # Use stock-free simple scores so zero/missing fallback traverses the
        # actual frozen corrected() without requiring its real episode engine.
        backend.baseline = lambda a, m, n, p: (["synthetic"][:n], a["baseline"][:n].copy())
        backend.adapter.score_and_choose = lambda *args, **kwargs: (
            ["synthetic"][:args[5]], kwargs["pre_score"][:args[5]].copy())
        with patch.object(r, "MONTHS", 3):
            orders, scores = r.panels(arrays, meta, backend, {})
        self.assertEqual(set(orders), set(r.NAMES))
        self.assertEqual(scores[r.NAMES[0]].shape, (3, 49))

    def test_prepare_review_required_before_load_or_returns(self):
        obj = object.__new__(r.Runner)
        obj.refresh = Mock()
        obj.require_prepare_review = Mock(side_effect=ValueError("review required"))
        obj.loaded = Mock()
        with self.assertRaises(ValueError):
            obj.select_first88()
        obj.loaded.assert_not_called()
        with self.assertRaises(ValueError):
            obj.diagnose_later34()
        obj.loaded.assert_not_called()

    def test_prepare_existing_cache_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = object.__new__(r.Runner)
            obj.out = Path(directory)
            obj.refresh = Mock()
            obj.backend_factory = Mock()
            (obj.out/"DATA.npz").write_bytes(b"partial synthetic")
            with self.assertRaises(ValueError):
                obj.prepare()
            obj.backend_factory.assert_not_called()

    def test_select_existing_outputs_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = object.__new__(r.Runner)
            obj.out = Path(directory)
            obj.refresh = Mock()
            obj.require_prepare_review = Mock()
            obj.loaded = Mock()
            (obj.out/"TRAIN_OPEN_TRIALS.json").write_text("{}")
            with self.assertRaises(ValueError):
                obj.select_first88()
            obj.loaded.assert_not_called()

    def test_refs_reject_missing_or_aliased_path(self):
        obj = object.__new__(r.Runner)
        obj.root = Path("/synthetic")
        obj.bound_path = lambda value: Path(value)
        with self.assertRaises(ValueError):
            obj._refs({}, {"x": "expected"})
        with patch.object(r, "sha", return_value="a"*64):
            with self.assertRaises(ValueError):
                obj._refs({"x": dict(path="/synthetic/other", sha256="a"*64)},
                          {"x": "expected"})

    def test_unregistered_source_closure_rejected(self):
        obj = object.__new__(r.Runner)
        obj.pins = {"/source": "a"*64}
        obj.verify = lambda value: value
        with self.assertRaises(ValueError):
            obj._closure({"/source": "b"*64})

    def test_no_identical_mask_requirement(self):
        args = feature_fixture()
        self.assertFalse(np.array_equal(args[0]["arrays"]["original_available"],
                                        args[0]["arrays"]["expanded_available"]))
        r.validate_features(*args)

    def test_frozen_files_include_open_training(self):
        self.assertIn("TRAIN_OPEN_TRIALS.json", r.FROZEN_FILES)
        self.assertEqual(len(r.CACHE_FILES), 6)
        self.assertEqual(len(r.NAMES), 4)

    def test_complete_catalog_exact_source_equality(self):
        data = source_fixture()[0]
        for row in data["latest_event_catalog"]:
            row.update(fiscal_quarter=4, publication_date="2020-01-01")
        census = dict(annual_documents=deepcopy(data["latest_event_catalog"]))
        for row in data["latest_event_catalog"]:
            row.update(clock_review=dict(verified=True),
                       processed_in_batch=row["event_id"] in {e["event_id"] for e in data["events"]})
        r.validate_catalog(data, census)
        data["latest_event_catalog"][-1]["publication_date"] = "2019-01-01"
        with self.assertRaises(ValueError):
            r.validate_catalog(data, census)

    def test_unprocessed_catalog_sentinel_preserved(self):
        data = source_fixture()[0]
        for row in data["latest_event_catalog"]:
            row["fiscal_quarter"] = 4
        census = dict(annual_documents=deepcopy(data["latest_event_catalog"]))
        for row in data["latest_event_catalog"]:
            row["processed_in_batch"] = True
        with self.assertRaises(ValueError):
            r.validate_catalog(data, census)

    def test_reference_arrays_read_both_frozen_hands_without_factor_call(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=object.__new__(r.Runner);obj.root=Path(directory)
            obj.require_legacy=Mock();obj.require_prior=Mock()
            meta=dict(synthetic=True);originals={"old"+str(i):np.zeros((122,49)) for i in range(27)}
            earnings=np.full((122,49),np.nan);earnings[:,0]=.2
            batch1=earnings.copy();batch1[:,1]=-.1
            s0=np.zeros((122,49));s1=np.ones((122,49));s2=np.full((122,49),2.)
            for folder in (r.LEGACY_OUT,r.PRIOR_OUT):
                out=obj.root/folder;out.mkdir(parents=True);(out/"META.json").write_text(json.dumps(meta))
            out=obj.root/r.LEGACY_OUT
            np.savez(out/"DATA.npz",**originals,earnings_bounded=earnings,common_available=np.isfinite(earnings))
            np.savez(out/"SCORES.npz",incumbent_local67=s0,earnings_control=s1)
            (out/"PRETRAIN_ORDERS.json").write_text(json.dumps(dict(incumbent_local67=["old0"],earnings_control=["old1"])))
            out=obj.root/r.PRIOR_OUT
            np.savez(out/"DATA.npz",**originals,earnings_original=earnings,original_available=np.isfinite(earnings),
                     earnings_expanded=batch1,expanded_available=np.isfinite(batch1))
            np.savez(out/"SCORES.npz",incumbent_local67=s0,original_earnings=s1,expanded_earnings=s2)
            (out/"PRETRAIN_ORDERS.json").write_text(json.dumps(dict(incumbent_local67=["old0"],original_earnings=["old1"],expanded_earnings=["prior2"])))
            with patch.object(r.old.factor,"build_features",side_effect=AssertionError("Old factor forbidden")), \
                 patch.object(r.previous.factor,"build",side_effect=AssertionError("Prior factor forbidden")):
                got,mask,got2,mask2,scores,orders=obj.reference_arrays(meta,originals)
            self.assertEqual(got.tobytes(),earnings.tobytes());self.assertEqual(got2.tobytes(),batch1.tobytes())
            self.assertEqual(orders[r.NAMES[2]],["prior2"])
            self.assertEqual(scores[r.NAMES[2]].tobytes(),s2.tobytes())
            originals["old0"][0,0]=1
            with self.assertRaises(ValueError):obj.reference_arrays(meta,originals)

    def test_batch2_complete_source_contract(self):
        self.assertEqual(r.validate_source(*source2_fixture(),batch="batch2"),
                         dict(events=19,admitted=14,held=5,catalog_events=32))

    def test_batch2_boundary_and_legacy_holds_are_required(self):
        args=source2_fixture();args[3]["event_reviews"][0]["boundary_assumption_verified"]=False
        with self.assertRaises(ValueError):r.validate_source(*args,batch="batch2")
        for year in (2011,2019):
            args=source2_fixture()
            args[0]["events"][3]["fiscal_year"]=args[8]["events"][3]["fiscal_year"]=year
            with self.assertRaises(ValueError):r.validate_source(*args,batch="batch2")

    def test_batch1_hand_or_mask_change_rejected(self):
        for key in ("earnings_batch1","batch1_available"):
            args=feature_fixture();args[0]["arrays"][key][0,1]=0
            with self.assertRaises(ValueError):r.validate_features(*args)

    def test_union_preserves_all_dictionaries_and_checks_refs(self):
        first=source_fixture()[0];second=source2_fixture()[0]
        # Distinct synthetic IDs across the independent packages.
        for e in second["events"]+second["latest_event_catalog"]:e["event_id"]="b2_"+e["event_id"]
        refs={b:{k:dict(path=p,sha256="a"*64) for k,p in paths.items()} for b,paths in r.COVERAGE_REFS.items()}
        prior=dict(records_canonical_sha256=r.canonical_hash(first["events"]),
                   catalog_canonical_sha256=r.canonical_hash(first["latest_event_catalog"]))
        for k,rk in (("path","dataset"),("admission_path","admission"),("coverage_path","coverage"),("manifest_path","manifest")):
            prior[k]=r.COVERAGE_REFS["batch1"][rk]
            prior["sha256" if k=="path" else k.replace("_path","_sha256")]="a"*64
        second["prior_batch1_reference"]=prior
        result=r.combine_sources(first,second,refs)
        self.assertEqual(result["events"],first["events"]+second["events"])
        self.assertEqual(len(result["latest_event_catalog"]),64)
        second["prior_batch1_reference"]["records_canonical_sha256"]="b"*64
        with self.assertRaises(ValueError):r.combine_sources(first,second,refs)

    def test_prior150_finalized_cache_and_source_chain(self):
        obj=object.__new__(r.Runner);obj.root=Path("/synthetic")
        obj.inputs={k:obj.root/k for k in r.INPUTS}
        obj.path=lambda p:Path(p) if Path(p).is_absolute() else obj.root/p
        refs={k:dict(sha256=k+"_hash") for k in r.LEGACY_FILES}
        obj.plan=dict(prior150=refs,legacy149={"fixed":True},coverage_sources={"batch1":{"fixed":True}},
                      raw_source_hashes={"fixed":"raw"})
        cache={n:n+"_hash" for n in r.CACHE_FILES}
        pins={str(obj.root/r.PRIOR_OUT/n):h for n,h in cache.items()}
        plan=dict(round=150,status="REGISTERED",arms=list(r.previous.NAMES),hashes={"old":"hash"},
                  legacy149=obj.plan["legacy149"],coverage_source=obj.plan["coverage_sources"]["batch1"],
                  inputs={k:dict(path=str(v),sha256="digest") for k,v in obj.inputs.items()},
                  raw_source_hashes=obj.plan["raw_source_hashes"])
        done=dict(round=150,plan_sha256="plan_hash",independent_outcome_sha256="outcome_hash")
        review=dict(plan_sha256="plan_hash",cache_sha256="cache_hash",passed=True,stage="evaluate",
                    artifact_hashes=pins,source_hashes={"independent":"hash"},orders_sha256=cache["PRETRAIN_ORDERS.json"])
        objects=dict(plan=plan,cache=cache,completion=done,outcome=review)
        obj._refs=lambda *args:(objects,{})
        obj._closure=lambda d:{str(obj.path(k)):v for k,v in d.items()}
        with patch.object(r,"sha",return_value="digest"):
            obj.require_prior()
            for key,value in (("cache_sha256","wrong"),("orders_sha256","wrong"),("passed",False)):
                prior=review[key];review[key]=value
                with self.assertRaises(ValueError):obj.require_prior()
                review[key]=prior
            plan["coverage_source"]={"swapped":True}
            with self.assertRaises(ValueError):obj.require_prior()

    def test_prepare_receipt_counts_are_exact(self):
        obj=object.__new__(r.Runner);obj.out=Path("/synthetic/out");obj.cache=lambda:{}
        receipt=dict(passed=True,stage="prepare",producer_modules_imported=False,
                     returns_or_nav_computed=False,monthly_label_models_fit=0,
                     orders_checked=488,source_cells_checked=5978,stock_cells_checked=3904)
        receipt.update({k:True for k in r.CHECKS})
        for key in ("orders_checked","source_cells_checked","stock_cells_checked"):
            saved=receipt[key];receipt[key]=saved-1
            with patch.object(r,"read",return_value=receipt):
                with self.assertRaisesRegex(ValueError,"population"):
                    obj.require_prepare_review()
            receipt[key]=saved

    def test_legacy_finalized_cache_hash_chain(self):
        obj = object.__new__(r.Runner)
        obj.root = Path("/synthetic")
        obj.inputs = {name: obj.root/name for name in r.INPUTS}
        obj.path = lambda p: Path(p) if Path(p).is_absolute() else obj.root/p
        ref = {key: dict(sha256=key+"_digest") for key in r.LEGACY_FILES}
        obj.plan = dict(legacy149=ref, raw_source_hashes={"raw": "hash"})
        cache = {name: name+"_digest" for name in r.CACHE_FILES}
        pins = {str(obj.root/r.LEGACY_OUT/name): digest for name, digest in cache.items()}
        plan = dict(round=149, status="REGISTERED", hashes={"original": "hash"},
                    inputs={name: dict(path=str(path), sha256="digest") for name, path in obj.inputs.items()},
                    raw_source_hashes={"raw": "hash"})
        done = dict(round=149, plan_sha256="plan_digest", independent_outcome_sha256="outcome_digest")
        review = dict(plan_sha256="plan_digest", cache_sha256="cache_digest", passed=True, stage="evaluate",
                      artifact_hashes=pins, source_hashes={"reviewer": "hash"},
                      orders_sha256=cache["PRETRAIN_ORDERS.json"])
        objects = dict(plan=plan, cache=cache, completion=done, outcome=review)
        obj._refs = lambda *args: (objects, {})
        obj._closure = lambda mapping: {str(obj.path(k)): v for k, v in mapping.items()}
        with patch.object(r, "sha", return_value="digest"):
            obj.require_legacy()
            review["cache_sha256"] = "different_cache"
            with self.assertRaises(ValueError):
                obj.require_legacy()


if __name__ == "__main__":
    unittest.main(verbosity=2)

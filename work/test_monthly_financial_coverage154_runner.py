"""Synthetic runner/closure tests. Never load real financial or market data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import monthly_financial_coverage154_runner as r
from test_monthly_financial_coverage154_selection import fixture as scoring_fixture


def proposal():
    return dict(round=154, arms=list(r.NAMES), primary_challenger=r.NAMES[6], dose=.01,
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
    values=[];current=np.full((3,49),np.nan)
    for i in range(6):
        current=current.copy();current[:,i]=(-1 if i==2 else 1)*.1*(i+1);values.append(current)
    names=["AAPL","AMZN","INTC","KO","MU","PLTR"]+["X"+str(i) for i in range(43)]
    meta=dict(dates=["2020-01-31","2020-02-28","2020-03-31"],symbols=[names.copy() for _ in range(3)])
    arrays={k:v.copy() for k,v in zip(r.HANDS,values)}
    arrays.update({k:np.isfinite(v) for k,v in zip(r.EXTRA[6:],values)})
    args=[dict(arrays=arrays)]
    for v in values[:5]:args.extend((v,np.isfinite(v)))
    return (*args,meta)


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


def source3_fixture():
    args=source_fixture();data,adm,policy,review,manifest,cov,refs,old,ext,plan=args
    for obj,key in ((data,"events"),(ext,"events"),(review,"event_reviews"),(plan,"records")):obj[key]=obj[key][:10]
    for i,(e,raw,rr) in enumerate(zip(data["events"],ext["events"],review["event_reviews"])):
        e["symbol"]=raw["symbol"]="KO" if i<6 else "LRCX";e["context_holds"]=raw["context_holds"]=["retained"]
        e["available_from"]=raw["available_from"]="2020-02-03"
        raw["identity"]={"source_anchor":"literal issuer","certified_daily_security_master":False}
        e["identity"]=dict(raw["identity"],issuer_id="issuer",security_id=e["symbol"],identity_valid_from=e["available_from"],identity_valid_through="2021-07-03",scope="source-use")
        e["ni_assets_admission"].update(admitted=i>=5,reasons=[] if i>=5 else ["hold"])
        rr.update(identity_verified=True,source=e["source"],retained_context_holds=e["context_holds"],boundary_assumption_verified=True,assumption_id="KO_calendar_annual_boundaries_v1" if i<6 else "LRCX_consecutive_annual_boundaries_v1",start_is_source_quoted=False,issuer_id="issuer",clock_verified=i>=5,accounting_review=dict(status="resolved",current_prior_common_basis=True,field_checks={f:True for f in r.factor.FIELDS}))
        e["source_review"]=dict(accounting_review=deepcopy(rr["accounting_review"]));e["clock_review"]=dict(verified=i>=5)
        if i==1:
            for f in r.factor.FIELDS:
                e["fields"][f]=raw["fields"][f]=None;rr["fields"][f]=dict(is_null=True,null_reason_verified=True,source_value_present=True)
    data["latest_event_catalog"]=[dict(event_id=e["event_id"],symbol=e["symbol"]) for e in data["events"]]+[dict(event_id="KO_missing"+str(i),symbol="KO") for i in range(10)]+[dict(event_id="LRCX_missing"+str(i),symbol="LRCX") for i in range(12)]
    policy["source_scope"]["symbols"]=["KO","LRCX"]
    adm["counts"].update(events=10,admitted=5,held=5,unprocessed_annual_catalog_events=22,prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9,prior_batch2_events=19,prior_batch2_admitted=14,prior_batch2_held=5,reviewed_nonnull_fields=27,reviewed_null_fields=3)
    return args


def source4_fixture(admitted_count=5):
    args=source3_fixture();data,adm,policy,review,manifest,cov,refs,old,ext,plan=args
    for obj,key in ((data,"events"),(ext,"events"),(review,"event_reviews"),(plan,"records")):
        obj[key]=obj[key][:9]
    for i,(e,raw,rr) in enumerate(zip(data["events"],ext["events"],review["event_reviews"])):
        e["symbol"]=raw["symbol"]="MU" if i<2 else "PEP"
        for name in r.factor.FIELDS:
            e["fields"][name]=dict(synthetic_only=True);raw["fields"][name]=dict(synthetic_only=True)
            rr["fields"][name]=dict(is_null=False,verified=True,source_value_present=True)
        e["identity"]["security_id"]=e["symbol"]
        ok=i<admitted_count
        e["ni_assets_admission"].update(admitted=ok,reasons=[] if ok else ["hold"])
        rr.update(assumption_id=e["symbol"]+"_consecutive_annual_boundaries_v1",clock_verified=ok)
        e["clock_review"]["verified"]=ok
    data["latest_event_catalog"]=[dict(event_id=e["event_id"],symbol=e["symbol"]) for e in data["events"]]+[dict(event_id="MU_missing"+str(i),symbol="MU") for i in range(5)]+[dict(event_id="PEP_missing"+str(i),symbol="PEP") for i in range(9)]
    policy["source_scope"]["symbols"]=["MU","PEP"]
    metadata=[{k:deepcopy(row.get(k)) for k in r.CATALOG_KEYS} for row in data["latest_event_catalog"]]
    policy["fixed_catalog_metadata"]=deepcopy(metadata)
    plan["catalog_metadata"]=deepcopy(metadata);ext["catalog_metadata"]=deepcopy(metadata)
    adm["counts"].update(events=9,admitted=admitted_count,held=9-admitted_count,annual_catalog_events=23,
                        unprocessed_annual_catalog_events=14,prior_batch3_events=10,prior_batch3_admitted=5,
                        prior_batch3_held=5,reviewed_nonnull_fields=27,reviewed_null_fields=0)
    return args


def source5_fixture(admitted_count=19):
    args=source4_fixture()
    data,adm,policy,review,manifest,cov,refs,old,ext,plan=args
    templates=[deepcopy(obj[key][0]) for obj,key in ((data,"events"),(ext,"events"),(review,"event_reviews"))]
    data["events"]=[];ext["events"]=[];review["event_reviews"]=[]
    for i in range(21):
        e,raw,rr=deepcopy(templates)
        symbol="PLTR" if i<6 else "PYPL" if i<11 else "TMO"
        event_id=symbol+":"+str(i)
        e.update(event_id=event_id,symbol=symbol);raw.update(event_id=event_id,symbol=symbol)
        rr.update(event_id=event_id,assumption_id=symbol+"_consecutive_annual_boundaries_v1")
        e["identity"]["security_id"]=symbol
        ok=i<admitted_count
        e["ni_assets_admission"].update(admitted=ok,reasons=[] if ok else ["hold"])
        rr["clock_verified"]=ok;e["clock_review"]["verified"]=ok
        data["events"].append(e);ext["events"].append(raw);review["event_reviews"].append(rr)
    plan["records"]=[dict(event_id=e["event_id"]) for e in data["events"]]
    data["latest_event_catalog"]=[dict(event_id=e["event_id"],symbol=e["symbol"]) for e in data["events"]]+[dict(event_id="PYPL_missing"+str(i),symbol="PYPL") for i in range(6)]+[dict(event_id="TMO_missing",symbol="TMO")]
    policy["source_scope"]["symbols"]=["PLTR","PYPL","TMO"]
    policy["known_time"]={"age_days":550,"first_public_version_verified":False}
    policy["preservation"]={"no_first_public_time_or_currency_ISO_or_security_master_certification":True}
    metadata=[{k:deepcopy(row.get(k)) for k in r.CATALOG_KEYS} for row in data["latest_event_catalog"]]
    policy["fixed_catalog_metadata"]=deepcopy(metadata)
    plan["catalog_metadata"]=deepcopy(metadata);ext["catalog_metadata"]=deepcopy(metadata)
    adm["counts"].update(events=21,admitted=admitted_count,held=21-admitted_count,annual_catalog_events=28,
                        unprocessed_annual_catalog_events=7,prior_batch4_events=9,prior_batch4_admitted=5,
                        prior_batch4_held=4,reviewed_nonnull_fields=63,reviewed_null_fields=0)
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
        for column in (0, 6):
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
        self.assertEqual(len(r.NAMES), 7)

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

    def test_reference_arrays_read_five_frozen_hands_without_factor_call(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=object.__new__(r.Runner);obj.root=Path(directory)
            obj.require_legacy=Mock();obj.require_prior=Mock()
            meta=dict(synthetic=True);originals={"old"+str(i):np.zeros((122,49)) for i in range(27)}
            values=[];a=np.full((122,49),np.nan)
            for i in range(5):a=a.copy();a[:,i]=.1*(i+1);values.append(a)
            scores=[np.full((122,49),float(i)) for i in range(6)]
            for round_ in range(149,154):
                folder=r.LEGACY_OUT if round_==149 else "outputs/monthly_financial_coverage"+str(round_)
                out=obj.root/folder;out.mkdir(parents=True);(out/"META.json").write_text(json.dumps(meta))
                extra={"earnings_bounded":values[0],"common_available":np.isfinite(values[0])} if round_==149 else {"earnings_original":values[0],"original_available":np.isfinite(values[0]),"earnings_expanded":values[round_-149],"expanded_available":np.isfinite(values[round_-149])}
                for j in range(1,round_-149):
                    extra["earnings_batch"+str(j)]=values[j];extra["batch"+str(j)+"_available"]=np.isfinite(values[j])
                np.savez(out/"DATA.npz",**originals,**extra)
                names=("incumbent_local67","earnings_control") if round_==149 else (*r.NAMES[:round_-149+1],"expanded_earnings")
                np.savez(out/"SCORES.npz",**dict(zip(names,scores)))
                (out/"PRETRAIN_ORDERS.json").write_text(json.dumps({name:["old"+str(i)] for i,name in enumerate(names)}))
            with patch.object(r.factor,"build",side_effect=AssertionError("New factor forbidden")),patch.object(r.previous.factor,"build",side_effect=AssertionError("Prior factor forbidden")),patch.object(r.old.factor,"build_features",side_effect=AssertionError("Old factor forbidden")):
                got=obj.reference_arrays(meta,originals)
            self.assertEqual(len(got),12)
            for i in range(5):self.assertEqual(got[2*i].tobytes(),values[i].tobytes())
            self.assertEqual(got[-1][r.NAMES[5]],["old5"])
            self.assertEqual(got[-2][r.NAMES[5]].tobytes(),scores[5].tobytes())
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

    def test_union_preserves_all_five_packages_and_refs(self):
        packages=[source_fixture()[0],source2_fixture()[0],source3_fixture()[0],source4_fixture()[0],source5_fixture()[0]]
        for i,data in enumerate(packages):
            for e in data["events"]+data["latest_event_catalog"]:e["event_id"]="batch"+str(i)+"_"+e["event_id"]
        refs={b:{k:dict(path=p,sha256="a"*64) for k,p in paths.items()} for b,paths in r.COVERAGE_REFS.items()}
        def reference(data,batch):
            result=dict(records_canonical_sha256=r.canonical_hash(data["events"]),catalog_canonical_sha256=r.canonical_hash(data["latest_event_catalog"]))
            for k,rk in (("path","dataset"),("admission_path","admission"),("coverage_path","coverage"),("manifest_path","manifest")):
                result[k]=r.COVERAGE_REFS[batch][rk];result["sha256" if k=="path" else k.replace("_path","_sha256")]="a"*64
            return result
        for i,data in enumerate(packages):
            for j in range(i):
                data["prior_batch"+str(j+1)+"_reference"]=reference(packages[j],"batch"+str(j+1))
        result=r.combine_sources(*packages,refs)
        self.assertEqual(result["events"],sum((p["events"] for p in packages),[]))
        self.assertEqual(len(result["latest_event_catalog"]),147)
        result["events"][-1]["symbol"]="changed"
        self.assertNotEqual(packages[-1]["events"][-1]["symbol"],"changed")
        for key in ("records_canonical_sha256","catalog_canonical_sha256","admission_sha256"):
            ref=packages[-1]["prior_batch4_reference"];saved=ref[key];ref[key]="wrong"
            with self.assertRaises(ValueError):r.combine_sources(*packages,refs)
            ref[key]=saved

    def test_prior153_and_nested152_151_150_finalized_chain(self):
        obj=object.__new__(r.Runner);obj.root=Path("/synthetic");obj.inputs={k:obj.root/k for k in r.INPUTS}
        obj.path=lambda p:Path(p) if Path(p).is_absolute() else obj.root/p
        refs={n:{k:dict(sha256=str(n)+"_"+k) for k in r.LEGACY_FILES} for n in range(150,154)}
        refs[153]["plan"]["sha256"]=r.PRIOR_PLAN_SHA;refs[153]["completion"]["sha256"]=r.PRIOR_COMPLETION_SHA
        obj.plan=dict(prior153=refs[153],legacy149={"fixed":True},coverage_sources={k:{"fixed":k} for k in r.COVERAGE_REFS},raw_source_hashes={"fixed":"raw"})
        groups={};module=r.previous
        for n in (153,152,151,150):
            rf=refs[n];folder="outputs/monthly_financial_coverage"+str(n)
            cache={k:k+"_hash" for k in r.CACHE_FILES};pins={str(obj.root/folder/k):v for k,v in cache.items()}
            plan=dict(round=n,status="REGISTERED",arms=list(module.NAMES),hashes={"old":"hash"},legacy149=obj.plan["legacy149"],inputs={k:dict(path=str(v),sha256="digest") for k,v in obj.inputs.items()},raw_source_hashes=obj.plan["raw_source_hashes"])
            if n>150:
                plan["prior"+str(n-1)]=refs[n-1]
                plan["coverage_sources"]={k:obj.plan["coverage_sources"][k] for k in ("batch1","batch2","batch3","batch4")[:n-149]}
                module=module.previous
            else:plan["coverage_source"]=obj.plan["coverage_sources"]["batch1"]
            done=dict(round=n,plan_sha256=rf["plan"]["sha256"],independent_outcome_sha256=rf["outcome"]["sha256"])
            rev=dict(plan_sha256=rf["plan"]["sha256"],cache_sha256=rf["cache"]["sha256"],passed=True,stage="evaluate",artifact_hashes=pins,source_hashes={"independent":"hash"},orders_sha256=cache["PRETRAIN_ORDERS.json"])
            groups[n]=dict(plan=plan,cache=cache,completion=done,outcome=rev)
        obj._refs=lambda rf,*args:(next(groups[n] for n in groups if rf is refs[n]),{})
        obj._closure=lambda d:{str(obj.path(k)):v for k,v in d.items()}
        with patch.object(r,"sha",return_value="digest"):
            obj.require_prior()
            for group_ in groups.values():
                for key,value in (("cache_sha256","wrong"),("orders_sha256","wrong"),("passed",False)):
                    before=group_["outcome"][key];group_["outcome"][key]=value
                    with self.assertRaises(ValueError):obj.require_prior()
                    group_["outcome"][key]=before
            groups[150]["plan"]["coverage_source"]={"swapped":True}
            with self.assertRaises(ValueError):obj.require_prior()

    def test_prepare_receipt_counts_are_exact(self):
        obj=object.__new__(r.Runner);obj.out=Path("/synthetic/out");obj.cache=lambda:{}
        receipt=dict(passed=True,stage="prepare",producer_modules_imported=False,
                     returns_or_nav_computed=False,monthly_label_models_fit=0,
                     orders_checked=854,source_cells_checked=5978,stock_cells_checked=3904)
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


class Batch3Tests(unittest.TestCase):
    def test_complete(self):self.assertEqual(r.validate_source(*source3_fixture(),batch="batch3"),dict(events=10,admitted=5,held=5,catalog_events=32))
    def test_raw_identity_fields_cannot_disappear(self):
        for action in ("delete","change"):
            a=source3_fixture();identity=a[0]["events"][0]["identity"]
            if action=="delete":identity.pop("source_anchor")
            else:identity["source_anchor"]="changed"
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch3")
    def test_identity_certification_not_upgraded(self):
        a=source3_fixture();a[0]["events"][0]["identity"]["certified_daily_security_master"]=True
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch3")
    def test_null_cannot_be_admitted(self):
        a=source3_fixture();a[0]["events"][1]["ni_assets_admission"].update(admitted=True,reasons=[])
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch3")
    def test_accounting_and_calendar_cannot_be_skipped(self):
        for part in ("clock","accounting","boundary"):
            a=source3_fixture();e=a[0]["events"][5];rr=a[3]["event_reviews"][5]
            if part=="clock":rr["clock_verified"]=False
            elif part=="boundary":rr["boundary_assumption_verified"]=False
            else:e["source_review"]["accounting_review"]["status"]="unresolved"
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch3")
    def test_full_catalog_and_prior_counts(self):
        for change in ("catalog","counts"):
            a=source3_fixture()
            if change=="catalog":a[0]["latest_event_catalog"].pop()
            else:a[1]["counts"]["prior_batch2_held"]=4
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch3")
    def test_batch2_control_cannot_change(self):
        for key in ("earnings_batch2","batch2_available"):
            a=feature_fixture();a[0]["arrays"][key][0,2]=0
            with self.assertRaises(ValueError):r.validate_features(*a)
    def test_old_eight_issuers_cannot_be_recomputed(self):
        a=feature_fixture();a[0]["arrays"]["earnings_expanded"][0,2]=.5
        with self.assertRaises(ValueError):r.validate_features(*a)


class Batch4Tests(unittest.TestCase):
    def test_complete_dynamic_count(self):
        for n in (0,4,5,9):
            self.assertEqual(r.validate_source(*source4_fixture(n),batch="batch4"),dict(events=9,admitted=n,held=9-n,catalog_events=23))

    def test_catalog_missing_or_duplicate_rejected(self):
        for change in ("missing","duplicate"):
            a=source4_fixture();rows=a[0]["latest_event_catalog"]
            if change=="missing":rows.pop()
            else:rows[-1]=deepcopy(rows[0])
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch4")

    def test_old_batch3_counts_and_raw_identity_preserved(self):
        for change in ("counts","identity","source","holds"):
            a=source4_fixture()
            if change=="counts":a[1]["counts"]["prior_batch3_held"]=4
            elif change=="identity":a[0]["events"][0]["identity"]["source_anchor"]="changed"
            elif change=="source":a[0]["events"][0]["source"]["sha256"]="bad"
            else:a[0]["events"][0]["context_holds"]=[]
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch4")

    def test_batch4_accounting_and_clock_fail_closed(self):
        for change in ("accounting","clock","boundary","assumption"):
            a=source4_fixture();rr=a[3]["event_reviews"][0]
            if change=="accounting":rr["accounting_review"]["current_prior_common_basis"]=False
            elif change=="clock":rr["clock_verified"]=False
            elif change=="boundary":rr["boundary_assumption_verified"]=False
            else:rr["assumption_id"]="wrong"
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch4")

    def test_batch3_controls_are_immutable(self):
        for key in ("earnings_batch3","batch3_available"):
            a=feature_fixture();a[0]["arrays"][key][0,3]=0
            with self.assertRaises(ValueError):r.validate_features(*a)

    def test_new_issuer_only_and_fixed_49_pool(self):
        self.assertEqual(set(r.factor.NEW_SYMBOLS),{"PLTR","PYPL","TMO"})
        self.assertEqual((len(r.NAMES),len(r.EXTRA),r.WIDTH),(7,12,49))
        a=feature_fixture();a[-1]["symbols"][0][5]="KO"
        with self.assertRaises(ValueError):r.validate_features(*a)

    def test_batch4_catalog_full_equal_and_sentinels(self):
        a=source4_fixture();data=a[0]
        for row in data["latest_event_catalog"]:row.update(fiscal_quarter=4,publication_date="2020-01-01")
        census=dict(annual_documents=deepcopy(data["latest_event_catalog"]))
        for row in data["latest_event_catalog"]:row.update(clock_review={},processed_in_batch=row["event_id"] in {e["event_id"] for e in data["events"]})
        r.validate_catalog(data,census,"batch4")
        data["latest_event_catalog"][-1]["processed_in_batch"]=True
        with self.assertRaises(ValueError):r.validate_catalog(data,census,"batch4")



class IntegrationTests(unittest.TestCase):
    def test_twelve_field_projection_must_match_policy_plan_and_extraction(self):
        for index in (2,8,9):
            args=source4_fixture()
            key="fixed_catalog_metadata" if index==2 else "catalog_metadata"
            args[index][key][0]["context_holds"]=["changed"]
            with self.assertRaisesRegex(ValueError,"projection"):
                r.validate_source(*args,batch="batch4")

    def test_published_nonnull_population_cannot_shrink(self):
        args=source4_fixture(0);field=r.factor.FIELDS[0]
        args[0]["events"][0]["fields"][field]=None
        args[8]["events"][0]["fields"][field]=None
        with self.assertRaisesRegex(ValueError,"Fixed27"):
            r.validate_source(*args,batch="batch4")

    def test_panels_visits_all123_prefixes_for_each_arm(self):
        arrays,meta,backend=scoring_fixture()
        for key,value in list(arrays.items()):arrays[key]=np.tile(value[:1],(122,1))
        meta["symbols"]=[meta["symbols"][0].copy() for _ in range(122)]
        backend.make_orders=lambda choices,scores,meta:scores.tolist()
        backend.baseline=lambda a,m,n,p:(["synthetic"]*n,a["baseline"][:n].copy())
        backend.adapter.score_and_choose=lambda *args,**kwargs:(["synthetic"]*args[5],kwargs["pre_score"][:args[5]].copy())
        original_choose=r.choose;calls=[]
        def traced(name,arrays,meta,n,backend,policy):
            calls.append((name,n));return original_choose(name,arrays,meta,n,backend,policy)
        with patch.object(r,"choose",side_effect=traced):
            orders,scores=r.panels(arrays,meta,backend,{})
        for name in r.NAMES:
            self.assertTrue(set(range(123)) <= {n for arm,n in calls if arm==name})
            self.assertEqual(len(orders[name]),122)
        self.assertEqual(set(scores),set(r.NAMES))

    def test_prepare_wires_all_five_immutable_controls_and_rejects_changed_old_score(self):
        # Entirely generated inputs; the real financial factor is never called.
        args=feature_fixture();result=deepcopy(args[0]);meta=args[-1]
        result["audit"]={}
        originals={"old"+str(i):np.zeros((3,49)) for i in range(27)}
        scores={name:np.full((3,49),float(i)) for i,name in enumerate(r.NAMES)}
        orders={name:[{"synthetic":name}]*3 for name in r.NAMES}
        ref_scores={name:value.copy() for name,value in scores.items() if name!=r.NAMES[-1]}
        ref_orders={name:deepcopy(value) for name,value in orders.items() if name!=r.NAMES[-1]}
        with tempfile.TemporaryDirectory() as tmp:
            obj=object.__new__(r.Runner);obj.root=Path(tmp);obj.out=Path(tmp)
            obj.refresh=Mock();obj.plan=dict(raw_source_hashes={},legacy149={},prior153={},coverage_sources={})
            obj.inputs={"meta":Path(tmp)/"meta","policy":Path(tmp)/"policy","original_orders":Path(tmp)/"orders"}
            backend=Mock();backend.load.return_value=(meta,originals,None,None);backend.price_parity.return_value={}
            obj.backend_factory=Mock(return_value=backend)
            obj.reference_arrays=Mock(return_value=(*args[1:-1],ref_scores,ref_orders))
            source={"synthetic_only":True};obj.require_coverage=Mock(return_value=source);obj.write=Mock()
            def read(path):
                return meta if path==obj.inputs["meta"] else orders[r.NAMES[0]] if path==obj.inputs["original_orders"] else {}
            with patch.object(r,"read",side_effect=read),patch.object(r.base,"validate_meta"),patch.object(r,"panels",return_value=(orders,scores)),patch.object(r.factor,"build",return_value=result) as build,patch.object(r,"sha",return_value="generated"):
                for name in r.NAMES[1:6]:
                    scores[name][0,0]+=1
                    with self.assertRaisesRegex(ValueError,"baseline/control"):
                        obj.prepare()
                    self.assertFalse((obj.out/"DATA.npz").exists())
                    scores[name][0,0]-=1
                obj.prepare()
                self.assertEqual(len(build.call_args.args),7)
                self.assertIs(build.call_args.args[0],meta)
                for i in range(5):self.assertIs(build.call_args.args[i+1],args[1+2*i])
                self.assertIs(build.call_args.args[-1],source)
                self.assertTrue(result["audit"]["prior153_control_scores_and_orders_byte_exact"])
                self.assertTrue(result["audit"]["prior153_batch4_earnings_cache_and_mask_verified"])



class Batch5Tests(unittest.TestCase):
    def test_complete_dynamic_count(self):
        for n in (0,1,19,21):
            self.assertEqual(r.validate_source(*source5_fixture(n),batch="batch5"),dict(events=21,admitted=n,held=21-n,catalog_events=28))

    def test_all_three_issuers_and_full_catalog_required(self):
        for change in ("missing","duplicate","wrong_symbol"):
            a=source5_fixture()
            if change=="missing":a[0]["latest_event_catalog"].pop()
            elif change=="duplicate":a[0]["events"][-1]=deepcopy(a[0]["events"][0])
            else:a[0]["events"][-1]["symbol"]="MU"
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_age_days_policy_is_exact(self):
        for value in (549,551,None):
            a=source5_fixture();a[2]["known_time"]["age_days"]=value
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_policy_and_dataset_uncertainty_must_both_remain_explicit(self):
        for which in ("dataset","publication","scope"):
            a=source5_fixture()
            if which=="dataset":a[0]["certified_daily_security_master"]=True
            elif which=="publication":a[2]["known_time"]["first_public_version_verified"]=True
            else:a[2]["preservation"]["no_first_public_time_or_currency_ISO_or_security_master_certification"]=False
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_batch4_counts_and_both_clock_holds_preserved(self):
        a=source5_fixture();a[1]["counts"]["prior_batch4_held"]=3
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")
        a=source5_fixture();a[0]["events"][-1]["ni_assets_admission"].update(admitted=True,reasons=[])
        a[1]["counts"].update(admitted=20,held=1)
        with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_accounting_boundary_and_money_field_reviews_required(self):
        for change in ("accounting","boundary","amount"):
            a=source5_fixture();rr=a[3]["event_reviews"][0]
            if change=="accounting":rr["accounting_review"]["current_prior_common_basis"]=False
            elif change=="boundary":rr["boundary_assumption_verified"]=False
            else:rr["fields"][r.factor.FIELDS[0]]["source_value_present"]=False
            with self.assertRaises(ValueError):r.validate_source(*a,batch="batch5")

    def test_exact_sixty_three_fields_cannot_be_dropped(self):
        a=source5_fixture(0);name=r.factor.FIELDS[0]
        a[0]["events"][0]["fields"][name]=None;a[8]["events"][0]["fields"][name]=None
        with self.assertRaisesRegex(ValueError,"Fixed63"):r.validate_source(*a,batch="batch5")

    def test_all_catalog_projections_bound(self):
        for index in (2,8,9):
            a=source5_fixture();key="fixed_catalog_metadata" if index==2 else "catalog_metadata"
            a[index][key][-1]["context_holds"]=["changed"]
            with self.assertRaisesRegex(ValueError,"projection"):r.validate_source(*a,batch="batch5")

    def test_frozen_batch4_values_and_mask_cannot_change(self):
        for key in ("earnings_batch4","batch4_available"):
            a=feature_fixture();a[0]["arrays"][key][0,4]=0
            with self.assertRaises(ValueError):r.validate_features(*a)

    def test_new_catalog_no_sentinel_loss(self):
        a=source5_fixture();data=a[0]
        for row in data["latest_event_catalog"]:row.update(fiscal_quarter=4,publication_date="2020-01-01")
        census=dict(annual_documents=deepcopy(data["latest_event_catalog"]))
        for row in data["latest_event_catalog"]:row.update(clock_review={},processed_in_batch=row["event_id"] in {e["event_id"] for e in data["events"]})
        r.validate_catalog(data,census,"batch5")
        data["latest_event_catalog"][-1]["processed_in_batch"]=True
        with self.assertRaises(ValueError):r.validate_catalog(data,census,"batch5")


if __name__ == "__main__":
    unittest.main(verbosity=2)

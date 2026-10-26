"""Registered coverage-only replay using the finalized149 earnings cache.

No old factor is rebuilt. Only a separately admitted AMZN/CSCO batch may add
earnings coverage; every original array and original149 earnings cell stays
unchanged. Actual stages require a frozen PLAN and independent source review.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import monthly_stock_comp149_runner as old
import monthly_financial_coverage150_factor as factor
import monthly_financial_coverage150_selection as selection

base = old.base
require, sha, read = base.require, base.sha, base.read
NAMES, HANDS, DOSE = selection.NAMES, selection.HANDS, selection.DOSE
EXTRA = (*HANDS, "original_available", "expanded_available")
TRAIN, MONTHS, WIDTH = 88, 122, 49
OUT = "outputs/monthly_financial_coverage150"
LEGACY_OUT = "outputs/monthly_stock_comp149"
CACHE_FILES = old.CACHE_FILES
FROZEN_FILES = old.FROZEN_FILES
TOLERANCES = dict(old.TOLERANCES)
INPUTS = ("proposal", "meta", "policy", "original_orders", "prices", "sessions", "protected_manifest")
LEGACY_FILES = dict(plan="PLAN.json", cache="CACHE.json",
                    completion="COMPLETION.json", outcome="INDEPENDENT_OUTCOME.json")
COVERAGE_REFS = {
    "dataset": "outputs/monthly_financial_coverage_admission_batch1_v1/DATASET.json",
    "admission": "outputs/monthly_financial_coverage_admission_batch1_v1/ADMISSION.json",
    "coverage": "outputs/monthly_financial_coverage_admission_batch1_v1/COVERAGE.json",
    "manifest": "outputs/monthly_financial_coverage_admission_batch1_v1/MANIFEST.json",
    "policy": "work/after149_batch1_admission_policy.json",
    "review": "work/after149_batch1_independent_review.json",
}
SOURCE_BATCH = "outputs/monthly_financial_coverage_batch1_v1"
ADDED_EVENT_KEYS = {"identity", "ni_assets_admission", "source_review", "clock_review"}
CHECKS = (
    "original27_arrays_byte_identical", "all_features_independently_reconstructed",
    "all_three_score_matrices_and_ordered_choices_independently_verified",
    "one_episode_and_prefix_isolation_verified", "zero_and_missing_exact_fallback",
    "original149_earnings_cache_and_mask_verified", "coverage_source_admission_verified",
    "old44_source_records_and_holds_preserved", "latest_catalog_blocking_independently_verified",
)
select = selection.select
choose = selection.choose
selection_contract = selection.selection_contract


def measurement_contract():
    return dict(
        method=factor.METHOD, definition=factor.DEFINITION,
        earnings="NI_current/((assets_current+assets_prior)/2)",
        bounded="h(x)=x/(1+abs(x))",
        exact_arithmetic="scaled_Fraction_then_one_bounded_float",
        original_control="verified_final149_CACHE_earnings_bounded_and_common_available",
        old_events=44, old_admitted=31, old_held=13,
        new_symbols=["AMZN", "CSCO"], source_events=26, annual_catalog_events=32,
        masks="distinct_original_and_expanded_with_original_cells_exactly_preserved",
        time="current_release_next_pinned_session_330_400_day_current_year_550_day_identity_window",
        latest_event_before_validity=True, invalid_latest_blocks_fallback=True,
        placement="add_0.01_h_to_original_pre_score_then_one_original_episode_selector",
        unchanged_original_arrays=27, monthly_equal_weight_positions=3,
        intramonth_trades=False, etf_financial_features=False,
        new_model_fits=0, no_parameter_search=True)


def validate_proposal(p):
    require(p.get("round") == 150 and p.get("arms") == list(NAMES)
            and p.get("primary_challenger") == NAMES[2] and p.get("dose") == DOSE,
            "Wrong prospective coverage experiment")
    require(p.get("status") == "FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION",
            "Final prospective proposal required")
    require(p.get("selection_contract") == selection_contract()
            and p.get("measurement") == measurement_contract(), "Prospective contract changed")
    for key in ("actual_new_factors", "actual_new_orders", "actual_new_returns"):
        require(type(p.get(key)) is int and p[key] == 0, "No actual calculations before registration")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_source(data, admission, policy, review, manifest, coverage, refs, old_data,
                    extraction, source_plan):
    """Source/schema validation only; no factor or source amount arithmetic."""
    require(data.get("schema_version") == 1 and data.get("method") == factor.METHOD,
            "Explicit expanded source dataset required")
    require(admission.get("passed") is True and
            admission.get("stage") == "conditional_source_admission_not_strategy_registration",
            "Conditional source admission required")
    require(manifest.get("status") == "conditional_source_package_no_strategy_registration",
            "Final source manifest required")
    require(manifest.get("outputs") == {
        "DATASET.json": refs["dataset"]["sha256"], "ADMISSION.json": refs["admission"]["sha256"],
        "COVERAGE.json": refs["coverage"]["sha256"]}, "Exact source output hashes required")
    require(manifest.get("input_hashes") == admission.get("input_hashes")
            and bool(manifest.get("input_hashes")), "Source input closures differ")
    for name, refkey in (("policy_sha256", "policy"), ("independent_review_sha256", "review")):
        require(data.get(name) == admission.get(name) == refs[refkey]["sha256"],
                "Source policy/review linkage differs")
    require(data.get("source_plan_sha256") == admission.get("source_plan_sha256")
            == extraction.get("plan_sha256") == review.get("plan_sha256"),
            "Source extraction plan linkage differs")
    require(review.get("passed") is True, "Independent physical source review required")
    require(policy.get("stage") == "prospective_source_policy_not_strategy_registration"
            and policy.get("required_fields") == list(factor.FIELDS)
            and policy.get("source_scope", {}).get("symbols") == ["AMZN", "CSCO"]
            and policy["source_scope"].get("no_automatic_admission") is True
            and policy.get("known_time", {}).get("maximum_age_days") == 550,
            "Source policy contract changed")
    for key in ("first_public_version_verified", "complete_correction_history_verified",
                "certified_daily_security_master"):
        require(data.get(key) is False and policy.get("preservation", {}).get(key) is False,
                "Source assumptions cannot be upgraded")
    for key in ("scores_computed", "ratios_computed", "returns_computed", "new_strategy_trials"):
        require(type(data.get(key)) is int and data[key] == 0, "Source-only dataset required")
    for key in ("actual_factors_calculated", "actual_returns_calculated", "new_strategy_trials"):
        require(type(admission.get(key)) is int and admission[key] == 0, "Source-only receipt required")
    events, catalog = data["events"], data["latest_event_catalog"]
    ids = [e["event_id"] for e in events]
    catalog_ids = [e["event_id"] for e in catalog]
    require(len(ids) == len(set(ids)) == 26 and
            Counter(e["symbol"] for e in events) == {"AMZN": 10, "CSCO": 16},
            "Complete fixed26 event population required")
    require(len(catalog_ids) == len(set(catalog_ids)) == 32 and set(ids) <= set(catalog_ids)
            and Counter(e["symbol"] for e in catalog) == {"AMZN": 16, "CSCO": 16},
            "Complete fixed32 annual catalog required")
    require(ids == [e["event_id"] for e in extraction["events"]]
            == [e["event_id"] for e in source_plan["records"]], "Extraction order changed")
    require([{k: v for k, v in e.items() if k not in ADDED_EVENT_KEYS} for e in events]
            == extraction["events"], "Original extraction values/metadata changed")
    reviews = review.get("event_reviews", [])
    reviewed = {e["event_id"]: e for e in reviews}
    require(len(reviewed) == len(reviews) == 26 and set(reviewed) == set(ids),
            "Independent review population differs")
    admitted = 0
    for event in events:
        gate = event["ni_assets_admission"]
        require(set(gate) == {"admitted", "reasons", "definition"} and
                type(gate["admitted"]) is bool and type(gate["reasons"]) is list and
                gate["definition"] == factor.DEFINITION, "New source gate schema changed")
        require(all(type(reason) is str and reason for reason in gate["reasons"]),
                "Explicit source reasons required")
        require(bool(gate["reasons"]) is (not gate["admitted"]), "Gate/reasons disagree")
        if event["fiscal_year"] == 2019:
            require(not gate["admitted"], "FY2019 accounting hold removed")
        admitted += gate["admitted"]
        rr = reviewed[event["event_id"]]
        require(rr.get("passed") is True and rr.get("metadata_verified") is True
                and set(rr.get("fields", {})) == set(factor.FIELDS), "Incomplete physical review")
        for name in factor.FIELDS:
            null = event["fields"][name] is None
            fr = rr["fields"][name]
            require(fr.get("is_null") is null and
                    (fr.get("null_reason_verified") is True if null else
                     fr.get("verified") is True and fr.get("source_value_present") is True),
                    "Source field not independently verified")
    counts = admission["counts"]
    require(all(counts.get(k) == v for k, v in dict(
        events=26, admitted=admitted, held=26-admitted, annual_catalog_events=32,
        unprocessed_annual_catalog_events=6, original_events=44,
        original_admitted=31, original_held=13).items()), "Source admission counts differ")
    old_events = old_data["events"]
    require(len(old_events) == len({e["event_id"] for e in old_events}) == 44
            and sum(e["stock_compensation_admission"]["admitted"] is True for e in old_events) == 31,
            "Original44/31/13 changed")
    ref = data["original_four_issuer_reference"]
    require(ref.get("events") == 44 and ref.get("admitted") == 31 and ref.get("held") == 13
            and ref.get("preserved_without_modification") is True
            and ref.get("records_canonical_sha256") == canonical_hash(old_events),
            "Original source records not preserved")
    require(coverage.get("counts", {}).get("months") == 122
            and coverage["counts"].get("stocks") == 73
            and coverage["counts"].get("stock_slots") == 3904,
            "Original candidate census changed")
    return dict(events=26, admitted=admitted, held=26-admitted, catalog_events=32)


def validate_features(result, original_earnings, original_mask, meta):
    arrays = result["arrays"]
    require(set(arrays) == set(EXTRA), "Exact coverage arrays required")
    shape = (len(meta["dates"]), WIDTH)
    for hand, mask in zip(HANDS, EXTRA[2:]):
        require(arrays[hand].shape == arrays[mask].shape == shape
                and arrays[hand].dtype == np.float64 and arrays[mask].dtype == bool
                and not np.isinf(arrays[hand]).any()
                and np.array_equal(np.isfinite(arrays[hand]), arrays[mask]),
                "Feature shape/type/mask mismatch")
        require(np.all(np.abs(arrays[hand][arrays[mask]]) <= 1), "Bounded earnings required")
    require(arrays[HANDS[0]].tobytes() == original_earnings.tobytes()
            and arrays[EXTRA[2]].tobytes() == original_mask.tobytes(), "Original149 hand/mask changed")
    require(np.all(~original_mask | arrays[EXTRA[3]])
            and arrays[HANDS[1]][original_mask].tobytes() == original_earnings[original_mask].tobytes(),
            "Old valid earnings coverage changed")
    for i, names in enumerate(meta["symbols"]):
        for j, name in enumerate(names):
            if name not in factor.NEW_SYMBOLS:
                require(arrays[HANDS[0]][i:i+1, j:j+1].tobytes()
                        == arrays[HANDS[1]][i:i+1, j:j+1].tobytes(), "Nonbatch coverage changed")


def validate_catalog(dataset, census):
    """The complete source event timeline, including unprocessed events, is fixed."""
    expected = [row for row in census["annual_documents"] if row.get("symbol") in factor.NEW_SYMBOLS]
    catalog = dataset["latest_event_catalog"]
    stripped = [{k: v for k, v in row.items() if k not in {"clock_review", "processed_in_batch"}}
                for row in catalog]
    require(stripped == expected and len(expected) == 32
            and all(row.get("fiscal_quarter") == 4 for row in expected),
            "Complete original annual catalog changed")
    extracted = {row["event_id"] for row in dataset["events"]}
    require(all(row.get("processed_in_batch") is (row["event_id"] in extracted) for row in catalog),
            "Unprocessed catalog sentinel changed")


def panels(arrays, meta, backend, policy):
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    orders, scores = {}, {}
    for name in NAMES:
        choices, score = choose(name, arrays, meta, MONTHS, backend, policy)
        orders[name], scores[name] = backend.make_orders(choices, score, meta), score
        for n in range(MONTHS+1):
            pc, ps = choose(name, arrays, meta, n, backend, policy)
            require(pc == choices[:n] and ps.tobytes() == score[:n].tobytes(),
                    "Future-prefix dependence")
    for value in (0., np.nan):
        null = dict(arrays)
        for hand in HANDS:
            null[hand] = np.full((MONTHS, WIDTH), np.nan)
            for i, names in enumerate(meta["symbols"]):
                for j, symbol in enumerate(names):
                    if symbol not in meta["etfs"]:
                        null[hand][i, j] = value
        for name in NAMES[1:]:
            c, s = choose(name, null, meta, MONTHS, backend, policy)
            require(backend.make_orders(c, s, meta) == orders[NAMES[0]]
                    and s.tobytes() == scores[NAMES[0]].tobytes(), "Zero/missing fallback changed")
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()},
            "Input arrays changed")
    return orders, scores


class Backend(old.Backend):
    def training(self, orders, meta, px, op):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            picked = [[meta["symbols"][i].index(s) for s in row["symbols"]]
                      for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(picked, blocks, valid)
            returns.append(rr)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
        opened = self.open_training(orders, meta, px, op, TRAIN, NAMES)
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **self.local.old.session.measure(rr, meta, bounds, n))
                      for name, rr in zip(NAMES, returns)]
            oprefix = self.open_training(orders, meta, px, op, n, NAMES)
            pick = select(prefix, oprefix)
            folds.append(dict(n=n, end=end, selection=pick, open_training=oprefix))
            stitched.extend(returns[NAMES.index(pick["primary"])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
        return rows, opened, dict(
            folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), meta["dates"][36], meta["ends"][87]),
            incumbent=rp.stats_rr(np.asarray(control), meta["dates"][36], meta["ends"][87]))

    def diagnostics(self, orders, meta, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (("close", 0., "same_close"), ("open", 0., "next_open_zero"),
                                     ("open", .0005, "next_open_cost")):
                met, eq, trades, holdings = self.local.ex.e.engine.replay(
                    px, op, orders[name], cost=cost, execution=mode,
                    start=pd.Timestamp(meta["dates"][0]), end=pd.Timestamp(meta["ends"][-1]))
                for row in met["annual_returns"]:
                    if row["year"] in (2016, 2026):
                        row["partial"] = True
                results[name][key] = dict(metrics=met, slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name+"_"+key+"_DAILY.json",
                       [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name+"_"+key+"_TRADES.json", trades)
                writer(name+"_"+key+"_HOLDINGS.json", holdings)
        return results


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root, self.plan_path = Path(root).resolve(), Path(plan_path).resolve()
        self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, "Frozen PLAN changed")
        self.plan = p = read(self.plan_path)
        require(p["round"] == 150 and p["status"] == "REGISTERED" and p["arms"] == list(NAMES)
                and p["primary_challenger"] == NAMES[2], "Registered150 PLAN required")
        require(p["dose"] == DOSE and p["tolerances"] == TOLERANCES, "Declared policy changed")
        require(p["runtime"] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                "Pinned runtime changed")
        self.pins = self.verify(p["hashes"])
        require(set(p["inputs"]) == set(INPUTS), "Exact named inputs required")
        self.inputs = {k: self.bound_path(v["path"]) for k, v in p["inputs"].items()}
        require(all(sha(self.inputs[k]) == v["sha256"] for k, v in p["inputs"].items()), "Input changed")
        proposal = read(self.inputs["proposal"])
        validate_proposal(proposal)
        require(p["legacy149"] == proposal["sources"]["legacy149"]
                and p["coverage_source"] == proposal["sources"]["coverage_source"],
                "Prospective source references changed")
        for path, digest in source_files(self.root).items():
            require(self.pins.get(path) == digest, "Implementation not registered")
        self.out = self.root / OUT
        require(self.plan_path == self.out/"PLAN.json" and Path(p["output_directory"]) == self.out,
                "Dedicated150 directory required")
        self.backend_factory = backend_factory
        self.require_legacy()
        self.require_coverage()

    def _closure(self, mapping):
        verified = self.verify(mapping)
        require(all(self.pins.get(k) == v for k, v in verified.items()), "Unregistered source closure")
        return verified

    def _refs(self, refs, expected):
        require(set(refs) == set(expected), "Exact source refs required")
        objects, paths = {}, {}
        for key, name in expected.items():
            ref = refs[key]
            require(set(ref) == {"path", "sha256"}, "Exact path/hash reference required")
            path = self.bound_path(ref["path"])
            require(path == self.root/name and sha(path) == ref["sha256"], "Reference path/hash changed")
            paths[key], objects[key] = path, read(path)
        return objects, paths

    def require_legacy(self):
        objects, paths = self._refs(
            self.plan["legacy149"], {k: LEGACY_OUT+"/"+v for k, v in LEGACY_FILES.items()})
        p, cache, done, review = (objects[k] for k in ("plan", "cache", "completion", "outcome"))
        ref = self.plan["legacy149"]
        require(p["round"] == done["round"] == 149 and p["status"] == "REGISTERED",
                "Completed149 predecessor required")
        require(done["plan_sha256"] == review["plan_sha256"] == ref["plan"]["sha256"]
                and done["independent_outcome_sha256"] == ref["outcome"]["sha256"]
                and review["cache_sha256"] == ref["cache"]["sha256"]
                and review.get("passed") is True and review.get("stage") == "evaluate",
                "Finalized149 audit/cache chain required")
        self._closure(p["hashes"])
        require(set(cache) == set(CACHE_FILES), "Exact149 cache required")
        cachepins = self._closure({LEGACY_OUT+"/"+name: digest for name, digest in cache.items()})
        audited = self._closure(review["artifact_hashes"])
        require(all(audited.get(path) == digest for path, digest in cachepins.items()),
                "149 cache not covered by final independent audit")
        require(review["orders_sha256"] == cache["PRETRAIN_ORDERS.json"], "149 order binding differs")
        for key in INPUTS:
            if key != "proposal":
                require(self.inputs[key] == self.path(p["inputs"][key]["path"])
                        and sha(self.inputs[key]) == p["inputs"][key]["sha256"], "Original input rewired")
        require(self.plan["raw_source_hashes"] == p["raw_source_hashes"], "Original price sources changed")
        self._closure(review["source_hashes"])
        return objects

    def require_coverage(self):
        objects, paths = self._refs(self.plan["coverage_source"], COVERAGE_REFS)
        refs = self.plan["coverage_source"]
        data, admission, review = (objects[k] for k in ("dataset", "admission", "review"))
        inputs = self._closure(admission["input_hashes"])
        self._closure(objects["manifest"]["input_hashes"])
        reviewed = self._closure(review["input_hashes"])
        source_plan_path = self.bound_path(SOURCE_BATCH+"/PLAN.json")
        extraction_path = self.bound_path(SOURCE_BATCH+"/EXTRACTIONS.json")
        source_plan_sha, extraction_sha = sha(source_plan_path), sha(extraction_path)
        require(data["source_plan_sha256"] == source_plan_sha
                and review["extractions_sha256"] == extraction_sha, "Extraction linkage changed")
        for path in (source_plan_path, extraction_path, paths["policy"], paths["review"]):
            require(inputs.get(str(path)) == sha(path), "Admission omitted mandatory dependency")
        for path in (source_plan_path, extraction_path):
            require(reviewed.get(str(path)) == sha(path), "Physical review omitted extraction binding")
        source_plan = read(source_plan_path)
        self._closure(source_plan["inputs"])
        self._closure(source_plan["code"])
        census_path = self.bound_path("outputs/monthly_financial_coverage_census_v1/CENSUS.json")
        require(inputs.get(str(census_path)) == sha(census_path), "Source omitted full candidate census")
        validate_catalog(data, read(census_path))
        ref = data["original_four_issuer_reference"]
        legacy = read(self.bound_path(LEGACY_OUT+"/PLAN.json"))
        require(self.path(ref["path"]) == self.path(legacy["sbc_dataset"]["path"])
                and ref["sha256"] == legacy["sbc_dataset"]["sha256"]
                and self.path(ref["admission_path"]) == self.path(legacy["sbc_admission"]["path"])
                and ref["admission_sha256"] == legacy["sbc_admission"]["sha256"],
                "Original149 source references changed")
        old_data = read(self.bound_path(ref["path"]))
        require(sha(self.bound_path(ref["admission_path"])) == ref["admission_sha256"],
                "Original149 source receipt changed")
        validate_source(data, admission, objects["policy"], review, objects["manifest"],
                        objects["coverage"], refs, old_data, read(extraction_path), source_plan)
        for event in data["events"]:
            path = self.bound_path(event["source"]["path"])
            require(sha(path) == event["source"]["sha256"]
                    and reviewed.get(str(path)) == sha(path), "New raw source not independently reviewed")
        return data

    def reference_arrays(self, meta, original):
        """Read the already finalized149 cache without invoking its factor."""
        self.require_legacy()
        directory = self.root/LEGACY_OUT
        require(read(directory/"META.json") == meta and len(original) == 27, "Original metadata changed")
        with np.load(directory/"DATA.npz", allow_pickle=False) as cache:
            require(all(k in cache.files and cache[k].dtype == v.dtype and cache[k].shape == v.shape
                        and cache[k].tobytes() == v.tobytes() for k, v in original.items()),
                    "Original27 arrays differ from149")
            earnings, mask = cache["earnings_bounded"].copy(), cache["common_available"].copy()
        require(earnings.dtype == np.float64 and earnings.shape == (MONTHS, WIDTH)
                and mask.dtype == bool and mask.shape == earnings.shape
                and np.array_equal(np.isfinite(earnings), mask) and not np.isinf(earnings).any(),
                "Original149 earnings/mask invalid")
        with np.load(directory/"SCORES.npz", allow_pickle=False) as cache:
            scores = {NAMES[0]: cache[old.NAMES[0]].copy(), NAMES[1]: cache["earnings_control"].copy()}
        old_orders = read(directory/"PRETRAIN_ORDERS.json")
        orders = {NAMES[0]: old_orders[old.NAMES[0]], NAMES[1]: old_orders["earnings_control"]}
        return earnings, mask, scores, orders

    def prepare(self):
        self.refresh()
        require(not any((self.out/x).exists() for x in (*CACHE_FILES, "CACHE.json")),
                "Preparation already started; preserve partial outputs")
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        base.validate_meta(meta)
        require(len(original) == 27 and meta == read(self.inputs["meta"]), "Original cache changed")
        parity = backend.price_parity(meta, px, op, self.plan["raw_source_hashes"])
        earnings, mask, ref_scores, ref_orders = self.reference_arrays(meta, original)
        result = factor.build(meta, earnings, self.require_coverage())
        validate_features(result, earnings, mask, meta)
        arrays = dict(original, **result["arrays"])
        orders, scores = panels(arrays, meta, backend, read(self.inputs["policy"]))
        require(orders[NAMES[0]] == read(self.inputs["original_orders"]), "Incumbent changed")
        for name in NAMES[:2]:
            require(orders[name] == ref_orders[name] and scores[name].tobytes() == ref_scores[name].tobytes(),
                    "Final149 baseline/control replay changed")
        result["audit"].update(
            legacy149=self.plan["legacy149"], coverage_source=self.plan["coverage_source"],
            original149_earnings_cache_and_mask_verified=True,
            original149_control_scores_and_orders_byte_exact=True)
        for filename, values in (("DATA.npz", arrays), ("SCORES.npz", scores)):
            with (self.out/filename).open("xb") as stream:
                np.savez_compressed(stream, **values)
        for filename, value in (("META.json", meta), ("PRETRAIN_ORDERS.json", orders),
                                ("PRICE_PARITY.json", parity), ("FEATURE_AUDIT.json", result["audit"])):
            self.write(filename, value)
        self.write("CACHE.json", {name: sha(self.out/name) for name in CACHE_FILES})

    def loaded(self):
        self.cache()
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        earnings, mask, _, _ = self.reference_arrays(meta, original)
        require(meta == read(self.out/"META.json"), "Original metadata changed")
        with np.load(self.out/"DATA.npz", allow_pickle=False) as data:
            require(set(data.files) == set(original)|set(EXTRA), "Prepared array names changed")
            require(all(data[k].dtype == v.dtype and data[k].shape == v.shape
                        and data[k].tobytes() == v.tobytes() for k, v in original.items()),
                    "Original27 arrays changed")
            validate_features(dict(arrays={k: data[k] for k in EXTRA}), earnings, mask, meta)
        return backend, meta, px, op

    def require_prepare_review(self):
        cache = self.cache()
        review = read(self.out/"INDEPENDENT_PREPARE.json")
        require(review.get("passed") is True and review.get("stage") == "prepare"
                and all(review.get(k) is True for k in CHECKS), "Independent prepare guarantees incomplete")
        require(review.get("producer_modules_imported") is False
                and review.get("returns_or_nav_computed") is False
                and type(review.get("monthly_label_models_fit")) is int
                and review["monthly_label_models_fit"] == 0, "Independent source/prepare isolation")
        require(self.verify(review["source_hashes"]) ==
                self.verify(self.plan["independent_prepare_source_hashes"]), "Reviewer closure changed")
        artifacts = self.verify(review["artifact_hashes"])
        required = dict(cache, **{str(self.out/n): sha(self.out/n) for n in ("PLAN.json", "CACHE.json")})
        require(all(artifacts.get(k) == v for k, v in required.items()), "Unreviewed prepared artifact")
        allowed = dict(self.pins, **required)
        require(all(allowed.get(k) == v for k, v in artifacts.items()), "Unbound reviewer artifact")
        for key, name in (("plan_sha256", "PLAN.json"), ("cache_sha256", "CACHE.json"),
                          ("orders_sha256", "PRETRAIN_ORDERS.json")):
            require(review.get(key) == sha(self.out/name), "Prepare review link changed")

    def select_first88(self):
        self.refresh()
        self.require_prepare_review()
        require(not any((self.out/x).exists() for x in
                        ("TRAIN_TRIALS.json", "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "FROZEN.json")),
                "Selection already started")
        backend, meta, px, op = self.loaded()
        rows, opened, walk = backend.training(read(self.out/"PRETRAIN_ORDERS.json"), meta, px, op)
        self.write("TRAIN_TRIALS.json", rows)
        self.write("TRAIN_OPEN_TRIALS.json", opened)
        self.write("WALKFORWARD.json", walk)
        self.write("FROZEN.json", dict(selection=select(rows, opened), pristine_oos=False,
            all_diagnostic_orders_frozen=True, hashes={name: sha(self.out/name) for name in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh()
        self.require_prepare_review()
        frozen = read(self.out/"FROZEN.json")
        require(set(frozen["hashes"]) == set(FROZEN_FILES), "Frozen artifact list changed")
        self.verify({str(self.out/k): v for k, v in frozen["hashes"].items()})
        require(frozen["selection"] == select(read(self.out/"TRAIN_TRIALS.json"),
                                             read(self.out/"TRAIN_OPEN_TRIALS.json")), "Selection changed")
        require(not (self.out/"DIAGNOSE_STARTED.json").exists(), "Diagnostic already started")
        self.write("DIAGNOSE_STARTED.json", dict(plan_sha256=self.plan_sha,
            started_utc=datetime.now(timezone.utc).isoformat()))
        backend, meta, px, op = self.loaded()
        results = backend.diagnostics(read(self.out/"PRETRAIN_ORDERS.json"), meta, px, op, self.write)
        for filename, mode in (("TRAIN_TRIALS.json", "same_close"), ("TRAIN_OPEN_TRIALS.json", "next_open_cost")):
            for row in read(self.out/filename):
                require(abs(results[row["name"]][mode]["slices"]["training2016_2023"]["cagr"]
                            - row["training"]["cagr"]) < TOLERANCES["training_cagr"],
                        "Training replay parity")
        self.write("RESULTS.json", results)
        self.write("VALIDATION.json", dict(selection=frozen["selection"], no_observer_writes=True,
            no_intramonth_trades=True, new_model_fits=0, independent_outcome_review_pending=True,
            pristine_oos=False))


def source_files(root):
    root = Path(root)
    files = (Path(__file__), Path(factor.__file__), Path(selection.__file__),
             root/"work/test_monthly_financial_coverage150_runner.py",
             root/"work/test_monthly_financial_coverage150_factor.py",
             root/"work/test_monthly_financial_coverage150_selection.py")
    return dict(old.source_files(root), **{str(p.resolve()): sha(p) for p in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "select_first88", "diagnose_later34"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha), args.phase)()

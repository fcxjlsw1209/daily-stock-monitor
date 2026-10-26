"""Register/close the fixed session-clustering experiment; no market computation.

Registration verifies the prospective source closure and inherited evidence.
Only explicit CLI phases write PLAN/continuation/completion records.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse
import importlib
import json
import os
import re
import tempfile
import sys
import hashlib
import fcntl
from contextlib import contextmanager
import monthly_session_cluster148_runner as r
import complete_monthly_same_report145 as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = "monthly_session_cluster148"
PREVIOUS = "monthly_asset_growth147"
STATE = "outputs/monthly_target70_research/CONTINUATION_STATUS.json"
PROPOSAL = "work/monthly_session_cluster148_proposal.json"
PROPOSAL_SHA = "04ddd0ae78d2a65c268cb98e5466905f15b6a50eb1203b56182854333e84714e"
PREVIOUS_PLAN = "61f94781edb5c4f6f98f9c4753db8a4d99fe887b795fb39e120270402c7549c1"
PREVIOUS_DONE = "ac0bc2e3b0aa5860c1878b3f1f0e746837cc591ea170682ec059c53fa7b2accb"
SUITE_SOURCES = {
    "factor": "work/test_monthly_session_cluster148_factor.py",
    "runner": "work/test_monthly_session_cluster148_runner.py",
    "prepare_review": "work/test_monthly_session_cluster148_prepare_independent.py",
    "outcome_review": "work/test_monthly_session_cluster148_outcome_independent.py",
    "lifecycle": "work/test_monthly_session_cluster148_lifecycle.py",
}
REVIEW_MODULES = {
    "prepare": "monthly_session_cluster148_prepare_independent",
    "outcome": "monthly_session_cluster148_outcome_independent",
}
PREREVIEWS = {
    "prepare": "work/monthly_session_cluster148_prepare_review_SYNTHETIC.json",
    "outcome": "work/monthly_session_cluster148_outcome_prereview.json",
}
require = r.require


def write(path, value):
    with Path(path).open("xb") as f:
        f.write(old.encoded(value))


def state_snapshot(path):
    raw = Path(path).read_bytes()
    return old.decode(raw), hashlib.sha256(raw).hexdigest()


@contextmanager
def state_lock(root):
    """Serialize lifecycle writers; root is the sole central-state writer."""
    path = Path(root)/"outputs/monthly_target70_research/.lifecycle.lock"
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def preserve_completion(path, done):
    if Path(path).exists():
        saved = r.read(path)
        require({k: v for k, v in saved.items() if k != "completed_utc"} ==
                {k: v for k, v in done.items() if k != "completed_utc"},
                "Existing completion conflicts with revalidated evidence")
        return saved
    write(path, done)
    return done


def predecessor(state):
    items = state["completed"]
    require(len(items) == 147 and sum(x["trials"] for x in items) == 943,
            "Expected147 completed rounds/943 trials")
    require(items[-1]["directory"] == PREVIOUS
            and items[-1]["completion"]["sha256"] == PREVIOUS_DONE,
            "Wrong preceding completion")
    require(not any(x["directory"] == STUDY for x in items), "Already completed148")


def update_state(path, expected, value):
    """Preserve an exact prior copy and fail if a concurrent writer intervened."""
    require(r.sha(path) == expected, "Concurrent continuation update")
    backup = path.parent/("PRE148_"+expected+".json")
    if not backup.exists():
        with backup.open("xb") as f:
            f.write(path.read_bytes())
    require(r.sha(backup) == expected, "Continuation backup changed")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(old.encoded(value))
        f.flush()
        os.fsync(f.fileno())
    try:
        require(r.sha(path) == expected, "Concurrent continuation commit")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_tests(doc, reader):
    require(doc.get("passed") is True and doc.get("synthetic_only") is True,
            "Prospective synthetic tests required")
    names, count = set(), 0
    for suite in doc["suites"]:
        name = suite["name"]
        require(name in SUITE_SOURCES and name not in names
                and type(suite["count"]) is int and suite["count"] > 0,
                "Distinct positive declared suites")
        names.add(name)
        require(reader.path(suite["source"]["path"]) == reader.path(SUITE_SOURCES[name]),
                "Wrong synthetic test source")
        for key in ("source", "log"):
            ref = suite[key]
            require(old.digest(ref["sha256"]) and reader.hash(ref["path"]) == ref["sha256"],
                    "Changed synthetic "+key)
        log = reader.path(suite["log"]["path"]).read_text()
        require(re.search(r"Ran "+str(suite["count"])+r" tests? in ", log)
                and log.rstrip().endswith("OK"), "Synthetic test failure")
        count += suite["count"]
    require(names == set(SUITE_SOURCES) and type(doc["test_count"]) is int
            and doc["test_count"] == count, "Complete synthetic scope")


def independent_sources(root):
    """Import only auditors' declarations; never call their actual audit phases."""
    return {name: importlib.import_module(module).source_files(root)
            for name, module in REVIEW_MODULES.items()}


def validate_prereview(reader, review, sources):
    require(review.get("passed") is True and review.get("actual_factors_calculated") == 0
            and review.get("actual_returns_calculated") == 0,
            "Prospective independent review")
    require(reader.verify(review["source_hashes"]) == reader.verify(sources),
            "Prospective review source closure differs")
    contracts = reader.verify(review["reviewed_contract_hashes"])
    for path, digest in ((PROPOSAL, PROPOSAL_SHA),
                         ("work/monthly_session_cluster148_runner.py",
                          reader.hash("work/monthly_session_cluster148_runner.py"))):
        require(contracts.get(str(reader.path(path))) == digest,
                "Prospective review omitted current contract")


def hypothesis_and_inputs(reader, proposal, previous):
    """Return exactly seven inputs, with no financial admission dependency."""
    r.validate_proposal(proposal)
    for key in ("concept", "novelty", "bounded_archive_novelty"):
        ref = proposal[key]
        require(reader.hash(ref["path"]) == ref["sha256"],
                "Prospective hypothesis evidence changed")
    expected = {
        "prices": dict(path=r.PRICE, sha256=r.PRICE_SHA),
        "sessions": dict(path=r.SESSIONS, sha256=r.SESSIONS_SHA),
        "calendar": dict(path=r.CALENDAR, sha256=r.CALENDAR_SHA),
        "predecessor": dict(path="outputs/"+PREVIOUS+"/COMPLETION.json", sha256=PREVIOUS_DONE),
    }
    require(proposal["sources"] == expected, "Exact original price/calendar provenance required")
    for ref in expected.values():
        require(reader.hash(ref["path"]) == ref["sha256"], "Price provenance changed")
    calendar = reader.read(r.CALENDAR)
    sessions = reader.read(r.SESSIONS)
    require(calendar["start"] == "2015-01-01" and calendar["end"] == "2027-12-31"
            and sessions["plan_sha256"] == r.CALENDAR_SHA, "Calendar provenance/coverage")
    reader.verify(calendar["bindings"])
    inputs = {key: deepcopy(previous["inputs"][key]) for key in r.INPUTS if key != "proposal"}
    inputs["proposal"] = dict(path=PROPOSAL, sha256=PROPOSAL_SHA)
    require(set(inputs) == set(r.INPUTS), "Exact seven named inputs")
    for key, ref in inputs.items():
        require(reader.hash(ref["path"]) == ref["sha256"], "Named input changed: "+key)
        ref["path"] = str(reader.path(ref["path"]))
    for key in ("prices", "sessions"):
        require(inputs[key] == dict(path=str(reader.path(expected[key]["path"])),
                                   sha256=expected[key]["sha256"]), "Inherited input differs: "+key)
    return inputs


def build_plan(root, tests):
    root = Path(root).resolve()
    reader = old.Reader(root)
    out = root/"outputs"/STUDY
    require(not any((out/name).exists() for name in
                    ("PLAN.json", "CACHE.json", "FROZEN.json", "RESULTS.json", "COMPLETION.json",
                     *r.CACHE_FILES)), "Registration must precede actual outputs")
    state, state_sha = state_snapshot(root/STATE)
    predecessor(state)
    require(state["registered_queue"] == [], "Pending experiment")
    prior_dir = "outputs/"+PREVIOUS+"/"
    require(reader.hash(prior_dir+"COMPLETION.json") == PREVIOUS_DONE,
            "Predecessor receipt changed")
    require(reader.hash(prior_dir+"PLAN.json") == PREVIOUS_PLAN, "Predecessor PLAN changed")
    previous = reader.read(prior_dir+"PLAN.json")
    done = reader.read(prior_dir+"COMPLETION.json")
    require(done["round"] == 147 and done["plan_sha256"] == PREVIOUS_PLAN
            and done["directory"] == PREVIOUS and done["trials"] == 4,
            "Wrong preceding completion metadata")
    require(reader.hash(prior_dir+"INDEPENDENT_OUTCOME.json") == done["independent_outcome_sha256"]
            and reader.hash(prior_dir+"PROTECTED_AFTER.json") == done["protected_sha256"],
            "Predecessor final evidence changed")
    reader.verify(previous["hashes"])
    require(len(previous["raw_source_hashes"]) == 64, "Original raw price scope")
    reader.verify(previous["raw_source_hashes"])
    reader.verify(r.source_files(root))
    for path in (Path(__file__), Path(old.__file__), root/SUITE_SOURCES["lifecycle"]):
        reader.hash(path)
    independent = {}
    for name, sources in independent_sources(root).items():
        independent[name] = {str(reader.path(path)): reader.hash(path) for path in sources}
        validate_prereview(reader, reader.read(PREREVIEWS[name]), independent[name])
    require(set(independent) == set(REVIEW_MODULES), "Both independent auditors required")
    require(reader.hash(PROPOSAL) == PROPOSAL_SHA, "Prospective proposal changed")
    proposal = reader.read(PROPOSAL)
    inputs = hypothesis_and_inputs(reader, proposal, previous)
    test_doc = reader.read(tests)
    validate_tests(test_doc, reader)
    # Mutable continuation is a checked registration precondition, never a
    # permanent strategy input. The inherited immutable closure stays intact.
    hashes = dict(reader.seen)
    hashes.pop(str(reader.path(STATE)), None)
    return dict(round=148, status="REGISTERED", registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE, sha256=state_sha),
        arms=list(r.NAMES), primary_challenger=r.NAMES[2], dose=r.DOSE,
        output_directory=str(out), runtime=dict(python=sys.version,
            numpy=r.np.__version__, pandas=r.pd.__version__), tolerances=r.TOLERANCES,
        hashes=hashes, inputs=inputs, raw_source_hashes=previous["raw_source_hashes"],
        independent_prepare_source_hashes=independent["prepare"],
        independent_outcome_source_hashes=independent["outcome"],
        actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0, actual_model_fits_before_registration=0,
        no_observer_writes=True, tests=str(reader.path(tests)))


def enqueue(root, digest):
    path = root/"outputs"/STUDY/"PLAN.json"
    require(r.sha(path) == digest, "PLAN changed")
    plan = r.read(path)
    require(plan["round"] == 148 and plan["status"] == "REGISTERED"
            and plan["arms"] == list(r.NAMES), "Wrong registration")
    reader = old.Reader(root)
    reader.verify(plan["hashes"])
    state, _ = state_snapshot(root/STATE)
    predecessor(state)
    entry = dict(round=148, directory=STUDY,
        plan=dict(path=str(path.relative_to(root)), sha256=digest),
        registered_utc=plan["registered_utc"])
    if state["registered_queue"] == [entry]:
        return
    require(not state["registered_queue"], "Conflicting registration")
    expected = plan["registration_precondition"]["sha256"]
    state.update(registered_queue=[entry], current=STUDY,
        current_status="registered_before_actual_preparation", updated_utc=plan["registered_utc"])
    update_state(root/STATE, expected, state)


def register(root, tests):
    plan = build_plan(root, tests)
    out = root/"outputs"/STUDY
    out.mkdir(exist_ok=True)
    write(out/"PLAN.json", plan)
    digest = r.sha(out/"PLAN.json")
    enqueue(root, digest)
    return dict(plan_sha256=digest, registered=True, hashes=len(plan["hashes"]))


def complete(root, plan_sha):
    out = root/"outputs"/STUDY
    runner = r.Runner(root, out/"PLAN.json", plan_sha)
    runner.require_prepare_review()
    reader = old.Reader(root)
    plan = reader.read(out/"PLAN.json")
    review = reader.read(out/"INDEPENDENT_OUTCOME.json")
    require(review.get("passed") is True and review.get("stage") == "evaluate"
            and review.get("plan_sha256") == plan_sha
            and review.get("execution_paths_verified") == 9
            and review.get("total_monthly_orders_verified") == 366,
            "Full independent outcome review")
    reader.verify(review["artifact_hashes"])
    require(reader.verify(review["source_hashes"])
            == reader.verify(plan["independent_outcome_source_hashes"]), "Outcome code changed")
    results = reader.read(out/"RESULTS.json")
    selection = r.select(reader.read(out/"TRAIN_TRIALS.json"),
                         reader.read(out/"TRAIN_OPEN_TRIALS.json"))
    require(reader.read(out/"FROZEN.json")["selection"] == selection
            and review["selected_name"] == selection["primary"], "Selection mismatch")
    required = ["PLAN.json", "CACHE.json", "INDEPENDENT_PREPARE.json", "RESULTS.json", "FROZEN.json",
                "TRAIN_TRIALS.json", "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "VALIDATION.json",
                "DIAGNOSE_STARTED.json", *r.CACHE_FILES]
    required += [name+"_"+mode+"_"+kind+".json" for name in r.NAMES
                 for mode in ("same_close", "next_open_zero", "next_open_cost")
                 for kind in ("DAILY", "TRADES", "HOLDINGS")]
    for name in required:
        require(review["artifact_hashes"].get(str(out/name)) == reader.hash(out/name),
                "Unreviewed outcome: "+name)
    manifest = reader.read(plan["inputs"]["protected_manifest"]["path"])
    require(len(manifest) == 21444, "Protected archive scope")
    reader.verify(manifest)
    proof = dict(passed=True, checked=len(manifest), verified=len(manifest),
        manifest_sha256=plan["inputs"]["protected_manifest"]["sha256"],
        changed=[], missing=[], read_or_identity_errors=[])
    old.protected_scan(reader, plan, proof)
    if not (out/"PROTECTED_AFTER.json").exists():
        write(out/"PROTECTED_AFTER.json", proof)
    else:
        require(reader.read(out/"PROTECTED_AFTER.json") == proof, "Changed protection proof")
    state, expected = state_snapshot(root/STATE)
    predecessor(state)
    require(len(state["registered_queue"]) == 1
            and state["registered_queue"][0]["directory"] == STUDY
            and state["registered_queue"][0]["plan"]["sha256"] == plan_sha,
            "Registration queue mismatch")
    primary = selection["primary"]
    metric = results[primary]["same_close"]["metrics"]
    require(old.number(metric["cagr"]) and old.number(metric["sharpe"])
            and abs(metric["cagr"]-review["primary_cagr"]) < 1e-12
            and review["primary_target_reached"] == (metric["cagr"] >= .7), "Target verification")
    done = dict(round=148, directory=STUDY, completed_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha, selection=selection, all_arm_results=results,
        primary=primary, primary_cagr=metric["cagr"], primary_sharpe=metric["sharpe"],
        target_reached=metric["cagr"] >= .7, trials=3, independent_execution_paths=9,
        protected_files=len(manifest), historical_price_vintage_qualified=True, pristine_oos=False,
        no_observer_writes=True,
        independent_outcome_sha256=reader.hash(out/"INDEPENDENT_OUTCOME.json"),
        protected_sha256=reader.hash(out/"PROTECTED_AFTER.json"))
    done = preserve_completion(out/"COMPLETION.json", done)
    ref = dict(path=str((out/"COMPLETION.json").relative_to(root)), sha256=r.sha(out/"COMPLETION.json"))
    entry = {k: done[k] for k in ("directory", "completed_utc", "primary", "primary_cagr",
        "primary_sharpe", "target_reached", "trials", "independent_execution_paths", "protected_files")}
    entry.update(completion=ref, new_admitted=int(selection["admitted"]))
    state["completed"].append(entry)
    require(len(state["completed"]) == 148 and sum(x["trials"] for x in state["completed"]) == 946,
            "Exact completed round/trial increment")
    state.update(current=STUDY, current_status="completed_independently_verified",
        registered_queue=[], latest_round_receipt=ref, updated_utc=done["completed_utc"],
        last_goal_turn_classification="progress")
    if selection["admitted"] and metric["cagr"] > state["best_training_selected_candidate"]["cagr"]:
        state["best_training_selected_candidate"] = dict(directory=STUDY,
            cagr=metric["cagr"], sharpe=metric["sharpe"], max_drawdown=metric["max_drawdown"],
            next_open_5bp_cagr=results[primary]["next_open_cost"]["metrics"]["cagr"],
            spec=dict(mode=primary), observer_promotion=False)
    update_state(root/STATE, expected, state)
    return dict(completion=ref, primary=primary, cagr=metric["cagr"],
                sharpe=metric["sharpe"], target_reached=done["target_reached"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["register", "enqueue", "complete"])
    p.add_argument("--plan-sha")
    p.add_argument("--tests")
    a = p.parse_args()
    with state_lock(ROOT):
        if a.phase == "register":
            require(a.tests, "Synthetic tests required")
            result = register(ROOT, a.tests)
        else:
            require(old.digest(a.plan_sha), "Exact PLAN SHA required")
            result = enqueue(ROOT, a.plan_sha) if a.phase == "enqueue" else complete(ROOT, a.plan_sha)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

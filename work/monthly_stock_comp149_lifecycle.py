"""Register/close only the fixed SBC-adjustment experiment, with checked state updates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import tempfile
import sys
import hashlib
import fcntl
from contextlib import contextmanager
import monthly_stock_comp149_runner as r
import complete_monthly_same_report145 as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = "monthly_stock_comp149"
STATE = "outputs/monthly_target70_research/CONTINUATION_STATUS.json"
PROPOSAL_SHA = None  # Explicit --proposal-sha is required before registration.
PREVIOUS_PLAN = "3574e88c562557961ba49ddd37ddd98192a2a4a8b195029b2c31a189fb7a37d6"
PREVIOUS_DONE = "3d8405758c531ee4c9c715a84dc8c71792e9c55b63a057d0e9ab46b13d3aeb03"
SOURCE_POLICY = "862c3696a2e39a476dbbd6ec450e7b27dd1c1e869eb8334902cd20d673bf5cb7"
require = r.require


def write(path, value):
    with Path(path).open("xb") as f:
        f.write(old.encoded(value))


def state_snapshot(path):
    raw=Path(path).read_bytes()
    return old.decode(raw),hashlib.sha256(raw).hexdigest()


@contextmanager
def state_lock(root):
    """Serialize this lifecycle's writers; root is the sole central-state writer."""
    path=Path(root)/"outputs/monthly_target70_research/.lifecycle.lock"
    with path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def preserve_completion(path, done):
    if Path(path).exists():
        saved=r.read(path)
        require({k:v for k,v in saved.items() if k!="completed_utc"} ==
                {k:v for k,v in done.items() if k!="completed_utc"},
                "Existing completion conflicts with revalidated evidence")
        return saved
    write(path,done)
    return done


def predecessor(state):
    items = state["completed"]
    require(len(items) == 148 and sum(x["trials"] for x in items) == 946,
            "Expected148 completed rounds/946 trials")
    require(items[-1]["directory"] == "monthly_session_cluster148"
            and items[-1]["completion"]["sha256"] == PREVIOUS_DONE,
            "Wrong preceding completion")
    require(not any(x["directory"] == STUDY for x in items), "Already completed149")


def update_state(path, expected, value):
    """Preserve an exact prior copy and fail if a concurrent writer intervened."""
    require(r.sha(path) == expected, "Concurrent continuation update")
    backup = path.parent/("PRE149_"+expected+".json")
    if not backup.exists():
        with backup.open("xb") as f:
            f.write(path.read_bytes())
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
    names = set()
    count = 0
    for suite in doc["suites"]:
        name = suite["name"]
        require(name not in names and suite["count"] > 0, "Distinct positive suites")
        names.add(name)
        for k in ("source", "log"):
            ref = suite[k]
            require(reader.hash(ref["path"]) == ref["sha256"], "Changed synthetic "+k)
        log = reader.path(suite["log"]["path"]).read_text()
        require(re.search(r"Ran "+str(suite["count"])+r" tests? in ", log)
                and log.rstrip().endswith("OK"), "Synthetic test failure")
        count += suite["count"]
    require(names == {"factor", "runner", "prepare_review", "outcome_review", "lifecycle", "proposal"}
            and doc["test_count"] == count, "Complete synthetic scope")


def build_plan(root, tests, source_review, proposal_sha=None):
    proposal_sha = proposal_sha or PROPOSAL_SHA
    require(type(proposal_sha) is str and re.fullmatch("[0-9a-f]{64}", proposal_sha),
            "Explicit final prospective proposal SHA required")
    import monthly_stock_comp149_prepare_review as prep
    import monthly_stock_comp149_outcome_review as outcome
    reader = old.Reader(root)
    out = root/"outputs"/STUDY
    require(not (out/"PLAN.json").exists(), "PLAN already frozen")
    state,state_sha = state_snapshot(root/STATE)
    predecessor(state)
    require(state["registered_queue"] == [], "Pending experiment")
    require(reader.hash("outputs/monthly_session_cluster148/COMPLETION.json") == PREVIOUS_DONE,
            "Predecessor receipt changed")
    require(reader.hash("outputs/monthly_session_cluster148/PLAN.json") == PREVIOUS_PLAN,
            "Predecessor PLAN changed")
    previous = reader.read("outputs/monthly_session_cluster148/PLAN.json")
    reader.verify(previous["hashes"])
    require(len(previous["raw_source_hashes"]) == 64, "Original raw price scope")
    reader.verify(previous["raw_source_hashes"])
    reader.verify(r.source_files(root))
    for path in [Path(__file__), root/"work/test_monthly_stock_comp149_lifecycle.py",
                 root/"work/prepare_stock_comp149_proposal.py",
                 root/"work/test_prepare_stock_comp149_proposal.py"]:
        reader.hash(path)
    independent = {}
    for name, module in (("prepare", prep), ("outcome", outcome)):
        sources = module.source_files(root)
        independent[name] = {str(p): reader.hash(p) for p in sources}
    for name, path in (("prepare", "work/monthly_stock_comp149_prepare_review_SYNTHETIC.json"),
                       ("outcome", "work/monthly_stock_comp149_outcome_prereview.json")):
        review=reader.read(path)
        require(review.get("passed") is True and review.get("actual_factors_calculated") == 0
                and review.get("actual_returns_calculated") == 0, "Prospective independent review")
        require(reader.verify(review["source_hashes"]) == independent[name],
                "Prospective independent reviewer code changed")
        contract = reader.verify(review["reviewed_contract_hashes"])
        for required_path in ("work/monthly_stock_comp149_runner.py",
                              "work/monthly_stock_comp149_factor.py",
                              "work/monthly_stock_comp149_proposal.json"):
            require(contract.get(str(reader.path(required_path))) == reader.hash(required_path),
                    "Prospective review omitted current contract")
    require(reader.hash("work/monthly_stock_comp149_proposal.json") == proposal_sha,
            "Prospective proposal changed")
    proposal = reader.read("work/monthly_stock_comp149_proposal.json")
    r.validate_proposal(proposal)
    for key in ("concept", "novelty"):
        reference = proposal[key]
        require(reader.hash(reference["path"]) == reference["sha256"],
                "Prospective hypothesis evidence changed")
    require(reader.hash(proposal["sources"]["policy"]["path"]) == SOURCE_POLICY,
            "Source policy changed")
    policy = reader.read(proposal["sources"]["policy"]["path"])
    require(reader.hash("work/cash_quality_source_admission_v1.py") == policy["implementation_sha256"],
            "Source admission implementation changed")
    for ref in policy["inputs"].values():
        require(reader.hash(ref["path"]) == ref["sha256"], "Source policy input changed")
    review = reader.read(source_review)
    require(review.get("passed") is True and review.get("admitted_events") == 31
            and review.get("held_events") == 13, "Independent actual admission review")
    reader.verify(review["input_hashes"])
    union_review = reader.read("outputs/monthly_cashflow_source_union_v1/INDEPENDENT_REVIEW.json")
    reader.verify(union_review["input_hashes"])
    test_doc = reader.read(tests)
    validate_tests(test_doc, reader)
    inputs = {k: previous["inputs"][k] for k in
        ("meta", "policy", "original_orders", "prices", "sessions", "protected_manifest")}
    inputs["proposal"] = dict(path=str(root/"work/monthly_stock_comp149_proposal.json"),
                              sha256=proposal_sha)
    inputs["admitted_dataset"] = proposal["sources"]["dataset"]
    inputs["source_admission"] = proposal["sources"]["admission"]
    for ref in inputs.values():
        require(reader.hash(ref["path"]) == ref["sha256"], "Named input changed")
        ref["path"] = str(reader.path(ref["path"]))
    admission = reader.read(inputs["source_admission"]["path"])
    reader.verify(admission["source_hashes"])
    require(admission["data_sha256"] == inputs["admitted_dataset"]["sha256"]
            and admission["admitted_events"] == 31 and admission["held_events"] == 13,
            "Admission population changed")
    source_refs = {key: proposal["sources"][key] for key in r.SBC_REFS}
    source_objects = {}
    for key, ref in source_refs.items():
        require(reader.path(ref["path"]) == root/r.SBC_REFS[key] and
                reader.hash(ref["path"]) == ref["sha256"], "Exact SBC source package required")
        source_objects[key] = reader.read(ref["path"])
    r.validate_sbc_admission(source_objects["sbc_dataset"], source_objects["sbc_admission"],
                             source_objects["sbc_policy"], source_objects["sbc_review"],
                             reader.read(inputs["admitted_dataset"]["path"]), admission,
                             source_refs)
    source_pins = {}
    for key in ("sbc_admission", "sbc_review"):
        mapping = source_objects[key].get("input_hashes")
        require(type(mapping) is dict and bool(mapping), "Nonempty new source closure")
        source_pins[key] = reader.verify(mapping)
    required_new = dict(source_pins["sbc_admission"],
                        **{str(reader.path(source_refs[k]["path"])): source_refs[k]["sha256"]
                           for k in ("sbc_dataset", "sbc_admission", "sbc_policy")})
    require(all(source_pins["sbc_review"].get(path) == digest
                for path, digest in required_new.items()), "New source independent review omitted evidence")
    # Continuation is deliberately mutable and therefore a checked precondition,
    # never a long-lived model input.
    hashes = dict(reader.seen)
    hashes.pop(str(reader.path(STATE)),None)
    return dict(round=149, status="REGISTERED", registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE, sha256=state_sha),
        arms=list(r.NAMES), primary_challenger=r.NAMES[3], dose=r.DOSE,
        output_directory=str(out), runtime=dict(python=sys.version,
            numpy=r.np.__version__, pandas=r.pd.__version__), tolerances=r.TOLERANCES,
        hashes=hashes, inputs=inputs, source_policy=proposal["sources"]["policy"],
        **source_refs,
        raw_source_hashes=previous["raw_source_hashes"],
        independent_prepare_source_hashes=independent["prepare"],
        independent_outcome_source_hashes=independent["outcome"],
        actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0, actual_model_fits_before_registration=0,
        no_observer_writes=True, tests=tests, source_admission_independent_review=source_review)


def enqueue(root, digest):
    path=root/"outputs"/STUDY/"PLAN.json"
    require(r.sha(path) == digest, "PLAN changed")
    plan=r.read(path)
    reader=old.Reader(root)
    reader.verify(plan["hashes"])
    state,_=state_snapshot(root/STATE)
    predecessor(state)
    entry=dict(round=149,directory=STUDY,plan=dict(path=str(path.relative_to(root)),sha256=digest),
               registered_utc=plan["registered_utc"])
    if state["registered_queue"] == [entry]:
        return
    require(not state["registered_queue"], "Conflicting registration")
    expected=plan["registration_precondition"]["sha256"]
    state.update(registered_queue=[entry],current=STUDY,
        current_status="registered_before_actual_preparation",updated_utc=plan["registered_utc"])
    update_state(root/STATE,expected,state)


def register(root, tests, source_review, proposal_sha=None):
    plan=build_plan(root,tests,source_review,proposal_sha)
    out=root/"outputs"/STUDY
    out.mkdir(exist_ok=True)
    write(out/"PLAN.json",plan)
    digest=r.sha(out/"PLAN.json")
    enqueue(root,digest)
    return dict(plan_sha256=digest,registered=True,hashes=len(plan["hashes"]))


def validate_outcome_receipt(review, plan_sha):
    require(review.get("passed") is True and review.get("stage") == "evaluate"
            and review.get("plan_sha256") == plan_sha, "Full independent outcome review")
    for key, expected in (("execution_paths_verified", 12),
                          ("total_monthly_orders_verified", 488),
                          ("stock_compensation_feature_matrices_verified", 4)):
        require(type(review.get(key)) is int and review[key] == expected,
                "Independent outcome scope: "+key)
    for key in ("stock_comp_source_admission_verified", "common_mask_subset_of146_verified"):
        require(review.get(key) is True, "Independent new source check: "+key)


def complete(root, plan_sha):
    out=root/"outputs"/STUDY
    runner=r.Runner(root,out/"PLAN.json",plan_sha)
    runner.require_prepare_review()
    reader=old.Reader(root)
    plan=reader.read(out/"PLAN.json")
    review=reader.read(out/"INDEPENDENT_OUTCOME.json")
    validate_outcome_receipt(review, plan_sha)
    reader.verify(review["artifact_hashes"])
    require(reader.verify(review["source_hashes"])
            == reader.verify(plan["independent_outcome_source_hashes"]), "Outcome code changed")
    results=reader.read(out/"RESULTS.json")
    selection=r.select(reader.read(out/"TRAIN_TRIALS.json"),
                       reader.read(out/"TRAIN_OPEN_TRIALS.json"))
    require(reader.read(out/"FROZEN.json")["selection"] == selection
            and review["selected_name"] == selection["primary"], "Selection mismatch")
    required=["RESULTS.json","FROZEN.json","TRAIN_TRIALS.json","TRAIN_OPEN_TRIALS.json",
              "WALKFORWARD.json","VALIDATION.json","DIAGNOSE_STARTED.json"]
    required += [name+"_"+mode+"_"+kind+".json" for name in r.NAMES
                 for mode in ("same_close","next_open_zero","next_open_cost")
                 for kind in ("DAILY","TRADES","HOLDINGS")]
    for name in required:
        require(review["artifact_hashes"].get(str(out/name)) == reader.hash(out/name),
                "Unreviewed outcome: "+name)
    manifest=reader.read(plan["inputs"]["protected_manifest"]["path"])
    require(len(manifest) == 21444, "Protected archive scope")
    reader.verify(manifest)
    proof=dict(passed=True,checked=len(manifest),verified=len(manifest),
        manifest_sha256=plan["inputs"]["protected_manifest"]["sha256"],
        changed=[],missing=[],read_or_identity_errors=[])
    old.protected_scan(reader,plan,proof)
    if not (out/"PROTECTED_AFTER.json").exists():
        write(out/"PROTECTED_AFTER.json",proof)
    else:
        require(reader.read(out/"PROTECTED_AFTER.json") == proof, "Changed protection proof")
    state,expected=state_snapshot(root/STATE)
    predecessor(state)
    require(len(state["registered_queue"]) == 1
            and state["registered_queue"][0]["plan"]["sha256"] == plan_sha, "Registration queue mismatch")
    primary=selection["primary"]
    metric=results[primary]["same_close"]["metrics"]
    require(abs(metric["cagr"]-review["primary_cagr"])<1e-12
            and review["primary_target_reached"] == (metric["cagr"]>=.7), "Target verification")
    done=dict(round=149,directory=STUDY,completed_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha,selection=selection,all_arm_results=results,
        primary=primary,primary_cagr=metric["cagr"],primary_sharpe=metric["sharpe"],
        target_reached=metric["cagr"]>=.7,trials=4,independent_execution_paths=12,
        protected_files=len(manifest),source_admission_conditional=True,
        independent_outcome_sha256=reader.hash(out/"INDEPENDENT_OUTCOME.json"),
        protected_sha256=reader.hash(out/"PROTECTED_AFTER.json"))
    done=preserve_completion(out/"COMPLETION.json",done)
    ref=dict(path=str((out/"COMPLETION.json").relative_to(root)),sha256=r.sha(out/"COMPLETION.json"))
    entry={k:done[k] for k in ("directory","completed_utc","primary","primary_cagr",
        "primary_sharpe","target_reached","trials","independent_execution_paths","protected_files")}
    entry.update(completion=ref,new_admitted=int(selection["admitted"]))
    state["completed"].append(entry)
    require(len(state["completed"]) == 149 and
            sum(item["trials"] for item in state["completed"]) == 950,
            "Final149 rounds/950 trials accounting")
    state.update(current=STUDY,current_status="completed_independently_verified",
        registered_queue=[],latest_round_receipt=ref,updated_utc=done["completed_utc"],
        last_goal_turn_classification="progress")
    if selection["admitted"] and metric["cagr"] > state["best_training_selected_candidate"]["cagr"]:
        state["best_training_selected_candidate"]=dict(directory=STUDY,
            cagr=metric["cagr"],sharpe=metric["sharpe"],max_drawdown=metric["max_drawdown"],
            next_open_5bp_cagr=results[primary]["next_open_cost"]["metrics"]["cagr"],
            spec=dict(mode=primary),observer_promotion=False)
    update_state(root/STATE,expected,state)
    return dict(completion=ref,primary=primary,cagr=metric["cagr"],
                sharpe=metric["sharpe"],target_reached=done["target_reached"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["register","enqueue","complete"])
    p.add_argument("--plan-sha")
    p.add_argument("--tests")
    p.add_argument("--source-review")
    p.add_argument("--proposal-sha")
    a=p.parse_args()
    with state_lock(ROOT):
        if a.phase=="register":
            require(a.tests and a.source_review,"Tests and source review required")
            result=register(ROOT,a.tests,a.source_review,a.proposal_sha)
        elif a.phase=="enqueue":
            result=enqueue(ROOT,a.plan_sha)
        else:
            result=complete(ROOT,a.plan_sha)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Register/close only the fixed cash-quality experiment, with checked state updates."""
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
import monthly_cash_quality146_runner as r
import complete_monthly_same_report145 as old
import monthly_cash_quality146_prepare_review as prep
import monthly_cash_quality146_outcome_review as outcome

ROOT = Path(__file__).resolve().parents[1]
STUDY = "monthly_cash_quality146"
STATE = "outputs/monthly_target70_research/CONTINUATION_STATUS.json"
PROPOSAL_SHA = "585fea23faef22c724b7977273f3d3c5994c673198c0e88115f12288accdf51a"
PREVIOUS_PLAN = "23135db73d4721fa4ee6d77a32e029825b2be5ac3792f2061ab4eae648c00bfc"
PREVIOUS_DONE = "3819dd665abf5bfc4b389b04546f76b27fed3463d364d707b761e2dab3e09bed"
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
    require(len(items) == 145 and sum(x["trials"] for x in items) == 935,
            "Expected145 completed rounds/935 trials")
    require(items[-1]["directory"] == "monthly_same_report145"
            and items[-1]["completion"]["sha256"] == PREVIOUS_DONE,
            "Wrong preceding completion")
    require(not any(x["directory"] == STUDY for x in items), "Already completed146")


def update_state(path, expected, value):
    """Preserve an exact prior copy and fail if a concurrent writer intervened."""
    require(r.sha(path) == expected, "Concurrent continuation update")
    backup = path.parent/("PRE146_"+expected+".json")
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
    require(names == {"factor", "runner", "prepare_review", "outcome_review", "lifecycle"}
            and doc["test_count"] == count, "Complete synthetic scope")


def build_plan(root, tests, source_review):
    reader = old.Reader(root)
    out = root/"outputs"/STUDY
    require(not (out/"PLAN.json").exists(), "PLAN already frozen")
    state,state_sha = state_snapshot(root/STATE)
    predecessor(state)
    require(state["registered_queue"] == [], "Pending experiment")
    require(reader.hash("outputs/monthly_same_report145/COMPLETION.json") == PREVIOUS_DONE,
            "Predecessor receipt changed")
    require(reader.hash("outputs/monthly_same_report145/PLAN.json") == PREVIOUS_PLAN,
            "Predecessor PLAN changed")
    previous = reader.read("outputs/monthly_same_report145/PLAN.json")
    reader.verify(previous["hashes"])
    require(len(previous["raw_source_hashes"]) == 64, "Original raw price scope")
    reader.verify(previous["raw_source_hashes"])
    reader.verify(r.source_files(root))
    for path in [Path(__file__), root/"work/test_monthly_cash_quality146_lifecycle.py"]:
        reader.hash(path)
    independent = {}
    for name, module in (("prepare", prep), ("outcome", outcome)):
        sources = module.source_files(root)
        independent[name] = {str(p): reader.hash(p) for p in sources}
    for path in ("work/monthly_cash_quality146_prepare_review_SYNTHETIC.json",
                 "work/monthly_cash_quality146_outcome_prereview.json"):
        review=reader.read(path)
        require(review.get("passed") is True and review.get("actual_factors_calculated") == 0
                and review.get("actual_returns_calculated") == 0, "Prospective independent review")
        reader.verify(review["source_hashes"])
        reader.verify(review["reviewed_contract_hashes"])
    require(reader.hash("work/monthly_cash_quality146_proposal.json") == PROPOSAL_SHA,
            "Prospective proposal changed")
    proposal = reader.read("work/monthly_cash_quality146_proposal.json")
    r.validate_proposal(proposal)
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
    inputs["proposal"] = dict(path=str(root/"work/monthly_cash_quality146_proposal.json"),
                              sha256=PROPOSAL_SHA)
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
    # Continuation is deliberately mutable and therefore a checked precondition,
    # never a long-lived model input.
    hashes = dict(reader.seen)
    hashes.pop(str(reader.path(STATE)),None)
    return dict(round=146, status="REGISTERED", registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE, sha256=state_sha),
        arms=list(r.NAMES), primary_challenger=r.NAMES[3], dose=r.DOSE,
        output_directory=str(out), runtime=dict(python=sys.version,
            numpy=r.np.__version__, pandas=r.pd.__version__), tolerances=r.TOLERANCES,
        hashes=hashes, inputs=inputs, source_policy=proposal["sources"]["policy"],
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
    entry=dict(round=146,directory=STUDY,plan=dict(path=str(path.relative_to(root)),sha256=digest),
               registered_utc=plan["registered_utc"])
    if state["registered_queue"] == [entry]:
        return
    require(not state["registered_queue"], "Conflicting registration")
    expected=plan["registration_precondition"]["sha256"]
    state.update(registered_queue=[entry],current=STUDY,
        current_status="registered_before_actual_preparation",updated_utc=plan["registered_utc"])
    update_state(root/STATE,expected,state)


def register(root, tests, source_review):
    plan=build_plan(root,tests,source_review)
    out=root/"outputs"/STUDY
    out.mkdir(exist_ok=True)
    write(out/"PLAN.json",plan)
    digest=r.sha(out/"PLAN.json")
    enqueue(root,digest)
    return dict(plan_sha256=digest,registered=True,hashes=len(plan["hashes"]))


def complete(root, plan_sha):
    out=root/"outputs"/STUDY
    runner=r.Runner(root,out/"PLAN.json",plan_sha)
    runner.require_prepare_review()
    reader=old.Reader(root)
    plan=reader.read(out/"PLAN.json")
    review=reader.read(out/"INDEPENDENT_OUTCOME.json")
    require(review.get("passed") is True and review.get("stage") == "evaluate"
            and review.get("plan_sha256") == plan_sha
            and review.get("execution_paths_verified") == 12
            and review.get("total_monthly_orders_verified") == 488,
            "Full independent outcome review")
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
    done=dict(round=146,directory=STUDY,completed_utc=datetime.now(timezone.utc).isoformat(),
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
    a=p.parse_args()
    with state_lock(ROOT):
        if a.phase=="register":
            require(a.tests and a.source_review,"Tests and source review required")
            result=register(ROOT,a.tests,a.source_review)
        elif a.phase=="enqueue":
            result=enqueue(ROOT,a.plan_sha)
        else:
            result=complete(ROOT,a.plan_sha)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()

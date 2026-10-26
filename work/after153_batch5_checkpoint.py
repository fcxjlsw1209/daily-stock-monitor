"""Record independently verified batch5 source progress without strategy work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "outputs/monthly_financial_coverage_batch5_v1/"
BATCH = "outputs/monthly_financial_coverage_admission_batch5_v1/"
REVIEW = "work/after153_batch5_independent_review.json"
DONE = "outputs/monthly_financial_coverage153/COMPLETION.json"
PINS = {
    DONE: "6828b1dfd341f34cccc77d3d20bb3135d0fb28a3a2f80d8eb6e29c52157dc560",
    SOURCE + "PLAN.json": "77eae28f976174c657b85d4160c47ab056fb5b3abdaf31d9c8c45cb410d49c6e",
    SOURCE + "EXTRACTIONS.json": "938c2528953d73340139e63a80bb0f4afed703020f8be7fbc73e1e1124d78fa0",
    "work/after153_batch5_admission_policy.json":
        "326da6e7f6abadd76e68436cccda4e7ffeee95b4fabd06699bf01b177fc5523c",
    SOURCE + "independent_review_repair_v1/checker_before.py":
        "2da15aa96cf10f7df5e84af1860b505cb882d1968e1064078038c2d25f1d1fec",
    SOURCE + "independent_review_repair_v1/test_before.py":
        "7d101b6e24e154cb3834e7b241fbe92501682a77e29831c7b7ef9129f11174a7",
    SOURCE + "independent_review_repair_v1/REPAIR_NOTE.txt":
        "b21a76fd032e72593dc45bd152463b1ed543a9b430c48c7ec346f7ee32c4a800",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--review-sha", required=True)
    args = parser.parse_args()
    pins = dict(PINS)
    pins.update({BATCH + "MANIFEST.json": args.manifest_sha,
                 REVIEW: args.review_sha})
    docs = {}

    def verify(name, digest):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Input escapes local root")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "Changed input: " + name)
        relative = str(path.relative_to(ROOT))
        require(relative not in pins or pins[relative] == digest, "Conflicting input")
        pins[relative] = digest
        return json.loads(raw) if path.suffix == ".json" else None

    for name, digest in list(pins.items()):
        docs[name] = verify(name, digest)
    manifest = docs[BATCH + "MANIFEST.json"]
    require(set(manifest["outputs"]) == {"DATASET.json", "ADMISSION.json", "COVERAGE.json"},
            "Complete source package required")
    for name, digest in manifest["outputs"].items():
        docs[BATCH + name] = verify(BATCH + name, digest)
    for name, digest in manifest["input_hashes"].items():
        verify(name, digest)
    admission = docs[BATCH + "ADMISSION.json"]
    coverage = docs[BATCH + "COVERAGE.json"]
    review = docs[REVIEW]
    require(admission["passed"] is True and review["passed"] is True,
            "Independent review and conditional admission must pass")
    require(admission["input_hashes"] == manifest["input_hashes"],
            "Manifest and admission closures differ")
    require(admission["independent_review_sha256"] == args.review_sha,
            "Admission binds another review")
    counts = admission["counts"]
    require(counts["events"] == 21 and counts["annual_catalog_events"] == 28
            and counts["unprocessed_annual_catalog_events"] == 7,
            "Complete fixed batch5 population required")
    require(counts["reviewed_nonnull_fields"] == 63 and counts["reviewed_null_fields"] == 0,
            "Unexpected source field population")
    require(coverage["counts"]["stock_slots"] == 3904, "Stock-month axes changed")
    for key in ("new_strategy_trials", "actual_factors_calculated", "actual_returns_calculated"):
        require(type(admission[key]) is int and admission[key] == 0,
                "Checkpoint is source-only")
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    require(len(state["completed"]) == 153
            and sum(x["trials"] for x in state["completed"]) == 968
            and state["registered_queue"] == [], "Final153 and empty queue required")
    require(state["latest_round_receipt"] == {"path": DONE, "sha256": PINS[DONE]},
            "Wrong finalized predecessor")
    checkpoint_path = ROOT / "migration/after153_batch5_checkpoint.json"
    require(not checkpoint_path.exists(), "Write-once checkpoint already exists")
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        stage="independently_reviewed_conditional_source_admission",
        completed_rounds=153, completed_trials=968, target_reached=False,
        previous_goal_turn_classification="progress: round153 completed and batch5 extracted",
        last_goal_turn_classification="progress",
        source_hashes=pins, source_counts=counts, coverage_counts=coverage["counts"],
        new_strategy_trials=0, new_factors_computed=0, frozen_observers_modified=False,
        next_experiment_status="not_registered",
        next_experiment_scope="Predeclare the final fixed-dose coverage contrast for PLTR/PYPL/TMO before calculating new scores or returns.",
        limitations=[
            "Conditional source admission is not first-public-vintage certification.",
            "Annual starts are explicit same-report consecutive-boundary assumptions.",
            "The as-reported GAAP purpose retains the PayPal carve-out and Thermo Fisher restatement limitations.",
            "Early calendar and unresolved holds remain; latest unprocessed entries block fallback.",
            "Source admission is not a strategy result or achievement of the70% target.",
        ])
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    before = resume_path.read_bytes()
    resume = json.loads(before)
    require(resume["round"] == 153, "Concurrent research checkpoint advanced")
    backup = ROOT / ("migration/RESEARCH_RESUME_before_batch5_" +
                     hashlib.sha256(before).hexdigest() + ".json")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(before)
    require(backup.read_bytes() == before, "Conflicting backup")
    with checkpoint_path.open("xb") as stream:
        stream.write(encoded(checkpoint))
    ref = dict(path=str(checkpoint_path.relative_to(ROOT)),
               sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    resume.update(
        after153_batch5_checkpoint=ref,
        latest_source_preparation=checkpoint["stage"],
        next_research_preparation=dict(
            status="not_registered", source_symbols=["PLTR", "PYPL", "TMO"],
            source_checkpoint=ref, values_extracted=63, retained_null_fields=0,
            actual_new_strategy_trials=0),
        next_research_status="batch5_conditionally_admitted_no_strategy_registration",
        next_step=checkpoint["next_experiment_scope"],
        last_goal_turn_classification="progress",
        previous_goal_turn_classification=checkpoint["previous_goal_turn_classification"],
        updated_utc=checkpoint["created_utc"])
    require(resume_path.read_bytes() == before, "Concurrent resume update")
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(resume_path.read_bytes() == before, "Concurrent resume commit")
        os.replace(temporary, resume_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(checkpoint=ref, source_counts=counts,
                          coverage_counts=coverage["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

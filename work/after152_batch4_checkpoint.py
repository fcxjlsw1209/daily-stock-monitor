"""Record verified batch4 source progress; no strategy calculation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "outputs/monthly_financial_coverage_batch4_v1/"
BATCH = "outputs/monthly_financial_coverage_admission_batch4_v1/"
REVIEW = "work/after152_batch4_independent_review.json"
DONE = "outputs/monthly_financial_coverage152/COMPLETION.json"
PINS = {
    DONE: "7b9925053304e38b94e8ad386e0dc74f7d220a26b9884bb3f99e189365e23016",
    SOURCE + "PLAN.json": "9b1016e51ad22eae4faf13629afeae7e06a8f7efe3b2f0e8d7e77dce0d232643",
    SOURCE + "EXTRACTIONS.json": "6855162969dd271aab13a8e62608684cbf9d285703fe8897913dfa3360afbe8c",
    "work/after152_batch4_admission_policy.json":
        "c71b7dacfa9446754fa8b876a8dc2a7e4fe5caded5df785a256c4bbef7999093",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-sha", required=True)
    p.add_argument("--review-sha", required=True)
    args = p.parse_args()
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
    require(counts["events"] == 9 and counts["annual_catalog_events"] == 23
            and counts["unprocessed_annual_catalog_events"] == 14,
            "Complete fixed batch4 population required")
    require(counts["reviewed_nonnull_fields"] == 27 and counts["reviewed_null_fields"] == 0,
            "Unexpected source field population")
    require(coverage["counts"]["stock_slots"] == 3904, "Stock-month axes changed")
    for key in ("new_strategy_trials", "actual_factors_calculated", "actual_returns_calculated"):
        require(type(admission[key]) is int and admission[key] == 0,
                "Checkpoint is source-only")
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    require(len(state["completed"]) == 152
            and sum(x["trials"] for x in state["completed"]) == 962
            and state["registered_queue"] == [], "Final152 and empty queue required")
    require(state["latest_round_receipt"] == {"path": DONE, "sha256": PINS[DONE]},
            "Wrong finalized predecessor")
    checkpoint_path = ROOT / "migration/after152_batch4_checkpoint.json"
    require(not checkpoint_path.exists(), "Write-once checkpoint already exists")
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        stage="independently_reviewed_conditional_source_admission",
        completed_rounds=152, completed_trials=962, target_reached=False,
        previous_goal_turn_classification="progress: round152 completed and batch4 extracted",
        last_goal_turn_classification="progress",
        source_hashes=pins, source_counts=counts, coverage_counts=coverage["counts"],
        new_strategy_trials=0, new_factors_computed=0, frozen_observers_modified=False,
        next_experiment_status="not_registered",
        next_experiment_scope="Predeclare the next fixed-dose source-coverage contrast before calculating any new scores or returns.",
        limitations=[
            "Conditional research source admission is not first-public-vintage certification.",
            "MU/PEP 364/371-day annual starts remain explicit same-report boundary assumptions.",
            "Early-calendar and unresolved accounting holds remain; latest unprocessed entries block fallback.",
            "Source admission is not a strategy result or evidence that the70% target was achieved.",
        ])
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    before = resume_path.read_bytes()
    resume = json.loads(before)
    require(resume["round"] == 152, "Concurrent research checkpoint advanced")
    backup = ROOT / ("migration/RESEARCH_RESUME_before_batch4_" +
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
        after152_batch4_checkpoint=ref,
        latest_source_preparation=checkpoint["stage"],
        next_research_preparation=dict(
            status="not_registered", source_symbols=["MU", "PEP"], source_checkpoint=ref,
            values_extracted=27, retained_null_fields=0, actual_new_strategy_trials=0),
        next_research_status="batch4_conditionally_admitted_no_strategy_registration",
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

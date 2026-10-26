"""Record verified source progress; no strategy result or observer is changed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BATCH = "outputs/monthly_financial_coverage_batch1_v1"
ADMISSION = "outputs/monthly_financial_coverage_admission_batch1_v1"


def read(name):
    return json.loads((ROOT/name).read_bytes())


def reference(name):
    return dict(path=name, sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False)+"\n").encode()


def main():
    state = read("outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert len(state["completed"]) == 149 and not state["registered_queue"]
    assert sum(row["trials"] for row in state["completed"]) == 950
    plan = read(BATCH+"/PLAN.json")
    extraction = read(BATCH+"/EXTRACTIONS.json")
    review_name = "work/after149_batch1_independent_review.json"
    review = read(review_name)
    manifest = read(ADMISSION+"/MANIFEST.json")
    admission = read(ADMISSION+"/ADMISSION.json")
    dataset = read(ADMISSION+"/DATASET.json")
    coverage = read(ADMISSION+"/COVERAGE.json")
    assert extraction["plan_sha256"] == reference(BATCH+"/PLAN.json")["sha256"]
    assert review["passed"] is True
    assert review["plan_sha256"] == extraction["plan_sha256"]
    assert review["extractions_sha256"] == reference(BATCH+"/EXTRACTIONS.json")["sha256"]
    assert admission["passed"] is True
    assert admission["independent_review_sha256"] == reference(review_name)["sha256"]
    for filename, expected in manifest["outputs"].items():
        assert reference(ADMISSION+"/"+filename)["sha256"] == expected
    for filename, expected in manifest["input_hashes"].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected
    assert dataset["new_strategy_trials"] == dataset["ratios_computed"] == dataset["returns_computed"] == 0
    assert admission["new_strategy_trials"] == admission["actual_factors_calculated"] == 0
    assert coverage["counts"]["stock_slots"] == 3904 and coverage["counts"]["stocks"] == 73
    refs = [BATCH+"/PLAN.json", BATCH+"/EXTRACTIONS.json", review_name,
            ADMISSION+"/MANIFEST.json", ADMISSION+"/ADMISSION.json",
            ADMISSION+"/DATASET.json", ADMISSION+"/COVERAGE.json",
            "work/after149_batch1_admission_policy.json",
            "migration/after149_coverage_checkpoint.json",
            "outputs/monthly_stock_comp149/COMPLETION.json"]
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="batch1_conditional_sources_verified_before_strategy_registration",
        previous_goal_turn_classification="progress",
        completed_research_rounds=149, strategy_evaluations=950,
        new_strategy_trials=0, target_70_reached=False,
        best_candidate=state["best_training_selected_candidate"],
        extraction_counts=extraction["counts"],
        synthetic_extraction_tests=sum(row["tests"] for row in plan["synthetic_tests"]),
        admission_counts=admission["counts"], incremental_coverage=coverage["counts"],
        references=[reference(name) for name in refs],
        next_steps=[
            "Complete and independently review the matched-dose three-arm coverage experiment.",
            "Freeze round150 before computing actual new ratios, scores, orders or returns.",
            "Keep the original149 earnings-control mask exact; add only conditionally admitted AMZN/CSCO data.",
            "Select on first88 with segmented guards, then diagnose later34 and separate execution-cost paths.",
            "Continue the retained73-stock source backlog without deleting unprocessed reports or prior holds.",
        ],
        limitations=[
            "Source admission is conditional historical research, not certification of first-public vintages or a continuous security master.",
            "Two frozen CSCO2011 extraction gaps remain null: the independent review found the original asset values, but the producer rejected unrelated non-GAAP context. The FY2019 accounting-transition holds also remain explicit.",
            "The new financial source package has not yet produced a strategy result.",
            "The repeatedly researched history is not pristine out-of-sample evidence;70% remains unachieved.",
        ])
    filename = "migration/after149_batch1_checkpoint.json"
    with (ROOT/filename).open("xb") as stream:
        stream.write(encoded(checkpoint))
    resume_path = ROOT/"migration/RESEARCH_RESUME.json"
    previous = resume_path.read_bytes()
    previous_sha = hashlib.sha256(previous).hexdigest()
    with (ROOT/("migration/before_after149_batch1_"+previous_sha+".json")).open("xb") as stream:
        stream.write(previous)
    resume = json.loads(previous)
    resume.update(updated_utc=checkpoint["created_utc"],
                  next_research_status="batch1_source_admitted_round150_not_registered",
                  next_mechanism_registered=False,
                  after149_batch1_checkpoint=reference(filename),
                  next_step=checkpoint["next_steps"][0],
                  previous_goal_turn_classification="progress: source-only26-report extraction and independent conditional admission; no new strategy trials")
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
    assert hashlib.sha256(resume_path.read_bytes()).hexdigest() == previous_sha
    os.replace(temporary, resume_path)
    print(json.dumps(dict(checkpoint=reference(filename), counts=admission["counts"],
                          coverage=coverage["counts"], new_trials=0), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Record verified source progress; never score a strategy or touch observers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "outputs/monthly_financial_coverage150/COMPLETION.json": "28546cec6459a8ca73df5b434d06878ca7bb16008fc0e13b1a6b870cee8ffb92",
    "outputs/monthly_financial_coverage_batch2_v1/PLAN.json": "465af6e6f1c6fbe9f1f045d9526b6b50b19f10f8fc9acc4267f99c702efc95cc",
    "outputs/monthly_financial_coverage_batch2_v1/EXTRACTIONS.json": "547db745817f20e2adb4a9b22a085757b71e149c8a07c38ad7c271cc47485834",
    "work/after150_batch2_independent_review.json": "45b174e6b7ac944c9fd9eedf3694021abeb03013a22fca5b6904694a3dc7028f",
    "outputs/monthly_financial_coverage_admission_batch2_v1/DATASET.json": "04a2d2410a35db0a02034fab75b781cd43e5f52f569fb5272f34f3219cee2b33",
    "outputs/monthly_financial_coverage_admission_batch2_v1/ADMISSION.json": "410f4cf9e096ce4069a83b0b42a4291837df73b244f72d5b3cd0e9d90f94ab95",
    "outputs/monthly_financial_coverage_admission_batch2_v1/COVERAGE.json": "55744f23ce9a1aefe410b6cba1cdcb97ff424c6464df77801227f2e8d83da4de",
    "outputs/monthly_financial_coverage_admission_batch2_v1/MANIFEST.json": "ad03affca28c9a955d0c2db63d8efad955bdef6f4d47d5cd266bd09f93a524ff",
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def main():
    docs = {}
    for name, digest in PINS.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Changed source: " + name)
        docs[name] = json.loads(raw)
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    if (len(state["completed"]) != 150
            or sum(x["trials"] for x in state["completed"]) != 953
            or state["registered_queue"]):
        raise ValueError("Source checkpoint requires completed150 and empty queue")
    admission = docs["outputs/monthly_financial_coverage_admission_batch2_v1/ADMISSION.json"]
    review = docs["work/after150_batch2_independent_review.json"]
    coverage = docs["outputs/monthly_financial_coverage_admission_batch2_v1/COVERAGE.json"]
    if admission["passed"] is not True or review["passed"] is not True:
        raise ValueError("Source verification incomplete")
    output = ROOT / "migration/after150_batch2_checkpoint.json"
    if output.exists():
        raise ValueError("Write-once checkpoint already exists")
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        last_goal_turn_classification="progress",
        stage="independently_reviewed_conditional_source_admission",
        completed_rounds=150, completed_trials=953, target_reached=False,
        source_hashes=PINS, source_counts=admission["counts"],
        coverage_counts=coverage["counts"],
        new_strategy_trials=0, new_factors_computed=0,
        next_experiment="monthly_financial_coverage151",
        next_experiment_status="pure_code_and_synthetic_test_preparation_not_registered",
        next_experiment_scope="Fixed-dose four-arm comparison: incumbent, original four-issuer earnings, prior six-issuer earnings, cumulative eight-issuer earnings.",
        limitations=[
            "Sources are conditional research inputs, not certified first-public versions.",
            "INTC consecutive annual boundaries are explicit source-use assumptions.",
            "KLAC2011 geometry/nulls, early-calendar holds and KLAC2019 ASC606 hold remain.",
            "Thirteen unprocessed annual catalog entries still block fallback to older reports.",
            "No new performance result; the verified incumbent remains67.66617372041686% historical CAGR.",
        ],
        frozen_observer_accounts_modified=False,
    )
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    before = resume_path.read_bytes()
    resume = json.loads(before)
    with output.open("xb") as stream:
        stream.write(encoded(checkpoint))
    ref = dict(path=str(output.relative_to(ROOT)),
               sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    resume.update(
        after150_batch2_checkpoint=ref,
        latest_source_preparation=checkpoint["stage"],
        next_research_preparation=dict(
            experiment=checkpoint["next_experiment"],
            status=checkpoint["next_experiment_status"],
            actual_new_strategy_trials=0),
        last_goal_turn_classification="progress",
    )
    backup = ROOT / ("migration/RESEARCH_RESUME_before_batch2_" +
                     hashlib.sha256(before).hexdigest() + ".json")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(before)
    elif backup.read_bytes() != before:
        raise ValueError("Checkpoint backup conflict")
    if resume_path.read_bytes() != before:
        raise ValueError("Concurrent research-resume update")
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if resume_path.read_bytes() != before:
            raise ValueError("Concurrent research-resume commit")
        os.replace(str(temporary), str(resume_path))
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(checkpoint=ref, counts=admission["counts"],
                          conditional_new_stock_months=coverage["counts"]["conditional_new_slots"])))


if __name__ == "__main__":
    main()

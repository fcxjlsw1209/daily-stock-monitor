"""Record independently verified batch3 progress without computing a strategy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BATCH = "outputs/monthly_financial_coverage_admission_batch3_v1/"
PINS = {
    "outputs/monthly_financial_coverage151/COMPLETION.json":
        "7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5",
    "outputs/monthly_financial_coverage_batch3_v1/PLAN.json":
        "6dd69ae328c0ddf0c995eda11962716b53ec85caff10e9493454e9adbd01c352",
    "outputs/monthly_financial_coverage_batch3_v1/EXTRACTIONS.json":
        "f023c218e88679b45bfb28b1ebc4c213220fe905b2f02e5fb0c730c51857abef",
    "work/after151_batch3_admission_policy.json":
        "5192981c1053f5abb3a6169a2f1e3ae8980365acd3875e6d0ca486c8c1b31758",
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--review-sha", required=True)
    args = parser.parse_args()
    pins = dict(PINS)
    pins[BATCH + "MANIFEST.json"] = args.manifest_sha
    pins["work/after151_batch3_independent_review.json"] = args.review_sha
    docs = {}

    def read_pinned(name, digest):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Checkpoint input escapes local root")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "Changed input: " + name)
        pins[name] = digest
        return json.loads(raw)

    for name, digest in list(pins.items()):
        docs[name] = read_pinned(name, digest)
    manifest = docs[BATCH + "MANIFEST.json"]
    require(set(manifest["outputs"]) == {"DATASET.json", "ADMISSION.json", "COVERAGE.json"},
            "Complete source package required")
    for name, digest in manifest["outputs"].items():
        docs[BATCH + name] = read_pinned(BATCH + name, digest)
    for name, digest in manifest["input_hashes"].items():
        read_pinned(name, digest) if str(name).endswith(".json") else require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
            "Changed non-JSON source input")
    admission = docs[BATCH + "ADMISSION.json"]
    coverage = docs[BATCH + "COVERAGE.json"]
    review = docs["work/after151_batch3_independent_review.json"]
    require(admission["passed"] is True and review["passed"] is True,
            "Independent review and conditional admission must both pass")
    require(admission["input_hashes"] == manifest["input_hashes"],
            "Manifest and admission input closures differ")
    require(admission["independent_review_sha256"] == args.review_sha,
            "Admission binds a different review")
    require(admission["counts"]["events"] == 10
            and admission["counts"]["annual_catalog_events"] == 32,
            "Complete batch3 event/catalog population required")
    require(coverage["counts"]["stock_slots"] == 3904, "Original stock-month axes changed")
    for key in ("new_strategy_trials", "actual_factors_calculated", "actual_returns_calculated"):
        require(type(admission[key]) is int and admission[key] == 0,
                "This checkpoint is source-only")
    state = json.loads((ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    require(len(state["completed"]) == 151
            and sum(x["trials"] for x in state["completed"]) == 957
            and state["registered_queue"] == [], "Final151 and empty queue required")
    require(state["latest_round_receipt"] == {
        "path": "outputs/monthly_financial_coverage151/COMPLETION.json",
        "sha256": PINS["outputs/monthly_financial_coverage151/COMPLETION.json"]},
        "Wrong finalized predecessor")
    out = ROOT / "migration/after151_batch3_checkpoint.json"
    require(not out.exists(), "Write-once checkpoint already exists")
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        stage="independently_reviewed_conditional_source_admission",
        completed_rounds=151, completed_trials=957, target_reached=False,
        previous_goal_turn_classification="progress: frozen batch3 source extraction",
        last_goal_turn_classification="progress",
        source_hashes=pins, source_counts=admission["counts"],
        coverage_counts=coverage["counts"],
        new_strategy_trials=0, new_factors_computed=0, frozen_observers_modified=False,
        next_experiment="monthly_financial_coverage152",
        next_experiment_status="not_registered",
        next_experiment_scope="Fixed1% cumulative coverage comparison; all predecessor controls retained.",
        limitations=[
            "These sources are conditional research inputs, not certified first-public vintages.",
            "KO FY2012 producer nulls remain null despite independently readable adjusted GAAP source.",
            "Annual boundaries inferred from the same report remain explicit source-use assumptions.",
            "Early-calendar and unresolved accounting holds remain; unprocessed annual events block fallback.",
            "Source admission is not a new strategy result or proof of 70% CAGR.",
        ],
    )
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    before = resume_path.read_bytes()
    resume = json.loads(before)
    with out.open("xb") as stream:
        stream.write(encoded(checkpoint))
    ref = dict(path=str(out.relative_to(ROOT)),
               sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    resume.update(
        after151_batch3_checkpoint=ref,
        latest_source_preparation=checkpoint["stage"],
        next_research_preparation=dict(
            experiment=checkpoint["next_experiment"], status="not_registered",
            source_symbols=["KO", "LRCX"], source_checkpoint=ref,
            values_extracted=27, retained_null_fields=3, actual_new_strategy_trials=0),
        next_research_status="batch3_conditionally_admitted_no_strategy_registration",
        next_step="Preregister the fixed1% batch3 cumulative-coverage contrast before computing new factors or returns; retain all prior controls.",
        last_goal_turn_classification="progress",
        previous_goal_turn_classification=checkpoint["previous_goal_turn_classification"],
        updated_utc=checkpoint["created_utc"],
    )
    backup = ROOT / ("migration/RESEARCH_RESUME_before_batch3_" +
                     hashlib.sha256(before).hexdigest() + ".json")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(before)
    require(backup.read_bytes() == before, "Conflicting checkpoint backup")
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
    print(json.dumps(dict(checkpoint=ref, source_counts=admission["counts"],
                          coverage_counts=coverage["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

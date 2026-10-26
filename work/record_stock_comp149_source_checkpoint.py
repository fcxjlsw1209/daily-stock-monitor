"""Record source-only progress without computing factors or touching accounts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def ref(name):
    return dict(path=name, sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())


def main():
    state_path = ROOT/"migration/RESEARCH_RESUME.json"
    old_bytes = state_path.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    state = json.loads(old_bytes)
    central = json.loads((ROOT/"outputs/monthly_target70_research/CONTINUATION_STATUS.json").read_bytes())
    assert len(central["completed"]) == 148 and not central["registered_queue"]
    assert not (ROOT/"outputs/monthly_stock_comp149/PLAN.json").exists()
    failed_policy = ref("outputs/monthly_stock_comp_source_admission_v1/POLICY.json")
    assert failed_policy["sha256"] == "fd36c7365458a3caa8c4fb0e8d0e41dea7b2990bd632fae54c751de26ed84fbe"
    assert not (ROOT/"outputs/monthly_stock_comp_source_admission_v1/DATASET.json").exists()
    paths = (
        "outputs/monthly_stock_comp_source_v1/PLAN.json",
        "outputs/monthly_stock_comp_source_v1/HTML_INDEPENDENT_REVIEW.json",
        "outputs/monthly_stock_comp_source_v1/PDF_VISUAL_REVIEW.json",
        "outputs/monthly_stock_comp_pdf_repair_v2/PLAN.json",
        "outputs/monthly_stock_comp_pdf_repair_v2/MANIFEST.json",
        "outputs/monthly_stock_comp_source_admission_v2/POLICY.json",
        "outputs/monthly_stock_comp_source_admission_v2/DATASET.json",
        "outputs/monthly_stock_comp_source_admission_v2/ADMISSION.json",
        "outputs/monthly_stock_comp_source_admission_v2/INDEPENDENT_REVIEW.json",
        "work/monthly_stock_comp149_proposal.json",
    )
    note = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        stage="source_admission_passed_before_round149_registration",
        source_evidence=[ref(path) for path in paths],
        original_events=44, independently_verified_present_values=34,
        conditionally_admitted_events=31, original_holds_retained=13,
        failed_v1_policy=failed_policy,
        v1_failure="Build rejected absolute local input hashes through a relative-only verifier; no dataset/admission was published.",
        v2_fix="A separate version uses strict local Evidence.pins with SHA checking and a final stable-input check. Original v1 files remain unchanged.",
        new_strategy_trials=0, real_new_factors=0, real_new_returns=0,
        next="Finish synthetic independent reviews and register the frozen four-arm149 proposal before calculating new factors.",
    )
    dest = ROOT/"migration/stock_comp149_source_checkpoint.json"
    with dest.open("xb") as stream:
        stream.write(encoded(note))
    backup = ROOT/("migration/pre_stock_comp149_source_" + old_sha + ".json")
    with backup.open("xb") as stream:
        stream.write(old_bytes)
    state.update(
        next_research_status="stock_compensation_source_admitted_pending_round149_registration",
        next_step=note["next"], next_mechanism_registered=False,
        latest_source_admission=ref(paths[7]),
        latest_source_independent_review=ref(paths[8]),
        next_source_resume_checkpoint=str(dest.relative_to(ROOT)),
        next_candidate_review=dict(path="work/monthly_stock_comp149_proposal.json",
                                   sha256=ref(paths[9])["sha256"],
                                   status="prospective_four_arm_proposal_not_registered",
                                   new_field_admitted=True, new_strategy_trials=0),
        updated_utc=note["recorded_utc"],
    )
    with tempfile.NamedTemporaryFile(dir=state_path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(encoded(state))
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == old_sha
    os.replace(tmp, state_path)
    print(json.dumps(ref(str(dest.relative_to(ROOT)))))


if __name__ == "__main__":
    main()

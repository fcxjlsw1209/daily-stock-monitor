"""Record completed source preparation without altering any experiment result."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def ref(relative):
    return {"path": relative,
            "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def main():
    census_path = "outputs/monthly_financial_coverage_census_v1/CENSUS.json"
    plan_path = "outputs/monthly_financial_coverage_census_v1/PLAN.json"
    review_path = "work/monthly_financial_census_independent_review.json"
    census, review = load(census_path), load(review_path)
    assert review["passed"] is True
    assert review["census_sha256"] == ref(census_path)["sha256"]
    assert review["plan_sha256"] == ref(plan_path)["sha256"] == census["plan_sha256"]
    state = load("outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert len(state["completed"]) == 149
    assert sum(item["trials"] for item in state["completed"]) == 950
    assert state["registered_queue"] == []
    amzn = load("work/after149_amzn_source_context.json")
    csco = load("work/after149_csco_source_context.json")
    assert amzn["counts"]["selected_html_originals"] == 10
    assert amzn["counts"]["financial_values_extracted"] == 0
    assert csco["passed"] is True and csco["financial_amounts_extracted"] == 0
    assert csco["returns_computed"] == 0
    references = [plan_path, census_path, review_path,
                  "work/after149_amzn_source_context.json",
                  "work/after149_csco_source_context.json",
                  "work/after149_guidance_audit.json",
                  "work/after149_next_mechanism_review.json",
                  "work/after149_coverage_extension_concept.json",
                  "outputs/monthly_stock_comp149/COMPLETION.json"]
    checkpoint = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="source_census_and_first_context_batch_complete",
        previous_goal_turn_classification="progress",
        completed_research_rounds=149, strategy_evaluations=950,
        new_strategy_trials=0, new_financial_admissions=0,
        target_70_reached=False,
        best_candidate=state["best_training_selected_candidate"],
        census_counts=census["counts"],
        context_batch=dict(
            selection="Alphabetical new issuers with complete primary text-label leads",
            AMZN=dict(original_html_documents=10, amounts_extracted=0),
            CSCO=dict(original_html_documents=16, amounts_extracted=0)),
        references={Path(path).name + ":" + str(index): ref(path)
                    for index, path in enumerate(references)},
        conclusion="Source coverage can be extended beyond the current four issuers. Context evidence exists, but no new financial feature or return has yet been admitted or computed.",
        next_steps=[
            "Freeze source-only NI/average-assets extraction for the first context batch with explicit source, date, currency, annual-period and accounting-change treatment.",
            "Test signed-cell and multirow-header parsing; independently verify current/comparative values and applicable identity evidence.",
            "Continue the unchanged73-stock source scope, including necessary-field-only leads and unscanned PDF attachments; do not treat the seven-label locator as final eligibility.",
            "After admission, preregister an expanded-coverage comparison with unchanged incumbent and original-coverage control, first88 selection and later34 diagnostics.",
        ],
        limitations=[
            "The 920 catalog records are not920 audited complete annual reports.",
            "The13 text-label issuers are neither13 admitted issuers nor an exhaustive count of usable reports.",
            "No financial field, first-public version or continuous issuer identity is newly certified by this checkpoint.",
            "The70% objective remains unmet; existing history has been repeatedly researched.",
        ],
    )
    output = ROOT / "migration/after149_coverage_checkpoint.json"
    with output.open("xb") as stream:
        stream.write(encoded(checkpoint))
    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    old_sha = hashlib.sha256(raw).hexdigest()
    backup = ROOT / ("migration/before_after149_coverage_" + old_sha + ".json")
    with backup.open("xb") as stream:
        stream.write(raw)
    resume = json.loads(raw)
    resume.update(
        updated_utc=checkpoint["created_utc"],
        next_research_status="source_context_preparation_after149",
        next_mechanism_registered=False,
        after149_coverage_checkpoint=ref(str(output.relative_to(ROOT))),
        after149_context_batch=["AMZN", "CSCO"],
        next_step=checkpoint["next_steps"][0],
        previous_goal_turn_classification="progress: completed73-stock source census and26 original-report context reviews; no new strategy trials",
    )
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
    assert hashlib.sha256(resume_path.read_bytes()).hexdigest() == old_sha
    os.replace(temporary, resume_path)
    print(json.dumps({"checkpoint": ref(str(output.relative_to(ROOT))),
                      "new_trials": 0, "target_70_reached": False}))


if __name__ == "__main__":
    main()

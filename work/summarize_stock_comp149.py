"""Summarize completed round149 evidence and update only its resume metadata."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/monthly_stock_comp149"
PLAN_SHA = "9b0095d8112cb4579e63f85e2f6b837fe8d7ebb5eef5e399fd926f41c682735d"


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    complete = read(OUT/"COMPLETION.json")
    assert complete["round"] == 149 and complete["plan_sha256"] == PLAN_SHA
    assert complete["independent_execution_paths"] == 12 and complete["protected_files"] == 21444
    state = read(ROOT/"outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    assert len(state["completed"]) == 149 and not state["registered_queue"]
    assert sum(row["trials"] for row in state["completed"]) == 950
    results = read(OUT/"RESULTS.json")
    orders = read(OUT/"PRETRAIN_ORDERS.json")
    independent = read(OUT/"INDEPENDENT_OUTCOME.json")
    feature = read(OUT/"FEATURE_AUDIT.json")
    assert independent["passed"] and independent["plan_sha256"] == PLAN_SHA
    assert independent["selected_name"] == complete["primary"]
    assert complete["all_arm_results"] == results
    assert ref(OUT/"INDEPENDENT_OUTCOME.json")["sha256"] == complete["independent_outcome_sha256"]
    arms = ("earnings_control", "cashflow_control", "comp_adjusted_cashflow")
    same_sets = all(
        [sorted(x["symbols"]) for x in orders[name]] ==
        [sorted(x["symbols"]) for x in orders[arms[0]]]
        for name in arms[1:])
    changes = [
        dict(date=base["date"], incumbent=base["symbols"], challenger=other["symbols"])
        for base, other in zip(orders["incumbent_local67"], orders["comp_adjusted_cashflow"])
        if set(base["symbols"]) != set(other["symbols"])
    ]
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), round=149,
        completed_rounds=149, total_trials=950, primary=complete["primary"],
        target_reached=complete["target_reached"],
        period=dict(start="2016-08-17", end="2026-09-24", months=122,
                    selection_months=88, later_diagnostic_months=34),
        results={name: dict(
            same_close_zero_cost=result["same_close"]["metrics"],
            next_open_5bp_each_side=result["next_open_cost"]["metrics"],
            slices=result["same_close"]["slices"])
            for name, result in results.items()},
        financial_arms_have_identical_constituent_sets=same_sets,
        changed_order_months=changes,
        feature_availability=dict(available=feature["common_available_cells"],
                                  original=feature["original_cash_quality_available_cells"],
                                  reasons=feature["reasons"]),
        conclusion="The fixed reported SBC adjustment added no portfolio discrimination beyond the two financial controls and did not pass the predeclared training guards. Incumbent retained; 70% target unmet.",
        plan=ref(OUT/"PLAN.json"), completion=ref(OUT/"COMPLETION.json"),
        independent=ref(OUT/"INDEPENDENT_OUTCOME.json"),
        orders=ref(OUT/"PRETRAIN_ORDERS.json"), protected=ref(OUT/"PROTECTED_AFTER.json"),
        inherited_limitations=[
            "Repeatedly inspected development history, not pristine out of sample.",
            "Conditional sparse four-issuer accounting source scope, not certified first-public versions.",
            "Same-close zero-cost returns are the research comparison; next-open and costs are reported separately.",
            "No observer strategy, ledger, or order plan is promoted or changed.",
        ],
    )
    with (OUT/"SUMMARY.json").open("xb") as stream:
        stream.write(encoded(summary))
    resume_path = ROOT/"migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    old_sha = hashlib.sha256(raw).hexdigest()
    resume = json.loads(raw)
    backup = ROOT/("migration/round149_resume_"+old_sha+".json")
    with backup.open("xb") as stream:
        stream.write(raw)
    artifacts = {name: ref(OUT/name) for name in (
        "PLAN.json", "INDEPENDENT_PREPARE.json", "INDEPENDENT_OUTCOME.json",
        "PROTECTED_AFTER.json", "COMPLETION.json", "SUMMARY.json")}
    resume.update(
        round=149, completed_research_rounds=149, strategy_evaluations=950,
        current_experiment="monthly_stock_comp149", current_experiment_artifacts=artifacts,
        formal_round149_completed=True, formal_round149_completion=artifacts["COMPLETION.json"],
        plan_sha256=PLAN_SHA, frozen_primary=complete["primary"],
        latest_protected_check=artifacts["PROTECTED_AFTER.json"],
        full_protection_check_passed=True,
        remaining_formal_archive_check="None for149;21444 protected files verified",
        target_70_reached=complete["target_reached"],
        next_mechanism_registered=False, next_research_status="round149_complete_no_next_mechanism_registered",
        next_step="Review distinct guidance-consistent mechanisms or independently justified source improvements before a new prospective registration. Do not rescale or tune the rejected SBC feature to the observed returns.",
        previous_goal_turn_classification="progress: completed149, source repair and independent12-path verification",
        resume_update_checkpoint=str(backup.relative_to(ROOT)),
        updated_utc=summary["created_utc"],
    )
    resume["next_candidate_review"].update(status="registered_and_completed_round149", new_strategy_trials=4)
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(encoded(resume))
    assert hashlib.sha256(resume_path.read_bytes()).hexdigest() == old_sha
    os.replace(tmp, resume_path)
    print(json.dumps(dict(summary=ref(OUT/"SUMMARY.json"),
                          financial_arms_same=same_sets, changed_months=len(changes),
                          target_reached=complete["target_reached"])))


if __name__ == "__main__":
    main()

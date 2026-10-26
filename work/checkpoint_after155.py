"""Record formally completed155 and its next research decision; metadata only.

Run after the completion-bound summary exists. This helper neither imports
research code nor reads prices, predictions, orders or portfolio-return arrays.
It is deliberately outside the registered experiment's execution closure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = "monthly_endpoint155"
DIRECTORY = "outputs/" + STUDY
STATE = "outputs/monthly_target70_research/CONTINUATION_STATUS.json"
NAMES = ("incumbent_local67", "matched_mean_return_ridge", "matched_loss_indicator_ridge")
NEXT_STEP = (
    "Re-evaluate a distinct, economically explained hypothesis against the completed "
    "research ledger and source availability before any new registration. Do not "
    "retune the same fixed-dose endpoint-target comparison after observing its "
    "outcomes. Preserve the long-momentum/skip-two-months monthly EW3 framework "
    "and frozen observation accounts; register any genuinely distinct experiment "
    "before new labels, fits, orders or returns."
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def preserve(path, raw):
    if path.exists():
        check(path.read_bytes() == raw, "Conflicting immutable checkpoint: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())


def main(completion_sha, summary_sha):
    check(all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x)
              for x in (completion_sha, summary_sha)), "Explicit final artifact digests required")
    observed = {}

    def read(relative, expected=None):
        path = ROOT / relative
        check(not path.is_symlink(), "Metadata input must not be a symlink")
        raw = path.read_bytes()
        value = digest(raw)
        check(expected is None or value == expected, "Changed input: " + relative)
        observed[relative] = value
        return json.loads(raw)

    def ref(relative):
        return dict(path=relative, sha256=observed[relative])

    def unchanged():
        for relative, expected in observed.items():
            check(digest((ROOT / relative).read_bytes()) == expected,
                  "Concurrent evidence update: " + relative)

    completion_path = DIRECTORY + "/COMPLETION.json"
    summary_path = DIRECTORY + "/SUMMARY.json"
    done = read(completion_path, completion_sha)
    summary = read(summary_path, summary_sha)
    check(done.get("round") == 155 and done.get("directory") == STUDY
          and done.get("status") == "completed_independently_verified"
          and done.get("trials") == 3 and done.get("protected_files") == 21444
          and done.get("independent_execution_paths") == 9, "Formal155 closure required")
    check(summary.get("round") == 155 and summary.get("status") == "completed_summary"
          and summary.get("completion_sha256") == completion_sha
          and summary.get("plan_sha256") == done["plan_sha256"], "Completion-bound summary required")
    plan = read(DIRECTORY + "/PLAN.json", done["plan_sha256"])
    check(plan.get("round") == 155 and plan.get("status") == "REGISTERED"
          and plan.get("arms") == list(NAMES), "Registered155 PLAN required")
    check(str(Path(__file__).resolve()) not in plan["hashes"]
          and "work/checkpoint_after155.py" not in plan["hashes"],
          "Checkpoint helper must remain outside execution dependencies")
    selection = done["selection"]
    selected, admitted, target = done["primary"], selection["admitted"], done["target_reached"]
    check(selected in NAMES and selection.get("primary") == selected
          and type(admitted) is bool and type(target) is bool
          and summary.get("selection") == selection and summary.get("primary") == selected
          and summary.get("admitted") == admitted and summary.get("target_reached") == target,
          "Selected/admitted/target metadata mismatch")
    arms = summary["arms"]
    check([row["name"] for row in arms] == list(NAMES), "Three ordered summary arms required")
    selected_row = next(row for row in arms if row["name"] == selected)
    selected_metrics = selected_row["full"]["same_close"]
    check(all(finite(selected_metrics.get(k)) for k in ("cagr", "sharpe", "max_drawdown"))
          and selected_metrics["cagr"] == done["primary_cagr"]
          and selected_metrics["sharpe"] == done["primary_sharpe"]
          and target == (selected_metrics["cagr"] >= .7), "Finite selected result metadata required")
    check(type(done.get("new_model_fits")) is int and done["new_model_fits"] >= 0
          and summary.get("producer_model_fits") == done["new_model_fits"]
          and summary.get("registered_trials") == 3 and summary.get("execution_paths") == 9
          and summary.get("monthly_orders") == 366
          and summary.get("additional_training_label_or_return_calculations") == 0,
          "Completed scope mismatch")
    for filename, expected in (("INDEPENDENT_OUTCOME.json", done["independent_outcome_sha256"]),
                               ("PROTECTED_AFTER.json", done["protected_sha256"])):
        report = read(DIRECTORY + "/" + filename, expected)
        check(report.get("passed") is True, "Passed completion evidence required: " + filename)
        if filename == "INDEPENDENT_OUTCOME.json":
            check(report.get("stage") == "evaluate" and report.get("plan_sha256") == done["plan_sha256"]
                  and report.get("selected_name") == selected
                  and report.get("primary_target_reached") == target, "Independent selected result mismatch")
        else:
            check(report.get("checked") == report.get("verified") == 21444
                  and all(report.get(k) == [] for k in ("changed", "missing", "read_or_identity_errors")),
                  "Full protected archive verification required")
    state = read(STATE)
    completed = state.get("completed", [])
    check(len(completed) == 155 and all(type(row.get("trials")) is int and row["trials"] >= 0
                                      for row in completed)
          and sum(row["trials"] for row in completed) == 978
          and state.get("registered_queue") == [] and state.get("current") == STUDY
          and state.get("current_status") == "completed_independently_verified",
          "Exact completed155/978 central state required")
    last = completed[-1]
    check(last.get("directory") == STUDY and last.get("completion") == ref(completion_path)
          and state.get("latest_round_receipt") == ref(completion_path)
          and last.get("primary") == selected and last.get("new_admitted") == int(admitted)
          and last.get("target_reached") == target and last.get("trials") == 3
          and last.get("primary_cagr") == done["primary_cagr"]
          and last.get("primary_sharpe") == done["primary_sharpe"], "Central completion differs")

    resume_path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = resume_path.read_bytes()
    resume = json.loads(raw)
    check(resume.get("round") in (154, 155), "Do not overwrite an unrelated/newer resume")
    if resume["round"] == 155:
        check(resume.get("formal_round155_completion") == ref(completion_path),
              "Conflicting round155 resume")
    checkpoint_path = "migration/after155_research_checkpoint.json"
    destination = ROOT / checkpoint_path
    saved = json.loads(destination.read_bytes()) if destination.exists() else None
    if saved:
        previous = saved["prior_resume"]
        check(re.fullmatch(r"migration/after155_resume_[0-9a-f]{64}\.json", previous["path"])
              and digest((ROOT / previous["path"]).read_bytes()) == previous["sha256"],
              "Original resume backup changed")
    else:
        previous = dict(path="migration/after155_resume_" + digest(raw) + ".json", sha256=digest(raw))
    record = dict(
        created_utc=saved["created_utc"] if saved else datetime.now(timezone.utc).isoformat(),
        stage="completed155_next_distinct_hypothesis_review", goal_turn_classification="progress",
        formal_completion=ref(completion_path), summary=ref(summary_path),
        plan=ref(DIRECTORY + "/PLAN.json"), central_state=ref(STATE), prior_resume=previous,
        completed_rounds=155, total_trials=978, selected=selected, admitted=admitted,
        target_70_reached=target, primary_metrics=selected_metrics,
        producer_model_fits=done["new_model_fits"], new_trials_in_next_design=0,
        next_mechanism_registered=False, next_step=NEXT_STEP,
        decision=dict(
            endpoint_retest="Completed fixed1% paired-target retest; do not rerun completed phases or tune its dose after seeing outcomes.",
            next_research="Check a distinct mechanism, prior experiments and source feasibility before creating a new prospective design.",
            limitations="Historical repetition, uncalibrated binary scores, inherited training mask and local pool/vintage limitations remain.",
            observations="Original frozen observation accounts and ledgers remain unchanged; no automatic promotion."),
        new_label_fit_order_return_calculations=0)
    if saved:
        check(saved == record, "Existing checkpoint conflicts with completed evidence")
    checkpoint_ref = dict(path=checkpoint_path, sha256=digest(encoded(record)))
    if resume.get("after155_research_checkpoint") == checkpoint_ref:
        check(resume.get("round") == 155 and resume.get("completed_research_rounds") == 155
              and resume.get("strategy_evaluations") == 978
              and resume.get("frozen_primary") == selected and resume.get("target_70_reached") == target
              and resume.get("next_step") == NEXT_STEP, "Partially applied checkpoint")
        unchanged()
        print(json.dumps(dict(checkpoint=checkpoint_ref, already_applied=True)))
        return
    unchanged()
    if not saved:
        preserve(ROOT / previous["path"], raw)
    preserve(destination, encoded(record))
    if "research_preparation_before155" not in resume:
        resume["research_preparation_before155"] = resume.get("next_research_preparation")
    artifacts = {name: ref(DIRECTORY + "/" + name) for name in (
        "PLAN.json", "COMPLETION.json", "SUMMARY.json", "INDEPENDENT_OUTCOME.json", "PROTECTED_AFTER.json")}
    resume.update(
        round=155, completed_research_rounds=155, strategy_evaluations=978,
        current_experiment=STUDY, current_experiment_artifacts=artifacts,
        formal_round155_completed=True, formal_round155_completion=ref(completion_path),
        plan_sha256=done["plan_sha256"], frozen_primary=selected, target_70_reached=target,
        latest_protected_check=ref(DIRECTORY + "/PROTECTED_AFTER.json"),
        full_protection_check_passed=True, remaining_formal_archive_check="None for155;21444 files verified",
        after155_research_checkpoint=checkpoint_ref, resume_update_checkpoint=previous["path"],
        next_research_preparation=dict(status="distinct_hypothesis_review_only", new_trials=0,
            prior_completed_experiment=ref(completion_path), registered=False, novelty_established=False),
        next_mechanism_registered=False, next_research_status="review_distinct_hypothesis_before_registration",
        next_step=NEXT_STEP, last_goal_turn_classification="progress", updated_utc=record["created_utc"])
    with tempfile.NamedTemporaryFile(dir=resume_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume)); stream.flush(); os.fsync(stream.fileno())
    try:
        unchanged()
        check(resume_path.read_bytes() == raw, "Concurrent resume update")
        os.replace(temporary, resume_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(checkpoint=checkpoint_ref, selected=selected, admitted=admitted,
                          target_reached=target, completed_rounds=155, total_trials=978)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-sha", required=True)
    parser.add_argument("--summary-sha", required=True)
    args = parser.parse_args()
    main(args.completion_sha, args.summary_sha)

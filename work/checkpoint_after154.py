"""Save the next research action after formal154 closure, without modelling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage154"
EVIDENCE = {
    "source_context": (
        "work/after154_source_context_review.json",
        "10725ea8952821863d801347c180d92f591a84adb974a05e2c8802912b4a4c4f",
    ),
    "novelty_review": (
        "work/after154_terminal_loss_novelty.json",
        "95b3bde9d26bcd1c46659000c619cdd12c663951b33dce229643d3064e24af64",
    ),
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def load(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def preserve(path, raw):
    if path.exists():
        check(path.read_bytes() == raw, "Conflicting checkpoint: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def main(design_sha):
    summary = load(OUT / "SUMMARY.json")
    done = load(OUT / "COMPLETION.json")
    check(summary["completion"] == ref(OUT / "COMPLETION.json"), "Summary binding")
    check(done["round"] == 154 and done["trials"] == 7
          and done["protected_files"] == 21444, "Formal154 closure required")
    state = load(ROOT / "outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 154 and not state["registered_queue"]
          and sum(item["trials"] for item in state["completed"]) == 975,
          "Unexpected continuation state")
    design_path = ROOT / "work/after154_endpoint_retest_design_v2.json"
    check(ref(design_path)["sha256"] == design_sha, "Design binding")
    design = load(design_path)
    check(design["registered"] is False
          and design["actual_label_calculations"] == 0
          and design["actual_model_fits"] == 0
          and design["actual_orders"] == 0
          and design["actual_returns"] == 0, "Design must precede calculation")
    evidence = {}
    for key, (name, digest) in EVIDENCE.items():
        path = ROOT / name
        check(ref(path)["sha256"] == digest, "Evidence binding: " + key)
        evidence[key] = ref(path)
    novelty = load(ROOT / EVIDENCE["novelty_review"][0])
    check(novelty["novelty_established"] is False, "Preserve novelty uncertainty")
    destination = ROOT / "migration/after154_research_checkpoint.json"
    saved = load(destination) if destination.exists() else None
    record = {
        "created_utc": saved["created_utc"] if saved else datetime.now(timezone.utc).isoformat(),
        "stage": "completed154_next_bounded_retest_design_not_registered",
        "goal_turn_classification": "progress",
        "formal_completion": ref(OUT / "COMPLETION.json"),
        "summary": ref(OUT / "SUMMARY.json"),
        "completed_rounds": 154,
        "total_trials": 975,
        "target_70_reached": done["target_reached"],
        "primary": done["primary"],
        "primary_cagr": done["primary_cagr"],
        "primary_sharpe": done["primary_sharpe"],
        "source_evidence": evidence,
        "next_design": ref(design_path),
        "new_trials_in_next_design": 0,
        "next_mechanism_registered": False,
        "decision": {
            "financial_coverage": "Retain verified data and remaining source gaps; do not automatically repeat the same fixed-dose coverage strategy experiment.",
            "next_primary": "Implement and independently review a matched natural-month return versus terminal-loss-indicator Ridge contrast, then register before new labels, fits, scores or returns.",
            "novelty": "Historical repetition is not excluded: round32 and round116 source files are outside the fully recovered protected archive. Treat the design explicitly as a bounded retest.",
            "source_backlog": "ACN nine original annual PDFs are context-verified leads, not admitted values. Missing FY2021/FY2023 catalog entries and FY2019 comparability need resolution before use.",
            "constraints": "Keep long momentum and skip42 formation, original pool, EW3 and fixed intramonth holdings; no frozen observer changes.",
        },
    }
    preserve(destination, encoded(record))
    path = ROOT / "migration/RESEARCH_RESUME.json"
    raw = path.read_bytes()
    resume = json.loads(raw)
    check(resume["round"] == 154 and resume["formal_round154_completed"] is True,
          "Report154 checkpoint required first")
    checkpoint_ref = ref(destination)
    if resume.get("after154_research_checkpoint") == checkpoint_ref:
        print(json.dumps({"checkpoint": checkpoint_ref, "already_applied": True}))
        return
    backup = ROOT / ("migration/after154_resume_" + hashlib.sha256(raw).hexdigest() + ".json")
    preserve(backup, raw)
    resume.update(
        after154_research_checkpoint=checkpoint_ref,
        next_research_preparation={
            "status": "bounded_terminal_loss_retest_design_only",
            "design": ref(design_path),
            "source_evidence": evidence,
            "new_trials": 0,
            "novelty_established": False,
        },
        next_mechanism_registered=False,
        next_research_status="implement_and_review_bounded_retest_before_registration",
        next_step=record["decision"]["next_primary"],
        last_goal_turn_classification="progress",
        updated_utc=record["created_utc"],
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(resume))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        check(path.read_bytes() == raw, "Concurrent resume update")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"checkpoint": checkpoint_ref, "target_reached": done["target_reached"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-sha", required=True)
    main(parser.parse_args().design_sha)

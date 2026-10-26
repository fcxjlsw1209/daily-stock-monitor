"""Preview, then explicitly commit, the metadata-only after156 research handoff.

Default execution is read-only. --apply-checkpoint-sha requires the exact digest
printed by a reviewed preview, with the same --asof. No strategy modules,
arrays, source collectors, registration functions or observers are imported.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STATE = "outputs/monthly_target70_research/CONTINUATION_STATUS.json"
RESUME = "migration/RESEARCH_RESUME.json"
CHECKPOINT = "migration/after156_research_checkpoint.json"
STUDY = "outputs/monthly_revenue_persistence156/"
PINS = {
    STUDY + "COMPLETION.json": "97177e6413d2227cb727e9efcc027b0cff68084a8c8e8a899f97d02b573665e2",
    STUDY + "SUMMARY.json": "30b7c0020797e743a6564aebee5ae1a7a0b9b3f3a1b971d23dfd7d0623b4ae20",
    STUDY + "SUMMARY.md": "d4ed50f52fc9340fe307fe02142a2582330613dcbcdf10450d4f6f54155e3619",
    "work/after156_filing_universe_inventory.json": "e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286",
    "work/after156_filing_drift_kernel_review.json": "f82cd8014106014b7e830007511682a542c176c2a7d0f26a73e1aa3f89aaace6",
    "work/after156_filing_drift_design.json": "4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75",
    "work/after156_filing_clock_independent.json": "1a87cd2fc21766806894ef4f9506bdc01dcc91c38fa6a2dd82a4328d93f849eb",
    "work/after156_filing_date_anchor_diagnostic.json": "4747ec5bf3473e1cf54ec1c78d716eddbcd93ed735af300a8c6122854613d78e",
    "work/after156_filing_date_reference.json": "64405c42ba692120714a82ade3f841e8a91959cfb7cc1dd3dba5ca0d48a1e362",
    "work/after156_filing_warmup.json": "535eaba07cff687330be47c998cbd614985f83b771db4de6bcb8d857fb065837",
    "work/after156_filing_event_feasibility.json": "1af27cbd0777d6e0542fe56c8306efc3b747e018ac00bd7e70d7400c99abc6de",
    "work/after155_sec_gap_discovery_collection.json": "78f988dbcb18c16a5cad6873f9c88e764d5dab6fc1c57eb5b62184a4c4129af2",
    "work/after156_score_margin_audit_v2.json": "28153e5a977ea9b2d2db54e27c855780899a17d51f2038a44fbf37b035af118b",
    "work/after156_source_path_review.json": "afee84acd1a388291d3f9b448648ed2ecb15c314d53e2f050bcf1436c6231711",
    "work/after156_filing_coverage_ceiling.json": "88c13206ad3901ee1cfc211d82fa2147f2068ef0f0018c6afd4bef8b6fd45113",
    "work/after156_identity_binding_batch.json": "a381874f329871a0e602606e3325fd923274feb35b7d8f8c663ec0a8941c921f",
}
NEXT = (
    "156 completed rounds / 982 trials; incumbent retained and 70% target unmet. "
    "Continue the unregistered SEC 10-Q/10-K filing-event research using the full "
    "original equity universe inventory. Reuse the 110 dated source bindings "
    "and conditional coverage screen to finish bounded identity and canonical "
    "chronology work; these are evidence progress, not full filing admission. Resolve "
    "historical identity, original provenance, complete original/amended chronology "
    "and public-availability timing before actual factors. Exact acceptance time "
    "is unadmitted; the date-only proposal is a separate conditional alternative, "
    "also unadmitted. Preserve dominant skip42 momentum, monthly EW3 and fixed "
    "intramonth holdings. Reconcile novelty and freeze an admissible protocol, "
    "common mask and independent reviews before prospective registration. No "
    "actual factor, rank, order or return calculations until those gates pass. "
    "Honor the SEC provider stop for 2026-10-20; no route/identity substitution "
    "or automatic retry. Frozen observation accounts remain unchanged."
)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def decoded(raw):
    # Preserve historical strings exactly in these mutable handoff documents.
    # The launcher's json.loads relocation must not rewrite unrelated fields.
    return json.JSONDecoder().decode(raw.decode("utf-8"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(root, name):
    p = Path(name)
    check(not p.is_absolute() and ".." not in p.parts, "Relative local path required")
    path = root / p
    check(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),
          "Unsafe checkpoint path: " + name)
    return path


def field_patch(before, patch):
    after = copy.deepcopy(before)
    after.update(copy.deepcopy(patch))
    check({k: v for k, v in after.items() if k not in patch} ==
          {k: v for k, v in before.items() if k not in patch}, "Unrelated fields changed")
    return after


def build(root, asof):
    stamp = datetime.fromisoformat(asof)
    check(stamp.utcoffset() is not None, "Timezone-aware --asof required")
    hashes, raw_inputs, data = {}, {}, {}

    def read(name, expected=None, parse=True):
        raw = local(root, name).read_bytes()
        observed = sha(raw)
        check(expected is None or observed == expected, "Changed evidence: " + name)
        check(name not in hashes or hashes[name] == observed, "Concurrent evidence change")
        hashes[name], raw_inputs[name] = observed, raw
        if parse:
            data[name] = decoded(raw)
            return data[name]
        return raw

    def ref(name):
        return {"path": name, "sha256": hashes[name]}

    for name, expected in PINS.items():
        read(name, expected, not name.endswith(".md"))
    for name in ("AGENTS.md", "LOCAL_SETUP.md", "run_local.sh", "local_run.py",
                 "local.sandbox.sb", "work/after156_research_checkpoint.py"):
        read(name, parse=False)
    done, summary = data[STUDY + "COMPLETION.json"], data[STUDY + "SUMMARY.json"]
    check(done["round"] == 156 and done["status"] == "completed_independently_verified"
          and done["trials"] == 4 and done["protected_files"] == 21444
          and done["independent_execution_paths"] == 12 and done["new_model_fits"] == 0,
          "Formal round156 closure required")
    for key, filename in (("plan", "PLAN.json"), ("completion", "COMPLETION.json"),
                          ("independent", "INDEPENDENT_OUTCOME.json"),
                          ("protected", "PROTECTED_AFTER.json")):
        check(summary[key]["path"] == STUDY + filename, "Unexpected summary binding")
        read(STUDY + filename, summary[key]["sha256"])
    read(STUDY + "INDEPENDENT_PREPARE.json", "71cde0326c69f1d6e3b8e37f54cc20935bd0ddab5e2cd497fba390c8e54c0604")
    check(summary["round"] == 156 and summary["total_trials"] == 982
          and summary["primary"] == done["primary"] == "incumbent_local67"
          and summary["selection"] == done["selection"]
          and done["selection"]["admitted"] is False
          and summary["target_reached"] is done["target_reached"] is False
          and done["primary_cagr"] == 0.6766617372041686
          and all(v == [] for v in summary["changed_order_months"].values())
          and summary["observer_changes"] == 0, "Completed result mismatch")
    outcome = data[STUDY + "INDEPENDENT_OUTCOME.json"]
    check(outcome["passed"] is True and outcome["stage"] == "evaluate"
          and outcome["plan_sha256"] == done["plan_sha256"] == hashes[STUDY + "PLAN.json"]
          and outcome["selected_name"] == done["primary"]
          and abs(outcome["primary_cagr"] - done["primary_cagr"]) < 1e-12
          and outcome["primary_target_reached"] is False
          and outcome["execution_paths_verified"] == 12
          and outcome["total_monthly_orders_verified"] == 488
          and outcome["first88_and_three_fold_selection_verified"] is True,
          "Independent outcome mismatch")
    protected = data[STUDY + "PROTECTED_AFTER.json"]
    check(protected["passed"] is True and protected["checked"] == protected["verified"] == 21444
          and all(protected[k] == [] for k in ("changed", "missing", "read_or_identity_errors")),
          "Completed protected receipt mismatch")
    diagnostic = data["work/after156_score_margin_audit_v2.json"]
    check(diagnostic["purpose"] == "posthoc_diagnostic_of_completed156_no_new_counterfactual_orders"
          and diagnostic["completed_orders_only"] is True
          and diagnostic["original_cache_artifacts_verified"] is True
          and all(diagnostic[k] == 0 for k in ("actual_new_strategy_trials",
              "actual_new_factor_calculations", "actual_new_returns_calculated", "actual_new_model_fits")),
          "Completed-score diagnostic scope changed")
    for group in (diagnostic["execution_hashes"], diagnostic["input_hashes"]):
        for name, expected in group.items():
            read(name, expected, False)
    superseded = diagnostic["supersedes_diagnostic"]
    read(superseded["path"], superseded["sha256"], False)
    segments = diagnostic["segments"]
    check([r["segment"] for r in segments] == ["early53", "recent35", "later34"]
          and [r["months"] for r in segments] == [53, 35, 34]
          and [r["eligible_available_cells"] for r in segments] == [0, 37, 57]
          and [r["available_source_cells"] for r in segments] == [0, 49, 68]
          and [r["original_selected_source_cells"] for r in segments] == [0, 1, 2],
          "Diagnostic source counts changed")
    for row in segments:
        check(len(row["eligible_source_saved_ranks"]) == row["eligible_available_cells"]
              and sum(r["selected"] for r in row["eligible_source_saved_ranks"]) == row["original_selected_source_cells"]
              and all(arm["changed_order_months"] == 0 for arm in row["arms"].values()),
              "Diagnostic row counts or decisions changed")
    source_path_review = data["work/after156_source_path_review.json"]
    check(source_path_review["stage"] == "READ_ONLY_SOURCE_PATH_AND_CONTRACT_REVIEW_AFTER156"
          and not any(source_path_review["admission"].values()), "Source-path review is not admission")
    for name, expected in source_path_review["input_hashes"].items():
        read(name, expected, False)
    kernel = data["work/after156_filing_drift_kernel_review.json"]
    check(kernel["passed"] is True and not any(kernel["admission"].values())
          and all(v == 0 for v in kernel["actual_activity"].values())
          and kernel["validation"]["producer_tests_passed"] == 25
          and kernel["validation"]["independent_tests_passed"] == 15,
          "Scoped synthetic kernel receipt required")
    for name, expected in kernel["input_hashes"].items():
        read(name, expected, False)
    inventory = data["work/after156_filing_universe_inventory.json"]
    clock = data["work/after156_filing_clock_independent.json"]
    dates = data["work/after156_filing_date_anchor_diagnostic.json"]
    warmup = data["work/after156_filing_warmup.json"]
    design = data["work/after156_filing_drift_design.json"]
    ceiling = data["work/after156_filing_coverage_ceiling.json"]
    identity = data["work/after156_identity_binding_batch.json"]
    check(inventory["summary"]["original_stocks"] == 73
          and inventory["summary"]["stock_candidate_slots"] == 3904
          and inventory["summary"]["clocks_admitted"] == 0
          and inventory["summary"]["fully_admitted_historical_security_mappings"] == 0
          and all(v == 0 for v in inventory["activity_counts"].values()), "Inventory is not admission")
    check(clock["passed"] is True and not any(clock["admission"].values())
          and not any(dates["admission"].values())
          and clock["independent_date_proposal_review"]["independent_rows_compared"] == 141
          and clock["independent_date_proposal_review"]["passed"] is True,
          "Clock and date proposal must remain unadmitted")
    check(warmup["summary"]["all122"] == {"axis_ready": 3904, "reasons": {"axis_ready_only": 3904}, "slots": 3904}
          and all(warmup[k] == 0 for k in ("factor_calculations", "new_trials", "price_array_values_read", "returns"))
          and design["prospective_round"] is None
          and design["activity"]["new_trials_registered"] == 0, "Prospective scope changed")
    screen = ceiling["summary"]["current_metadata_links"]["conditional_date_envelope_screen"]
    check(ceiling["status"] == "SOURCE_ONLY_OPTIMISTIC_BOUNDS_NOT_ADMISSION"
          and all(v == 0 for v in ceiling["activity_counts"].values())
          and [(screen[k]["screened_slots"], screen[k]["stock_candidate_slots"])
               for k in ("first88", "early53", "recent35")] == [(1897, 2816), (1109, 1696), (788, 1120)]
          and all(r["actual_source_identity_clock_gate_passed"] is False
                  for group in ceiling["numerical_floor_comparisons"].values() for r in group.values()),
          "Conditional coverage screen must remain unadmitted")
    check(identity["status"] == "SOURCE_BINDINGS_BUILT_NOT_HISTORICAL_SECURITY_OR_FILING_CLOCK_ADMISSION"
          and identity["summary"]["bindings"] == len(identity["bindings"]) == 110
          and identity["summary"]["symbols"] == 14
          and identity["summary"]["historical_security_epochs_admitted"] == 0
          and identity["summary"]["new_filing_gates_satisfied"] == 0
          and identity["synthetic_validation"]["passed"] is True
          and all(v == 0 for v in identity["activity_counts"].values()),
          "Dated identity evidence is not complete historical admission")
    # Recheck producer/test code and cross-receipt bindings, not the full raw-source
    # inventory or price archive. Those original audit scopes remain explicit.
    for report in (inventory, clock, dates, warmup, ceiling, identity):
        for name, expected in report["input_hashes"].items():
            if name in PINS or name.endswith(".py") or name in ("AGENTS.md", "LOCAL_SETUP.md", "run_local.sh", "local.sandbox.sb"):
                read(name, expected, False)
    collection = data["work/after155_sec_gap_discovery_collection.json"]
    failure = collection["receipts"][0]
    check(failure["SEC_provider_stopped_for_UTC_date"] == "2026-10-20"
          and failure["http_status"] == 403 and collection["direct_requests"] == 1,
          "Bound SEC stop receipt required")
    read(failure["path"], failure["sha256"], False)
    stop = {"provider": "SEC", "day": "2026-10-20", "day_timezone": "UTC",
            "created_utc": failure["request_utc"], "http_status": 403,
            "request_receipt": ref("work/after155_sec_gap_discovery_collection.json"),
            "failure_body": ref(failure["path"]), "recovery_guaranteed": False,
            "no_automatic_retry": True,
            "scope": "No further SEC requests on this date through alternate routes or identities; no recovery inferred from elapsed time."}
    state, resume = read(STATE), read(RESUME)
    check(len(state["completed"]) == 156 and sum(r["trials"] for r in state["completed"]) == 982
          and state["registered_queue"] == [] and state["current"] == "monthly_revenue_persistence156"
          and state["current_status"] == "completed_independently_verified"
          and state["completed"][-1]["completion"] == ref(STUDY + "COMPLETION.json"),
          "Exact central 156/982 completed state required")
    check(resume["completed_research_rounds"] == 156 and resume["strategy_evaluations"] == 982
          and resume["round"] == 156 and resume["formal_round156_completion"] == ref(STUDY + "COMPLETION.json"),
          "Exact resume 156/982 state required")
    proposal = dict(ref("work/after156_filing_drift_design.json"),
                    status="conditional_protocol_not_registered", registered=False,
                    new_strategy_trials=0, new_factor_values=0, new_strategy_returns=0)
    source = dict(ref("work/after156_filing_universe_inventory.json"),
                  status="full_universe_metadata_only_source_clock_identity_gates_pending",
                  original_equities=73, stock_candidate_slots=3904,
                  original_10q_10k_observed_for_equities=58,
                  admitted_historical_mappings=0, clocks_admitted=0,
                  inventory_candidate_missing_required_history_pages=100,
                  missing_page_limit="Across alternative snapshots and locating CIKs; not a required minimum download count.",
                  source_path_review=ref("work/after156_source_path_review.json"),
                  coverage_screen=ref("work/after156_filing_coverage_ceiling.json"),
                  identity_evidence=ref("work/after156_identity_binding_batch.json"),
                  conditional_current_CIK_screen={k: {"slots": screen[k]["screened_slots"],
                      "denominator": screen[k]["stock_candidate_slots"]}
                      for k in ("first88", "early53", "recent35")},
                  source_identity_bindings=110, source_identity_symbols=14,
                  new_complete_filing_gates_satisfied=0,
                  evidence_limits="Coverage is a conditional routing screen, not clock/identity admission or a guaranteed lower bound. Identity rows are dated source bindings, not continuous security epochs.",
                  next_actions="Join 110 dated source bindings to consumed intervals, select receipt-backed canonical snapshots and adjudicate full original/amended chronology; independently resolve a bounded clock policy without redundant per-event downloads unless necessary.",
                  date_proposal=ref("work/after156_filing_date_anchor_diagnostic.json"),
                  clock_review=ref("work/after156_filing_clock_independent.json"))
    numerical = dict(ref("work/after156_filing_drift_kernel_review.json"),
                     status="synthetic_review_passed_actual_source_gates_pending",
                     real_strategy_registered=False, new_strategy_trials=0,
                     producer_tests=25, independent_tests=15)
    novelty = dict(ref("work/after156_filing_drift_design.json"), section="bounded_novelty",
                   status="bounded_prior_plan_comparison_not_exhaustive_novelty_proof")
    state_patch = {"next": NEXT, "next_step": NEXT, "next_direction_evidence": novelty,
                   "next_direction_feasibility": source, "next_experiment_proposal": proposal,
                   "next_numerical_preflight": numerical, "next_prospective_specification": proposal,
                   "next_research_preflight": numerical, "next_source_preflight": source,
                   "source_access_stop": stop, "last_goal_turn_classification": "progress",
                   "updated_utc": asof, "after156_research_checkpoint": CHECKPOINT}
    resume_patch = {"next_step": NEXT, "next_candidate_review": numerical,
                    "next_mechanism_deduplication": novelty, "next_mechanism_registered": False,
                    "next_research_preparation": {
                        "status": "filing_event_source_and_protocol_review_only", "registered": False,
                        "new_trials": 0, "novelty_established": False,
                        "prior_completed_experiment": ref(STUDY + "COMPLETION.json"),
                        "proposal": proposal},
                    "next_research_status": "after156_filing_research_unregistered_source_gates_pending",
                    "next_source_preparation": source, "next_source_resume_checkpoint": CHECKPOINT,
                    "resume_update_checkpoint": CHECKPOINT, "after156_research_checkpoint": CHECKPOINT,
                    "source_access_stop": stop, "previous_goal_turn_classification": "progress",
                    "last_goal_turn_classification": "progress", "updated_utc": asof}
    patches = {STATE: state_patch, RESUME: resume_patch}
    outputs = {name: encoded(field_patch(data[name], patch)) for name, patch in patches.items()}
    after_state = decoded(outputs[STATE])
    check(after_state["completed"] == state["completed"] and
          after_state["best_training_selected_candidate"] == state["best_training_selected_candidate"],
          "Completed/best records must be identical")
    checkpoint = {
        "schema_version": 1, "created_utc": asof, "status": "AFTER156_METADATA_HANDOFF",
        "completed_rounds": 156, "strategy_trials": 982, "previous_goal_turn_classification": "progress",
        "activity": {"new_trials": 0, "factor_calculations": 0, "price_values_read": 0,
                     "orders": 0, "returns": 0, "network_requests": 0, "observer_changes": 0},
        "completed_result": {"completion": ref(STUDY + "COMPLETION.json"),
                             "summary": ref(STUDY + "SUMMARY.json"), "primary": done["primary"],
                             "primary_cagr": done["primary_cagr"], "target_reached": False,
                             "selection_admitted": False, "protected_files": 21444,
                             "independent_execution_paths": 12, "monthly_orders_verified": 488,
                             "saved_score_diagnostic": ref("work/after156_score_margin_audit_v2.json"),
                             "diagnostic_scope": "Saved post-episode scores only; gaps are not pre-score margins or counterfactual selection thresholds. All observed holding decisions remain equal; no new dose or variant tested.",
                             "eligible_source_cells_early53_recent35_later34": [0, 37, 57],
                             "held_source_cells_early53_recent35_later34": [0, 1, 2]},
        "next_research": {"proposal": proposal, "sources": source, "synthetic_review": numerical,
                          "date_reference": ref("work/after156_filing_date_reference.json"),
                          "warmup_axis_only": ref("work/after156_filing_warmup.json"),
                          "registered": False, "source_gate_passed": False},
        "source_access_stop": stop, "input_hashes": hashes,
        "verification_scope": "Exact named evidence, receipt bindings and relevant producer/test code checked, including completed-score diagnostic source bytes and source-path review inputs. No repeat raw-source inventory audit, protected-member scan, price-array validation or new numerical research. Historical paused status inside immutable kernel review is preserved; current handoff records resumed research only.",
        "preserved_completed_sha256": sha(encoded(state["completed"])),
        "preserved_best_sha256": sha(encoded(state["best_training_selected_candidate"])),
        "updates": {name: {"before_sha256": hashes[name], "after_sha256": sha(outputs[name]),
                           "backup": "migration/after156_before_" + Path(name).stem + "_" + hashes[name] + ".json",
                           "before_fields": {k: {"present": k in data[name], "value": data[name].get(k)} for k in patch},
                           "patch": patch} for name, patch in patches.items()},
    }
    # Before returning, detect any concurrent change among every byte we consumed.
    for name, expected in hashes.items():
        check(sha(local(root, name).read_bytes()) == expected, "Concurrent update: " + name)
    return checkpoint, outputs, raw_inputs


def preserve(path, raw):
    if path.exists():
        check(path.read_bytes() == raw, "Conflicting immutable checkpoint: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_replace(path, raw, expected):
    check(sha(path.read_bytes()) == expected, "Concurrent state edit: " + str(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=".after156-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        check(sha(path.read_bytes()) == expected, "Concurrent state edit: " + str(path))
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink()


def commit(root, checkpoint, outputs, raw_inputs, expected):
    raw = encoded(checkpoint)
    check(sha(raw) == expected, "Preview digest mismatch; review a fresh preview")
    for name, value in checkpoint["input_hashes"].items():
        check(sha(local(root, name).read_bytes()) == value, "Changed preview input: " + name)
    for name, update in checkpoint["updates"].items():
        preserve(local(root, update["backup"]), raw_inputs[name])
    # Immutable write-ahead record plus raw backups allow explicit recovery if the
    # process stops between these two atomic document replacements. No rollback
    # overwrites concurrent work. The script never invents a completion receipt.
    preserve(local(root, CHECKPOINT), raw)
    for name, update in checkpoint["updates"].items():
        atomic_replace(local(root, name), outputs[name], update["before_sha256"])
    for name, update in checkpoint["updates"].items():
        check(sha(local(root, name).read_bytes()) == update["after_sha256"], "Post-write hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", required=True, help="Fixed timezone-aware preview timestamp")
    parser.add_argument("--apply-checkpoint-sha", help="Explicitly apply this reviewed checkpoint digest")
    parser.add_argument("--show-patches", action="store_true")
    args = parser.parse_args()
    check(not (ROOT / CHECKPOINT).exists(), "Checkpoint already exists; inspect its update hashes and backups, do not overwrite")
    if args.apply_checkpoint_sha:
        with (ROOT / "outputs/monthly_target70_research/.lifecycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checkpoint, outputs, raw_inputs = build(ROOT, args.asof)
            commit(ROOT, checkpoint, outputs, raw_inputs, args.apply_checkpoint_sha)
    else:
        checkpoint, outputs, raw_inputs = build(ROOT, args.asof)
    result = {"mode": "applied" if args.apply_checkpoint_sha else "read_only_preview",
              "checkpoint": CHECKPOINT, "checkpoint_sha256": sha(encoded(checkpoint)),
              "input_files_checked": len(checkpoint["input_hashes"]),
              "new_trials": 0, "completed_rounds": 156, "strategy_trials": 982,
              "updates": {name: {"before_sha256": value["before_sha256"],
                                  "after_sha256": value["after_sha256"],
                                  "fields": sorted(value["patch"])}
                          for name, value in checkpoint["updates"].items()}}
    if args.show_patches:
        result["proposed_patches"] = {name: value["patch"] for name, value in checkpoint["updates"].items()}
    print(encoded(result).decode(), end="")


if __name__ == "__main__":
    main()

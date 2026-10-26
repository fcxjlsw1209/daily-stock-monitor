"""Bind completed source-only registration review to the final synthetic receipt.

This script reads source text and validation metadata only. It never imports
the new producer modules or opens market arrays, account ledgers, or outcomes.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import coskew162_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "work/coskew162_registration_tests.json"
EXPECTED_RECEIPT = "e9aae5990bb4f017c2a34cf5ffd4daecac8618347fde11bf02ba4f48e47ebc25"
DESTINATION = ROOT / "work/coskew162_registration_review.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    item = Path(value)
    item = (item if item.is_absolute() else ROOT / item).resolve()
    require(item.is_relative_to(ROOT), "Review input must be inside local research root")
    require(not any(part in str(item) for part in ("OneDrive", "CloudStorage", "stock-observer-runtime")), "Forbidden input")
    return item


def digest(value):
    return hashlib.sha256(path(value).read_bytes()).hexdigest()


def read_pinned(value, expected):
    raw = path(value).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Input changed: " + str(value))
    return json.loads(raw)


def main():
    require(not DESTINATION.exists(), "Write-once registration review already exists")
    require(not (ROOT / "outputs/monthly_coskew162/PLAN.json").exists(), "Review must precede registration")
    independent.prepare.independent_imports()
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / "work" / name for name in (
        "coskew162_host.py", "coskew162_kernel.py", "coskew162_runner.py",
        "test_coskew162_kernel.py", "test_coskew162_runner.py", "test_coskew162_host.py",
        "coskew162_validate_code.py")]))
    implementation = {str(item): digest(item) for item in expected_paths}
    receipt = read_pinned(RECEIPT, EXPECTED_RECEIPT)
    require(receipt["passed"] is True and receipt["synthetic_only"] is True and
            receipt["real_features_orders_or_returns_computed"] is False, "Synthetic-only aggregate required")
    require(receipt["implementation_hashes"] == implementation, "Exact complete validation_paths closure required")
    suites = {"kernel": 20, "runner": 34, "host": 23, "independent": 40}
    require([row["suite"] for row in receipt["suites"]] == list(suites), "Four exact suites required")
    evidence = {str(RECEIPT): EXPECTED_RECEIPT}
    for row in receipt["suites"]:
        require(row["tests_run"] == suites[row["suite"]], "Final suite test count changed")
        item = read_pinned(row["receipt"]["path"], row["receipt"]["sha256"])
        require(item["suite"] == row["suite"] and item["tests_run"] == row["tests_run"], "Suite identity/count mismatch")
        require(item["passed"] is True and item["synthetic_only"] is True and item["code_unchanged"] is True,
                "Suite must pass with unchanged implementation")
        require(item["failures"] == item["errors"] == item["skipped"] == 0, "No failed or skipped tests")
        require(item["implementation_hashes"] == implementation, "Suite was run against different code")
        require(item["log"] == row["log"] and digest(row["log"]["path"]) == row["log"]["sha256"], "Suite log binding")
        evidence[str(path(row["receipt"]["path"]))] = row["receipt"]["sha256"]
        evidence[str(path(row["log"]["path"]))] = row["log"]["sha256"]
    require(receipt["tests_run"] == sum(suites.values()) == 117, "117 final tests required")
    source_paths = [ROOT / "work" / name for name in (
        "coskew162_source_probe.py", "coskew162_proposal.json", "coskew162_design_review.json")]
    source_pins = {str(item): digest(item) for item in source_paths}
    design = read_pinned(source_paths[2], source_pins[str(source_paths[2])])
    require(design["proposal_sha256"] == source_pins[str(source_paths[1])], "Current proposal binding")
    require(design["input_hashes"] == design["source_hashes"], "Source closure maps must agree")
    require(all(design["input_hashes"].get(str(item)) == source_pins[str(item)] for item in source_paths[:2]),
            "Current source probe/proposal must match reviewed hashes")
    require(design["passed"] is True and design["summary"]["market_proxy_present"] is True, "Source identity review required")
    require(all(type(design[key]) is int and design[key] == 0 for key in (
        "actual_features_computed", "actual_orders_computed", "actual_returns_computed")), "Source probe must be metadata only")
    report = dict(
        passed=True, schema="coskew162_registration_review_v1", round=162,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT / "work/coskew162_host.py")],
        registration_tests_sha256=EXPECTED_RECEIPT, implementation_hashes=implementation,
        tests_run=117, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        reviewer_scope="Independent host, registration and source-binding review; reviewer previously implemented the kernel and its synthetic tests.",
        checks={
            "complete_current_validation_paths_match_all_four_suites": True,
            "tests_bind_before_and_after_implementation_hashes": True,
            "registration_before_actual_features_orders_and_outcomes": True,
            "source_probe_decodes_only_price_dates_and_symbols": True,
            "current_market_proxy_source_field_consistent": True,
            "official_calendar_and_original_axes_bound": True,
            "first88_orders_and_return_endpoints_only_for_selection": True,
            "later34_diagnostics_follow_frozen_selection": True,
            "full_21444_manifest_required_before_completion": True,
            "original_history_preserved_and_completed_counts_162_1001": True,
            "new_best_requires_training_admission_and_verified_full_reference_gain": True,
            "completion_and_central_best_share_verified_expected_object": True,
            "write_once_artifacts_and_locked_atomic_state_commit": True,
        },
        resolved_findings=[
            "Independent source review now reads market_proxy_present, matching the probe and host.",
            "Completion now records best_after and commits the identical expected best to central state; completed-state validation permits only that verified object.",
        ],
        limitations=[
            "This approves registration mechanics and the reviewed fixed design, not actual numerical coverage, a profitable strategy, or achievement of the return target.",
            "Actual preparation, all 5978 cells/366 decisions/123 prefixes and the nine outcome paths still require their registered independent checks.",
            "The 21444-file archive check is enforced in completion code; this review did not read or rehash the protected archive.",
            "Historical price vintages and estimated historical pool remain conditional research inputs; later34 has been reused in prior development.",
            "The central best is a descriptive full-period reference among training-admitted candidates; it does not change this round's frozen first88 winner or promote observation accounts.",
        ],
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        producer_modules_imported=False, strategy_admitted=False, observer_promotion=False,
        review_generator=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
    )
    require(implementation == {str(item): digest(item) for item in expected_paths}, "Code changed during binding review")
    require(digest(RECEIPT) == EXPECTED_RECEIPT, "Aggregate changed during binding review")
    require(source_pins == {str(item): digest(item) for item in source_paths}, "Design changed during review")
    independent.prepare.independent_imports()
    with DESTINATION.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=117,
                         path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == "__main__":
    main()

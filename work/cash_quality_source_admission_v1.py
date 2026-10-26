"""Conditional research admission of already reviewed cash-flow source cells.

Preserves every raw event, gap and older rejection. No ratios, scores or returns.
The explicit policy and completed independent union review must be hash-bound.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
METHOD = "current_annual_cash_quality_v1"
DEFINITION = "consolidated_gaap_cfo_net_income_average_assets_v1"
REQUIRED = (("current", "operating_cash_flow"), ("current", "net_income"),
            ("current", "total_assets"), ("comparison", "total_assets"))
OUT = "outputs/monthly_cash_quality_source_admission_v1"
SELF = "work/cash_quality_source_admission_v1.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, path):
    p = (root / path).resolve()
    require(p.is_relative_to(root.resolve()), "Path outside research root")
    require("CloudStorage" not in str(p), "No cloud fallback")
    return p


def read_bound(root, ref):
    p = local(root, ref["path"])
    require(digest(p) == ref["sha256"], "Changed input: " + ref["path"])
    return json.loads(p.read_text())


def canonical_day(value):
    require(isinstance(value, str), "Date string required")
    result = date.fromisoformat(value)
    require(result.isoformat() == value, "Canonical date required")
    return result


def assess(event, sessions):
    """All reasons are source/metadata based; no outcome-dependent decisions."""
    reasons = []
    if event.get("prior_field_admitted") is not True or event.get("inherited_rejection_codes"):
        reasons.append("preserved_inherited_source_hold")
    if event.get("metadata_gaps"):
        reasons.append("unresolved_source_metadata")
    try:
        released = canonical_day(event["release_date"])
        available = canonical_day(event["available_from"])
        j = bisect_right(sessions, released.isoformat())
        require(j < len(sessions) and sessions[j] == available.isoformat(),
                "not_first_session_after_release")
        identity = event["identity"]
        require(identity["security_id"] == event["symbol"] and identity["issuer_id"],
                "missing_or_conflicting_identity")
        require(identity["identity_valid_from"] == available.isoformat(),
                "identity_availability_mismatch")
        periods = event["periods"]
        ends = {}
        for side in ("current", "comparison"):
            p = periods[side]
            start, end = canonical_day(p["period_start"]), canonical_day(p["period_end"])
            days = (end - start).days + 1
            require(330 <= days <= 400 and type(p["days_inclusive"]) is int
                    and p["days_inclusive"] == days, "annual_duration_invalid")
            ends[side] = (start, end)
        require(ends["comparison"][1] + timedelta(days=1) == ends["current"][0],
                "noncontiguous_annual_periods")
        require(ends["current"][1] <= released, "release_precedes_fiscal_end")
        require(identity["identity_valid_through"] ==
                (ends["current"][1] + timedelta(days=550)).isoformat(),
                "identity_expiry_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append("period_identity_or_clock:" + str(exc))
    currencies = []
    for side, field in REQUIRED:
        cell = event.get("fields", {}).get(side, {}).get(field)
        if cell is None:
            reasons.append("missing_required:" + side + "." + field)
            continue
        try:
            require(cell["source"] == event["source"], "different_release_source")
            require(cell["available_from"] == event["available_from"], "different_availability")
            require(cell["period"] == event["periods"][side], "different_period")
            require(cell["currency_literal"] == "$", "unrecognized_literal_currency")
            require(cell.get("currency_iso") in (None, "USD"), "conflicting_explicit_currency")
            currencies.append(cell.get("currency_iso"))
            value, scale = Decimal(cell["value_decimal"]), Decimal(cell["scale"])
            require(value.is_finite() and scale.is_finite() and scale > 0, "invalid_amount_scale")
            require(field != "total_assets" or value > 0, "nonpositive_assets")
            require(cell.get("fact_admitted") is False, "raw_admission_flag_changed")
            require(cell["locator"]["period_end"] == event["periods"][side]["period_end"],
                    "locator_period_mismatch")
            if field == "net_income":
                require(cell["locator"]["ni_scope"] == "indirect_method_consolidated_start",
                        "income_scope_not_consolidated_reconciliation_start")
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            reasons.append("invalid_required:" + side + "." + field + ":" + str(exc))
    return dict(admitted=not reasons, reasons=reasons, definition=DEFINITION)


def validate_policy(policy):
    require(policy.get("schema_version") == 1 and policy.get("method") == METHOD,
            "Explicit cash quality policy required")
    require(policy.get("definition") == DEFINITION, "Definition changed")
    require(policy.get("preserve_all_inherited_holds") is True, "Cannot lift legacy holds")
    require(policy.get("same_report_literal_currency_assumption") is True,
            "Currency assumption must remain explicit")
    require(policy.get("use_comparison_flows") is False, "Comparison flow use forbidden")
    require(policy.get("annual_days") == [330, 400] and policy.get("max_age_days") == 550,
            "Time contract changed")
    require(policy.get("financial_ratios_permitted") is False
            and policy.get("strategy_registration_permitted") is False,
            "Admission is source-only")
    require(policy.get("expected_events") == 44, "Full fixed source census required")


def validate_accounting(accounting, union, reference):
    require(accounting.get("stage") == "source_only_field_scoped_accounting_review"
            and accounting.get("source_only") is True, "Wrong accounting review stage")
    records = accounting.get("events")
    require(isinstance(records, list), "Accounting events required")
    indexed = {e["event_id"]: e for e in records}
    require(len(indexed) == len(records) == len(union["events"]), "Accounting census incomplete")
    for event in union["events"]:
        provenance = event["union_provenance"]
        require(provenance["accounting_review_reference"] == reference,
                "Accounting review differs from independently checked union")
        expected = indexed.get(event["event_id"])
        require(expected is not None and provenance["accounting_event_review"] == expected,
                "Accounting event differs from independently checked union")
        require(expected["inherited_rejection_codes"] == event["inherited_rejection_codes"]
                and expected["inherited_rejections_unchanged"] is True,
                "Accounting review changed inherited rejection")


def build(root, policy_path, policy_sha, verify=False):
    policy = read_bound(root, dict(path=policy_path, sha256=policy_sha))
    validate_policy(policy)
    require(policy["runtime"] == sys.version, "Runtime changed")
    require(digest(local(root, SELF)) == policy["implementation_sha256"], "Implementation changed")
    refs = policy["inputs"]
    require(set(refs) == {"union_plan", "union_manifest", "union_data", "union_independent",
                          "accounting", "sessions", "concept", "tests"},
            "Source admission closure required")
    inputs = {k: read_bound(root, v) for k, v in refs.items()}
    union, plan, manifest = inputs["union_data"], inputs["union_plan"], inputs["union_manifest"]
    review = inputs["union_independent"]
    require(union["plan_sha256"] == refs["union_plan"]["sha256"] ==
            manifest["plan_sha256"], "Union plan mismatch")
    require(manifest["output_hashes"]["RAW_FACTS.json"] == refs["union_data"]["sha256"],
            "Union data mismatch")
    require(review.get("passed") is True, "Completed independent union review required")
    pins = dict(review.get("input_hashes", {}), **review.get("source_hashes", {}))
    require(all(pins.get(refs[k]["path"]) == refs[k]["sha256"]
                for k in ("union_plan", "union_manifest", "union_data")),
            "Independent review does not bind completed union")
    for group in ("source_hashes", "source_code_hashes"):
        for path, sha in plan[group].items():
            require(digest(local(root, path)) == sha, "Changed union provenance: " + path)
    require(inputs["tests"].get("passed") is True
            and inputs["tests"]["source_sha256"] == policy["implementation_sha256"],
            "Current source-admission synthetic tests required")
    validate_accounting(inputs["accounting"], union, refs["accounting"])
    accounting = {e["event_id"]: e for e in inputs["accounting"]["events"]}
    require(len(accounting) == 44, "Accounting census incomplete")
    events = union["events"]
    require(len(events) == len({e["event_id"] for e in events}) == 44
            and set(accounting) == {e["event_id"] for e in events}, "Event census mismatch")
    sessions = inputs["sessions"]["sessions"]
    require(sessions == sorted(set(sessions)), "Invalid session calendar")
    for day in sessions:
        canonical_day(day)
    rows = []
    for original in events:
        require(all(original.get(k) is False for k in
                    ("fact_admitted", "model_admitted", "strategy_admitted")),
                "Raw union admission changed")
        row = deepcopy(original)
        require("cash_quality_admission" not in row, "Existing admission cannot be overwritten")
        row["cash_quality_admission"] = assess(row, sessions)
        require({k: v for k, v in row.items() if k != "cash_quality_admission"} == original,
                "Original source event changed")
        rows.append(row)
    dataset = dict(schema_version=1, method=METHOD, events=rows,
                   policy_sha256=policy_sha, conditional_research_only=True)
    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n").encode()
    data_sha = hashlib.sha256(encoded(dataset)).hexdigest()
    admitted = [e["event_id"] for e in rows if e["cash_quality_admission"]["admitted"]]
    receipt = dict(schema_version=1, stage="conditional_cash_quality_source_admission",
        method=METHOD, passed=True, policy_sha256=policy_sha, data_sha256=data_sha,
        events=44, admitted_events=len(admitted), held_events=44-len(admitted),
        admitted_event_ids=admitted,
        holds=[dict(event_id=e["event_id"], **e["cash_quality_admission"])
               for e in rows if not e["cash_quality_admission"]["admitted"]],
        source_hashes={v["path"]: v["sha256"] for v in refs.values()},
        original_raw_events_preserved=True, original_rejections_preserved=True,
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False, currency_iso_not_newly_certified=True,
        financial_ratios=0, factor_calculations=0, portfolio_calculations=0,
        return_calculations=0, network_requests=0, observer_changes=0)
    for name, value in (("DATASET.json", dataset), ("ADMISSION.json", receipt)):
        path = local(root, OUT + "/" + name)
        if verify:
            require(path.read_bytes() == encoded(value), "Nonreproducible admission: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                f.write(encoded(value))
    return {k: receipt[k] for k in ("passed", "events", "admitted_events", "held_events", "data_sha256")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--policy-sha", required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    print(json.dumps(build(ROOT, args.policy, args.policy_sha, args.verify), sort_keys=True))

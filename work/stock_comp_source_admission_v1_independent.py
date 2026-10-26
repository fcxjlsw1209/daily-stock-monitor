"""Independent source-package acceptance; no producer import or financial ratio.

Reconstruct the three-cell patch, retain all original holds, and verify each
conditional field decision against the source and independent cell receipts.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = "outputs/monthly_stock_comp_source_admission_v2"
SELF = "work/stock_comp_source_admission_v1_independent.py"
TEST = "work/test_stock_comp_source_admission_v1_independent.py"
METHOD = "current_annual_stock_compensation_adjusted_cash_flow_v1"
DEFINITION = "current_annual_reported_stock_compensation_addback_v1"
SOURCE_PLAN = "ef563d83e864c3e592c75f232dccc1327aed9b1fc95b5a2115c703ad51df45ba"
SOURCE_MANIFEST = "5e25899f13b31bc631d1ef5bebc3d5f76e0a14bbb5866bd41ff104b288736f31"
OLD_DATA = "3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5"
OLD_ADMISSION = "e9d2bf123cdc351fecf1c794fd3a0e929b1d1a6307232d094f402d29b7463494"
PATCH_IDS = {"AAPL:FY2021:2021-10-28", "AAPL:FY2022:2022-10-27", "AAPL:FY2023:2023-11-02"}
META = ("event_id", "symbol", "fiscal_year", "release_date", "available_from",
        "identity", "periods", "source", "inherited_rejection_codes", "cash_quality_admission")
RAW_FLAGS = ("fact_admitted", "model_admitted", "strategy_admitted")
CERTS = ("first_public_version_verified", "complete_correction_history_verified", "certified_daily_security_master")
KNOWN_FLAGS = {"new_fields_independent_source_review_required",
               "cashflow_accounting_change_context_unreviewed",
               "negative_stock_compensation_addback_requires_review"}
LABEL = re.compile(r"^(?:stock|share)[\s\-\u2010-\u2015]+based\s+compensation(?:\s+expense)?$", re.I)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def index(events):
    require(type(events) is list, "Event list required")
    result = {e["event_id"]: e for e in events}
    require(len(result) == len(events), "Duplicate event")
    return result


def decimal(value):
    require(type(value) is str and re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value),
            "Explicit finite decimal string required")
    result = Decimal(value)
    require(result.is_finite(), "Finite decimal required")
    return result


def policy_contract(policy):
    expected = dict(schema_version=1, method=METHOD, purpose=METHOD, definition=DEFINITION,
                    expected_events=44, same_current_report=True, latest_event_before_validity=True,
                    invalid_latest_blocks_older_fallback=True, preserve_all_13_original_holds=True,
                    annual_days=[330, 400], max_age_days=550, same_report_literal_currency_assumption=True,
                    financial_ratios_permitted=False, strategy_registration_permitted=False,
                    conditional_research_only=True, first_public_version_verified=False,
                    complete_correction_history_verified=False, certified_daily_security_master=False,
                    iso_currency_certified=False)
    require(all(policy.get(key) == value and type(policy.get(key)) is type(value)
                for key, value in expected.items()), "Conditional source policy contract mismatch")
    for key in ("accounting", "accounting_changes"):
        require(type(policy.get(key)) is str and bool(policy[key]), "Explicit accounting scope required")


def patched_candidates(raw_events, patch_events):
    result = deepcopy(index(raw_events))
    patches = index(patch_events)
    require(len(result) == 44 and set(patches) == PATCH_IDS, "Exact 44-event / three-patch census")
    for eid, patch in patches.items():
        require(eid in result and result[eid]["symbol"] == "AAPL", "Only declared Apple targets")
        original = result[eid]["stock_compensation"]
        require(original["value"] is None and original["gaps"] == [
            dict(reason="invalid_or_ambiguous_current_monetary_cell", error_type="ValueError")],
            "Patch may replace only declared failed cells")
        require(set(patch) == {"event_id", "result"}, "Patch metadata injection")
        result[eid]["stock_compensation"] = deepcopy(patch["result"])
    return result


def decision(original, candidate, verified_ids):
    """Independently determine only field eligibility; never calculate a ratio."""
    require(all(key in candidate and candidate[key] == original[key] for key in META), "Original metadata changed")
    require(all(candidate.get(key) is False for key in RAW_FLAGS), "Original raw flags changed")
    component = candidate["stock_compensation"]
    require(component["field"] == "stock_compensation_addback" and component["candidate_only"] is True,
            "Explicit unadmitted raw field required")
    require(all(component.get(key) is False for key in RAW_FLAGS + CERTS), "Raw component certification changed")
    reasons = [] if original["cash_quality_admission"]["admitted"] is True else ["preserved_original_cash_quality_hold"]
    value = component["value"]
    if value is None:
        require(type(component["gaps"]) is list and bool(component["gaps"]), "Missing value has no explicit gap")
        return reasons + ["stock_compensation_value_missing"]
    require(component["gaps"] == [], "Observed value conflicts with gap")
    require(value["source"] == original["source"] and value["source_sha256"] == original["source"]["sha256"],
            "Current source mismatch")
    require(value["period"] == original["periods"]["current"] and value["available_from"] == original["available_from"],
            "Current report period or availability mismatch")
    require(value["fact_admitted"] is False and value["currency_iso"] is None, "Premature field certification")
    amount, scale = decimal(value["value_decimal"]), decimal(value["scale"])
    require(scale > 0, "Positive scale required")
    loc = value["locator"]
    require(loc["field_scope"] == "explicit_component_of_consolidated_indirect_operating_reconciliation" and
            LABEL.fullmatch(loc["row_label"]), "Standalone consolidated cash-flow component required")
    require(loc["period_end"] == original["periods"]["current"]["period_end"] and
            decimal(loc["scale"]) == scale and loc["currency_literal"] == value["currency_literal"] == "$",
            "Exact current period and same-report units required")
    cfo = original["fields"]["current"]["operating_cash_flow"]
    if cfo is not None:
        cloc = cfo["locator"]
        require(decimal(cfo["scale"]) == scale and cfo["currency_literal"] == "$", "CFO units differ")
        require(loc["format"] == cloc["format"] and loc["statement_title"] == cloc["statement_title"],
                "CFO statement differs")
        require(loc.get("statement_segment") == cloc.get("statement_segment"), "CFO statement segment differs")
        keys = ("page_1based", "financial_column_index_0based") if loc["format"] == "pdf" else (
            "table_index_zero_based", "logical_columns")
        require(loc["format"] in ("html", "pdf") and all(loc[key] == cloc[key] for key in keys),
                "CFO annual column differs")
    codes = [flag["code"] for flag in component["accounting_review_flags"]]
    require(all(code in KNOWN_FLAGS for code in codes), "New accounting issue lacks a policy disposition")
    require(amount >= 0 or "negative_stock_compensation_addback_requires_review" in codes,
            "Negative component lost its explicit review flag")
    if original["event_id"] not in verified_ids:
        reasons.append("independent_source_value_review_missing")
    return reasons


def verify_output_events(original_events, candidates, output_events, verified_ids):
    prior, actual = index(original_events), index(output_events)
    require(len(prior) == len(actual) == 44 and list(prior) == list(candidates) == list(actual),
            "Complete ordered 44-event population required")
    admitted, holds = [], []
    for eid, original in prior.items():
        result = actual[eid]
        require(set(result) == set(candidates[eid]) | {"stock_compensation_admission"},
                "New dataset fields differ from candidate plus admission")
        require({key: val for key, val in result.items() if key != "stock_compensation_admission"} == candidates[eid],
                "Source candidate or accounting flags changed")
        reasons = decision(original, candidates[eid], verified_ids)
        expected = dict(admitted=not reasons, reasons=reasons, definition=DEFINITION)
        require(result["stock_compensation_admission"] == expected, "Source eligibility decision differs")
        (holds if reasons else admitted).append(dict(event_id=eid, **expected) if reasons else eid)
    require(sum(e["cash_quality_admission"]["admitted"] is not True for e in prior.values()) == 13,
            "Original thirteen holds changed")
    require(len(admitted) <= 31 and all(prior[eid]["cash_quality_admission"]["admitted"] is True for eid in admitted),
            "New gate widened old field availability")
    return admitted, holds


def bound_path(root, name, expected=None, external_pins=None):
    """Permit external PDF libraries only when pinned by the fixed source PLAN."""
    root = Path(root).resolve()
    text = str(name)
    require(text and ".." not in Path(text).parts and
            not any(part in text.lower() for part in ("onedrive", "cloudstorage")),
            "Bound local path required")
    path = Path(text) if Path(text).is_absolute() else root / text
    require(path.resolve() == path, "Input symlink or path alias forbidden")
    if root in path.parents:
        return path, str(path.relative_to(root))
    runtime = Path("/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages")
    require(runtime in path.parents and path.relative_to(runtime).parts[0] in
            ("pypdf", "pypdfium2", "pypdfium2_raw"), "Input outside bound local runtime")
    require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected) and
            (external_pins or {}).get(str(path)) == expected, "External runtime file lacks fixed PLAN pin")
    return path, str(path)


def verify(root, policy_sha):
    root = Path(root).resolve()
    seen = {}
    external_pins = {}

    def bytes_at(path, expected=None):
        path, key = bound_path(root, path, expected, external_pins)
        raw = path.read_bytes()
        sha = digest(raw)
        require(expected is None or sha == expected, "Input digest mismatch: " + key)
        require(key not in seen or seen[key] == sha, "Input changed during audit")
        seen[key] = sha
        return raw

    def read(path, expected=None):
        return json.loads(bytes_at(path, expected))

    def closure(hashes):
        for path, sha in hashes.items():
            bytes_at(path, sha)

    require(type(policy_sha) is str and re.fullmatch("[0-9a-f]{64}", policy_sha), "Explicit source policy SHA required")
    policy = read(OUT + "/POLICY.json", policy_sha)
    policy_contract(policy)
    source_ref = policy["inputs"]["source_plan"]
    require(source_ref["sha256"] == SOURCE_PLAN, "Wrong fixed source PLAN")
    fixed_source_plan = read(source_ref["path"], SOURCE_PLAN)
    external_pins.update(fixed_source_plan["pdf_runtime_hashes"])
    receipt = read(OUT + "/ADMISSION.json")
    closure(receipt["input_hashes"])
    closure(policy["source_code_hashes"])
    data = read(OUT + "/DATASET.json", receipt["data_sha256"])
    require(receipt["passed"] is True and receipt["stage"] == "conditional_stock_compensation_source_admission",
            "Wrong admission receipt")
    require(data["method"] == METHOD and data["conditional_research_only"] is True and
            data["policy_sha256"] == receipt["policy_sha256"] == policy_sha, "Data/policy binding")
    require(all(receipt[key] == 0 and type(receipt[key]) is int for key in
                ("financial_ratios", "factor_calculations", "model_fits", "return_calculations", "observer_changes")),
            "Source-only counter changed")
    require(all(receipt[key] is False for key in CERTS), "Unsupported certification")
    require(receipt["source_values_independently_reviewed"] is True and
            receipt["accounting_flags_preserved"] is True and
            receipt["accounting_flag_disposition"] == policy["accounting_changes"], "Accounting disposition mismatch")
    refs = policy["inputs"]
    obj = {key: read(ref["path"], ref["sha256"]) for key, ref in refs.items()}
    for ref in refs.values():
        require(receipt["input_hashes"].get(ref["path"]) == ref["sha256"], "Admission omits frozen input")
    require(refs["source_plan"]["sha256"] == SOURCE_PLAN and refs["source_manifest"]["sha256"] == SOURCE_MANIFEST and
            refs["old_dataset"]["sha256"] == OLD_DATA and refs["old_admission"]["sha256"] == OLD_ADMISSION,
            "Wrong inherited dataset or source baseline")
    require(obj["source_manifest"]["plan_sha256"] == SOURCE_PLAN and
            obj["source_manifest"]["output_hashes"]["RAW_FACTS.json"] == refs["raw"]["sha256"], "Raw source manifest link")
    for label in ("source_plan", "patch_plan"):
        closure(obj[label]["source_hashes"])
        closure(obj[label]["source_code_hashes"])
    for label in ("html_review", "old_review"):
        require(obj[label]["passed"] is True, "Missing independent source review")
        closure(obj[label]["input_hashes"])
    require(obj["html_review"]["plan_sha256"] == SOURCE_PLAN, "Wrong HTML source plan")
    for key in ("raw", "source_plan"):
        require(obj["html_review"]["input_hashes"].get(refs[key]["path"]) == refs[key]["sha256"],
                "HTML review refers to different input")
    psha = refs["patch_plan"]["sha256"]
    require(obj["patch_plan"]["source_plan_sha256"] == SOURCE_PLAN and
            obj["patch_plan"]["source_manifest_sha256"] == SOURCE_MANIFEST, "Wrong patch source baseline")
    for key in ("patch", "patch_audit", "patch_manifest"):
        require(obj[key]["plan_sha256"] == psha, "Patch PLAN link")
    require(set(obj["patch_manifest"]["output_hashes"]) == {"PATCH.json", "AUDIT.json"} and
            obj["patch_manifest"]["output_hashes"]["PATCH.json"] == refs["patch"]["sha256"] and
            obj["patch_manifest"]["output_hashes"]["AUDIT.json"] == refs["patch_audit"]["sha256"] and
            obj["patch_audit"]["patch_sha256"] == refs["patch"]["sha256"] and obj["patch_audit"]["passed"] is True,
            "Patch manifest or audit link")
    require(obj["patch"]["source_plan_sha256"] == SOURCE_PLAN and
            obj["patch"]["source_manifest_sha256"] == SOURCE_MANIFEST, "Patched object source link")

    original = index(obj["old_dataset"]["events"])
    candidates = patched_candidates(obj["raw"]["events"], obj["patch"]["events"])
    html_rows = index(obj["html_review"]["event_reviews"])
    require(set(html_rows) == set(original), "HTML review must retain all events")
    verified = set()
    for eid, row in html_rows.items():
        if row["status"] == "HTML_physical_cell_verified":
            require(row["cell"]["verified"] is True and original[eid]["symbol"] != "AAPL" and
                    row["cell"]["source_sha256"] == original[eid]["source"]["sha256"],
                    "Invalid HTML physical source proof")
            verified.add(eid)
    visual = obj["pdf_visual"]
    pdf_rows = index(visual["events"])
    require(visual["passed_visual_identification"] is True and visual["pages_reviewed"] == 10 and
            set(pdf_rows) == {eid for eid, event in original.items() if event["symbol"] == "AAPL"},
            "All ten exact Apple visual proofs required")
    for eid, row in pdf_rows.items():
        value = candidates[eid]["stock_compensation"]["value"]
        require(value is not None and value["value_decimal"] == row["value_decimal"] and
                value["source"] == row["source"] and value["scale"] == row["scale"] and
                value["period"]["period_end"] == row["current_period_end"], "PDF visual/value mismatch")
        for label in ("source", "rendered_image"):
            bytes_at(row[label]["path"], row[label]["sha256"])
        verified.add(eid)
    admitted, holds = verify_output_events(obj["old_dataset"]["events"], candidates, data["events"], verified)
    require(receipt["events"] == 44 and receipt["admitted_events"] == len(admitted) and
            receipt["held_events"] == len(holds) and receipt["admitted_event_ids"] == admitted and receipt["holds"] == holds,
            "Admission counts or exact event decisions changed")
    for path in (SELF, TEST):
        bytes_at(path)
    return dict(passed=True, stage="independent_conditional_stock_compensation_source_review",
                events=44, admitted_events=len(admitted), held_events=len(holds), inherited_holds=13,
                present_source_values=sum(e["stock_compensation"]["value"] is not None for e in candidates.values()),
                admitted_event_ids=admitted, holds=holds, source_only=True, input_hashes=seen,
                policy_sha256=policy_sha, all_source_candidates_preserved=True,
                original_rejections_preserved=True, independent_decisions_match=True,
                three_patch_cells_bound_to_original_sources=True,
                all_present_values_match_independent_physical_or_visual_proof=True,
                financial_ratios=0, factor_calculations=0, return_calculations=0,
                producer_admission_code_imported=False, **{key: False for key in CERTS})


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--policy-sha", required=True)
    parsed = args.parse_args()
    result = verify(ROOT, parsed.policy_sha)
    result["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / OUT / "INDEPENDENT_REVIEW.json"
    with target.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps(dict(passed=True, admitted=result["admitted_events"], held=result["held_events"],
                          receipt_sha256=digest(target.read_bytes()))))


if __name__ == "__main__":
    main()

"""Conditional source-use gate for reported SBC; never computes financial ratios."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys

import stock_comp_source_v1_runner as src
import cash_quality_source_admission_v1_independent as old

ROOT = Path(__file__).resolve().parents[1]
OUT = "outputs/monthly_stock_comp_source_admission_v2"
PATCH = "outputs/monthly_stock_comp_pdf_repair_v2"
SELF = "work/stock_comp_source_admission_v2.py"
TEST = "work/test_stock_comp_source_admission_v2.py"
METHOD = "current_annual_stock_compensation_adjusted_cash_flow_v1"
DEFINITION = "current_annual_reported_stock_compensation_addback_v1"
SOURCE_PLAN = "ef563d83e864c3e592c75f232dccc1327aed9b1fc95b5a2115c703ad51df45ba"
NEED_REPAIR = {"AAPL:FY2021:2021-10-28", "AAPL:FY2022:2022-10-27",
               "AAPL:FY2023:2023-11-02"}
CERTS = ("first_public_version_verified", "complete_correction_history_verified",
         "certified_daily_security_master")
FLAGS = ("fact_admitted", "model_admitted", "strategy_admitted")
require = src.require


def policy_fields():
    return dict(
        schema_version=1, method=METHOD, purpose=METHOD, definition=DEFINITION,
        expected_events=44, same_current_report=True,
        latest_event_before_validity=True, invalid_latest_blocks_older_fallback=True,
        preserve_all_13_original_holds=True, annual_days=[330, 400], max_age_days=550,
        timing="First strictly later pinned session following the current release date; no backfill.",
        field="Explicit standalone current annual consolidated indirect operating-cash-flow stock/share-based compensation addback.",
        excluded="Quarterly and non-GAAP adjustments, mixed/net-tax lines, inferred zeros, issuance and buyback cash flows.",
        accounting="Retain signed reported addback. CFO minus this noncash addback is a shareholder economic-cost research proxy, not reported GAAP CFO, actual cash spending, share dilution or free cash flow.",
        accounting_changes="Current release values on their reported basis, including adoption/restatement; no organic or constant-accounting-basis claim. Same current-release CFO and SBC are required. Original comparison-asset gate remains intact.",
        scope="Only a subset of the original 31 conditionally admitted events; all 44 events and 13 original holds are retained.",
        same_report_literal_currency_assumption=True, financial_ratios_permitted=False,
        strategy_registration_permitted=False, conditional_research_only=True,
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False, iso_currency_certified=False,
    )


def check_cell(original, candidate):
    """Hard fail altered provenance; return reasons for genuine source gaps."""
    require(all(candidate.get(k) == original.get(k) for k in src.METADATA),
            "Original event metadata changed")
    require(all(candidate.get(k) is False for k in FLAGS), "Raw event flag changed")
    result = candidate["stock_compensation"]
    require(result["field"] == "stock_compensation_addback" and
            result["candidate_only"] is True, "Explicit raw component required")
    require(all(result.get(k) is False for k in FLAGS + CERTS),
            "Raw component certification changed")
    reasons = []
    if original["cash_quality_admission"]["admitted"] is not True:
        reasons.append("preserved_original_cash_quality_hold")
    value = result.get("value")
    if value is None:
        require(bool(result["gaps"]), "Unexplained missing source value")
        reasons.append("stock_compensation_value_missing")
        return reasons
    require(result["gaps"] == [], "Value coexists with ambiguity")
    require(value["source"] == original["source"] and
            value["source_sha256"] == original["source"]["sha256"],
            "Current report source mismatch")
    require(value["period"] == original["periods"]["current"] and
            value["available_from"] == original["available_from"], "Current period/clock mismatch")
    require(value["fact_admitted"] is False and value["currency_iso"] is None,
            "Raw value certification changed")
    require(type(value["value_decimal"]) is str and type(value["scale"]) is str,
            "Exact source decimal strings required")
    amount, scale = Decimal(value["value_decimal"]), Decimal(value["scale"])
    require(amount.is_finite() and scale.is_finite() and scale > 0, "Invalid monetary literal")
    loc = value["locator"]
    require(loc["field_scope"] == "explicit_component_of_consolidated_indirect_operating_reconciliation",
            "Wrong component scope")
    require(loc["period_end"] == original["periods"]["current"]["period_end"] and
            Decimal(loc["scale"]) == scale and
            value["currency_literal"] == loc["currency_literal"] == "$",
            "Locator unit or period disagreement")
    # Some preserved rejected events have no old CFO cell; never promote them.
    cfo = original["fields"]["current"]["operating_cash_flow"]
    if cfo is not None:
        require(Decimal(cfo["scale"]) == scale and cfo["currency_literal"] == "$",
                "CFO and SBC units must match")
        cashloc = cfo["locator"]
        require(loc["format"] == cashloc["format"] and
                loc["statement_title"] == cashloc["statement_title"], "Same statement required")
        keys = (("page_1based", "financial_column_index_0based") if loc["format"] == "pdf"
                else ("table_index_zero_based", "logical_columns"))
        require(all(loc[k] == cashloc[k] for k in keys), "Same annual column required")
        require(loc.get("statement_segment") == cashloc.get("statement_segment"),
                "Same statement segment required")
    allowed = {"new_fields_independent_source_review_required",
               "cashflow_accounting_change_context_unreviewed",
               "negative_stock_compensation_addback_requires_review"}
    require(all(x.get("code") in allowed for x in result["accounting_review_flags"]),
            "Unreviewed new accounting flag")
    return reasons


def admitted_event(original, candidate, verified_ids):
    reasons = check_cell(original, candidate)
    if candidate["stock_compensation"]["value"] is not None and candidate["event_id"] not in verified_ids:
        reasons.append("independent_source_value_review_missing")
    result = deepcopy(candidate)
    result["stock_compensation_admission"] = dict(
        admitted=not reasons, reasons=reasons, definition=DEFINITION)
    return result


def code_hashes(root):
    return dict(src.code_hashes(root), **{
        p: src.base.sha(root, p) for p in (SELF, TEST,
            "work/cash_quality_source_admission_v1_independent.py",
            "work/cashflow_source_union_v1_independent.py")})


def freeze(root, patch_plan_sha, html_review_path):
    """Freeze the full evidence set before granting conditional field use."""
    src.load_plan(root, SOURCE_PLAN)
    tests = src.base.read(root, OUT + "/TESTS.json")
    require(tests["passed"] is True and tests["synthetic_only"] is True
            and tests["source_code_hashes"] == code_hashes(root), "Current passing synthetic tests required")
    refs = {
        "raw": src.OUT + "/RAW_FACTS.json", "source_plan": src.OUT + "/PLAN.json",
        "source_manifest": src.OUT + "/MANIFEST.json", "html_review": html_review_path,
        "pdf_visual": src.OUT + "/PDF_VISUAL_REVIEW.json",
        "patch_plan": PATCH + "/PLAN.json", "patch": PATCH + "/PATCH.json",
        "patch_audit": PATCH + "/AUDIT.json", "patch_manifest": PATCH + "/MANIFEST.json",
        "old_dataset": src.ADMISSION + "/DATASET.json",
        "old_admission": src.ADMISSION + "/ADMISSION.json",
        "old_review": src.ADMISSION + "/INDEPENDENT_REVIEW.json",
        "sessions": src.base.SESSIONS, "tests": OUT + "/TESTS.json",
    }
    src.base.read(root, refs["patch_plan"], patch_plan_sha)
    inputs = {k: dict(path=p, sha256=src.base.sha(root, p)) for k, p in refs.items()}
    policy = dict(policy_fields(), created_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=sys.version, inputs=inputs, source_code_hashes=code_hashes(root))
    src.base.publish(root, OUT + "/POLICY.json", policy)
    return dict(policy_sha256=src.base.sha(root, OUT + "/POLICY.json"))


def assemble(root, policy_sha):
    evidence = old.Evidence(root)
    policy = src.base.read(root, OUT + "/POLICY.json", policy_sha)
    require(all(policy.get(k) == v for k, v in policy_fields().items())
            and policy["runtime"] == sys.version, "Policy/runtime changed")
    require(policy["source_code_hashes"] == code_hashes(root), "Admission code changed")
    hashes = dict(policy["source_code_hashes"])
    hashes[OUT + "/POLICY.json"] = policy_sha
    def read_ref(ref):
        value = src.base.read(root, ref["path"], ref["sha256"])
        hashes[ref["path"]] = ref["sha256"]
        return value
    obj = {k: read_ref(ref) for k, ref in policy["inputs"].items()}
    require(policy["inputs"]["source_plan"]["sha256"] == SOURCE_PLAN, "Wrong original source plan")
    plan = src.load_plan(root, SOURCE_PLAN)
    for group in (plan["source_hashes"], plan["source_code_hashes"],
                  obj["old_review"]["input_hashes"], obj["html_review"]["input_hashes"]):
        evidence.pins(group)
        hashes.update(group)
    require(obj["old_review"]["passed"] is True and
            obj["old_review"]["admitted_events"] == 31 and
            obj["old_review"]["held_events"] == 13, "Original source review required")
    require(obj["html_review"]["passed"] is True and
            obj["html_review"]["plan_sha256"] == SOURCE_PLAN, "Independent HTML review required")
    for ref_key in ("raw", "source_plan"):
        ref = policy["inputs"][ref_key]
        require(obj["html_review"]["input_hashes"][ref["path"]] == ref["sha256"],
                "Independent HTML receipt input mismatch")
    require(obj["source_manifest"]["plan_sha256"] == SOURCE_PLAN and
            obj["source_manifest"]["output_hashes"]["RAW_FACTS.json"]
            == policy["inputs"]["raw"]["sha256"], "Original source output links")
    require(obj["pdf_visual"]["passed_visual_identification"] is True and
            obj["pdf_visual"]["pages_reviewed"] == 10, "Full PDF visual review required")
    original = src.indexed(obj["old_dataset"]["events"])
    candidate = src.indexed(obj["raw"]["events"])
    require(len(candidate) == 44 and list(candidate) == list(original), "Exact ordered 44-event census")
    require(obj["patch_audit"]["passed"] is True, "Passing source patch audit required")
    patch_sha = policy["inputs"]["patch_plan"]["sha256"]
    for key in ("patch_manifest", "patch_audit", "patch"):
        require(obj[key]["plan_sha256"] == patch_sha, "Patch plan binding mismatch")
    for ref_key, name in (("patch", "PATCH.json"), ("patch_audit", "AUDIT.json")):
        require(obj["patch_manifest"]["output_hashes"][name] == policy["inputs"][ref_key]["sha256"],
                "Patch manifest output binding mismatch")
    patch_rows = obj["patch"]["events"]
    patch_map = src.indexed(patch_rows)
    require(set(patch_map) == NEED_REPAIR, "Exact declared three-event repair")
    for eid, row in patch_map.items():
        require(candidate[eid]["stock_compensation"]["value"] is None, "Patch would overwrite an existing value")
        candidate[eid]["stock_compensation"] = deepcopy(row["result"])
    visual = src.indexed(obj["pdf_visual"]["events"])
    require(len(visual) == 10 and set(visual) ==
            {e["event_id"] for e in original.values() if e["symbol"] == "AAPL"},
            "Exactly ten original AAPL visual reviews required")
    verified_ids = {row["event_id"] for row in obj["html_review"]["event_reviews"]
                    if row["status"] == "HTML_physical_cell_verified"}
    for row in obj["html_review"]["event_reviews"]:
        if row["event_id"] in verified_ids:
            require(row["cell"]["verified"] is True and
                    row["cell"]["source_sha256"] == original[row["event_id"]]["source"]["sha256"],
                    "HTML verified source mismatch")
    for eid, row in visual.items():
        value = candidate[eid]["stock_compensation"]["value"]
        require(value is not None and
                value["value_decimal"] == row["value_decimal"] and
                value["scale"] == row["scale"] and
                value["source"] == row["source"] and
                value["period"]["period_end"] == row["current_period_end"],
                "Patched/existing PDF value differs from independent visual reading")
        hashes[row["source"]["path"]] = row["source"]["sha256"]
        hashes[row["rendered_image"]["path"]] = row["rendered_image"]["sha256"]
        verified_ids.add(eid)
    events = []
    for eid, prior_event in original.items():
        old.audit_event(prior_event, obj["sessions"]["sessions"])
        events.append(admitted_event(prior_event, candidate[eid], verified_ids))
    require(sum(not x["cash_quality_admission"]["admitted"] for x in events) == 13,
            "Original holds changed")
    require(not any(x["stock_compensation_admission"]["admitted"]
                    and not x["cash_quality_admission"]["admitted"] for x in events), "Old hold promoted")
    # Preserve complete patch provenance; recognized manifest hashes must match the files.
    for name, sha in obj["patch_manifest"]["output_hashes"].items():
        p = PATCH + "/" + name
        evidence.pins({p: sha})
        hashes[p] = sha
    for key in ("source_hashes", "source_code_hashes"):
        group = obj["patch_plan"][key]
        evidence.pins(group)
        hashes.update(group)
    evidence.pins(hashes)
    evidence.stable()
    return policy, events, hashes


def build(root, policy_sha):
    policy, events, hashes = assemble(root, policy_sha)
    data = dict(schema_version=1, method=METHOD, policy_sha256=policy_sha,
                conditional_research_only=True, events=events)
    selected = [e["event_id"] for e in events if e["stock_compensation_admission"]["admitted"]]
    holds = [dict(event_id=e["event_id"], **e["stock_compensation_admission"])
             for e in events if not e["stock_compensation_admission"]["admitted"]]
    receipt = dict(
        passed=True, stage="conditional_stock_compensation_source_admission",
        events=len(events), admitted_events=len(selected), held_events=len(holds),
        admitted_event_ids=selected, holds=holds,
        data_sha256=src.safe.digest(src.safe.encoded(data)), policy_sha256=policy_sha,
        input_hashes=hashes, source_values_independently_reviewed=True,
        accounting_flags_preserved=True,
        accounting_flag_disposition=policy["accounting_changes"],
        financial_ratios=0, factor_calculations=0, model_fits=0, return_calculations=0,
        observer_changes=0, **{key: False for key in CERTS})
    src.base.publish(root, OUT + "/DATASET.json", data)
    src.base.publish(root, OUT + "/ADMISSION.json", receipt)
    return dict(passed=True, admitted=len(selected), held=len(holds),
                data_sha256=receipt["data_sha256"],
                admission_sha256=src.base.sha(root, OUT + "/ADMISSION.json"))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("freeze", "build"))
    cli.add_argument("--patch-plan-sha")
    cli.add_argument("--html-review-path")
    cli.add_argument("--policy-sha")
    args = cli.parse_args()
    result = (freeze(ROOT, args.patch_plan_sha, args.html_review_path)
              if args.phase == "freeze" else build(ROOT, args.policy_sha))
    print(src.safe.encoded(result).decode())


if __name__ == "__main__":
    main()

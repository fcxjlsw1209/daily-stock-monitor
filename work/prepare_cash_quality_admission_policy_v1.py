"""Freeze the reviewed field-specific policy before any source admission."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

root = Path.cwd()
out = root / "outputs/monthly_cash_quality_source_admission_v1"


def ref(path):
    return dict(path=path, sha256=hashlib.sha256((root/path).read_bytes()).hexdigest())


union = "outputs/monthly_cashflow_source_union_v1/"
inputs = {key: ref(path) for key, path in (
    ("union_plan", union+"PLAN.json"), ("union_manifest", union+"MANIFEST.json"),
    ("union_data", union+"RAW_FACTS.json"), ("union_independent", union+"INDEPENDENT_REVIEW.json"),
    ("accounting", "work/cashflow_accounting_review_v1.json"),
    ("sessions", "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json"),
    ("concept", "work/after145_cashflow_experiment_concept.json"),
    ("tests", "outputs/monthly_cash_quality_source_admission_v1/TESTS.json"))}
assert inputs["union_plan"]["sha256"] == "61abd53de8f2c99c2e979d77ff7712ddd9f3536d8fdedb370949240eb5ee75d3"
assert inputs["concept"]["sha256"] == "4fe507528756a3da14b88d24929fb638a88b785c8294d476418bc8768a8ba4b2"
code_review = ref("work/cash_quality_source_admission_v1_review.json")
assert code_review["sha256"] == "6848e5299fbe666dfcd15a559e40e212405306462e3e686f602775929a0c0f97"
policy = dict(schema_version=1, method="current_annual_cash_quality_v1",
    definition="consolidated_gaap_cfo_net_income_average_assets_v1",
    created_utc=datetime.now(timezone.utc).isoformat(), runtime=sys.version,
    expected_events=44, preserve_all_inherited_holds=True,
    same_report_literal_currency_assumption=True, use_comparison_flows=False,
    annual_days=[330, 400], max_age_days=550,
    financial_ratios_permitted=False, strategy_registration_permitted=False,
    implementation_sha256=ref("work/cash_quality_source_admission_v1.py")["sha256"],
    independent_code_review=code_review, inputs=inputs,
    scope="Current consolidated GAAP CFO and cash-flow-start NI, divided only later in a "
          "registered experiment by average current/prior same-report assets.",
    clock="First pinned exchange session strictly after current report release date.",
    accounting_policy="Retain all six legacy source holds. Current field-specific values may be "
        "conditionally admitted despite missing or revised unused comparison flows. All "
        "44 accounting reviews and findings remain attached; no claim of organic comparability.",
    limitations=["First-public source bytes and full revision history are not certified.",
        "Historical security identity is evidence-based, not a certified daily security master.",
        "Only four technology-heavy issuers; no whole-pool financial coverage claim.",
        "Dollar literal is used consistently within each report; no new ISO currency certification.",
        "Working capital, noncash expenses, taxes and acquisitions may drive CFO-minus-NI.",
        "Raw gaps are not filled. Missing/invalid latest events block older feature fallback."])
out.mkdir(parents=True, exist_ok=True)
with (out/"POLICY.json").open("x") as f:
    json.dump(policy, f, sort_keys=True, indent=2)
    f.write("\n")
print(json.dumps(ref(str((out/"POLICY.json").relative_to(root)))))

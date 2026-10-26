"""Write-once source census for reported annual stock-compensation addbacks.

No financial ratios, price inputs, model fitting, rankings, orders or returns.
Freeze records source locators; build alone may extract the single new field.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import stock_comp_source_v1 as parser

base = parser.base
safe = base.safe
require = base.require
ROOT = Path(__file__).resolve().parents[1]
SELF = "work/stock_comp_source_v1_runner.py"
TEST = "work/test_stock_comp_source_v1_runner.py"
OUT = "outputs/monthly_stock_comp_source_v1"
UNION = "outputs/monthly_cashflow_source_union_v1"
ADMISSION = "outputs/monthly_cash_quality_source_admission_v1"
METADATA = (
    "event_id", "symbol", "fiscal_year", "release_date", "available_from",
    "identity", "periods", "source", "inherited_rejection_codes",
    "cash_quality_admission",
)
ZERO_COUNTS = (
    "fact_admissions", "model_admissions", "strategy_admissions",
    "financial_calculations", "ratios_computed", "ranks_computed",
    "model_fits", "orders_computed", "returns_computed",
)
PINS = {
    UNION + "/RAW_FACTS.json": "fb26792299d6e6b75ab8ae321e1bcca65fc978876ddf4f1e0a0e983b25c41c06",
    UNION + "/MANIFEST.json": "40915dc0e562da20163864d9da2a46e6ac47bcc30f30ffb7561ad7a7b86b56b2",
    UNION + "/INDEPENDENT_REVIEW.json": "fc8b18dfc5db027d80ec68ab983b73e5c4ac5a76b4a25c8fa69a44daa2384993",
    ADMISSION + "/ADMISSION.json": "e9d2bf123cdc351fecf1c794fd3a0e929b1d1a6307232d094f402d29b7463494",
    ADMISSION + "/DATASET.json": "3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5",
    base.OUT + "/PLAN.json": "4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472",
    base.PDF_PLAN: "cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50",
    base.SESSIONS: "c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4",
    "work/after148_next_mechanism_review.json": "dc4237f5c4382f40800bb8cbb2b8268c069e80ba10703fac60d860408f70c1de",
}


def code_hashes(root):
    return dict(parser.code_hashes(root), **{
        p: base.sha(root, p) for p in (SELF, TEST)
    })


def indexed(events):
    require(type(events) is list, "Event list required")
    result = {e["event_id"]: e for e in events}
    require(len(result) == len(events), "Duplicate event")
    return result


def metadata_contract(raw, inherited, admission, sessions):
    """Validate old evidence and retain every event; do not calculate money."""
    rmap, imap = indexed(raw["events"]), indexed(inherited["events"])
    require(len(rmap) == 44 and set(rmap) == set(imap), "Exact original 44-event census required")
    require(dict(Counter(e["symbol"] for e in imap.values())) ==
            dict(AAPL=10, AMD=10, MSFT=12, NVDA=12), "Exact issuer census required")
    holds = {h["event_id"]: h for h in admission["holds"]}
    require(len(holds) == len(admission["holds"]) == 13, "All 13 inherited holds required")
    allowed = set(admission["admitted_event_ids"])
    require(len(allowed) == 31 and allowed.isdisjoint(holds) and
            allowed | set(holds) == set(imap), "Old 31/13 partition required")
    axis = safe.session_axis(sessions["sessions"])
    specs = []
    for old in raw["events"]:
        event = imap[old["event_id"]]
        require({k: v for k, v in event.items() if k != "cash_quality_admission"} == old,
                "Inherited raw event changed")
        decision = event["cash_quality_admission"]
        require(type(decision["admitted"]) is bool, "Explicit old admission required")
        require(decision["admitted"] == (event["event_id"] in allowed), "Inherited hold changed")
        if not decision["admitted"]:
            require(dict(event_id=event["event_id"], **decision) == holds[event["event_id"]],
                    "Inherited hold reasons changed")
        require(safe.next_session(event["release_date"], axis) == event["available_from"],
                "Current-release next-session clock changed")
        require(safe.day(event["periods"]["current"]["period_end"]) <
                safe.day(event["release_date"]), "Release must follow annual period")
        require(event["source"]["url"].startswith("https://"), "Pinned original source URL required")
        specs.append({k: deepcopy(event[k]) for k in METADATA})
    return specs


def source_inputs(root):
    safe.verify_hashes(root, PINS)
    oldplan = base.read(root, base.OUT + "/PLAN.json")
    require(oldplan["runtime"] == sys.version, "Pinned runtime mismatch")
    safe.verify_hashes(root, oldplan["source_hashes"])
    safe.verify_hashes(root, oldplan["source_code_hashes"])
    pdf = base.read(root, base.PDF_PLAN)
    base.configure_pdf(pdf)
    adm = base.read(root, ADMISSION + "/ADMISSION.json")
    require(adm["data_sha256"] == PINS[ADMISSION + "/DATASET.json"], "Admission data binding mismatch")
    policy = base.read(root, ADMISSION + "/POLICY.json", adm["policy_sha256"])
    require(policy["runtime"] == sys.version and policy["preserve_all_inherited_holds"] is True,
            "Inherited admission contract mismatch")
    specs = metadata_contract(base.read(root, UNION + "/RAW_FACTS.json"),
                              base.read(root, ADMISSION + "/DATASET.json"),
                              adm, base.read(root, base.SESSIONS))
    hashes = dict(PINS)
    hashes[ADMISSION + "/POLICY.json"] = adm["policy_sha256"]
    for group in (oldplan["source_hashes"], oldplan["source_code_hashes"]):
        for path, sha in group.items():
            path = str(Path(path).relative_to(root)) if Path(path).is_absolute() else path
            require(path not in hashes or hashes[path] == sha, "Conflicting inherited pin")
            hashes[path] = sha
    for event in specs:
        source = event["source"]
        for path, sha in ((source["path"], source["sha256"]),
                          (source["receipt"]["path"], source["receipt"]["sha256"])):
            require(path not in hashes or hashes[path] == sha, "Conflicting original source pin")
            hashes[path] = sha
    safe.verify_hashes(root, hashes)
    return specs, pdf, hashes


def policy():
    return dict(
        field=parser.FIELD, events=44, maximum_new_cells=44, inherited_holds=13,
        scope="Only the explicit current full-year consolidated indirect operating-cash-flow stock/share-based compensation addback.",
        excluded="Non-GAAP adjustments, income-statement expense, quarterly columns, mixed or net-tax lines, buybacks, issuance proceeds, inferred dilution and inferred zeros.",
        value="Preserve signed displayed decimal and original scale; no scaling, financial subtraction or ratio at this stage.",
        availability="Inherit current annual report release to first strictly later pinned exchange session. Retrieved vintages do not certify original first-public bytes.",
        admission="Every new field remains unadmitted until independent source/accounting review. All 13 inherited holds remain in force; new validity may only narrow usable events.",
        interpretation="Potential shareholder economic-cost proxy, not an actual cash expenditure, measured dilution or free cash flow.",
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False, iso_currency_certified=False,
        financial_calculations_permitted=False, strategy_registration_permitted=False,
        network_requests_permitted=False,
    )


def test_contract(root):
    checks = [
        ("PARSER_TESTS.json", parser.code_hashes(root)),
        ("TESTS.json", code_hashes(root)),
    ]
    hashes = {}
    for name, code in checks:
        rel = OUT + "/" + name
        report = base.read(root, rel)
        require(report.get("passed") is True and report.get("synthetic_only") is True,
                "Current passing synthetic tests required")
        require(report.get("source_code_hashes") == code, "Stale synthetic tests")
        require(report.get("tests_run", 0) > 0, "Nonempty test suite required")
        require(all(type(report.get(k)) is int and report[k] == 0
                    for k in ("errors", "failures", "real_source_cells_extracted")),
                "Clean synthetic-only test counters required")
        hashes[rel] = base.sha(root, rel)
    return hashes


def freeze(root):
    require(not safe.local(root, OUT + "/PLAN.json").exists(), "Plan already frozen")
    hashes = test_contract(root)
    specs, pdf, sources = source_inputs(root)
    hashes.update(sources)
    for spec in specs:
        source = spec["source"]
        raw = safe.local(root, source["path"]).read_bytes()
        require(safe.digest(raw) == source["sha256"], "Source changed during inventory")
        spec["locator_inventory"] = parser.inventory(raw, spec["periods"], pdf)
    safe.verify_hashes(root, hashes)
    plan = dict(schema_version=1, stage="frozen_stock_compensation_source_census",
                created_utc=datetime.now(timezone.utc).isoformat(), runtime=sys.version,
                source_hashes=hashes, source_code_hashes=code_hashes(root),
                pdf_runtime=pdf["pdf_runtime"], pdf_runtime_hashes=pdf["pdf_runtime_hashes"],
                policy=policy(), events=specs, financial_amounts_extracted=0)
    base.publish(root, OUT + "/PLAN.json", plan)
    return dict(plan_sha256=base.sha(root, OUT + "/PLAN.json"), events=len(specs),
                unique_locators=sum(len(e["locator_inventory"]["candidates"]) == 1 and
                                    not e["locator_inventory"]["gaps"] for e in specs),
                financial_amounts_extracted=0)


def load_plan(root, sha):
    base.expect_sha(sha)
    plan = base.read(root, OUT + "/PLAN.json", sha)
    require(plan["stage"] == "frozen_stock_compensation_source_census", "Wrong source plan")
    require(plan["runtime"] == sys.version and plan["policy"] == policy(), "Source policy/runtime changed")
    require(plan["source_code_hashes"] == code_hashes(root), "Frozen source code changed")
    safe.verify_hashes(root, plan["source_hashes"])
    safe.verify_hashes(root, plan["source_code_hashes"])
    base.configure_pdf(plan)
    specs, _, _ = source_inputs(root)
    require([{k: e[k] for k in METADATA} for e in plan["events"]] == specs,
            "Frozen event metadata changed")
    return plan


def products(root, plan, sha):
    rows, coverage = [], []
    for spec in plan["events"]:
        source = spec["source"]
        raw = safe.local(root, source["path"]).read_bytes()
        require(safe.digest(raw) == source["sha256"], "Source changed during extraction")
        result = parser.extract_current(raw, spec["periods"], spec["locator_inventory"], plan)
        event = {k: deepcopy(spec[k]) for k in METADATA}
        event["stock_compensation"] = result
        event.update(fact_admitted=False, model_admitted=False, strategy_admitted=False)
        if result["value"] is not None:
            value = result["value"]
            require(value["source_sha256"] == source["sha256"] and
                    value["period"] == spec["periods"]["current"], "New cell provenance mismatch")
            value["available_from"] = spec["available_from"]
            value["source"] = deepcopy(source)
        coverage.append(dict(event_id=spec["event_id"], symbol=spec["symbol"],
                             present=result["value"] is not None,
                             inherited_hold=not spec["cash_quality_admission"]["admitted"],
                             new_field_admitted=False, gaps=deepcopy(result["gaps"])))
        rows.append(event)
    require(len(rows) == 44 and len(indexed(rows)) == 44, "Dropped original event")
    require(sum(e["inherited_hold"] for e in coverage) == 13, "Dropped inherited hold")
    counts = dict(events=44, requested_new_cells=44,
                  extracted_new_cells=sum(e["present"] for e in coverage),
                  gap_cells=sum(not e["present"] for e in coverage), inherited_holds=13,
                  **{k: 0 for k in ZERO_COUNTS})
    data = dict(schema_version=1, stage="raw_stock_compensation_candidates_not_admitted",
                plan_sha256=sha, counts=counts, events=rows)
    cov = dict(plan_sha256=sha, counts=counts, events=coverage)
    return data, cov


def build(root, sha):
    plan = load_plan(root, sha)
    names = ("RAW_FACTS.json", "COVERAGE.json", "AUDIT.json", "MANIFEST.json")
    require(not any(safe.local(root, OUT + "/" + n).exists() for n in names),
            "Source outputs are write-once; use verify for existing build")
    data, coverage = products(root, plan, sha)
    safe.verify_hashes(root, plan["source_hashes"])
    safe.verify_hashes(root, plan["source_code_hashes"])
    hashes = {name: safe.digest(safe.encoded(obj)) for name, obj in
              (("RAW_FACTS.json", data), ("COVERAGE.json", coverage))}
    audit = dict(passed=True, stage="source_process_audit_not_field_admission",
                 plan_sha256=sha, counts=data["counts"],
                 data_sha256=hashes["RAW_FACTS.json"],
                 coverage_sha256=hashes["COVERAGE.json"],
                 all_original_events_preserved=True, all_inherited_holds_preserved=True,
                 independent_field_and_visual_review_pending=True)
    for name, obj in (("RAW_FACTS.json", data), ("COVERAGE.json", coverage), ("AUDIT.json", audit)):
        base.publish(root, OUT + "/" + name, obj)
        hashes[name] = base.sha(root, OUT + "/" + name)
    base.publish(root, OUT + "/MANIFEST.json", dict(plan_sha256=sha, output_hashes=hashes))
    return dict(passed=True, counts=data["counts"], manifest_sha256=base.sha(root, OUT + "/MANIFEST.json"))


def verify(root, sha):
    plan = load_plan(root, sha)
    manifest = base.read(root, OUT + "/MANIFEST.json")
    require(manifest["plan_sha256"] == sha, "Manifest plan mismatch")
    require(set(manifest["output_hashes"]) == {"RAW_FACTS.json", "COVERAGE.json", "AUDIT.json"},
            "Exact source output manifest required")
    safe.verify_hashes(root, {OUT + "/" + k: v for k, v in manifest["output_hashes"].items()})
    data, coverage = products(root, plan, sha)
    require(data == base.read(root, OUT + "/RAW_FACTS.json"), "Raw source reproduction mismatch")
    require(coverage == base.read(root, OUT + "/COVERAGE.json"), "Coverage reproduction mismatch")
    audit = base.read(root, OUT + "/AUDIT.json")
    require(audit["passed"] is True and audit["counts"] == data["counts"], "Audit count mismatch")
    require(audit["plan_sha256"] == sha and
            audit["data_sha256"] == manifest["output_hashes"]["RAW_FACTS.json"] and
            audit["coverage_sha256"] == manifest["output_hashes"]["COVERAGE.json"],
            "Audit source bindings mismatch")
    require(audit["independent_field_and_visual_review_pending"] is True and
            audit["all_original_events_preserved"] is True and
            audit["all_inherited_holds_preserved"] is True, "Audit scope mismatch")
    safe.verify_hashes(root, plan["source_hashes"])
    safe.verify_hashes(root, plan["source_code_hashes"])
    return dict(passed=True, counts=data["counts"], source_only=True)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("freeze", "build", "verify"))
    cli.add_argument("--plan-sha")
    args = cli.parse_args()
    result = freeze(ROOT) if args.phase == "freeze" else (
        build(ROOT, args.plan_sha) if args.phase == "build" else verify(ROOT, args.plan_sha))
    print(safe.encoded(result).decode())


if __name__ == "__main__":
    main()

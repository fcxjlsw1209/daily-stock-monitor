"""Conditional INTC/KLAC NI/assets source gate; no financial arithmetic.

All original records are retained.  The separate first batch and four-issuer
source packages are referenced unchanged, not re-admitted or merged.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import InvalidOperation
import hashlib
import json
from pathlib import Path
import re

import after149_batch1_admission as old

ROOT = Path(__file__).resolve().parents[1]
OUT = "outputs/monthly_financial_coverage_admission_batch2_v1"
BATCH = "outputs/monthly_financial_coverage_batch2_v1"
BATCH1 = old.OUT
POLICY = "work/after150_batch2_admission_policy.json"
REVIEW = "work/after150_batch2_independent_review.json"
CONTEXT = "work/after150_next_source_context.json"
CENSUS, META, SESSIONS = old.CENSUS, old.META, old.SESSIONS
OLD_DATA, OLD_ADMISSION = old.OLD_DATA, old.OLD_ADMISSION
FIELDS, METHOD, DEFINITION = old.FIELDS, old.METHOD, old.DEFINITION
SYMBOLS = frozenset(("INTC", "KLAC"))
MAX_AGE = 550
ASSUMPTION = "INTC_consecutive_annual_boundaries_v1"
INTC_HOLD = "annual_start_not_directly_quoted_requires_source_use_assumption"
check, day, decimal, canonical, digest = old.check, old.day, old.decimal, old.canonical, old.digest
normalize_sessions, first_session, select_latest = old.normalize_sessions, old.first_session, old.select_latest


class Evidence(old.Evidence):
    def read(self, name, expected=None):
        # Decode precisely the bytes whose digest enters this closure.
        p = self.path(name)
        raw = p.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        check(expected is None or actual == expected, "Hash mismatch: "+str(name))
        key = str(p.relative_to(self.root))
        check(key not in self.seen or self.seen[key] == actual, "Source changed: "+key)
        self.seen[key] = actual
        return json.loads(raw)


def policy_contract(policy):
    check(policy.get("schema_version") == 1 and
          policy.get("policy_id") == "intc_klac_annual_ni_same_report_assets_batch2_v1" and
          policy.get("stage") == "prospective_source_policy_not_strategy_registration",
          "Wrong prospective policy")
    check(policy.get("definition") == DEFINITION and policy.get("required_fields") == list(FIELDS),
          "Measurement or fields changed")
    scope = policy.get("source_scope", {})
    check(scope.get("symbols") == ["INTC", "KLAC"] and scope.get("catalog_events") == 32 and
          scope.get("html_context_reviewed_events") == 19 and scope.get("no_automatic_admission") is True,
          "Fixed source scope changed")
    check(policy.get("known_time", {}).get("maximum_age_days") == MAX_AGE, "Age limit changed")
    a = policy.get("periods", {}).get("INTC_52_53_week_assumption", {})
    check(a.get("enabled") is True and a.get("not_a_quoted_fact") is True and
          a.get("assumption_id") == ASSUMPTION, "Explicit INTC boundary policy required")
    for name in ("KLAC_FY2011", "KLAC_FY2019"):
        check(str(policy.get("accounting", {}).get(name, "")).startswith("Hold "),
              "Required KLAC hold removed")
    for key in ("first_public_version_verified", "complete_correction_history_verified",
                "certified_daily_security_master"):
        check(policy.get("preservation", {}).get(key) is False, "Unsupported certification")


def context_map(context):
    docs = context["documents"]
    check(len(docs) == 32 and len({x["event_id"] for x in docs}) == 32 and
          Counter(x["symbol"] for x in docs) == {"INTC": 16, "KLAC": 16}, "Full32 context required")
    chosen = [x for x in docs if x.get("html_context_reviewed") is True]
    check(len(chosen) == 19 and Counter(x["symbol"] for x in chosen) == {"INTC": 3, "KLAC": 16},
          "Fixed19 context required")
    result = {}
    for row in chosen:
        cik = context["identity_leads"][row["symbol"]]["cik"]
        check(re.fullmatch(r"\d{10}", cik) is not None, "Reviewed CIK required")
        result[row["event_id"]] = (row["symbol"], cik, row)
    return result


def verify_independent_review(review, events, plan_sha, extraction_sha):
    reviewed = old.verify_independent_review(review, events, plan_sha, extraction_sha)
    for event in events:
        row = reviewed[event["event_id"]]
        check(row.get("symbol") == event["symbol"] and row.get("fiscal_year") == event["fiscal_year"],
              "Review identity mismatch")
        check(row.get("identity_verified") is True, "Independent identity evidence required")
        check(row.get("retained_context_holds") == event.get("context_holds", []),
              "Review did not preserve context holds")
        check(row.get("source") == event["source"] and row.get("available_from") == event["available_from"] and
              row.get("release_date") == event["publication_date"], "Review source/time mismatch")
        nonnull = any(event["fields"][f] is not None for f in FIELDS)
        if nonnull:
            check(row.get("periods_verified") is True and row.get("currency_verified") is True,
                  "Independent periods/currency evidence required")
        for name in FIELDS:
            cell = row["fields"][name]
            if event["fields"][name] is not None:
                for flag in ("value_and_sign_verified", "units_verified", "period_verified", "source_locator_verified"):
                    check(cell.get(flag) is True, "Independent field evidence missing: "+flag)
            else:
                check(type(cell.get("null_classification")) is str and cell["null_classification"],
                      "Precise null classification required")
        if event["symbol"] == "INTC":
            check(row.get("boundary_assumption_verified") is True and row.get("assumption_id") == ASSUMPTION and
                  row.get("start_is_source_quoted") is False, "INTC boundary assumption not verified")
            evidence = row.get("boundary_evidence", {})
            check(set(evidence) == set(FIELDS), "INTC boundary locators required")
            for name in FIELDS:
                check(type(evidence[name].get("table")) is int and evidence[name].get("headers") and
                      evidence[name].get("date") == event["fields"][name]["as_of"],
                      "INTC boundary locator mismatch")
            check(row.get("derived_start") == event["fields"]["annual_net_income"]["period"]["period_start"],
                  "INTC reviewed start mismatch")
    return reviewed


def assess_event(event, context_item, reviewed, sessions):
    result = deepcopy(event)
    check(not any(k in event for k in ("ni_assets_admission", "identity", "source_review", "clock_review")),
          "Gate already present")
    symbol, cik, context = context_item
    check(symbol == event["symbol"] and event["event_id"] == context["event_id"] and
          event["fiscal_year"] == context["fiscal_year"], "Context identity mismatch")
    check(set(event.get("fields", {})) == set(FIELDS), "Exact three fields required")
    check(all(event.get(k) is False for k in ("fact_admitted", "model_admitted", "strategy_admitted")),
          "Premature extraction admission")
    source = event["source"]
    check(source["path"] == context["evidence_path"] and source["sha256"] == context["evidence_sha256"] and
          source["url"] == context["source_url"], "Source/context mismatch")
    if "receipt" in source:
        check(source["receipt"] == context["context"]["receipt"]["path"], "Receipt/context mismatch")
    release, available = event["publication_date"], event["available_from"]
    check(release == context["publication_date"] and available == context["available_session"],
          "Event/context clock mismatch")
    check(day(available) > day(release), "Availability must follow publication")
    derived = first_session(release, sessions)
    check(reviewed.get("clock_verified") is (derived is not None and derived == available),
          "Independent clock finding disagrees with pinned calendar")
    reasons = []
    if derived is None:
        reasons.append("unverified_exchange_calendar")
    elif derived != available:
        reasons.append("availability_not_first_pinned_session")
    year = event["fiscal_year"]
    if symbol == "KLAC" and year == 2011:
        check(all(event["fields"][f] is None for f in FIELDS), "Frozen KLAC2011 null was replaced")
        reasons.append("KLAC_FY2011_PRE_geometry_unparsed")
    if symbol == "KLAC" and year == 2019:
        reasons.append("KLAC_FY2019_ASC606_transition_unresolved")
    holds = event.get("context_holds", [])
    check(isinstance(holds, list) and all(type(x) is str for x in holds), "Context holds malformed")
    for hold in holds:
        if symbol == "INTC" and hold == INTC_HOLD:
            if reviewed.get("boundary_assumption_verified") is not True or reviewed.get("assumption_id") != ASSUMPTION:
                reasons.append("INTC_boundary_assumption_unverified")
        else:
            reasons.append("retained_context_hold:"+hold)
    if symbol == "INTC" and (INTC_HOLD not in holds or
            reviewed.get("boundary_assumption_verified") is not True or reviewed.get("assumption_id") != ASSUMPTION):
        reasons.append("INTC_boundary_assumption_unverified")
    for f in FIELDS:
        if event["fields"][f] is None:
            reasons.append("missing_"+f)
    start = end = None
    try:
        period = event["fields"]["annual_net_income"]["period"]
        start, end = day(period["period_start"]), day(period["period_end"])
        count = (end-start).days+1
        check(type(period["days_inclusive"]) is int and period["days_inclusive"] == count and 330 <= count <= 400,
              "Invalid annual duration")
        check(end <= day(release) and end.isoformat() == context["official_actual_period_end"],
              "Annual end mismatch")
        meta = context["context_summary"]
        prior = day(meta["comparison_assets_date"])
        check(start-timedelta(days=1) == prior and meta["current_annual_period"]["end"] == str(end),
              "Context endpoints mismatch")
        check(event["periods"]["current"] == period, "Top-level period mismatch")
        if symbol == "INTC":
            check(count in (364, 371) and meta["current_annual_period"]["start"] is None and
                  meta["current_annual_period"]["candidate_start_from_prior_end_plus_one"] == str(start),
                  "INTC consecutive boundary mismatch")
            check(reviewed.get("derived_start") == str(start) and reviewed.get("start_is_source_quoted") is False and
                  event["periods"].get("start_assumption_certified") is False and
                  event["periods"].get("start_basis") == "derived_same_report_prior_annual_end_plus_one_not_directly_quoted",
                  "INTC derivation not explicit")
        else:
            check(count in (365, 366) and str(end) == f"{year}-06-30" and str(start) == f"{year-1}-07-01" and
                  str(prior) == f"{year-1}-06-30" and meta["current_annual_period"]["start"] == str(start),
                  "KLAC calendar annual mismatch")
    except (KeyError, TypeError, ValueError):
        reasons.append("invalid_current_annual_period")
    currencies = set()
    for name, cell in event["fields"].items():
        if cell is None:
            continue
        try:
            check(cell["source"] == source and cell["available_from"] == available and cell.get("fact_admitted") is False,
                  "Cell source/time/certification mismatch")
            check(type(cell["displayed_amount"]) is str and cell["displayed_amount"].strip() and
                  isinstance(cell["locator"], dict) and cell["locator"], "Cell provenance missing")
            value, scale = decimal(cell["value_decimal"]), decimal(cell["scale"])
            check(scale > 0 and (name == "annual_net_income" or value > 0), "Nonpositive scale/assets")
            check(cell["currency_literal"] == "$" and cell.get("currency_iso") is None, "Uncertified currency change")
            currencies.add(cell["currency_literal"])
            check(start is not None and end is not None, "No annual endpoints")
            expected = start-timedelta(days=1) if name == "prior_total_assets" else end
            check(day(cell["as_of"]) == expected, "Cell endpoint mismatch")
            if name != "annual_net_income":
                check(cell.get("period") is None, "Asset must be an instant")
        except (KeyError, TypeError, ValueError, InvalidOperation):
            reasons.append("invalid_"+name)
    result["identity"] = dict(issuer_id=cik, security_id=symbol, identity_valid_from=available,
        identity_valid_through=(end+timedelta(days=MAX_AGE)).isoformat() if end else None,
        scope="source_local_report_binding_and_use_window_not_continuous_security_master",
        certified_daily_security_master=False)
    result["ni_assets_admission"] = dict(admitted=not reasons, reasons=list(dict.fromkeys(reasons)), definition=DEFINITION)
    result["source_review"] = dict(event_id=reviewed["event_id"], metadata_verified=True,
        all_three_positions_reviewed=True, boundary_assumption_verified=reviewed.get("boundary_assumption_verified", False),
        original_context_holds_preserved=True)
    result["clock_review"] = dict(first_pinned_session=derived, verified=derived is not None and derived == available)
    return result


def annual_catalog(census, contexts, sessions):
    rows = [deepcopy(x) for x in census["annual_documents"] if x["symbol"] in SYMBOLS]
    check(len(rows) == 32 and len({x["event_id"] for x in rows}) == 32 and
          Counter(x["symbol"] for x in rows) == {"INTC": 16, "KLAC": 16}, "Complete32 catalog required")
    check(set(contexts) <= {x["event_id"] for x in rows}, "Context outside catalog")
    for row in rows:
        check(row["fiscal_quarter"] == 4, "Nonannual catalog item")
        derived = first_session(row["publication_date"], sessions)
        advertised = row.get("available_session")
        check(advertised is None or day(advertised) > day(row["publication_date"]), "Invalid catalog time")
        row["clock_review"] = dict(first_pinned_session=derived, verified=derived is not None and derived == advertised)
        row["processed_in_batch"] = row["event_id"] in contexts
    return rows


def coverage(meta, census, catalog, events, strict=True):
    # Reuse only the old complete-axis validation. Replace its batch1-specific
    # availability results with this independent batch2 view on the same axis.
    result = old.coverage(meta, census, catalog, events, strict)
    by_id = {e["event_id"]: e for e in events}
    for row in result["monthly_candidate_coverage"]:
        details = (select_latest(catalog, by_id, row["symbol"], row["session"]) if row["symbol"] in SYMBOLS else
                   dict(available=False, reason="outside_new_batch_prior_evidence_preserved", event_id=None))
        signal, symbol = row["session"], row["symbol"]
        row.clear()
        row.update(session=signal, symbol=symbol, **details)
    rows = result["monthly_candidate_coverage"]
    result["scope"] = "incremental_batch2_source_availability_not_strategy_eligibility"
    result["counts"]["conditional_new_slots"] = sum(x["available"] for x in rows)
    result["counts"]["reasons"] = dict(Counter(x["reason"] for x in rows))
    result["prior_batch1_coverage"] = "Unchanged and separately referenced; no old source gate is reopened."
    return result


def _bound(evidence, mapping, name, expected):
    check(any(evidence.path(k) == evidence.path(name) and v == expected for k, v in mapping.items()),
          "Mandatory binding missing: "+name)


def preserved_references(evidence):
    # Verify the existing manifest closure before interpreting any old record.
    manifest = evidence.read(BATCH1+"/MANIFEST.json")
    evidence.pins(manifest["input_hashes"])
    old_data = evidence.read(OLD_DATA)
    old_summary = old.old_population(old_data)
    old_adm = evidence.read(OLD_ADMISSION)
    check(old_adm.get("passed") is True and old_adm.get("data_sha256") == evidence.pin(OLD_DATA) and
          [old_adm.get(k) for k in ("events", "admitted_events", "held_events")] == [44, 31, 13],
          "Original source reference mismatch")
    check(set(manifest["outputs"]) == {"DATASET.json", "COVERAGE.json", "ADMISSION.json"},
          "Batch1 manifest output set changed")
    package = {name: evidence.read(BATCH1+"/"+name, h) for name, h in manifest["outputs"].items()}
    data, adm = package["DATASET.json"], package["ADMISSION.json"]
    evidence.pins(adm["input_hashes"])
    check(adm.get("passed") is True and data.get("method") == METHOD, "Batch1 source package not accepted")
    check(len(data["events"]) == 26 and len({x["event_id"] for x in data["events"]}) == 26 and
          Counter(x["symbol"] for x in data["events"]) == {"AMZN": 10, "CSCO": 16}, "Batch1 population changed")
    check(sum(x["ni_assets_admission"]["admitted"] is True for x in data["events"]) == 17 and
          len(data["latest_event_catalog"]) == 32 and
          adm["counts"]["events"] == 26 and adm["counts"]["admitted"] == 17 and adm["counts"]["held"] == 9,
          "Batch1 admissions/catalog changed")
    reference = data["original_four_issuer_reference"]
    check(reference["path"] == OLD_DATA and reference["sha256"] == evidence.pin(OLD_DATA) and
          reference["records_canonical_sha256"] == old_summary["records_canonical_sha256"] and
          reference["admission_sha256"] == evidence.pin(OLD_ADMISSION), "Batch1/original reference mismatch")
    return deepcopy(reference), dict(
        path=BATCH1+"/DATASET.json", sha256=evidence.pin(BATCH1+"/DATASET.json"),
        admission_path=BATCH1+"/ADMISSION.json", admission_sha256=evidence.pin(BATCH1+"/ADMISSION.json"),
        manifest_path=BATCH1+"/MANIFEST.json", manifest_sha256=evidence.pin(BATCH1+"/MANIFEST.json"),
        coverage_path=BATCH1+"/COVERAGE.json", coverage_sha256=evidence.pin(BATCH1+"/COVERAGE.json"),
        events=26, admitted=17, held=9, annual_catalog_events=32,
        records_canonical_sha256=hashlib.sha256(canonical(data["events"])).hexdigest(),
        catalog_canonical_sha256=hashlib.sha256(canonical(data["latest_event_catalog"])).hexdigest(),
        preserved_without_modification=True)


def build(root, plan_sha, extraction_sha, review_sha, policy_sha):
    check(all(re.fullmatch(r"[0-9a-f]{64}", x or "") for x in
              (plan_sha, extraction_sha, review_sha, policy_sha)), "Four explicit SHA256 values required")
    evidence = Evidence(root)
    plan = evidence.read(BATCH+"/PLAN.json", plan_sha)
    evidence.pins(plan["inputs"])
    evidence.pins(plan["code"])
    for name in (BATCH1+"/DATASET.json", BATCH1+"/ADMISSION.json"):
        _bound(evidence, plan["inputs"], name, evidence.pin(name))
    policy = evidence.read(POLICY, policy_sha)
    _bound(evidence, plan["inputs"], POLICY, policy_sha)
    policy_contract(policy)
    evidence.pins(policy["input_hashes"])
    context = evidence.read(CONTEXT)
    cref = {"path": CONTEXT, "sha256": evidence.pin(CONTEXT)}
    check(policy["source_scope"]["source_context"] == cref == plan["catalog_ref"] and
          plan["context"] == CONTEXT and plan["policy"] == POLICY, "Context/policy plan linkage")
    cmap = context_map(context)
    extraction = evidence.read(BATCH+"/EXTRACTIONS.json", extraction_sha)
    check(extraction["plan_sha256"] == plan_sha and extraction["catalog_ref"] == cref and
          extraction["status"] == "source_values_extracted_pending_independent_review", "Extraction linkage")
    for k in ("new_financial_admissions", "new_strategy_trials", "ratios_computed", "returns_computed"):
        check(type(extraction.get(k)) is int and extraction[k] == 0, "Premature financial processing")
    events = extraction["events"]
    check([e["event_id"] for e in events] == list(cmap) == [x["event_id"] for x in plan["records"]],
          "Exact19 extraction order/population mismatch")
    catalog_ids = [x["event_id"] for x in context["documents"]]
    check(plan["catalog_ids"] == extraction["catalog_ids"] == catalog_ids and plan["catalog_events"] == 32,
          "Full32 catalog missing")
    review = evidence.read(REVIEW, review_sha)
    check(review.get("stage") == "independent_source_value_and_null_classification_review",
          "Wrong independent review stage")
    for name in ("financial_calculations", "factor_calculations", "returns_computed", "new_admissions"):
        check(type(review.get(name)) is int and review[name] == 0, "Review exceeded source-only scope")
    evidence.pins(review["input_hashes"])
    evidence.pins(review["source_hashes"])
    for name in ("work/after150_batch2_independent_review.py", "work/test_after150_batch2_independent_review.py"):
        _bound(evidence, review["source_hashes"], name, evidence.pin(name))
    for name, h in review["source_hashes"].items():
        _bound(evidence, review["input_hashes"], name, h)
    for name in (BATCH+"/PLAN.json", BATCH+"/EXTRACTIONS.json", POLICY, CONTEXT, SESSIONS):
        _bound(evidence, review["input_hashes"], name, evidence.pin(name))
    reviewed = verify_independent_review(review, events, plan_sha, extraction_sha)
    for event in events:
        row = cmap[event["event_id"]][2]
        for name, h in ((row["evidence_path"], row["evidence_sha256"]),
                        (row["context"]["receipt"]["path"], row["context"]["receipt"]["sha256"])):
            evidence.pin(name, h)
            _bound(evidence, review["input_hashes"], name, h)
    sessions = normalize_sessions(evidence.read(SESSIONS))
    admitted = [assess_event(e, cmap[e["event_id"]], reviewed[e["event_id"]], sessions) for e in events]
    census, meta = evidence.read(CENSUS), evidence.read(META)
    catalog = annual_catalog(census, cmap, sessions)
    raw_catalog = [x for x in census["annual_documents"] if x["symbol"] in SYMBOLS]
    check([x["event_id"] for x in raw_catalog] == catalog_ids, "Catalog order mismatch")
    projection = [dict(event_id=x["event_id"], symbol=x["symbol"], fiscal_year=x["fiscal_year"],
        publication_date=x["publication_date"], available_session=x["available_session"],
        official_actual_period_end=x["official_actual_period_end"], source_path=x["evidence_path"],
        source_sha256=x["evidence_sha256"], html_context_reviewed=x.get("html_context_reviewed") is True,
        retained_context_gap=x.get("gap")) for x in context["documents"]]
    check(policy["fixed_catalog_metadata"] == projection, "Policy catalog differs from fixed context")
    for row, ctx in zip(raw_catalog, context["documents"]):
        check(all(ctx.get(k) == v for k, v in row.items()), "Frozen census/context metadata mismatch")
    for event in admitted:
        row = next(x for x in raw_catalog if x["event_id"] == event["event_id"])
        check(row["publication_date"] == event["publication_date"] and row["available_session"] == event["available_from"],
              "Catalog/event clock mismatch")
    old_ref, prior_ref = preserved_references(evidence)
    dataset = dict(schema_version=1, method=METHOD, policy_sha256=policy_sha, source_plan_sha256=plan_sha,
        independent_review_sha256=review_sha, events=admitted, latest_event_catalog=catalog,
        original_four_issuer_reference=old_ref, prior_batch1_reference=prior_ref,
        conditional_research_only=True, first_public_version_verified=False,
        complete_correction_history_verified=False, certified_daily_security_master=False,
        scores_computed=0, ratios_computed=0, returns_computed=0, new_strategy_trials=0)
    monthly = coverage(meta, census, catalog, admitted)
    for name in ("work/after150_batch2_admission.py", "work/test_after150_batch2_admission.py",
                 "work/after149_batch1_admission.py"):
        evidence.pin(name)
    evidence.stable()
    counts = dict(events=len(events), admitted=sum(x["ni_assets_admission"]["admitted"] for x in admitted),
        held=sum(not x["ni_assets_admission"]["admitted"] for x in admitted),
        reviewed_nonnull_fields=sum(e["fields"][f] is not None for e in events for f in FIELDS),
        reviewed_null_fields=sum(e["fields"][f] is None for e in events for f in FIELDS),
        annual_catalog_events=32, unprocessed_annual_catalog_events=13,
        original_events=44, original_admitted=31, original_held=13,
        prior_batch1_events=26, prior_batch1_admitted=17, prior_batch1_held=9)
    admission = dict(schema_version=1, stage="conditional_source_admission_not_strategy_registration",
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=policy_sha,
        source_plan_sha256=plan_sha, independent_review_sha256=review_sha,
        counts=counts, input_hashes=dict(evidence.seen),
        numerical_work="Finiteness/sign/period checks only; no ratio, sum of assets, score or return.",
        actual_factors_calculated=0, actual_returns_calculated=0, new_strategy_trials=0)
    return {"DATASET.json": dataset, "COVERAGE.json": monthly, "ADMISSION.json": admission}, evidence


def publish(root, objects, evidence):
    directory = Path(root)/OUT
    directory.mkdir(parents=True, exist_ok=True)
    check(set(objects) == {"DATASET.json", "COVERAGE.json", "ADMISSION.json"}, "Unexpected output set")
    check(not any((directory/name).exists() for name in list(objects)+["MANIFEST.json"]), "Write-once output exists")
    evidence.stable()
    for name, obj in objects.items():
        with (directory/name).open("xb") as f:
            f.write((json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode())
    evidence.stable()
    manifest = dict(schema_version=1, input_hashes=dict(evidence.seen),
                    outputs={name: digest(directory/name) for name in objects},
                    status="conditional_source_package_no_strategy_registration")
    with (directory/"MANIFEST.json").open("xb") as f:
        f.write((json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "extractions", "review", "policy"):
        p.add_argument("--"+arg+"-sha", required=True)
    args = p.parse_args()
    objects, evidence = build(ROOT, args.plan_sha, args.extractions_sha, args.review_sha, args.policy_sha)
    manifest = publish(ROOT, objects, evidence)
    print(json.dumps(dict(status="conditional_source_package_complete",
                         counts=objects["ADMISSION.json"]["counts"], outputs=manifest["outputs"]), sort_keys=True))


if __name__ == "__main__":
    main()

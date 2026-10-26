"""Prespecified KO/LRCX financial-source batch; no ranking or return calculations."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import importlib
import io
import unittest

import after149_source_batch as shared

ROOT = shared.ROOT
OUT = ROOT / "outputs/monthly_financial_coverage_batch3_v1"
CONTEXT = "work/after151_next_source_context.json"
POLICY = "work/after151_batch3_admission_policy.json"
CONTEXT_SHA = "1ff5c44b906554fcf15785bd047fc9ff924ad5d97a1e639d32d77f056e40114d"
PREDECESSOR = "outputs/monthly_financial_coverage151/COMPLETION.json"
PREDECESSOR_SHA = "7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5"
READY = "standard_HTML_annual_header_context_located_not_admitted"
MODULES = {"KO": "after151_ko_extract", "LRCX": "after151_lrcx_extract"}
FIELDS = shared.FIELDS
check = shared.check


def records(context):
    docs = context["events"]
    check(len(docs) == 32 and len({x["event_id"] for x in docs}) == 32,
          "Complete distinct32-event catalog required")
    check(Counter(x["symbol"] for x in docs) == {"KO": 16, "LRCX": 16},
          "Changed catalog issuer scope")
    chosen = [x for x in docs if x.get("context_status") == READY]
    check([(x["symbol"], x["fiscal_year"]) for x in chosen] ==
          [("KO", y) for y in range(2011, 2017)] +
          [("LRCX", y) for y in range(2013, 2017)],
          "Changed prespecified ten-report scope")
    return chosen


def catalog_metadata(context):
    return [dict(event_id=x["event_id"], symbol=x["symbol"], fiscal_year=x["fiscal_year"],
                 publication_date=x["publication_date"], available_session=x["available_session"],
                 official_actual_period_end=x["official_actual_period_end"],
                 source=x["source"], context_status=x["context_status"])
            for x in context["events"]]


def make_policy(context):
    return dict(
        schema_version=1, policy_id="ko_lrcx_annual_ni_same_report_assets_batch3_v1",
        stage="prospective_source_policy_not_strategy_registration",
        created_utc=shared.now(),
        definition="current_annual_net_income_average_assets_v1",
        purpose="Audit current consolidated annual NI and same-report annual asset endpoints.",
        source_scope=dict(
            symbols=list(MODULES), catalog_events=32, html_context_reviewed_events=10,
            source_context=dict(path=CONTEXT, sha256=CONTEXT_SHA),
            event_selection="Next alphabetic existing-source issuers; no returns or holdings selection.",
            extraction_scope="Ten enumerated HTML reports; other22 remain explicitly unprocessed.",
            catalog_requirement="Retain all32 entries in source order; latest held/unprocessed event blocks fallback.",
            independent_review="Independently parse raw values, signs, headers, units, identity and receipts.",
            no_automatic_admission=True),
        required_fields=list(FIELDS),
        fixed_catalog_metadata=catalog_metadata(context),
        net_income_scope=dict(
            required="Consolidated GAAP annual total net income before parent/noncontrolling attribution.",
            KO="Exact Consolidated Net Income row, not Net Income Attributable to Shareowners.",
            LRCX="Exact Net income or Net income (loss) in annual GAAP operations column, not quarterly or non-GAAP.",
            prohibited="No EPS, operating income, quarterly summing, annualization, synthetic TTM, or adjustments.",
            negative_and_zero="Finite annual loss or zero is valid; no profitable-company filter."),
        periods=dict(
            assets="Same current report: closing assets at NI annual end and opening assets at preceding annual end.",
            income="Current complete fiscal year only; no quarter, stub, outlook or nonconsecutive comparison.",
            LRCX_52_53_week_assumption=dict(
                enabled=True, not_a_quoted_fact=True,
                assumption_id="LRCX_consecutive_annual_boundaries_v1",
                formula="derived start = same-report prior annual assets date + one calendar day",
                required_checks=[
                    "Explicit annual NI header; NI end, current balance-sheet end and catalog end agree.",
                    "Prior balance-sheet date is immediately previous annual end, not March quarter end.",
                    "Independent date/header evidence for both endpoints and exactly364 or371 inclusive days.",
                    "No conflicting quoted start, changed fiscal end or stub interval.",
                    "Review explicitly sets boundary_assumption_verified with source locators."]),
            KO_calendar_annual="Derive Jan1 from same-report previous Dec31; require365/366 days and both Dec31 endpoints.",
            representation="All source context remains unchanged; derived starts are not quoted or certified facts."),
        numeric=dict(
            income="Finite signed decimal with original displayed literal and sign geometry.",
            assets="Finite strictly positive consolidated Total assets.",
            scale="Field-specific printed scale with physical evidence; KO millions, LRCX thousands.",
            currency="Consistent same-report dollar unit; ISO remains null without independent evidence.",
            source_only="No ratio, rank, model score, holding or return computation."),
        known_time=dict(
            available_from="First pinned exchange session strictly after publication, never the reported period date.",
            unverified_early_calendar="Outside pinned calendar remains a time hold; do not infer weekdays or holidays.",
            catalog_known_rule="Verified available_session<=signal; unverified publication<signal is a conservative blocker.",
            monthly_selection="Latest known catalog event before source validity/age. Ties unresolved hold. No old-report fallback.",
            maximum_age_days=550, age_from="current annual period end",
            identity="Report-local reviewed ticker/CIK only; valid from availability through annual end+550days, not continuous security master."),
        accounting=dict(
            interpretation="Reported consolidated GAAP NI and total assets, not organic or economically adjusted profitability.",
            review="Preserve original context_holds; record separate field-specific decisions with exact source evidence.",
            unresolved="Unresolved changes to selected NI/assets comparability remain purpose-specific holds.",
            KO="Examine retirement-method retrospective adjustment, segment reclassification and held-for-sale notes by actual field scope.",
            irrelevant_flags="Cashflow-only or segment-only warnings do not automatically invalidate consolidated NI/assets.",
            no_keyword_clearance="Neither absence nor presence of generic keywords establishes comparability."),
        independent_review_contract=dict(
            top_level=["passed", "plan_sha256", "extractions_sha256", "input_hashes", "event_reviews"],
            event_flags=["passed", "metadata_verified", "identity_verified", "periods_verified", "currency_verified"],
            field_flags="Nonnull: verified/source_value_present/value_and_sign_verified/units_verified/period_verified/source_locator_verified. Null: independently verified precise cause.",
            population="All10 extracted events and all32 catalog identities; no successful-row subset.",
            source_binding="Exact PLAN, policy, context, raw reports/receipts, pinned calendar and reviewer code/tests.",
            scope="Audit is not admission. Early time holds, unresolved accounting and absent fields still block use."),
        preservation=dict(
            previous_batches="Reference all prior records, admissions, holds, catalogs and3904 stock-month slots unchanged.",
            all_context_holds=True, no_observer_changes=True, no_new_strategy_trials=True,
            first_public_version_verified=False, complete_correction_history_verified=False,
            certified_daily_security_master=False),
        input_hashes={p: shared.sha(p) for p in [
            CONTEXT, "work/after151_next_source_feasibility.json",
            "work/after149_coverage_extension_concept.json",
            "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json", PREDECESSOR,
            "outputs/monthly_financial_coverage_admission_batch2_v1/DATASET.json",
            "outputs/monthly_financial_coverage_admission_batch2_v1/ADMISSION.json"]})


def validate_policy(policy, context, context_sha):
    check(policy["source_scope"]["source_context"] == dict(path=CONTEXT, sha256=context_sha),
          "Changed policy/context binding")
    check(policy["fixed_catalog_metadata"] == catalog_metadata(context), "Changed catalog")
    check(policy["required_fields"] == list(FIELDS), "Changed fields")
    a = policy["periods"]["LRCX_52_53_week_assumption"]
    check(a["enabled"] is True and a["not_a_quoted_fact"] is True and
          a["assumption_id"] == "LRCX_consecutive_annual_boundaries_v1",
          "Changed boundary assumption")
    check(policy["known_time"]["maximum_age_days"] == 550, "Changed source expiry")


def validate_record(value, context):
    for key in ("symbol", "event_id", "fiscal_year"):
        check(value.get(key) == context[key], "Changed identity: " + key)
    for key in ("fact_admitted", "model_admitted", "strategy_admitted"):
        check(value.get(key) is False, "Extraction may not admit facts or strategies")
    check(value.get("source") == context["source"], "Changed source/receipt")
    check(value.get("context_holds") == context["context_holds"], "Lost original holds")
    check(value.get("publication_date") == context["publication_date"] and
          value.get("release_date", context["publication_date"]) == context["publication_date"],
          "Changed publication clock")
    check(value.get("available_from") == context["available_session"], "Changed availability")
    check(set(value["fields"]) == set(FIELDS), "Changed field scope")
    p = value["periods"]
    c = context["periods"]
    check(p["start_basis"] == "derived_same_report_prior_annual_end_plus_one_not_directly_quoted" and
          p["start_assumption_certified"] is False, "Lost derived boundary disclosure")
    start = date.fromisoformat(c["proposed_current_start"])
    end = date.fromisoformat(c["current_end"])
    prior = date.fromisoformat(c["prior_end"])
    days = (end-start).days+1
    check(prior+timedelta(days=1) == start and end.isoformat() == context["official_actual_period_end"],
          "Inconsistent context periods")
    check(days in ((365, 366) if context["symbol"] == "KO" else (364, 371)),
          "Unexpected annual interval")
    if context["symbol"] == "KO":
        check((start.month, start.day, end.month, end.day, prior.month, prior.day) ==
              (1, 1, 12, 31, 12, 31), "Unexpected calendar-year endpoints")
    current = dict(period_start=start.isoformat(), period_end=end.isoformat(), days_inclusive=days)
    check(p["current"] == current and p["comparison"]["period_end"] == prior.isoformat(),
          "Changed extracted periods")
    for name, field in value["fields"].items():
        if field is None:
            check(any(g.get("field") == name and g.get("reason") for g in value.get("gaps", [])),
                  "Null field needs its own explicit gap")
            continue
        check(field.get("fact_admitted") is False, "Premature field admission")
        check(field.get("source") == context["source"], "Changed field source")
        check(field.get("available_from") == context["available_session"], "Changed field clock")
        number, scale = Decimal(str(field["value_decimal"])), Decimal(str(field["scale"]))
        check(number.is_finite() and scale.is_finite() and scale > 0, "Invalid decimal/scale")
        check(field.get("currency_literal") == "$" and "currency_iso" in field and
              field["currency_iso"] is None, "Unsupported currency certification")
        check(isinstance(field.get("displayed_amount"), str) and field["displayed_amount"] and
              bool(field.get("locator")), "Literal/physical locator missing")
        check(field.get("as_of") == (prior.isoformat() if name == "prior_total_assets" else end.isoformat()),
              "Changed field endpoint")
        if name == "annual_net_income":
            check(field["period"] == current, "NI is not current annual period")
        else:
            check(number > 0 and field.get("period") is None, "Invalid instant assets")


def synthetic_tests():
    results = []
    for name in ["test_after151_source_batch", "test_after151_batch_parser_interface"] + [
            "test_" + m for m in MODULES.values()]:
        stream = io.StringIO()
        r = unittest.TextTestRunner(stream=stream, verbosity=0).run(
            unittest.defaultTestLoader.loadTestsFromName(name))
        check(r.wasSuccessful() and r.testsRun > 0, name + "\n" + stream.getvalue())
        results.append(dict(module=name, tests=r.testsRun, passed=True, output=stream.getvalue()))
    return results


def freeze():
    check(shared.sha(CONTEXT) == CONTEXT_SHA, "Changed prepared context")
    context = shared.load(CONTEXT)
    selected = records(context)
    policy = shared.load(POLICY)
    validate_policy(policy, context, CONTEXT_SHA)
    bindings = {CONTEXT: CONTEXT_SHA, POLICY: shared.sha(POLICY)}
    for mapping in (context["input_hashes"], policy["input_hashes"]):
        for path, digest in mapping.items():
            path = shared.relative(path)
            check(shared.sha(path) == digest, "Changed source: " + path)
            bindings[path] = digest
    for event in context["events"]:
        for pkey, skey in (("path", "sha256"), ("receipt_path", "receipt_sha256")):
            path, digest = event["source"][pkey], event["source"][skey]
            check(shared.sha(path) == digest, "Changed catalog evidence")
            bindings[shared.relative(path)] = digest
    for path in (
        "outputs/monthly_financial_coverage_census_v1/PLAN.json",
        "outputs/monthly_financial_coverage_census_v1/CENSUS.json",
        "work/monthly_financial_census_independent_review.json",
        "outputs/monthly_local_linear70/META.json"):
        bindings[path] = shared.sha(path)
    state = shared.load("outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 151 and not state["registered_queue"], "Changed research stage")
    check(state["completed"][-1]["directory"] == "monthly_financial_coverage151" and
          state["completed"][-1]["completion"]["sha256"] == PREDECESSOR_SHA and
          shared.sha(PREDECESSOR) == PREDECESSOR_SHA, "Predecessor not verified")
    files = [__file__, "work/test_after151_source_batch.py",
             "work/test_after151_batch_parser_interface.py"]
    for module in MODULES.values():
        files += ["work/"+module+".py", "work/test_"+module+".py"]
    code = shared.code_closure(files)
    tests = synthetic_tests()
    check(code == shared.code_closure(files), "Code changed during tests")
    plan = dict(
        schema_version=1, created_utc=shared.now(), runtime=shared.runtime(),
        purpose="Source-only KO/LRCX annual NI and same-report annual total assets",
        context=CONTEXT, policy=POLICY, modules=MODULES, fields=list(FIELDS),
        catalog_ref=dict(path=CONTEXT, sha256=CONTEXT_SHA), catalog_events=32,
        catalog_ids=[x["event_id"] for x in context["events"]],
        records=[dict(symbol=x["symbol"], event_id=x["event_id"], fiscal_year=x["fiscal_year"],
                      source_path=x["source"]["path"], source_sha256=x["source"]["sha256"])
                 for x in selected], inputs=bindings, code=code, synthetic_tests=tests,
        no_admission_by_extraction=True, new_strategy_trials=0,
        preserve_all_prior_source_holds=True, preserve_unprocessed_catalog_events=True,
        no_observer_changes=True, no_network=True, no_scores_or_returns=True)
    shared.write_new(OUT/"PLAN.json", plan)
    print(shared.json.dumps(dict(plan=str((OUT/"PLAN.json").relative_to(ROOT)),
                                 sha256=shared.sha(OUT/"PLAN.json"), records=len(selected))))


def run(plan_sha):
    check(shared.sha(OUT/"PLAN.json") == plan_sha, "Changed PLAN")
    plan = shared.load(OUT/"PLAN.json")
    check(plan["runtime"] == shared.runtime(), "Changed runtime")
    bindings = dict(plan["inputs"], **plan["code"])
    for path, digest in bindings.items():
        check(shared.sha(path) == digest, "Changed frozen dependency: " + path)
    check(not (OUT/"EXTRACTIONS.json").exists() and not (OUT/"EXTRACTION_ERRORS.json").exists(),
          "Completed/failed extraction is immutable")
    context = {x["event_id"]: x for x in records(shared.load(CONTEXT))}
    modules = {s: importlib.import_module(m) for s, m in MODULES.items()}
    values, errors = [], []
    for r in plan["records"]:
        try:
            ctx = context[r["event_id"]]
            v = modules[r["symbol"]].extract_document(shared.path(r["source_path"]).read_bytes(), ctx)
            validate_record(v, ctx)
            values.append(v)
        except Exception as exc:
            errors.append(dict(event_id=r["event_id"], error_type=type(exc).__name__, error=str(exc)))
    if errors:
        shared.write_new(OUT/"EXTRACTION_ERRORS.json", dict(
            created_utc=shared.now(), plan_sha256=plan_sha, errors=errors,
            successful_events_not_published=len(values), new_financial_admissions=0,
            new_strategy_trials=0))
        raise ValueError("Extraction errors retained; no partial dataset published")
    check(len(values) == 10, "Incomplete batch")
    for path, digest in bindings.items():
        check(shared.sha(path) == digest, "Dependency changed during extraction")
    result = dict(
        schema_version=1, created_utc=shared.now(), plan_sha256=plan_sha,
        status="source_values_extracted_pending_independent_review",
        catalog_ref=plan["catalog_ref"], catalog_ids=plan["catalog_ids"],
        counts=dict(events=len(values), by_symbol=dict(Counter(x["symbol"] for x in values)),
                    nonnull_fields=sum(x["fields"][f] is not None for x in values for f in FIELDS),
                    unprocessed_catalog_events=22),
        events=values, new_financial_admissions=0, new_strategy_trials=0, ratios_computed=0,
        returns_computed=0, limitations=[
            "Extraction is not independent audit, financial admission or strategy testing.",
            "Original context holds, derived annual-start assumptions and all32 catalog records retained.",
            "First-public vintage, ISO currency and daily security master not certified."])
    shared.write_new(OUT/"EXTRACTIONS.json", result)
    print(shared.json.dumps(dict(result=str((OUT/"EXTRACTIONS.json").relative_to(ROOT)),
                                 sha256=shared.sha(OUT/"EXTRACTIONS.json"), counts=result["counts"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["policy", "tests", "freeze", "extract"])
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.phase == "policy":
        check(shared.sha(CONTEXT) == CONTEXT_SHA, "Changed prepared context")
        ctx = shared.load(CONTEXT)
        records(ctx)
        shared.write_new(POLICY, make_policy(ctx))
        print(shared.json.dumps(dict(policy=POLICY, sha256=shared.sha(POLICY))))
    elif args.phase == "tests":
        print(shared.json.dumps(synthetic_tests()))
    elif args.phase == "freeze":
        freeze()
    else:
        check(args.plan_sha is not None, "Exact plan digest required")
        run(args.plan_sha)

"""Register and extract the next alphabetic source batch, without strategy work."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import importlib
import io
import unittest

import after149_source_batch as shared

ROOT = shared.ROOT
OUT = ROOT / "outputs/monthly_financial_coverage_batch2_v1"
CONTEXT = "work/after150_next_source_context.json"
POLICY = "work/after150_batch2_admission_policy.json"
MODULES = {"INTC": "after150_intc_extract", "KLAC": "after150_klac_extract"}
FIELDS = shared.FIELDS
check = shared.check


def records(context):
    """Preserve the full catalog; extract only the already reviewed HTML scope."""
    docs = context["documents"]
    check(len(docs) == 32 and len({x["event_id"] for x in docs}) == 32,
          "Expected the complete distinct32-event catalog")
    check(Counter(x["symbol"] for x in docs) == {"INTC": 16, "KLAC": 16},
          "Catalog issuer scope changed")
    chosen = [x for x in docs if x.get("html_context_reviewed") is True]
    check(Counter(x["symbol"] for x in chosen) == {"INTC": 3, "KLAC": 16},
          "Expected exactly19 previously reviewed HTML sources")
    return chosen


def source_path(record):
    source = record.get("source", {})
    return source.get("path", record.get("evidence_path"))


def source_sha(record):
    source = record.get("source", {})
    return source.get("sha256", record.get("evidence_sha256"))


def catalog_metadata(context):
    return [dict(
        event_id=x["event_id"], symbol=x["symbol"], fiscal_year=x["fiscal_year"],
        publication_date=x["publication_date"], available_session=x["available_session"],
        official_actual_period_end=x["official_actual_period_end"],
        source_path=shared.relative(source_path(x)), source_sha256=source_sha(x),
        html_context_reviewed=x.get("html_context_reviewed") is True,
        retained_context_gap=x.get("gap"),
    ) for x in context["documents"]]


def validate_policy(policy, context, context_sha):
    check(policy["source_scope"]["source_context"] ==
          dict(path=CONTEXT, sha256=context_sha), "Policy/context linkage changed")
    check(policy["fixed_catalog_metadata"] == catalog_metadata(context),
          "Policy changed the complete catalog")
    check(policy["required_fields"] == list(FIELDS), "Policy field scope changed")


def validate_record(value, context):
    """Validate transport/provenance only; independent admission is a later step."""
    for key in ("symbol", "event_id", "fiscal_year"):
        check(value.get(key) == context[key], "Changed event identity: " + key)
    for key in ("fact_admitted", "model_admitted", "strategy_admitted"):
        check(value.get(key) is False, "Extraction cannot admit facts or strategies")
    expected_path = shared.relative(source_path(context))
    expected_sha = source_sha(context)
    check(shared.relative(value["source"]["path"]) == expected_path and
          value["source"]["sha256"] == expected_sha, "Changed event source")
    check(value.get("available_from") == context["available_session"],
          "Changed source availability")
    check(value.get("publication_date") == context["publication_date"] and
          value.get("release_date", context["publication_date"]) == context["publication_date"],
          "Changed publication date")
    check(set(value["fields"]) == set(FIELDS), "Incorrect field scope")
    holds = value.get("context_holds", [])
    if context["symbol"] == "INTC":
        check("annual_start_not_directly_quoted_requires_source_use_assumption" in holds,
              "Lost INTC boundary hold")
        if value.get("periods"):
            check(value["periods"].get("start_basis") ==
                  "derived_same_report_prior_annual_end_plus_one_not_directly_quoted" and
                  value["periods"].get("start_assumption_certified") is False,
                  "Lost derived boundary metadata")
    if context["symbol"] == "KLAC" and context["fiscal_year"] == 2011:
        check("KLAC_FY2011_preformatted_column_geometry_unverified" in holds and
              all(x is None for x in value["fields"].values()), "Lost PRE geometry hold")
    if context["symbol"] == "KLAC" and context["fiscal_year"] == 2019:
        check("KLAC_FY2019_ASC606_current_comparison_not_restated__average_assets_use_hold"
              in holds, "Lost accounting transition hold")
    for name, field in value["fields"].items():
        if field is None:
            check(any(g.get("field") == name and g.get("reason") for g in value.get("gaps", [])),
                  "Null requires a field-specific gap")
            continue
        check(field.get("fact_admitted") is False, "Field prematurely admitted")
        check(shared.relative(field["source"]["path"]) == expected_path and
              field["source"]["sha256"] == expected_sha, "Field source changed")
        check(field.get("available_from") == context["available_session"],
              "Field clock changed")
        number, scale = Decimal(str(field["value_decimal"])), Decimal(str(field["scale"]))
        check(number.is_finite() and scale.is_finite() and scale > 0,
              "Invalid numeric transport")
        check(isinstance(field.get("displayed_amount"), str) and field["displayed_amount"] and
              bool(field.get("locator")), "Missing literal or physical locator")
        check(field.get("currency_literal") == "$" and "currency_iso" in field and
              field["currency_iso"] is None, "Changed raw currency claim")
        if name == "annual_net_income":
            period = field["period"]
            check(period == value["periods"]["current"], "Field/event annual period differs")
            start, end = (date.fromisoformat(period[k]) for k in ("period_start", "period_end"))
            check(330 <= (end-start).days+1 <= 400 and
                  period["days_inclusive"] == (end-start).days+1,
                  "Not an internally consistent annual period")
            check(end.isoformat() == context["official_actual_period_end"],
                  "NI endpoint differs from catalog")
            check((start-timedelta(days=1)).isoformat() ==
                  value["periods"]["comparison"]["period_end"], "Annual boundary mismatch")
        else:
            check(number > 0, "Nonpositive assets")
            date.fromisoformat(field["as_of"])
        if name == "current_total_assets":
            check(field["as_of"] == context["official_actual_period_end"],
                  "Current assets endpoint differs")
        if name == "prior_total_assets":
            check(field["as_of"] == value["periods"]["comparison"]["period_end"] ==
                  context["context_summary"]["comparison_assets_date"],
                  "Prior assets endpoint differs")


def synthetic_tests():
    output = []
    for module in ["test_after150_source_batch", "test_after150_batch_parser_interface"] + [
            "test_" + x for x in MODULES.values()]:
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(
            unittest.defaultTestLoader.loadTestsFromName(module))
        check(result.wasSuccessful() and result.testsRun > 0, stream.getvalue())
        output.append(dict(module=module, tests=result.testsRun, passed=True,
                           output=stream.getvalue()))
    return output


def freeze():
    context = shared.load(CONTEXT)
    chosen = records(context)
    bindings = {CONTEXT: shared.sha(CONTEXT), POLICY: shared.sha(POLICY)}
    policy = shared.load(POLICY)
    validate_policy(policy, context, bindings[CONTEXT])
    for name, digest in policy["input_hashes"].items():
        name = shared.relative(name)
        check(shared.sha(name) == digest, "Policy source changed: " + name)
        bindings[name] = digest
    for name, digest in context["input_hashes"].items():
        name = shared.relative(name)
        check(shared.sha(name) == digest, "Context source changed: " + name)
        bindings[name] = digest
    for record in chosen:
        name = shared.relative(source_path(record))
        check(shared.sha(name) == source_sha(record), "HTML source changed")
        bindings[name] = source_sha(record)
    for name in (
        "outputs/monthly_financial_coverage_census_v1/PLAN.json",
        "outputs/monthly_financial_coverage_census_v1/CENSUS.json",
        "work/after149_coverage_extension_concept.json",
        "work/monthly_financial_census_independent_review.json",
        "outputs/monthly_financial_coverage150/COMPLETION.json",
        "work/after150_batch2_admission_policy_review.json",
        "outputs/monthly_financial_coverage_admission_batch1_v1/DATASET.json",
        "outputs/monthly_financial_coverage_admission_batch1_v1/ADMISSION.json",
        "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json",
        "outputs/monthly_local_linear70/META.json",
    ):
        bindings[name] = shared.sha(name)
    state = shared.load("outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 150 and not state["registered_queue"],
          "Expected completed150 and no registered trial")
    check(state["completed"][-1]["directory"] == "monthly_financial_coverage150" and
          state["completed"][-1]["completion"]["sha256"] ==
          bindings["outputs/monthly_financial_coverage150/COMPLETION.json"],
          "Unexpected preceding experiment or completion")
    files = [__file__, "work/test_after150_source_batch.py",
             "work/test_after150_batch_parser_interface.py"]
    for module in MODULES.values():
        files += ["work/" + module + ".py", "work/test_" + module + ".py"]
    code = shared.code_closure(files)
    tests = synthetic_tests()
    check(code == shared.code_closure(files), "Implementation changed during tests")
    plan = dict(
        schema_version=1, created_utc=shared.now(), runtime=shared.runtime(),
        purpose="Source-only INTC/KLAC annual NI and same-report assets extraction",
        selection="Next alphabetic existing-HTML issuers; no holdings/return-based selection",
        context=CONTEXT, policy=POLICY, modules=MODULES, fields=list(FIELDS),
        catalog_events=32,
        catalog_ref=dict(path=CONTEXT, sha256=bindings[CONTEXT]),
        catalog_ids=[x["event_id"] for x in context["documents"]],
        records=[dict(symbol=x["symbol"], event_id=x["event_id"], fiscal_year=x["fiscal_year"],
                      source_path=shared.relative(source_path(x)), source_sha256=source_sha(x))
                 for x in chosen],
        inputs=bindings, code=code, synthetic_tests=tests,
        no_admission_by_extraction=True, new_strategy_trials=0,
        preserve_all_prior_source_holds=True, preserve_unprocessed_catalog_events=True,
        no_observer_changes=True, no_network=True, no_scores_or_returns=True)
    shared.write_new(OUT / "PLAN.json", plan)
    print(shared.json.dumps(dict(plan=str((OUT/"PLAN.json").relative_to(ROOT)),
                                 sha256=shared.sha(OUT/"PLAN.json"), records=len(chosen))))


def run(plan_sha):
    check(shared.sha(OUT/"PLAN.json") == plan_sha, "Changed PLAN")
    plan = shared.load(OUT/"PLAN.json")
    check(plan["runtime"] == shared.runtime(), "Changed runtime")
    bindings = dict(plan["inputs"], **plan["code"])
    for path, digest in bindings.items():
        check(shared.sha(path) == digest, "Changed frozen dependency: " + path)
    check(not (OUT/"EXTRACTIONS.json").exists() and not (OUT/"EXTRACTION_ERRORS.json").exists(),
          "Do not rerun a completed or failed extraction")
    context = {x["event_id"]: x for x in records(shared.load(plan["context"]))}
    modules = {symbol: importlib.import_module(name) for symbol, name in MODULES.items()}
    values, errors = [], []
    for record in plan["records"]:
        try:
            ctx = context[record["event_id"]]
            value = modules[record["symbol"]].extract_document(
                shared.path(record["source_path"]).read_bytes(), ctx)
            validate_record(value, ctx)
            values.append(value)
        except Exception as exc:
            errors.append(dict(event_id=record["event_id"], symbol=record["symbol"],
                               error_type=type(exc).__name__, error=str(exc)))
    if errors:
        shared.write_new(OUT/"EXTRACTION_ERRORS.json",
                         dict(created_utc=shared.now(), plan_sha256=plan_sha, errors=errors,
                              successful_events_not_published=len(values),
                              new_financial_admissions=0, new_strategy_trials=0))
        raise ValueError("Recorded extraction errors; no partial dataset published")
    check(len(values) == 19, "Incomplete extraction")
    for path, digest in bindings.items():
        check(shared.sha(path) == digest, "Source changed during extraction")
    result = dict(
        schema_version=1, created_utc=shared.now(), plan_sha256=plan_sha,
        status="source_values_extracted_pending_independent_review",
        catalog_ref=plan["catalog_ref"], catalog_ids=plan["catalog_ids"],
        counts=dict(events=len(values), by_symbol=dict(Counter(x["symbol"] for x in values)),
                    nonnull_fields=sum(x["fields"][f] is not None for x in values for f in FIELDS)),
        events=values, new_financial_admissions=0, new_strategy_trials=0,
        ratios_computed=0, returns_computed=0,
        limitations=["Extraction is not independent review or admission.",
                     "Derived annual boundaries remain explicitly identified assumptions.",
                     "Unprocessed sources and all original holds remain.",
                     "First-public versions, ISO currency and daily security identity are not certified."])
    shared.write_new(OUT/"EXTRACTIONS.json", result)
    print(shared.json.dumps(dict(result=str((OUT/"EXTRACTIONS.json").relative_to(ROOT)),
                                 sha256=shared.sha(OUT/"EXTRACTIONS.json"), counts=result["counts"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else run(args.plan_sha)

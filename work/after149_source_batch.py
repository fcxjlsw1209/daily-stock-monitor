"""Freeze and extract a source-only annual financial batch; no strategy work."""
import argparse
import ast
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage_batch1_v1"
CONTEXTS = {
    "AMZN": "work/after149_amzn_source_context.json",
    "CSCO": "work/after149_csco_source_context.json",
}
MODULES = {"AMZN": "after149_amzn_extract", "CSCO": "after149_csco_extract"}
FIELDS = ("annual_net_income", "current_total_assets", "prior_total_assets")
BASE_INPUTS = [
    "outputs/monthly_financial_coverage_census_v1/PLAN.json",
    "outputs/monthly_financial_coverage_census_v1/CENSUS.json",
    "work/monthly_financial_census_independent_review.json",
    "work/after149_coverage_extension_concept.json",
    "work/after149_batch1_admission_policy.json",
    "outputs/monthly_earnings_repair/EVENTS.json",
    "outputs/monthly_earnings_repair/VALIDATION.json",
    "outputs/monthly_earnings_repair/SCOPED_READINESS.json",
    "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json",
    "outputs/monthly_local_linear70/META.json",
    "outputs/monthly_stock_comp_source_admission_v2/DATASET.json",
    "outputs/monthly_stock_comp_source_admission_v2/ADMISSION.json",
]


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    check(not p.is_symlink(), "Symlink input: " + str(p))
    p = p.resolve()
    p.relative_to(ROOT)
    check("CloudStorage" not in p.parts, "Cloud input")
    return p


def sha(p):
    return hashlib.sha256(path(p).read_bytes()).hexdigest()


def relative(p):
    return str(path(p).relative_to(ROOT))


def load(p):
    return json.loads(path(p).read_bytes())


def write_new(p, obj):
    path(p).parent.mkdir(parents=True, exist_ok=True)
    with path(p).open("xb") as stream:
        stream.write((json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                 indent=2, allow_nan=False) + "\n").encode())


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    return {"python": sys.version, "beautifulsoup4": importlib.metadata.version("beautifulsoup4")}


def code_closure(files):
    """Bind directly imported local helpers recursively; no imports executed."""
    pending, found = list(files), {}
    while pending:
        item = relative(pending.pop())
        if item in found:
            continue
        raw = path(item).read_bytes()
        found[item] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            candidate = ROOT / "work" / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                pending.append(candidate)
    return found


def synthetic_tests():
    records = []
    for module in ["test_after149_source_batch"] + ["test_" + m for m in MODULES.values()]:
        suite = unittest.defaultTestLoader.loadTestsFromName(module)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        check(result.wasSuccessful() and result.testsRun > 0,
              "Synthetic test failure in " + module + ": " + stream.getvalue())
        records.append({"module": module, "tests": result.testsRun, "passed": True,
                        "output": stream.getvalue()})
    return records


def context_records():
    result, bindings = [], {}
    for symbol, source in CONTEXTS.items():
        context = load(source)
        bindings[source] = sha(source)
        supplied = context.get("input_hashes", context.get("source_hashes"))
        check(isinstance(supplied, dict), "Context missing input pins")
        for name, expected in supplied.items():
            name = relative(name)
            check(sha(name) == expected, "Changed context evidence: " + name)
            bindings[name] = expected
        records = context["documents"] if symbol == "AMZN" else context["events"]
        for record in records:
            raw = record if symbol == "AMZN" else record["source"]
            source_path = relative(raw["path"])
            expected = raw["sha256"]
            check(sha(source_path) == expected, "Changed raw source: " + source_path)
            bindings[source_path] = expected
            receipt = record["receipt"] if symbol == "AMZN" else raw["receipt_path"]
            receipt = relative(receipt)
            bindings[receipt] = sha(receipt)
            result.append({"symbol": symbol, "event_id": record["event_id"],
                           "fiscal_year": record["fiscal_year"],
                           "context_path": source, "source_path": source_path,
                           "source_sha256": expected, "receipt_path": receipt})
    check(len(result) == 26, "Expected exactly26 context-reviewed sources")
    check(len({r["event_id"] for r in result}) == 26, "Duplicate event")
    return result, bindings


def freeze():
    records, bindings = context_records()
    for p in BASE_INPUTS:
        bindings[p] = sha(p)
    review = load(BASE_INPUTS[2])
    check(review["passed"] is True, "Independent census review missing")
    check(review["census_sha256"] == sha(BASE_INPUTS[1]), "Census review mismatch")
    state = load("outputs/monthly_target70_research/CONTINUATION_STATUS.json")
    check(len(state["completed"]) == 149 and not state["registered_queue"],
          "Research state changed; inspect before freezing")
    files = [__file__, "work/test_after149_source_batch.py"]
    for module in MODULES.values():
        files += ["work/" + module + ".py", "work/test_" + module + ".py"]
    code = code_closure(files)
    tests = synthetic_tests()
    check(code == code_closure(files), "Code changed during synthetic tests")
    plan = {
        "schema_version": 1, "created_utc": now(), "runtime": runtime(),
        "purpose": "Source-only annual NI and same-report asset extraction, no new factor or backtest",
        "selection": "First alphabetic new issuers with existing reviewed primary HTML context; no return selection",
        "records": records, "inputs": bindings, "code": code,
        "synthetic_tests": tests,
        "modules": MODULES, "fields": FIELDS,
        "policy": {
            "current_annual_NI": "Consolidated GAAP net income starting the annual indirect cashflow statement",
            "assets": "Current and comparative total assets in the same original report, never assembled across later releases",
            "currency": "Retain literal common-dollar evidence; do not infer an ISO currency or perform FX conversion",
            "known_time": "All fields share the current release next-session availability; early calendar gaps explicitly retained",
            "latest_event": "This extraction does not select monthly events; eventual use selects latest known before validity and age checks",
            "comparability_holds": ["AMZN FY2019 lease transition",
                                    "CSCO FY2019 ASC606 transition",
                                    "Any unresolved source/period/value/identity/clock ambiguity"],
            "no_false_missing": "Cashflow/SBC-only classification flags do not automatically invalidate NI/assets",
            "prior_hold_preservation": "All existing44 source records and13 prior holds remain unchanged",
            "all_universe_retained": "The remaining73-stock scope, unprocessed reports and PDFs remain gaps, not excluded stocks",
            "no_admission_by_extraction": True,
            "no_network": True, "no_scores": True, "no_returns": True,
            "no_observer_changes": True,
        },
    }
    write_new(OUT / "PLAN.json", plan)
    print(json.dumps({"plan": str((OUT / "PLAN.json").relative_to(ROOT)),
                      "sha256": sha(OUT / "PLAN.json"), "records": len(records)}))


def validate_record(record, expected=None, context=None):
    """Checks transport safety, not financial admission."""
    check(record.get("symbol") in MODULES, "Unknown symbol")
    check(isinstance(record.get("event_id"), str), "No event_id")
    check(record.get("fact_admitted") is False, "Unexpected fact admission")
    check(record.get("model_admitted") is False, "Unexpected model admission")
    check(record.get("strategy_admitted") is False, "Unexpected strategy admission")
    fields = record.get("fields")
    check(isinstance(fields, dict) and set(fields) == set(FIELDS), "Wrong field keys")
    if expected is not None:
        check(record["symbol"] == expected["symbol"] and
              record["event_id"] == expected["event_id"] and
              record["fiscal_year"] == expected["fiscal_year"], "Identity mismatch")
        check(record["source"]["sha256"] == expected["source_sha256"] and
              relative(record["source"]["path"]) == expected["source_path"], "Source mismatch")
        check(record["available_from"] == context["available_session"], "Availability mismatch")
        periods = context.get("periods")
        ends = ({"current": periods["current"]["period_end"],
                 "prior": periods["comparison"]["period_end"]} if periods else
                {"current": context["current_end"], "prior": context["comparison_end"]})
    for name in FIELDS:
        value = fields[name]
        if value is None:
            check(bool(record.get("gaps")), "Missing field without explicit gap")
            continue
        number, scale = Decimal(str(value["value_decimal"])), Decimal(str(value["scale"]))
        check(number.is_finite() and scale.is_finite() and scale > 0, "Nonfinite or invalid scale")
        check(isinstance(value.get("displayed_amount"), str), "No raw amount")
        check(bool(value.get("locator")), "No source locator")
        check(value.get("fact_admitted") is False, "Unexpected field admission")
        if expected is not None:
            check(value["source"]["sha256"] == expected["source_sha256"] and
                  relative(value["source"]["path"]) == expected["source_path"], "Field source mismatch")
            check(value["available_from"] == context["available_session"], "Field availability mismatch")
            wanted = ends["prior" if name == "prior_total_assets" else "current"]
            actual = value.get("as_of") or (value.get("period") or {}).get("period_end")
            check(actual == wanted, "Field endpoint mismatch")
            if name == "annual_net_income" and periods:
                check(value["period"] == periods["current"], "NI annual period mismatch")
            elif name == "annual_net_income":
                current = date.fromisoformat(ends["current"])
                start = current.replace(month=1, day=1)
                wanted_period = {"period_start": start.isoformat(),
                                 "period_end": current.isoformat(),
                                 "days_inclusive": (current - start).days + 1}
                check(value["period"] == wanted_period, "AMZN annual period mismatch")
        if name != "annual_net_income":
            check(number > 0, "Nonpositive total assets")


def run(expected):
    check(sha(OUT / "PLAN.json") == expected, "Wrong PLAN digest")
    plan = load(OUT / "PLAN.json")
    check(plan["runtime"] == runtime(), "Runtime changed")
    for name, expected_sha in {**plan["inputs"], **plan["code"]}.items():
        check(sha(name) == expected_sha, "Frozen dependency changed: " + name)
    check(not (OUT / "EXTRACTIONS.json").exists(), "Extraction already completed")
    modules = {s: importlib.import_module(m) for s, m in plan["modules"].items()}
    contexts = {s: load(p) for s, p in CONTEXTS.items()}
    extraction, errors = [], []
    for row in plan["records"]:
        symbol = row["symbol"]
        records = contexts[symbol]["documents" if symbol == "AMZN" else "events"]
        record = next(r for r in records if r["event_id"] == row["event_id"])
        try:
            value = modules[symbol].extract_document(path(row["source_path"]).read_bytes(), record)
            validate_record(value, row, record)
            extraction.append(value)
        except Exception as exc:
            errors.append({"symbol": symbol, "event_id": row["event_id"],
                           "error_type": type(exc).__name__, "error": str(exc)})
    if errors:
        write_new(OUT / "EXTRACTION_ERRORS.json", {
            "created_utc": now(), "plan_sha256": expected,
            "errors": errors, "successful_events_not_published": len(extraction),
            "financial_admissions": 0, "new_strategy_trials": 0})
        raise ValueError("Extraction errors recorded; no partial dataset published")
    check(len(extraction) == 26, "Missing extraction")
    # Revalidate inputs before publication; no protected bytes may be changed.
    for name, expected_sha in {**plan["inputs"], **plan["code"]}.items():
        check(sha(name) == expected_sha, "Dependency changed during extraction: " + name)
    result = {
        "schema_version": 1, "created_utc": now(), "plan_sha256": expected,
        "status": "source_values_extracted_pending_independent_review",
        "counts": {"events": len(extraction),
                   "by_symbol": dict(Counter(r["symbol"] for r in extraction)),
                   "nonnull_fields": sum(r["fields"][f] is not None for r in extraction for f in FIELDS)},
        "events": extraction, "new_financial_admissions": 0, "new_strategy_trials": 0,
        "ratios_computed": 0, "returns_computed": 0,
        "limitations": [
            "Extraction is not independent validation or financial/factor admission.",
            "First-public source vintages and complete correction history are not certified.",
            "The same-report dollar basis does not certify ISO currency.",
            "All existing data holds and unprocessed source gaps are preserved.",
        ],
    }
    write_new(OUT / "EXTRACTIONS.json", result)
    print(json.dumps({"result": str((OUT / "EXTRACTIONS.json").relative_to(ROOT)),
                      "sha256": sha(OUT / "EXTRACTIONS.json"), "counts": result["counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run"])
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else run(args.plan_sha)

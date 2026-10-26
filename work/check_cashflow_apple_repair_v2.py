"""Independent source-value check against the root's prior pixel inspection.

This checker never supplies values to an extractor or admits financial facts.
"""
import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
V1 = Path("outputs/monthly_cashflow_source_preflight_v1")
V2 = Path("outputs/monthly_cashflow_pdf_repair_v2")
VISUAL_SHA = "1610b77c1ffa81ceb9727eeaf70b84a1b13feb4370fd47eb93f793142b79a26f"
V1_RAW_SHA = "4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87"
FIELDS = {
    "total_assets": "visually_observed_total_assets",
    "net_income": "visually_observed_cash_flow_starting_net_income",
    "operating_cash_flow": "visually_observed_cash_generated_by_operating_activities",
}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load(path, expected=None):
    if expected is not None:
        assert sha(path) == expected, str(path)
    return json.loads((ROOT / path).read_text())


def main(plan_sha):
    assert len(plan_sha) == 64
    visual = load(V1 / "APPLE_VISUAL_REVIEW.json", VISUAL_SHA)
    original = load(V1 / "RAW_FACTS.json", V1_RAW_SHA)
    plan = load(V2 / "PLAN.json", plan_sha)
    assert plan["stage"] == "frozen_cashflow_pdf_repair_plan"
    for group in ("source_hashes", "source_code_hashes"):
        for path, expected in plan[group].items():
            assert sha(path) == expected, path
    manifest = load(V2 / "MANIFEST.json")
    assert manifest["plan_sha256"] == plan_sha
    for filename, expected in manifest["output_hashes"].items():
        assert sha(V2 / filename) == expected, filename
    data = load(V2 / "RAW_FACTS.json")
    assert data["plan_sha256"] == plan_sha
    old = {x["event_id"]: x for x in original["events"] if x["symbol"] == "AAPL"}
    seen = set()
    checked = []
    expected_rows = {x["event_id"]: x for x in visual["events"]}
    assert len(expected_rows) == len(data["events"]) == len(old) == 10
    for event in data["events"]:
        eid = event["event_id"]
        assert eid not in seen and eid in old and eid in expected_rows
        seen.add(eid)
        observed = expected_rows[eid]
        assert event["source"] == old[eid]["source"] == observed["source"]
        assert sha(event["source"]["path"]) == event["source"]["sha256"]
        for name in ("available_from", "release_date", "periods", "identity",
                     "inherited_rejection_codes"):
            assert event[name] == old[eid][name], (eid, name)
        for flag in ("fact_admitted", "model_admitted", "strategy_admitted",
                     "first_public_version_verified", "complete_correction_history_verified",
                     "certified_daily_security_master"):
            assert event[flag] is False, (eid, flag)
        for column, side in enumerate(("current", "comparison")):
            for field, observation_field in FIELDS.items():
                cell = event["fields"][side][field]
                assert cell is not None, (eid, side, field, "still missing")
                assert Decimal(cell["value_decimal"]) == Decimal(observed[observation_field][column]), (eid, side, field)
                assert Decimal(cell["scale"]) == Decimal(1000000)
                assert cell["currency_literal"] == "$" and cell["currency_iso"] is None
                assert cell["fact_admitted"] is False
                assert cell["pdfium_text_agrees"] is True
                assert cell["available_from"] == event["available_from"]
                assert cell["period"] == event["periods"][side]
                loc = cell["locator"]
                assert loc["period_end"] == observed["period_end_dates"][column]
                assert loc["financial_column_index_0based"] == column
                assert loc["page_1based"] == (2 if field == "total_assets" else 3)
                geo = cell["pdfium_geometry"]
                assert geo["selected_cell"] == geo["pdfium_all_row_cells"][column]
                checked.append({"event_id": eid, "side": side, "field": field,
                                "displayed_amount": cell["displayed_amount"],
                                "value_decimal": cell["value_decimal"],
                                "visual_match": True})
    assert len(checked) == 60 and seen == set(expected_rows)
    inputs = {str(p): sha(p) for p in (
        V1 / "APPLE_VISUAL_REVIEW.json", V1 / "RAW_FACTS.json",
        V2 / "PLAN.json", V2 / "RAW_FACTS.json", V2 / "AUDIT.json", V2 / "MANIFEST.json",
        Path("work/check_cashflow_apple_repair_v2.py"),
    )}
    result = {
        "stage": "independent_raw_field_visual_crosscheck_not_admission",
        "passed": True, "events": 10, "cells_checked": 60,
        "checks": checked, "input_hashes": inputs,
        "original_availability_periods_identity_and_rejection_codes_preserved": True,
        "source_admitted": False, "accounting_review_still_separate": True,
        "first_public_version_verified": False,
        "financial_ratios_computed": 0, "rankings_or_returns_computed": 0,
    }
    target = ROOT / V2 / "INDEPENDENT_VISUAL_CHECK.json"
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    assert inputs == {p: sha(p) for p in inputs}
    print(json.dumps({"passed": True, "cells_checked": len(checked),
                      "output": str(target.relative_to(ROOT)), "sha256": sha(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha", required=True)
    main(parser.parse_args().plan_sha)

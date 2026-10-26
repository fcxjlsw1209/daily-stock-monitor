"""Read-only independent source/clock review; only writes its review JSON."""
from collections import Counter
from datetime import date, datetime, timezone
from html.parser import HTMLParser
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/after155_event_source_independent_review.json"
MSFT = "outputs/monthly_msft_event_evidence156/"
CLOCK = "outputs/monthly_consumed_clock156/"
MSFT_SPEC_SHA = "a24ec71fa8a5773350cab49b65cc14dcfca03f6137f8f9802caa6d9c708dfc98"
PINS = {}
DATE_LITERAL = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def path(name):
    p = Path(name)
    p = p if p.is_absolute() else ROOT / p
    check(p.is_relative_to(ROOT) and p.resolve() == p, "Nonlocal source")
    check("CloudStorage" not in p.parts, "Cloud forbidden")
    return p


def raw(name):
    p = path(name)
    before = p.stat()
    data = p.read_bytes()
    after = p.stat()
    check((before.st_ino, before.st_mtime_ns, before.st_size) ==
          (after.st_ino, after.st_mtime_ns, after.st_size), "Concurrent source change")
    key = str(p.relative_to(ROOT))
    observed = hashlib.sha256(data).hexdigest()
    check(key not in PINS or PINS[key] == observed, "Source changed within review")
    PINS[key] = observed
    return data


def read(name):
    return json.loads(raw(name))


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.ignored -= 1

    def handle_data(self, value):
        if not self.ignored:
            self.parts.append(value)


def independent_page(data):
    parser = VisibleText()
    parser.feed(data.decode("utf-8-sig"))
    text = " ".join(" ".join(parser.parts).split())
    title = re.search(r"\bEarnings\s+Release\s+FY(\d\d)\s+Q([1-4])\b", text, re.I)
    check(title is not None, "Missing fiscal title")
    dateline = re.search(r"REDMOND,\s*Wash\.\s*[\u2014\u2013-]?\s*(" + DATE_LITERAL + ")", text, re.I)
    check(dateline is not None, "Missing Redmond original dateline")
    after = text[dateline.end():dateline.end()+2000]
    period = re.search(r"\b(?:quarter|fiscal year)\s+ended\s+(" + DATE_LITERAL + ")", after, re.I)
    check(period is not None, "Missing original results period")
    parse = lambda value: datetime.strptime(value, "%B %d, %Y").date().isoformat()
    return {
        "fiscal_year": int(title.group(1)) + 2000, "fiscal_quarter": int(title.group(2)),
        "title": title.group(), "publication_date": parse(dateline.group(1)),
        "period_end": parse(period.group(1)), "dateline_literal": dateline.group(1),
        "period_literal": period.group(1), "visible_text_characters": len(text),
        "has_pressreleasecontent_container": b"pressreleasecontent" in data,
        "has_income_statement_marker": bool(re.search(r"(?:income statements?|statements? of income)", text, re.I)),
    }


def independent_clock(event, schedule):
    """Separate pandas localization + linear date/close filter, no producer import."""
    publication_day = pd.Timestamp(event["publication_date"])
    check(pd.Timestamp(event["official_actual_period_end"]) <= publication_day, "Future fiscal period")
    zone = event["publication_timezone"]
    if event["precision"] == "date_only":
        midnight = (publication_day + pd.DateOffset(days=1)).tz_localize(
            zone, ambiguous="raise", nonexistent="raise")
        upper = midnight.tz_convert("UTC") - pd.Timedelta(1, unit="ns")
        candidates = [(day, close) for day, close in schedule
                      if day > event["publication_date"] and close > upper]
        check(bool(candidates), "Exhausted calendar")
        day, close = candidates[0]
        return {"available_session": day, "known_ns": close.value, "upper_ns": upper.value}
    check(event["precision"] == "exact", "Unknown timestamp precision")
    stamp = pd.Timestamp(event["publication_at"])
    check(stamp.tz is not None, "Naive exact timestamp")
    check(str(stamp.tz_convert(zone).date()) == event["publication_date"], "Exact local day disagreement")
    stamp = stamp.tz_convert("UTC")
    candidates = [(day, close) for day, close in schedule if close >= stamp]
    check(bool(candidates), "Exhausted exact timestamp calendar")
    return {"available_session": candidates[0][0], "known_ns": stamp.value, "upper_ns": None}


def synthetic_checks(schedule):
    def record(day):
        return dict(precision="date_only", publication_date=day,
                    publication_timezone="America/Los_Angeles", official_actual_period_end="2023-12-31")
    check(independent_clock(record("2024-03-10"), schedule)["upper_ns"] ==
          pd.Timestamp("2024-03-11T07:00:00Z").value-1, "Spring DST")
    check(independent_clock(record("2024-11-03"), schedule)["upper_ns"] ==
          pd.Timestamp("2024-11-04T08:00:00Z").value-1, "Fall DST")
    check(independent_clock(record("2024-11-28"), schedule)["known_ns"] ==
          pd.Timestamp("2024-11-29T18:00:00Z").value, "Early close")
    check(independent_clock(record("2024-11-29"), schedule)["available_session"] ==
          "2024-12-02", "Weekend")
    check(independent_clock(record("2025-01-08"), schedule)["available_session"] ==
          "2025-01-10", "Official 2025 closure")
    precise = record("2024-11-29")
    precise.update(precision="exact", publication_at="2024-11-29T18:00:00Z")
    check(independent_clock(precise, schedule)["available_session"] == "2024-11-29", "Exact close boundary")
    precise["publication_at"] = "2024-11-29T18:00:00.000000001Z"
    check(independent_clock(precise, schedule)["available_session"] == "2024-12-02", "One ns after close")
    fake = ("<h3>Earnings Release FY20 Q2</h3><p>REDMOND, Wash. — January 29, 2020 — "
            "Microsoft results for the quarter ended December 31, 2019.</p>")
    parsed = independent_page(fake.encode())
    check(parsed["publication_date"] == "2020-01-29" and parsed["period_end"] == "2019-12-31", "Independent HTML parsing")
    return 8


def validate_product(prefix, expected_spec=None):
    spec = read(prefix+"SPEC.json")
    if expected_spec:
        check(PINS[prefix+"SPEC.json"] == expected_spec, "Unexpected spec")
    for name, expected in spec["hashes"].items():
        check(hashlib.sha256(raw(name)).hexdigest() == expected, "Pinned source mismatch: "+name)
    manifest = read(prefix+"MANIFEST.json")
    check(manifest["spec_sha256"] == PINS[prefix+"SPEC.json"], "Manifest spec binding")
    for name, expected in manifest["outputs"].items():
        check(hashlib.sha256(raw(prefix+name)).hexdigest() == expected, "Product hash mismatch: "+name)
    return spec


def main():
    raw(__file__)
    msft_spec = validate_product(MSFT, MSFT_SPEC_SHA)
    clock_spec = validate_product(CLOCK)
    evidence = read(MSFT+"EVENT_EVIDENCE.json")
    summary = read(MSFT+"SUMMARY.json")
    clock_rows = read(CLOCK+"EVENT_CLOCKS.json")
    clock_summary = read(CLOCK+"SUMMARY.json")
    frozen = read("outputs/monthly_paired_target143/CALENDAR.json")
    mod_path = path("outputs/monthly_earnings_repair/research_calendar.py")
    imported = importlib.util.spec_from_file_location("review_only_calendar", mod_path)
    calendar_module = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(calendar_module)
    calendar = calendar_module.market_calendar(frozen["start"], frozen["end"])
    schedule = [(str(day.date()), close) for day, close in zip(
        calendar.schedule.index, calendar.schedule["close"])]
    check([day for day, _ in schedule] == frozen["sessions"], "Session axis")
    checks = synthetic_checks(schedule)
    parent = read("outputs/monthly_earnings_repair/EVENTS.json")
    added = read("outputs/monthly_event_patch156/ADDED_EVENTS.json")
    all_events = {r["event_id"]: r for r in parent + added}
    check(len(all_events) == len(parent+added) == 3775, "Event uniqueness")
    slots = read("outputs/monthly_event_scope156/SLOTS.json")
    scope_manifest = read("outputs/monthly_event_scope156/MANIFEST.json")
    for name in ("SLOTS.json", "INTERVALS.json"):
        p = "outputs/monthly_event_scope156/"+name
        check(hashlib.sha256(raw(p)).hexdigest() == scope_manifest["output_hashes"][name], "Original scope binding")
    msft_slots = [r for r in slots if r["symbol"] == "MSFT"]
    check(len(msft_slots) == len({r["slot_id"] for r in msft_slots}) == 122, "MSFT 122 original slots")
    consumed_ids = {eid for s in msft_slots for eid in s["existing_event_ids"]}
    interval = [r for r in read("outputs/monthly_event_scope156/INTERVALS.json") if r["symbol"] == "MSFT"]
    check(len(interval) == 1, "MSFT single interval")
    check(consumed_ids == set(interval[0]["existing_event_ids"]), "Independent slot-to-interval union")
    check(set(interval[0]["source_slot_ids"]) == {r["slot_id"] for r in msft_slots}, "Original slot identities")
    check(len(evidence) == len({r["event_id"] for r in evidence}) == 48, "48 unique pages")
    actual_pages = []
    for row in evidence:
        source = row["source"]
        data = raw(source["path"])
        check(hashlib.sha256(data).hexdigest() == source["sha256"], "Page source hash")
        receipt = read(source["receipt"])
        check(receipt["sha256"] == source["sha256"] and receipt["http_status"] == 200, "Original receipt")
        check(pd.Timestamp(receipt["retrieved_at"]).tz is not None, "Receipt retrieval clock")
        parsed = independent_page(data)
        y, q = parsed["fiscal_year"], parsed["fiscal_quarter"]
        check((y,q) == (row["fiscal_year"], row["fiscal_quarter"]), "Fiscal heading")
        expected_end = [date(y-1,9,30), date(y-1,12,31), date(y,3,31), date(y,6,30)][q-1]
        check(parsed["period_end"] == str(expected_end), "MSFT fiscal period")
        check(parsed["period_end"] <= parsed["publication_date"], "Period before publication")
        expected_url = f"https://www.microsoft.com/en-us/Investor/earnings/FY-{y}-Q{q}/press-release-webcast"
        check(source["url"] == receipt["url"] == expected_url, "Issuer canonical source URL")
        check(row["event_id"] == "MSFT_"+parsed["publication_date"], "Independent event id")
        check(parsed["dateline_literal"] == row["identity_evidence"]["dateline_literal"] and
              parsed["period_literal"] == row["identity_evidence"]["period_end_literal"], "Source literals")
        check(parsed["has_pressreleasecontent_container"] and parsed["visible_text_characters"] > 1000,
              "Full cached results-page structure")
        event = dict(publication_date=parsed["publication_date"], official_actual_period_end=parsed["period_end"],
                     publication_timezone="America/Los_Angeles", precision="date_only")
        own_clock = independent_clock(event, schedule)
        observed = row["clock"]
        check(observed["publication_date"] == parsed["publication_date"] and
              observed["official_actual_period_end"] == parsed["period_end"] and
              observed["publication_timezone"] == "America/Los_Angeles" and
              observed["precision"] == "date_only", "Clock identity")
        check(own_clock == dict(available_session=observed["available_session"],
              known_ns=pd.Timestamp(observed["known_by_at"]).value,
              upper_ns=pd.Timestamp(observed["date_only_upper_bound_at"]).value), "Independent MSFT clock")
        prior = all_events.get(row["event_id"])
        check(bool(prior) == row["parent_event_present"], "Parent event membership")
        if prior:
            for key in observed:
                check(prior[key] == observed[key], "Parent metadata mismatch: "+key)
        check(row["consumed"] == (row["event_id"] in consumed_ids), "Consumed flag")
        check(row["first_public_time_verified"] is False, "First-public overclaim")
        actual_pages.append(dict(event_id=row["event_id"], fiscal_year=y, fiscal_quarter=q,
            path=source["path"], sha256=source["sha256"], receipt=source["receipt"],
            title=parsed["title"], dateline_literal=parsed["dateline_literal"],
            period_literal=parsed["period_literal"], release_date=parsed["publication_date"],
            period_end=parsed["period_end"], available_session=own_clock["available_session"],
            consumed=row["consumed"], source_has_full_results_container=True,
            has_income_statement_marker=parsed["has_income_statement_marker"]))
    check({(r["fiscal_year"],r["fiscal_quarter"]) for r in actual_pages} ==
          {(y,q) for y in range(2015,2027) for q in range(1,5)}, "Declared fiscal source scope")
    consumed = [r for r in actual_pages if r["consumed"]]
    check(len(consumed) == 44 and {r["event_id"] for r in consumed} == consumed_ids, "44 consumed pages")
    ordinals = sorted(4*r["fiscal_year"]+r["fiscal_quarter"] for r in consumed)
    check(ordinals == list(range(ordinals[0], ordinals[-1]+1)), "Consecutive fiscal quarters")
    check(summary["source_pages_verified"] == 48 and summary["consumed_events_verified"] == 44
          and summary["original_slots_covered"] == 122, "MSFT summary counts")
    overlay = read("outputs/monthly_event_overlay_scope156/SLOT_AUDIT.json")
    overlay_manifest = read("outputs/monthly_event_overlay_scope156/MANIFEST.json")
    check(overlay_manifest["output_hashes"]["SLOT_AUDIT.json"] ==
          PINS["outputs/monthly_event_overlay_scope156/SLOT_AUDIT.json"], "Overlay manifest")
    check(len(overlay) == 3904 and Counter((s["slot_id"],s["symbol"],s["signal_date"]) for s in overlay) ==
          Counter((s["slot_id"],s["symbol"],s["signal_date"]) for s in slots), "All original overlay slots")
    all_consumed = {eid for s in overlay for eid in s["after"]["event_ids"]}
    check(len(all_consumed) == 1681, "Consumed scope")
    clock_map = {r["event_id"]:r for r in clock_rows}
    check(len(clock_map) == len(clock_rows) == 1681 and set(clock_map) == all_consumed, "Clock output identities")
    failures, precision, zones = [], Counter(), Counter()
    for eid in sorted(all_consumed):
        event = all_events[eid]
        own = independent_clock(event,schedule)
        observed = dict(available_session=event["available_session"],
            known_ns=pd.Timestamp(event["known_by_at"]).value,
            upper_ns=None if event.get("date_only_upper_bound_at") is None
                     else pd.Timestamp(event["date_only_upper_bound_at"]).value)
        record = clock_map[eid]
        if not (own == observed == record["expected"] == record["recorded"] and record["matches"]):
            failures.append({"event_id":eid,"independent":own,"observed":observed})
        precision[event["precision"]] += 1
        zones[event["publication_timezone"]] += 1
    check(not failures, "Independent consumed clocks differ")
    check(dict(precision) == clock_summary["precision_counts"] and clock_summary["passed"] and
          clock_summary["checked"] == 1681 and not clock_summary["mismatches"], "Clock summary counts")
    for sm in (summary, clock_summary):
        check(sm["calendar_completeness_verified"] is False and sm["factor_admitted"] is False, "Scope overclaim")
    # Verify every observed input unchanged through completion, including producer code pins.
    for name, expected in list(PINS.items()):
        check(hashlib.sha256(path(name).read_bytes()).hexdigest() == expected, "Final source mutation")
    result = {
        "schema_version":1, "status":"PASS_BOUNDED_SOURCE_IDENTITY_AND_CONSUMED_CLOCK_REVIEW",
        "review_method":"Independent html.parser extraction, pandas localization and linear session-close selection; neither producer nor event_schema normalization imported.",
        "msft": {"source_pages":48,"consumed_events":44,"original_slots":122,
            "original_slot_identity_match":True,"original_interval_union_match":True,
            "all_declared_fiscal_quarters_present":True,"source_receipt_hash_date_period_mismatches":0,
            "required_anchor_window":interval[0]["required_anchor_window"],
            "source_pages_fiscal_range":[2015,2026],"pages":actual_pages},
        "consumed_clock":{"events":1681,"original_slots_preserved":3904,
            "precision_counts":dict(precision),"timezone_counts":dict(zones),"mismatches":failures,
            "summary_matches":True,"independent_synthetic_cases":checks,
            "source_date_truth_reauthenticated_for_all_issuers":False},
        "algorithm_review":{
            "date_only":"Correct for current zones: next local calendar midnight minus 1ns, then first XNYS close strictly beyond the bound and on a later session date.",
            "exact":"Correct on current event: preserves the instant and selects first close at or after it; source timestamp remains inherited evidence.",
            "calendar":"Same frozen research calendar and official closure inputs independently hash checked; exact close/early close/weekend/DST/2025-01-09 closure behavior checked.",
            "nonblocking_generalization_limits":[
                "Producer uses datetime.fromisoformat for exact timestamps, which truncates sub-microsecond fractions; sole current exact event is whole-second precision, so no observed impact.",
                "Producer ZoneInfo midnight construction does not explicitly reject ambiguous/nonexistent local midnight; all current consumed zones/dates passed the stricter pandas localization."
            ]},
        "source_hashes":dict(sorted(PINS.items())),
        "admission_decision":{"listed_msft_identity_dates_periods_and_conservative_clocks_verified":True,
            "consumed_clock_arithmetic_verified":True,"historical_first_public_bytes_verified":False,
            "all_original_publication_dates_reauthenticated":False,
            "full_event_calendar_completeness_verified":False,"all_preliminary_or_correction_events_enumerated":False,
            "financial_values_admitted":False,"factor_admitted":False,"strategy_admitted":False},
        "remaining_requirements":[
            "Declare periodic-release taxonomy and independently enumerate relevant issuer archive intervals, including distinct preliminary/correction releases.",
            "Reconcile vendor date leads; lack of a missing fiscal quarter alone does not prove all-event completeness.",
            "Original source pages were retrieved later; archived historical first-public versions and exact publication times are not established by this review."
        ],
        "operations":{"downloads":0,"price_reads":0,"financial_values_extracted":0,
            "factor_calculations":0,"model_fits":0,"orders":0,"returns":0,"new_strategy_trials":0,
            "producer_or_frozen_files_modified":False,"observation_accounts_modified":False},
    }
    if OUT.exists():
        check(read(OUT) == result, "Review already exists with different content")
    else:
        with OUT.open("x") as stream:
            json.dump(result,stream,sort_keys=True,indent=2,ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"status":result["status"],"source_pages":48,"msft_slots":122,
                     "consumed_clocks":1681,"mismatches":0,"review_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

"""Independent source-only review of two already downloaded CAT releases.

Uses DOM-bounded issuer evidence rather than the producer's full-page parser.
Does not import producer, event_schema, or any price/model code.
"""
import calendar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import exchange_calendars as xcals
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "outputs/monthly_event_cat_wire156/"
OUTPUT = ROOT / "work/after155_cat_wire_independent_review.json"
PINS = {
    BASE + "SOURCE_SPEC.json": "14fdd4700a2897a9960dd41093137e7764bfada01a26c6a753be85d48cde01ea",
    BASE + "SOURCE_PROPOSAL_V2.json": "ddce16b7e0249029601b6c713cd2d76b8b7a52bcdbe01845a80d616a22bc3357",
    BASE + "SOURCE_PROPOSAL.json": "bccfe3630c546162da7cc1b207b34e00faa2769ce7ff9ef804265e869c588159",
    BASE + "COLLECTION.json": "a1d65c6107aedfdc884e4ec2e27c719485f7970b53fb4a607801d4a13ef43b8a",
    BASE + "CAT_2025Q1.receipt.json": "33a4705b29c4dd5051c099d170bf68377b35f107a8e8d329fa14f45ab3113c7b",
    BASE + "CAT_2025Q2.receipt.json": "a3b8507be12ce71a62d2245f2f9638e597c2e893ed6d4e6e5a69389403d8c24b",
    BASE + "CAT_2025Q1.html": "023e0b957a7f353ce7219965d89fbbf79188016fbd0317e52ac47968ef66eac5",
    BASE + "CAT_2025Q2.html": "0225a79282017f85d9485f8f6a3639a80fb55cda8b0fe9fc76896a9ca6ce87a4",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def path(name):
    p = Path(name)
    p = p if p.is_absolute() else ROOT / p
    require(p.is_relative_to(ROOT) and p.resolve() == p, "Nonlocal path")
    return p


def raw(name):
    p = path(name)
    a = p.stat()
    result = p.read_bytes()
    b = p.stat()
    require((a.st_ino, a.st_mtime_ns, a.st_size) == (b.st_ino, b.st_mtime_ns, b.st_size), "Input mutated")
    return result


def sha(name):
    return hashlib.sha256(raw(name)).hexdigest()


def read(name):
    return json.loads(raw(name))


def text(node):
    return " ".join(node.get_text(" ", strip=True).split())


def source_date(value):
    for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError("Unrecognized source date: " + value)


def ref(node, quote=None):
    value = text(node)
    require(quote is None or quote in value, "Excerpt not in DOM node")
    return {"html_line_1based": node.sourceline, "tag": node.name,
            "quote": value if quote is None else quote,
            "node_text_sha256": hashlib.sha256(value.encode()).hexdigest()}


def parse_source(blob):
    soup = BeautifulSoup(blob, "html.parser")
    articles = soup.select("article.news-release")
    require(len(articles) == 1, "Unique release article required")
    article = articles[0]
    header = article.select_one("header.release-header")
    body = article.select_one("section.release-body")
    require(header is not None and body is not None, "Bounded header/body required")
    h1 = header.find("h1")
    title = text(h1)
    title_match = re.fullmatch(r"Caterpillar Reports (First|Second)-Quarter (\d{4}) Results", title)
    require(title_match is not None, "Actual quarterly title required")
    ordinal, year = title_match.group(1).lower(), int(title_match.group(2))
    quarter = {"first": 1, "second": 2}[ordinal]
    provider_label = next(n for n in header.find_all("h2") if text(n) == "News provided by")
    provider = provider_label.find_next("strong")
    require(text(provider) == "Caterpillar Inc.", "Issuer-provided article required")
    display = header.select_one("p.mb-no")
    display_text = text(display)
    display_match = re.fullmatch(r"([A-Za-z]+ \d{1,2}, \d{4}), (\d{2}:\d{2}) ET", display_text)
    require(display_match is not None, "Printed date/time required")
    datelines = [n for n in body.find_all("p") if re.match(r"IRVING,\s*Texas", text(n))]
    require(len(datelines) == 1, "Unique article dateline required")
    dateline = datelines[0]
    location = dateline.select_one("span.xn-location")
    date_node = dateline.select_one("span.xn-chron")
    require(text(location) == "IRVING, Texas", "Issuer location mismatch")
    publication_date = source_date(text(date_node))
    require(source_date(display_match.group(1)) == publication_date, "Printed and body date differ")
    plain_dateline = text(dateline)
    actual = re.search(r"Caterpillar Inc\.\s*\(NYSE:\s*CAT\s*\)\s+announced\s+(first|second)-quarter\s+(\d{4})\s+results\.", plain_dateline)
    require(actual is not None and actual.group(1) == ordinal and int(actual.group(2)) == year,
            "Issuer and actual results must coexist in original dateline paragraph")
    periods = []
    for node in body.find_all("p"):
        m = re.match(r"Three Months Ended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s*-\s*U\.S\. GAAP", text(node))
        if m:
            d = source_date(m.group(1))
            if d.startswith(str(year)):
                periods.append((d, node, m.group(0)))
    require(periods and len({p[0] for p in periods}) == 1, "Unambiguous current GAAP quarter header required")
    period_end, period_node, period_quote = periods[0]
    end = datetime.fromisoformat(period_end)
    require(end.month == quarter * 3 and end.day == calendar.monthrange(year, end.month)[1], "Quarter/period mismatch")
    require(period_end < publication_date, "Results must follow period end")
    return {"symbol": "CAT", "issuer": "Caterpillar Inc.", "exchange": "XNYS",
            "fiscal_year": year, "fiscal_quarter": quarter,
            "official_actual_period_end": period_end, "publication_date": publication_date,
            "publication_timezone": "America/Chicago", "source_display_timestamp": display_text,
            "evidence": {"title": ref(h1), "provided_by": ref(provider), "displayed_timestamp": ref(display),
                         "dateline": ref(dateline, plain_dateline[:actual.end()]),
                         "location": ref(location), "date": ref(date_node),
                         "actual_results_and_issuer": ref(dateline, actual.group(0)),
                         "current_quarter_gaap_header": ref(period_node, period_quote)},
            "source_parser_scope": "Only the unique article header and release-body; related-news boxes and full-page regex matches are excluded."}


def conservative_clock(date, sessions, xnys):
    following_midnight = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize("America/Chicago", ambiguous="raise", nonexistent="raise")
    upper = following_midnight.tz_convert("UTC") - pd.Timedelta(nanoseconds=1)
    later = [s for s in sessions if s > date]
    require(later, "No next frozen session")
    first = later[0]
    close = xnys.session_close(pd.Timestamp(first))
    require(close > upper, "Next close must follow issuer-local end of day")
    return {"date_only_upper_bound_at": upper.isoformat(), "available_session": first,
            "known_by_at": close.isoformat(), "precision": "date_only", "publication_at": None,
            "first_public_time_verified": False, "known_by_is_actual_publication": False}


def flatten(row):
    return row.get("source_event_audit", row)


def conflict_keys(a, b):
    names = []
    if a.get("event_id") == b.get("event_id"):
        names.append("event_id")
    if a.get("symbol") == b.get("symbol"):
        if (a.get("fiscal_year"), a.get("fiscal_quarter")) == (b.get("fiscal_year"), b.get("fiscal_quarter")):
            names.append("issuer_fiscal_quarter")
        for field in ("publication_date", "official_actual_period_end"):
            if a.get(field) is not None and a.get(field) == b.get(field):
                names.append(field)
    return names


def synthetic_checks(sessions, xnys):
    checks = {}
    checks["abbreviated_dateline"] = source_date("Aug. 5, 2025") == "2025-08-05"
    checks["zero_padded_display_day"] = source_date("Aug 05, 2025") == "2025-08-05"
    checks["full_month_dateline"] = source_date("April 30, 2025") == "2025-04-30"
    checks["holiday_next_close"] = conservative_clock("2025-07-03", sessions, xnys)["available_session"] == "2025-07-07"
    checks["early_close_respected"] = conservative_clock("2025-07-02", sessions, xnys)["known_by_at"] == "2025-07-03T17:00:00+00:00"
    checks["official_mourning_closure_respected"] = conservative_clock("2025-01-08", sessions, xnys)["available_session"] == "2025-01-10"
    a = dict(symbol="CAT", event_id="new", fiscal_year=2025, fiscal_quarter=1, publication_date="2025-04-30", official_actual_period_end="2025-03-31")
    checks["quarter_collision_despite_different_id"] = "issuer_fiscal_quarter" in conflict_keys(a, dict(a, event_id="old"))
    checks["period_collision_despite_different_date"] = "official_actual_period_end" in conflict_keys(a, dict(a, event_id="old", publication_date="2025-05-02", fiscal_quarter=2))
    require(all(checks.values()), "Synthetic check failed")
    return checks


def main():
    pins = dict(PINS)
    for name, expected in pins.items():
        require(sha(name) == expected, "Pinned input mismatch: " + name)
    spec = read(BASE + "SOURCE_SPEC.json")
    pins.update(spec["input_hashes"])
    for name, expected in pins.items():
        require(sha(name) == expected, "Source dependency mismatch: " + name)
    calendar_doc = read("outputs/monthly_paired_target143/CALENDAR.json")
    for name, expected in calendar_doc["provenance"]["source_hashes"].items():
        require(sha(name) == expected, "Frozen calendar implementation mismatch")
        pins[str(path(name).relative_to(ROOT))] = expected
    sessions = calendar_doc["sessions"]
    require(sessions == sorted(set(sessions)), "Calendar ordering")
    xnys = xcals.get_calendar("XNYS", start="2025-01-01", end="2025-12-31")
    checks = synthetic_checks(sessions, xnys)
    v1 = read(BASE + "SOURCE_PROPOSAL.json")
    v2 = read(BASE + "SOURCE_PROPOSAL_V2.json")
    collection = read(BASE + "COLLECTION.json")
    require(v2["original_proposal_sha256"] == pins[BASE + "SOURCE_PROPOSAL.json"], "V1 not bound")
    require(v2["source_spec_sha256"] == pins[BASE + "SOURCE_SPEC.json"], "Spec not bound")
    require(v2["collection_sha256"] == pins[BASE + "COLLECTION.json"], "Collection not bound")
    require(sha(v2["parser_path"]) == v2["parser_sha256"], "Parser identity")
    pins[v2["parser_path"]] = v2["parser_sha256"]
    catalogs = {name: read(name) for name in ("outputs/monthly_earnings_repair/EVENTS.json", "outputs/monthly_event_patch156/ADDED_EVENTS.json", "outputs/monthly_event_scope156/EVENTS.json")}
    require([len(v) for v in catalogs.values()] == [3771, 4, 1677], "Bounded catalog counts")
    require(len(spec["targets"]) == len(v1["results"]) == len(v2["results"]) == len(collection["receipts"]) == 2, "Exact two sources")
    results = []
    for target, old, proposed, collected in zip(spec["targets"], v1["results"], v2["results"], collection["receipts"]):
        key = target["key"]
        receipt = read(BASE + key + ".receipt.json")
        require(receipt == collected == proposed["receipt"] == old["receipt"], "Same original receipt required")
        require(receipt["key"] == key == old["key"] == proposed["key"], "Source identity")
        require(receipt["requested_url"] == receipt["final_url"] == target["url"], "Unexpected redirect")
        require(receipt["http_status"] == 200 and receipt["status"] == "downloaded", "Successful source required")
        blob = raw(receipt["path"])
        require(len(blob) == receipt["bytes"] and sha(receipt["path"]) == receipt["sha256"], "Raw bytes mismatch")
        require(old["candidate"] is None and not old["source_checks"]["dateline"]["found"], "Original parser failure retained")
        require(all(v["found"] for k,v in old["source_checks"].items() if k != "dateline"), "V1 failure beyond claimed dateline")
        parsed = parse_source(blob)
        for k in ("fiscal_year", "fiscal_quarter", "publication_date"):
            require(parsed[k] == target[k], "Frozen target mismatch: " + k)
        require(parsed["official_actual_period_end"] == target["period_end"], "Actual period mismatch")
        clock = conservative_clock(parsed["publication_date"], sessions, xnys)
        parsed.update(clock)
        parsed["event_id"] = parsed["symbol"] + "_" + parsed["publication_date"]
        for k in ("symbol", "fiscal_year", "fiscal_quarter", "official_actual_period_end", "publication_date", "publication_timezone", "event_id", "source_display_timestamp", *clock):
            require(parsed[k] == proposed["candidate"][k], "Independent candidate mismatch: " + k)
        conflicts = {}
        for name, rows in catalogs.items():
            matches = [{"event_id": flatten(r).get("event_id"), "keys": conflict_keys(parsed, flatten(r))} for r in rows if conflict_keys(parsed, flatten(r))]
            conflicts[name] = matches
        require(not any(conflicts.values()), "Existing event conflict")
        require(not results or not conflict_keys(parsed, results[0]), "Two new candidates conflict")
        parsed.update({"source_path": receipt["path"], "source_sha256": receipt["sha256"], "source_url": target["url"],
                       "receipt_sha256": pins[BASE + key + ".receipt.json"], "retrieved_utc": receipt["request_utc"],
                       "http_last_modified": receipt["last_modified"], "catalog_conflicts": conflicts,
                       "single_event_source_admissible": True, "is_actual_results_release": True,
                       "is_advance_notice": False, "source_display_timestamp_used_as_first_public_time": False})
        results.append(parsed)
    for name, expected in pins.items():
        require(sha(name) == expected, "Input changed during review: " + name)
    report = {"schema_version": 1, "status": "pass_two_CAT_events_for_new_derived_catalog_only",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "reviewer_code_sha256": sha(__file__), "verified_input_sha256": pins,
              "independent_method": "BeautifulSoup DOM-bounded article extraction; independent date/quarter parsing; pandas issuer-midnight bound; first later frozen session and pinned XNYS session close. No producer or event_schema imports.",
              "events": results, "synthetic_checks": checks,
              "catalog_counts": {k:len(v) for k,v in catalogs.items()},
              "v1_preservation": {"original_failed_proposal_unchanged": True, "same_download_receipts_and_raw_bytes": True,
                                  "correction": "Whitespace between location span and comma; zero-padding in displayed dates. The original failed parser remains preserved. No second download occurred within this review or the recorded V2 transform.",
                                  "unrecorded_external_network_attempts_audited": False},
              "producer_nonblocking_locator_issue": "Producer full-page issuer-regex offsets point to related-news content because the dateline hyperlink inserts whitespace before ')'. This independent review instead verifies Caterpillar Inc., NYSE CAT and the actual-results verb together inside the release-body dateline at HTML lines 2608 and 2641; the original issuer identity is confirmed.",
              "vintage_limit": "The saved HTML was retrieved in October 2026 and HTTP Last-Modified is also later than the 2025 release dates. Original title, printed date and body dateline consistently establish the claimed release; no first-public-byte or exhaustive revision-history certification is made. Last-Modified and retrieval timestamps are not substituted for publication dates.",
              "time_limit": "Printed 06:30 ET is retained as source metadata only. Date-only America/Chicago end-of-day plus first later session close remains conservative. Use known_by <= actual decision cutoff; do not backdate retrieval or certify exact first publication.",
              "admission_decision": {"these_two_events_may_enter_new_derived_catalog": True,
                                     "old_events_must_remain_unchanged": True,
                                     "recheck_conflicts_at_append_time": True,
                                     "derived_catalog_created_by_this_review": False,
                                     "whole_window_archive_completeness_verified": False,
                                     "complete_event_calendar_verified": False,
                                     "factor_admitted": False, "strategy_admitted": False},
              "network_requests": 0, "financial_values_extracted": 0, "prices_read": 0,
              "growth_or_factors_computed": 0, "model_fits": 0, "returns_computed": 0, "observer_modified": False}
    with OUTPUT.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status":report["status"], "synthetic_checks_passed":len(checks), "events":len(results), "sha256":sha(OUTPUT)}))


if __name__ == "__main__":
    main()

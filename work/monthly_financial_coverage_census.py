"""Offline source census for the unchanged monthly candidate universe.

This inventories evidence, never admits financial cells or calculates scores.
The universe comes from the complete monthly META, never holdings or winners.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_financial_coverage_census_v1"
INPUTS = {
    "universe": "outputs/monthly_local_linear70/META.json",
    "events": "outputs/monthly_earnings_repair/EVENTS.json",
    "event_validation": "outputs/monthly_earnings_repair/VALIDATION.json",
    "event_scope": "outputs/monthly_earnings_repair/SCOPED_READINESS.json",
    "identity_leads": "outputs/monthly_profitability141_sources/DATA_PLAN.json",
    "companyfacts_inventory": "work/profitability141_migrated_cache_inventory_complete_20261019T202336581108Z.json",
    "admitted_financial_sources": "outputs/monthly_stock_comp_source_admission_v2/DATASET.json",
}
LABELS = {
    "balance_statement": r"\b(?:consolidated\s+)?(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)\b",
    "cashflow_statement": r"\bstatements?\s+of\s+cash\s+flows?\b",
    "total_assets": r"\btotal\s+assets\b",
    "operating_cashflow": r"\bnet\s+cash\b.{0,120}\boperating\s+activities\b",
    "net_income": r"\bnet\s+(?:income|loss)\b",
    "stock_compensation": r"\b(?:stock|share)[ -]based\s+(?:compensation|payments?)\b",
    "annual_period": r"\b(?:(?:twelve|12)\s+months?\s+ended|(?:fiscal\s+)?years?\s+ended)\b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(data):
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def local_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path.relative_to(ROOT)
    require("CloudStorage" not in path.parts, "Cloud source forbidden")
    current = ROOT
    for part in path.relative_to(ROOT).parts:
        require(part not in ("..", "."), "Traversal forbidden")
        current = current / part
        require(not current.is_symlink(), "Symlink source forbidden")
    require(path.resolve() == path, "Noncanonical source path")
    return path


def read_bytes(path):
    path = local_path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_mtime_ns, before.st_size) ==
            (after.st_ino, after.st_mtime_ns, after.st_size),
            "Source changed during read")
    return raw


def read_json(path):
    return json.loads(read_bytes(path), parse_constant=lambda value: (
        (_ for _ in ()).throw(ValueError("Nonfinite JSON"))))


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def universe_rows(meta):
    dates, panels, etfs = meta["dates"], meta["symbols"], meta["etfs"]
    require(len(dates) == len(panels) and bool(dates), "Universe shape")
    require(dates == sorted(set(dates)), "Universe dates")
    etfs = set(etfs)
    memberships = defaultdict(list)
    for session, symbols in zip(dates, panels):
        date.fromisoformat(session)
        require(len(symbols) == len(set(symbols)), "Duplicate candidate")
        require(etfs.issubset(symbols), "ETF universe changed")
        for symbol in symbols:
            if symbol not in etfs:
                memberships[symbol].append(session)
    return dict(sorted(memberships.items()))


def scan_body(raw):
    """Presence checks are deliberately only leads, not accounting validation."""
    if raw.startswith(b"%PDF"):
        return dict(format="pdf", text_scanned=False,
                    labels={}, status="pdf_requires_separate_source_review")
    text = raw.decode("utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).lower()
    return dict(format="html_or_text", text_scanned=True,
                labels={name: bool(re.search(pattern, text))
                        for name, pattern in LABELS.items()},
                status="labels_only_not_statement_context_or_values")


def full_body_claim(kind):
    """Classify a locator claim; this does not authenticate the document."""
    return (isinstance(kind, str) and kind.startswith("issuer_") and
            not any(part in kind for part in (
                "excerpt", "observation", "not_full", "notice_of_release_issuance")) and
            ("public_index" not in kind or "full_bytes" in kind))


def latest_annual(events, session):
    valid = []
    unknown = 0
    unknown_clock_barrier = False
    for event in events:
        value = event.get("available_session")
        try:
            date.fromisoformat(value)
        except (ValueError, TypeError):
            unknown += 1
            publication = event.get("publication_date")
            try:
                date.fromisoformat(publication)
                unknown_clock_barrier |= publication <= session
            except (ValueError, TypeError):
                unknown_clock_barrier = True
            continue
        if value <= session:
            valid.append(event)
    if unknown_clock_barrier:
        return None, "unknown_clock_barrier", unknown
    if not valid:
        return None, "no_prior_dated_annual_event", unknown
    latest_date = max(event["available_session"] for event in valid)
    latest = [event for event in valid if event["available_session"] == latest_date]
    if len(latest) != 1:
        return None, "ambiguous_latest_annual_event", unknown
    return latest[0], "latest_selected_before_validity", unknown


def date_lead_covers(event, session):
    """Date availability is necessary but not proof of financial admission."""
    if event.get("status") != "ready":
        return False
    if event.get("fiscal_quarter") != 4:
        return False
    try:
        asof = date.fromisoformat(session)
        available = date.fromisoformat(event.get("available_session"))
        period_end = date.fromisoformat(event.get("official_actual_period_end"))
        lag = (asof - period_end).days
    except (ValueError, TypeError):
        return False
    return available <= asof and 0 <= lag <= 550


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {name: dict(path=path, sha256=digest(read_bytes(path)))
              for name, path in INPUTS.items()}
    code = "work/monthly_financial_coverage_census.py"
    tests = "work/test_monthly_financial_coverage_census.py"
    plan = dict(
        schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="All-candidate source feasibility; zero financial admission or strategy trials",
        inputs=inputs,
        code={path: digest(read_bytes(path)) for path in (code, tests)},
        scope=dict(universe="All stocks in all122 original candidate panels before ranking",
                   no_holdings_or_return_selection=True,
                   inherited_cik_links="Locators only; no historical identity admission",
                   annual_lead="Existing fiscal_quarter=4 metadata; verify original text later",
                   stock_compensation_source="Existing final admission preserved, including all holds"),
        policy=dict(
            body_checks="Seven broad text-presence checks, no values or financial context admitted",
            age_for_coverage_leads_days=550,
            age_measured_from="official_actual_period_end",
            unavailable_latest="Report latest fiscal-quarter4 metadata first; never bypass invalid latest",
            source_errors="Retain explicit gaps and proceed with census; not with trading",
            companyfacts="Cached metadata locator only; never accounting source or financial admission",
            no_network=True, no_cloud=True, no_scores=True, no_returns=True,
            no_observer_changes=True,
        ),
    )
    write_new(OUT / "PLAN.json", plan)
    print(json.dumps({"plan_sha256": digest(read_bytes(OUT / "PLAN.json"))}))


def run(plan_sha):
    require(digest(read_bytes(OUT / "PLAN.json")) == plan_sha, "Plan digest")
    plan = read_json(OUT / "PLAN.json")
    for path, expected in plan["code"].items():
        require(digest(read_bytes(path)) == expected, "Census code changed")
    inputs = {}
    for name, ref in plan["inputs"].items():
        require(digest(read_bytes(ref["path"])) == ref["sha256"], "Input changed: " + name)
        inputs[name] = read_json(ref["path"])
    memberships = universe_rows(inputs["universe"])
    event_sha = plan["inputs"]["events"]["sha256"]
    require(inputs["event_validation"]["output_sha256"]["EVENTS.json"] == event_sha,
            "Event validation binding")
    require(inputs["event_scope"]["inputs"]["EVENTS.json"] == event_sha,
            "Event scope binding")
    require(len(inputs["universe"]["dates"]) == 122, "Original122-month scope")
    require(len(memberships) == 73, "Original73-stock scope")
    require(len(inputs["universe"]["etfs"]) == 17, "Original17-ETF scope")
    require(sum(map(len, memberships.values())) == 3904, "Original3904-stock-slot scope")
    require(all(len(panel) == 49 for panel in inputs["universe"]["symbols"]),
            "Original49-candidate scope")
    require(set(inputs["identity_leads"]["candidate_universe"]) == set(memberships),
            "Identity plan candidate universe differs")
    cik_leads = {row["symbol"]: row["candidate_ciks"]
                 for row in inputs["identity_leads"]["candidate_cik_locating_evidence"]}
    facts = defaultdict(list)
    for item in inputs["companyfacts_inventory"]["candidates"]:
        facts[item["cik"]].append({
            "path": item["relative_path"], "sha256": item["sha256"],
            "metadata_only_not_original_statement": True,
        })
    admitted = defaultdict(list)
    for event in inputs["admitted_financial_sources"]["events"]:
        admitted[event["symbol"]].append({
            "event_id": event["event_id"],
            "available_from": event.get("available_from"),
            "cash_quality_admission": event.get("cash_quality_admission"),
            "stock_compensation_admission": event.get("stock_compensation_admission"),
            "fact_admitted": event.get("fact_admitted"),
            "financial_attachment_source": event.get("source"),
        })
    all_events = defaultdict(list)
    for event in inputs["events"]:
        if event.get("symbol") in memberships:
            all_events[event["symbol"]].append(event)
    body_cache, documents, rows, month_rows = {}, [], [], []
    for symbol, sessions in memberships.items():
        source_events = all_events[symbol]
        annual = [event for event in source_events if event.get("fiscal_quarter") == 4]
        converted = []
        for event in annual:
            path = event.get("evidence_path")
            row = {key: event.get(key) for key in (
                "event_id", "symbol", "status", "fiscal_year", "fiscal_quarter",
                "publication_date", "available_session", "official_actual_period_end", "source_url",
                "source_kind", "first_public_time_verified", "evidence_sha256")}
            row["evidence_path"] = path
            row["source_hash_verified"] = False
            row["catalog_claims_issuer_full_body"] = full_body_claim(event.get("source_kind"))
            try:
                require(isinstance(path, str) and bool(path), "Missing source path")
                safe_path = local_path(path)
                row["evidence_path"] = str(safe_path.relative_to(ROOT))
                key = (str(safe_path), event.get("evidence_sha256"))
                if key not in body_cache:
                    raw = read_bytes(safe_path)
                    require(digest(raw) == event.get("evidence_sha256"), "Source hash mismatch")
                    body_cache[key] = scan_body(raw)
                row.update(body_cache[key])
                row["source_hash_verified"] = True
            except (ValueError, OSError, TypeError) as error:
                row["status_of_source_check"] = type(error).__name__ + ": " + str(error)
                row["labels"] = {}
            row["complete_text_label_lead"] = (
                row["source_hash_verified"] and row["catalog_claims_issuer_full_body"] and
                all(row["labels"].get(key, False) for key in LABELS))
            row["financial_cells_admitted_by_census"] = False
            converted.append(row)
        documents.extend(converted)
        by_event = {id(event): checked for event, checked in zip(annual, converted)}
        for session in sessions:
            latest, latest_status, unknown_clocks = latest_annual(annual, session)
            latest_check = by_event[id(latest)] if latest else {}
            month_rows.append(dict(
                symbol=symbol, session=session,
                latest_annual_date_lead=latest.get("event_id") if latest else None,
                latest_status=latest_status, annual_events_with_unknown_clocks=unknown_clocks,
                date_lead_within550=bool(latest and date_lead_covers(latest, session)),
                evidence_source_hash_verified=latest_check.get("source_hash_verified", False),
                complete_text_label_lead=bool(
                    latest and date_lead_covers(latest, session) and
                    latest_check.get("complete_text_label_lead", False)),
                financial_cells_admitted_by_census=False,
            ))
        labels = [row for row in converted if row["complete_text_label_lead"]]
        symbol_months = [row for row in month_rows if row["symbol"] == symbol]
        rows.append(dict(
            symbol=symbol, candidate_months=len(sessions),
            first_candidate_session=sessions[0], last_candidate_session=sessions[-1],
            locating_ciks=cik_leads.get(symbol, []),
            companyfacts_metadata_leads=[
                row for cik in cik_leads.get(symbol, []) for row in facts[cik]],
            event_records=len(source_events), annual_q4_metadata_records=len(annual),
            annual_evidence_hash_verified=sum(row["source_hash_verified"] for row in converted),
            catalog_claimed_full_bodies=sum(
                row["source_hash_verified"] and row["catalog_claims_issuer_full_body"]
                for row in converted),
            complete_text_label_leads=len(labels),
            complete_text_label_years=sorted({str(row["fiscal_year"]) for row in labels}),
            candidate_months_with_complete_text_lead=sum(
                row["complete_text_label_lead"] for row in symbol_months),
            previous_admission_records=admitted.get(symbol, []),
            caveat="All new evidence remains a lead; context, currency, periods, identity and clock require source admission",
        ))
    counts = dict(
        stocks=len(rows), monthly_stock_slots=len(month_rows),
        etfs_excluded_from_accounting_census=len(inputs["universe"]["etfs"]),
        all_event_records=sum(row["event_records"] for row in rows),
        annual_q4_metadata_records=len(documents),
        annual_evidence_hash_verified=sum(row["source_hash_verified"] for row in documents),
        catalog_claimed_full_bodies=sum(
            row["source_hash_verified"] and row["catalog_claims_issuer_full_body"]
            for row in documents),
        annual_evidence_read_or_hash_failures=sum(not row["source_hash_verified"] for row in documents),
        catalog_not_full_body=sum(not row["catalog_claims_issuer_full_body"] for row in documents),
        distinct_verified_primary_evidence_path_hashes=len(body_cache),
        stocks_with_complete_text_label_leads=sum(row["complete_text_label_leads"] > 0 for row in rows),
        complete_text_label_leads=sum(row["complete_text_label_leads"] for row in rows),
        monthly_stock_slots_with_complete_text_leads=sum(
            row["complete_text_label_lead"] for row in month_rows),
        new_financial_admissions=0, new_strategy_trials=0,
        source_kind_counts=dict(Counter(row["source_kind"] for row in documents)),
    )
    report = dict(
        schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha, counts=counts, stocks=rows,
        annual_documents=documents, monthly_candidate_coverage=month_rows,
        status="SOURCE_LEAD_CENSUS_COMPLETE_NOT_FINANCIAL_ADMISSION",
        limitations=[
            "Q4 metadata and text labels do not establish complete annual statements.",
            "Existing validated financial attachment sources are retained separately; body label scans only cover the unified event's primary evidence.",
            "A catalog full-body claim is not independent authentication; index excerpts and observations never qualify as full text label leads.",
            "Presence in separate tables does not certify matching fiscal periods, currency or scope.",
            "Raw evidence hash checks do not authenticate first-public versions or complete corrections.",
            "CIK links inherited from the locator plan are not historical issuer/security admission.",
            "Unknown quarter metadata, untranslated filings and PDF-only text remain explicit gaps.",
            "No source is chosen using holdings, ranks, returns, or whether it would cross the70% target.",
        ],
    )
    write_new(OUT / "CENSUS.json", report)
    print(json.dumps({"counts": counts, "report_sha256": digest(read_bytes(OUT / "CENSUS.json"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        run(args.plan_sha)

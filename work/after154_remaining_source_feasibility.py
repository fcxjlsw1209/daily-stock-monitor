"""Metadata-only next-source inventory; never parses amounts or strategy data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = "outputs/monthly_financial_coverage_census_v1/CENSUS.json"
CENSUS_SHA = "de95906ef3d752890d28027a9b52cfd7decc7aeb1bc0f09f2785f1826a3500fc"
OUT = "work/after154_remaining_source_feasibility.json"
CONCEPT = "work/after149_coverage_extension_concept.json"
FINANCIAL_REVIEW_QUEUE = frozenset(("BAC", "BRK.B", "C", "HSBC", "JPM", "WFC"))
# These prior gates are read only for report identities and admission status.
GATES = [
    "outputs/monthly_financial_coverage_admission_batch%d_v1/DATASET.json" % n
    for n in range(1, 6)
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def qualify(row):
    labels = row.get("labels") or {}
    return (row.get("complete_text_label_lead") is False and
            labels.get("net_income") is True and labels.get("total_assets") is True)


def snapshot(row, processed_pairs, processed_issuers, pins):
    path = row["evidence_path"]
    raw = (ROOT / path).read_bytes()
    actual = digest(raw)
    if actual != row["evidence_sha256"]:
        raise ValueError("Source digest changed: " + path)
    pins[path] = actual
    source = ROOT / path
    receipts = []
    for receipt in (source.with_name(source.name + ".meta.json"), source.with_suffix(".meta.json")):
        if receipt.is_file() and str(receipt.relative_to(ROOT)) not in {r["path"] for r in receipts}:
            name = str(receipt.relative_to(ROOT))
            hashed = digest(receipt.read_bytes())
            pins[name] = hashed
            receipts.append(dict(path=name, sha256=hashed))
    pair = (row["symbol"], int(row["fiscal_year"]))
    return dict(
        symbol=row["symbol"], fiscal_year=row["fiscal_year"], event_id=row["event_id"],
        publication_date=row.get("publication_date"),
        available_session=row.get("available_session"),
        official_actual_period_end=row.get("official_actual_period_end"),
        source=dict(path=path, sha256=actual, url=row.get("source_url"),
                    source_kind=row.get("source_kind"), format=row.get("format"),
                    bytes_verified=len(raw), receipt_candidates=receipts),
        text_scanned=row.get("text_scanned"), labels=row.get("labels"),
        catalog_claims_issuer_full_body=row.get("catalog_claims_issuer_full_body"),
        complete_text_label_lead=row.get("complete_text_label_lead"),
        already_extracted_in_original_or_batch_gate=pair in processed_pairs,
        issuer_has_prior_coverage_gate=row["symbol"] in processed_issuers,
        required_three_field_context_verified=False, admitted=False,
    )


def main():
    pins = {}
    def read(name, expected=None):
        raw=(ROOT/name).read_bytes()
        actual=digest(raw)
        if expected is not None and actual != expected:
            raise ValueError("Changed input: "+name)
        pins[name]=actual
        return json.loads(raw)
    census=read(CENSUS,CENSUS_SHA)
    read(CONCEPT, "60c3afd837cd32b75a70733b70c777084a9c124cf917e8052628947f1cb7d6ee")
    pins["work/after154_remaining_source_feasibility.py"]=digest(Path(__file__).read_bytes())
    records=census["annual_documents"]
    processed_pairs=set()
    processed_issuers=set()
    original=[]
    # Original source references already belong to the immutable census.
    for stock in census["stocks"]:
        for event in stock.get("previous_admission_records",[]):
            parts=event["event_id"].split(":")
            if len(parts)<2 or not parts[1].startswith("FY"):
                raise ValueError("Unknown original report identity")
            processed_pairs.add((stock["symbol"],int(parts[1][2:])))
            processed_issuers.add(stock["symbol"])
            original.append(dict(symbol=stock["symbol"],fiscal_year=int(parts[1][2:]),
                                 event_id=event["event_id"]))
    gate_refs=[]
    for name in GATES:
        data=read(name)
        events=data["events"]
        for event in events:
            processed_pairs.add((event["symbol"],int(event["fiscal_year"])))
            processed_issuers.add(event["symbol"])
        gate_refs.append(dict(path=name,sha256=pins[name],events=len(events),
                              metadata_projection_only=True))
    omitted=[r for r in records if qualify(r)]
    new_issuers=[r for r in omitted if r["symbol"] not in processed_issuers]
    full=[r for r in new_issuers if r["catalog_claims_issuer_full_body"]]
    excerpts=[r for r in new_issuers if not r["catalog_claims_issuer_full_body"]]
    full.sort(key=lambda r:(r["symbol"],int(r["fiscal_year"]),r["event_id"]))
    excerpts.sort(key=lambda r:(r["symbol"],int(r["fiscal_year"]),r["event_id"]))
    snapshots=[snapshot(r,processed_pairs,processed_issuers,pins) for r in full]
    excerpt_snapshots=[snapshot(r,processed_pairs,processed_issuers,pins) for r in excerpts]
    unscanned=[r for r in records if r["symbol"] not in processed_issuers and
               r.get("format")=="pdf" and not r.get("text_scanned") and
               r.get("catalog_claims_issuer_full_body")]
    pdf_counts=Counter(r["symbol"] for r in unscanned)
    # Verify the first two alphabetical non-bank PDF issuer sets in full.
    # Other sets remain census-only, explicitly not reverified in this pass.
    first_pdf=sorted(s for s in pdf_counts if s not in FINANCIAL_REVIEW_QUEUE)[:2]
    pdf_snapshots=[snapshot(r,processed_pairs,processed_issuers,pins) for r in unscanned
                   if r["symbol"] in first_pdf]
    pdf_snapshots.sort(key=lambda r:(r["symbol"],int(r["fiscal_year"]),r["event_id"]))
    missing_by_symbol={}
    for row in full:
        missing_by_symbol.setdefault(row["symbol"],Counter()).update(
            k for k,v in row["labels"].items() if not v)
    strong=[s for s in sorted(set(r["symbol"] for r in full))
            if any(r["symbol"]==s and r["labels"].get("annual_period") and
                   r["labels"].get("balance_statement") for r in full)]
    result=dict(
        schema_version=1,stage="metadata_only_remaining_source_feasibility_not_admission",
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_selection="Alphabetical issuer order using only existing source-format and text-label metadata; no holdings, score, rank or return inspection.",
        original_gate_identities=original,
        prior_batch_gate_refs=gate_refs,
        prior_covered_issuers=sorted(processed_issuers),
        counts=dict(census_reports=len(records),census_issuers=len(census["stocks"]),
                    omitted_by_seven_label_screen=len(omitted),
                    omitted_labels_new_issuer_reports=len(new_issuers),
                    full_body_label_leads_new_issuers=len(full),
                    index_only_label_leads_new_issuers=len(excerpts),
                    previously_covered_issuer_label_reports=len(omitted)-len(new_issuers),
                    unscanned_claimed_full_pdf_reports_new_issuers=len(unscanned),
                    verified_candidate_source_documents=len({r["source"]["path"] for r in snapshots+excerpt_snapshots+pdf_snapshots}),
                    new_financial_admissions=0,new_strategy_trials=0),
        next_strong_text_label_issuer_order=strong,
        preceding_weak_label_issue=dict(
            symbol="GE",fiscal_year=2012,
            reason="NI and assets labels exist but no balance-statement label; metadata cannot establish matching consolidated scope, and historical finance activities require accounting review."),
        full_body_label_leads=snapshots,
        omitted_label_reasons={s:dict(v) for s,v in sorted(missing_by_symbol.items())},
        index_only_label_leads=excerpt_snapshots,
        unscanned_pdf_issuer_queue=[dict(symbol=s,count=pdf_counts[s],
             reverified_in_this_pass=s in first_pdf,
             separate_financial_definition_review=s in FINANCIAL_REVIEW_QUEUE)
             for s in sorted(pdf_counts)],
        first_two_alphabetical_nonbank_pdf_issuers=first_pdf,
        verified_unscanned_pdf_sources=pdf_snapshots,
        duplicate_report_identities=[dict(symbol=r["symbol"],fiscal_year=r["fiscal_year"],
            event_id=r["event_id"],previously_extracted=(r["symbol"],int(r["fiscal_year"])) in processed_pairs)
            for r in omitted if r["symbol"] in processed_issuers],
        next_action="First review GE's metadata weakness and ACN/ASML PDF structure without extracting amounts; for directly text-located annual NI/assets candidate pairs, MCD is the first new issuer, followed by NKE, PG and XOM. Retain full annual catalogs and every existing hold. A separate predeclared source-context step is required before selecting physical cells.",
        limitations=[
            "Census flags detect labels anywhere in a document; they do not prove annual versus quarterly NI, two asset dates in one report, monetary units, comparable scope or physical numeric cells.",
            "No new document text or financial amount was parsed in this pass; raw bytes were used for checksum verification only.",
            "An issuer-full-body metadata claim is not independent original-document authentication. Index excerpts remain non-admissible even if they contain all required labels.",
            "Unscanned PDF rows have unknown NI/assets labels, not absent financial statements. ACN/ASML checks verify bytes only.",
            "Financial-review queue is a research triage assumption, not a point-in-time sector classification or exclusion from the stock pool.",
            "Existing source gates were consulted for identities only; no old held source is reopened and no unprocessed same-issuer report is silently called new.",
            "Release/availability fields are inherited census metadata; trading-calendar, identity, currency, original-publication vintage and complete correction history still require separate verification.",
        ],
        financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,
        orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0,
        input_hashes=pins)
    for name,value in pins.items():
        if digest((ROOT/name).read_bytes())!=value:raise ValueError("Concurrent source mutation: "+name)
    with (ROOT/OUT).open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(path=OUT,sha256=digest((ROOT/OUT).read_bytes()),
                         counts=result["counts"],next_order=strong,pdf_first=first_pdf)))


if __name__=="__main__":
    main()

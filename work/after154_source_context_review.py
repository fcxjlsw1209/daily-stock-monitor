"""Save a bounded visual/source review; never extract values or run strategies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/after154_source_context_review.json"
PINS = {
    "work/after154_remaining_source_feasibility.json":
        "e3b25e067a4157ded4aba9bb980fd0c1056c6eb51e343f50c2a1151818f06a4c",
    "work/after154_source_context_probe.json":
        "f4e1fe0e63441eb70f71fbd7dd9840f4a17e6ddb6b4a6fa4d971e2a2ac8aaa2b",
    "work/after154_source_context_details.json":
        "9d3d75ef2867bc0b808da42ef9a9971bbc98177756ff8d66b6abd918a2805f3b",
}


def main():
    pins = {}

    def verify(path, expected=None):
        full = ROOT / path
        if full.is_symlink():
            raise ValueError("Symlink not accepted")
        raw = full.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected and digest != expected:
            raise ValueError("Changed source: " + path)
        pins[path] = digest
        return raw

    records = {p: json.loads(verify(p, h)) for p, h in PINS.items()}
    feasibility = records["work/after154_remaining_source_feasibility.json"]
    details = records["work/after154_source_context_details.json"]
    for path in (
        "work/after154_source_context_probe.py",
        "work/after154_source_context_details.py",
        "work/after154_source_context_review.py",
    ):
        verify(path)
    acn = [d for d in details["documents"] if d["symbol"] == "ACN"]
    asml = [d for d in details["documents"] if d["symbol"] == "ASML"]
    expected_years = [2015, 2016, 2017, 2018, 2019, 2020, 2022, 2024, 2025]
    if sorted(d["fiscal_year"] for d in acn) != expected_years:
        raise ValueError("ACN review scope changed")
    if any(d["symbol"] in feasibility["prior_covered_issuers"] for d in acn):
        raise ValueError("ACN is already covered")
    images = []
    for document in details["documents"]:
        source = document["source"]
        verify(source["path"], source["sha256"])
        for receipt in document["receipts_verified"]:
            verify(receipt["path"], receipt["sha256"])
        for image in document["visual_artifacts"]:
            verify(image["path"], image["sha256"])
            images.append(image)
    if len(images) != 10:
        raise ValueError("Visual scope changed")
    for row in acn:
        if not row["semantic_context_located"] or not row["publication_date_matches"]:
            raise ValueError("ACN semantic/date check absent")
    if len(asml) != 2 or any(d["total_assets_label_found"] for d in asml):
        raise ValueError("ASML scope changed")
    annual_catalog_years = sorted({r["fiscal_year"] for r in details["full_catalogs"]["ACN"]})
    missing = sorted(set(range(2015, 2026)) - set(annual_catalog_years))
    if missing != [2021, 2023]:
        raise ValueError("ACN catalog gap changed")
    ge = details["ge_review"]
    ge_source = ge["source"]
    ge_raw = json.loads(verify(ge_source["path"], ge_source["sha256"]))
    if ge_raw.get("original_bytes_available") is not False:
        raise ValueError("GE original-byte status changed")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "bounded_semantic_and_visual_source_review_not_admission",
        "scope": {
            "selection_basis": "Existing source census; nonbank issuer alphabetical order. No holdings, ranks, scores or performance inspected.",
            "pdf_documents_text_checked": 11,
            "acn_reports_with_required_statement_context": 9,
            "representative_pages_visually_reviewed_by_assistant": 10,
            "visual_claim_scope": "ACN FY2015/FY2019/FY2025 statement pairs; ASML FY2024/FY2025 release pages 1 and 3. Other ACN years received text-context and provenance checks only.",
            "visual_artifacts": images,
        },
        "acn": {
            "proposed_next_source_batch": expected_years,
            "proposed_batch_role": "Source extraction and independent semantic/numeric validation only; not a factor or trial registration.",
            "already_in_prior_original_or_five_batch_issuer_coverage": False,
            "candidate_reports": [
                {
                    "fiscal_year": r["fiscal_year"],
                    "source": r["source"],
                    "receipts": r["receipts_verified"],
                    "income_page": r["income_page"],
                    "balance_page": r["balance_page"],
                    "publication_date": r["publication_date"],
                    "first_session_strictly_after_publication": r["first_session_after_publication"],
                    "income_end": r["annual_current_period_end"],
                    "prior_assets_end": r["prior_assets_end"],
                } for r in acn
            ],
            "visual_findings": [
                "Formal consolidated income statements distinguish quarterly and full-year domains; use only the full-year domain.",
                "Each time domain also has percentage-of-revenue columns; financial values require explicit physical-column mapping.",
                "Consolidated total NET INCOME occurs before noncontrolling-interest attribution and parent-only net income. EPS numerator and adjusted reconciliations are different quantities.",
                "Consolidated balance sheets provide current and preceding August 31 TOTAL ASSETS in thousands of US dollars, matching the income statement monetary unit.",
                "FY2019 includes ASU 2014-09 adoption and prior-presentation revision/reclassification footnotes. Field-specific comparability must be checked; this review does not certify no effect on NI or asset endpoints.",
            ],
            "catalog_gaps": {
                "scope": "Existing pinned CENSUS ACN projection only; not an assertion that these releases cannot be found elsewhere.",
                "missing_fiscal_years_inside_2015_2025": missing,
                "required_action": "Find or explicitly identify the missing FY2021/FY2023 release events before continuous use. Do not treat absent census records as proof that no newer event existed; preserve unavailable-latest-event blocking and do not silently carry an older report through these holes.",
            },
            "not_yet_verified": [
                "Actual monetary values, signs, annual-column ownership and independent numeric agreement.",
                "Literal income-period start: prior September 1 is inferred from the annual August 31 endpoints and must remain an explicit assumption.",
                "Exact intraday publication timestamps; the proposed clock is the next pinned trading session strictly after the verified publication date.",
                "First-public-release version and complete restatement/correction history.",
                "FY2019 accounting/presentation change comparability for each required field.",
            ],
        },
        "asml": {
            "disposition": "Hold pending same-event full US GAAP statement attachments.",
            "reports": [
                {
                    "fiscal_year": r["fiscal_year"], "source": r["source"],
                    "publication_date": r["publication_date"],
                    "statement_attachment_links": r["statement_attachment_links"],
                } for r in asml
            ],
            "visual_findings": [
                "Both four-page releases contain rounded summary quarterly and annual net income but no total-assets table.",
                "The summary and financial-reporting sections explicitly refer readers to separately published quarterly US GAAP statements.",
                "US GAAP and IFRS are not interchangeable; the releases identify accounting differences.",
                "Annual-report publication dates mentioned in these releases are later. Such annual filings cannot be substituted as if available on the earlier earnings-release date.",
            ],
            "existing_older_catalog_scope": "FY2020-FY2023 local JSON sources explicitly lack original bytes; they remain holds, not extracted sources.",
        },
        "ge_fy2012": {
            "source": ge_source,
            "disposition": "Hold: the raw object explicitly says original_bytes_available=false and public_search_index_not_original_bytes.",
            "source_representation": ge_raw["source_representation"],
            "retrieved_at": ge_raw.get("retrieved_at"),
            "metadata_correction": "A census full-body label is not proof of original-file availability. Existing census bytes were preserved and this separate review records the conflict.",
            "clock_limit": "The January 2013 release predates the pinned 2015-onward session calendar used here; no new known-time session was inferred.",
        },
        "later_queue": {
            "issuer_order": ["MCD", "NKE", "PG", "XOM"],
            "counts_of_existing_claimed_full_body_label_leads": {"MCD": 4, "NKE": 15, "PG": 2, "XOM": 1},
            "status": "Metadata leads only; not semantically or numerically admitted by this review.",
        },
        "next_actions": [
            "Declare a separate ACN source-extraction scope covering all nine available annual PDFs, independent of later strategy results.",
            "Locate FY2021/FY2023 ACN event records or keep continuous coverage on hold across unavailable newer reports.",
            "Map the three required monetary cells per event with physical annual columns and independently validate labels, units, dates, signs and provenance.",
            "Resolve FY2019 accounting/presentation comparability and retain the actual available-time convention.",
            "Locate same-release ASML US GAAP statement attachments; keep GE FY2012 on hold until original evidence is available.",
            "Continue MCD, NKE, PG and XOM in the predetermined order only after this bounded batch is resolved.",
        ],
        "financial_values_extracted": 0,
        "ratios_computed": 0,
        "new_factors_computed": 0,
        "orders_or_returns_computed": 0,
        "new_admissions": 0,
        "new_strategy_trials": 0,
        "source_downloads": 0,
        "input_hashes": pins,
    }
    for path, expected in list(pins.items()):
        verify(path, expected)
    with OUT.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "path": str(OUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "verified_inputs": len(pins),
        "acn_source_scope": len(acn),
        "missing_acn_catalog_years": missing,
        "new_admissions": 0,
    }))


if __name__ == "__main__":
    main()

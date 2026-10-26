"""Independent count/clock/source-scope checks; no financial values or strategy."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def years(items, cutoff):
    return sorted({r["fiscal_year"] for r in items
                   if isinstance(r.get("fiscal_year"), int) and
                   isinstance(r.get("available_session"), str) and
                   r["available_session"] <= cutoff})


def assert_cov(actual, records, cutoff):
    expected = years(records, cutoff)
    assert actual["fiscal_years"] == expected
    assert actual["distinct_annual_years"] == len(expected)
    assert actual["at_least_four_annual_years"] == (len(expected) >= 4)
    assert actual["known_source_ids"] == sorted({
        r.get("source_id", r.get("accession")) for r in records
        if isinstance(r.get("fiscal_year"), int) and
        isinstance(r.get("available_session"), str) and
        r["available_session"] <= cutoff})


def no_amounts(value):
    if isinstance(value, dict):
        assert not set(value).intersection({
            "val", "displayed_amount", "annual_current_usd_millions",
            "annual_prior_usd_millions", "return", "returns", "growth", "cagr", "sharpe"})
        for item in value.values():
            no_amounts(item)
    elif isinstance(value, list):
        for item in value:
            no_amounts(item)


def main():
    summary = read(OUT / "SUMMARY.json")
    spec = read(OUT / "SPEC.json")
    assert summary["spec_sha256"] == sha(OUT / "SPEC.json")
    assert sha(ROOT / spec["code"]["path"]) == spec["code"]["sha256"]
    for entry in spec["inputs"].values():
        assert sha(ROOT / entry["path"]) == entry["sha256"]
    for entry in summary["outputs"].values():
        assert sha(ROOT / entry["path"]) == entry["sha256"]
    reports = read(OUT / "ANNUAL_REPORTS.json")
    filings = read(OUT / "REVENUE_FILING_METADATA.json")
    slots = read(OUT / "SLOTS.json")
    original_slots = read(ROOT / spec["inputs"]["slots"]["path"])
    issuers = read(OUT / "ISSUERS.json")
    assert len(slots) == 3904 and len(issuers) == 73
    assert Counter((s["slot_id"], s["symbol"], s["information_cutoff"]) for s in slots) == Counter(
        (s["slot_id"], s["symbol"], s["signal_date"]) for s in original_slots)
    assert len({s["information_cutoff"] for s in slots}) == 122
    ids = {r["symbol"]: r for r in issuers}
    checked_sources = {(r["source_path"], r["sha256"])
                       for r in read(ROOT / spec["inputs"]["msft_review"]["path"])["reports"]}
    good = [r for r in reports if r["present_hash_verified"] and
            r["catalog_claims_full_body"] and r["status"] == "ready"]
    for report in reports:
        assert report["total_revenue_source_cells_independently_verified"] == (
            report["symbol"] == "MSFT" and (report["path"], report["sha256"]) in checked_sources)
        assert report["accounting_chain_admitted"] is False
    for slot in slots:
        cutoff, symbol = slot["information_cutoff"], slot["symbol"]
        primary = [r for r in good if r["symbol"] == symbol]
        verified = [r for r in primary if r["total_revenue_source_cells_independently_verified"]]
        assert_cov(slot["primary_report_presence"], primary, cutoff)
        assert_cov(slot["independently_verified_revenue_reports"], verified, cutoff)
        ciks = ids[symbol]["current_ciks"] + ids[symbol]["additional_locating_only_ciks"]
        assert set(slot["companyfacts_filing_metadata_by_cik_not_joined"]) == set(ciks)
        for cik in ciks:
            assert_cov(slot["companyfacts_filing_metadata_by_cik_not_joined"][cik],
                       [r for r in filings if r["cik"] == cik], cutoff)
        assert slot["four_report_same_basis_chain_admitted"] is False
        assert slot["factor_admitted"] is False and slot["strategy_admitted"] is False
    for filing in filings:
        assert filing["filed"] < filing["available_session"]
        assert all(row["end"] <= filing["filed"] for row in filing["metadata"])
        assert filing["accounting_chain_admitted"] is False
    for name in summary["outputs"]:
        no_amounts(read(OUT / name))
    scope = []
    present = {(r["path"], r["sha256"]) for r in reports}
    # The five previously processed source batches use 147 catalog records.
    # Check their original-file references are already represented: no new parse.
    for n in range(1, 6):
        p = ROOT / f"outputs/monthly_financial_coverage_admission_batch{n}_v1/DATASET.json"
        catalog = read(p)["latest_event_catalog"]
        absent = [(r["evidence_path"], r["evidence_sha256"]) for r in catalog
                  if (r["evidence_path"], r["evidence_sha256"]) not in present]
        assert not absent
        scope.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p),
                      "catalog_records": len(catalog), "missing_source_references": absent})
    assert sum(r["catalog_records"] for r in scope) == 147
    counts = summary["counts"]
    assert counts["slots_with_four_primary_report_years"] == sum(
        s["primary_report_presence"]["at_least_four_annual_years"] for s in slots)
    assert counts["slots_with_four_verified_revenue_report_years"] == sum(
        s["four_report_total_revenue_presence_established"] for s in slots)
    assert counts["slots_with_four_current_cik_revenue_metadata_years"] == sum(
        s["any_current_cik_has_four_metadata_years"] for s in slots)
    result = {
        "status": "PASS_METADATA_COUNTS_CLOCKS_AND_BOUNDED_SOURCE_COVERAGE",
        "verification_code_sha256": sha(__file__), "spec_sha256": sha(OUT / "SPEC.json"),
        "summary_sha256": sha(OUT / "SUMMARY.json"),
        "original_slots_preserved": 3904, "symbols_preserved": 73, "cutoffs_preserved": 122,
        "current_and_locating_ciks_checked_separately": True,
        "no_exported_financial_amount_or_return_fields": True,
        "new_financial_or_strategy_calculations": 0,
        "all_five_completed_source_batch_catalog_references_already_included": scope,
        "accounting_chain_admission": False,
        "limitation": "Metadata-only cross-check; does not re-authenticate original statement values, issuer continuity or accounting comparability.",
    }
    target = OUT / "VALIDATION.json"
    if target.exists():
        assert read(target) == result
    else:
        with target.open("x") as f:
            json.dump(result, f, sort_keys=True, indent=2)
            f.write("\n")
    print(json.dumps({"status": result["status"], "validation_sha256": sha(target)}))


if __name__ == "__main__":
    main()

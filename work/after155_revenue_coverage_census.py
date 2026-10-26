"""Offline source-metadata census; no financial values or strategy calculations.

Annual source files, SEC revenue-tagged filing metadata, and independently
checked revenue cells are distinct coverage levels. None admits a four-year
accounting chain. Only explicitly catalogued files are inspected.
"""
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monthly_revenue_coverage156"
CODE = "work/after155_revenue_coverage_census.py"
INPUTS = {
    "scope_spec": "outputs/monthly_event_scope156/SPEC.json",
    "scope_manifest": "outputs/monthly_event_scope156/MANIFEST.json",
    "slots": "outputs/monthly_event_scope156/SLOTS.json",
    "events": "outputs/monthly_earnings_repair/EVENTS.json",
    "old_census": "outputs/monthly_financial_coverage_census_v1/CENSUS.json",
    "identity": "work/monthly_profitability141_identity_review.json",
    "inventory": "work/profitability141_migrated_cache_inventory_complete_20261019T202336581108Z.json",
    "calendar": "outputs/monthly_paired_target143/CALENDAR.json",
    "msft_review": "work/after155_revenue_extended_review.json",
    "msft_proposal": "outputs/monthly_revenue_source156_preparation/v2/SOURCE_ADMISSION_PROPOSAL.json",
}
TOTAL_TAGS = {
    "us-gaap": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "ifrs-full": ["Revenue"],
}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def local_path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    p.relative_to(ROOT)
    require("CloudStorage" not in p.parts and ".." not in p.parts, "Nonlocal path")
    cursor = ROOT
    for part in p.relative_to(ROOT).parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "Symlink source forbidden")
    require(p.resolve() == p, "Noncanonical path")
    return p


def relative(value):
    return str(local_path(value).relative_to(ROOT))


def raw_read(value):
    p = local_path(value)
    a = p.stat()
    raw = p.read_bytes()
    b = p.stat()
    require((a.st_size, a.st_mtime_ns, a.st_ino) ==
            (b.st_size, b.st_mtime_ns, b.st_ino), "Source changed during read")
    return raw


def read_json(value, drop_amounts=False):
    # Never expose companyfacts financial amount keys to census logic.
    hook = (lambda pairs: {k: v for k, v in pairs if k != "val"}) if drop_amounts else None
    return json.loads(raw_read(value), object_pairs_hook=hook)


def write_once(name, value):
    p = OUT / name
    with p.open("xb") as handle:
        handle.write(encode(value))
    return {"path": str(p.relative_to(ROOT)), "sha256": digest(raw_read(p))}


def valid_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def next_session(value, sessions):
    if not valid_date(value):
        return None
    i = bisect.bisect_right(sessions, value)
    return sessions[i] if i < len(sessions) else None


def full_body(kind):
    return (isinstance(kind, str) and kind.startswith("issuer_") and
            not any(s in kind for s in ("excerpt", "observation", "not_full",
                                        "notice_of_release_issuance")) and
            ("public_index" not in kind or "full_bytes" in kind))


def format_from_head(head):
    if head.startswith(b"%PDF"):
        return "pdf_requires_statement_or_visual_review"
    h = head.lstrip().lower()
    if h.startswith((b"{", b"[")):
        return "json_or_api_payload_requires_body_review"
    if b"<" in h:
        return "html_or_markup_requires_statement_review"
    return "text_or_binary_requires_format_review"


def inspect_file(path):
    p = local_path(path)
    if not p.exists():
        return {"status": "missing", "sha256": None, "bytes": None, "format": None}
    try:
        before = p.stat()
        h = hashlib.sha256()
        with p.open("rb") as handle:
            head = handle.read(512)
            h.update(head)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        after = p.stat()
        require((before.st_size, before.st_ino, before.st_mtime_ns) ==
                (after.st_size, after.st_ino, after.st_mtime_ns), "File changed")
        return {"status": "present", "sha256": h.hexdigest(), "bytes": after.st_size,
                "format": format_from_head(head)}
    except OSError as error:
        return {"status": "unreadable", "sha256": None, "bytes": None,
                "format": None, "error_type": type(error).__name__}


def check_scope(inputs):
    slots = inputs["slots"]
    require(len(slots) == 3904 and len({s["symbol"] for s in slots}) == 73,
            "Expected exactly 3904 original slots and 73 symbols")
    require(len({s["slot_id"] for s in slots}) == len(slots), "Duplicate slot id")
    require(len({s["signal_date"] for s in slots}) == 122, "Expected 122 monthly cutoffs")
    require(all(valid_date(s["signal_date"]) for s in slots), "Invalid cutoff")
    require(inputs["scope_spec"]["scope"]["stock_slots"] == len(slots), "Scope disagreement")
    require(inputs["scope_manifest"]["spec_sha256"] == digest(raw_read(INPUTS["scope_spec"])),
            "Scope manifest/spec binding")
    require(inputs["scope_manifest"]["output_hashes"]["SLOTS.json"] ==
            digest(raw_read(INPUTS["slots"])), "Original slot digest binding")
    for key in ("events", "calendar"):
        require(inputs["scope_spec"]["source_hashes"][INPUTS[key]] ==
                digest(raw_read(INPUTS[key])), "Scope input binding: " + key)
    require(len(inputs["events"]) == 3771, "Parent event catalog changed")
    sessions = inputs["calendar"]["sessions"]
    require(sessions == sorted(set(sessions)), "Calendar order")
    review = inputs["msft_review"]
    require(review["status"] == "pass_24_source_cells_and_dates_comparability_remains_bounded",
            "MSFT independent review not passed")
    require(review["proposal_comparison"]["sha256"] ==
            digest(raw_read(INPUTS["msft_proposal"])), "MSFT proposal binding")
    require(review["availability_verification"]["calendar_sha256"] ==
            digest(raw_read(INPUTS["calendar"])), "MSFT calendar binding")
    return set(s["symbol"] for s in slots)


def build_source_refs(inputs, symbols):
    """Only bounded catalog entries, no directory traversal or body parsing."""
    refs = {}

    def add(path, expected, role, symbol):
        if not path:
            return
        path = relative(path)
        entry = refs.setdefault(path, {"path": path, "expected_sha256s": set(),
                                       "roles": set(), "symbols": set()})
        if expected:
            entry["expected_sha256s"].add(expected)
        entry["roles"].add(role)
        entry["symbols"].add(symbol)

    for event in inputs["events"]:
        if event.get("symbol") in symbols and event.get("fiscal_quarter") == 4:
            add(event.get("evidence_path"), event.get("evidence_sha256"),
                "annual_event_primary", event["symbol"])
    for stock in inputs["old_census"]["stocks"]:
        if stock["symbol"] not in symbols:
            continue
        for item in stock.get("previous_admission_records", []):
            source = item.get("financial_attachment_source") or {}
            add(source.get("path"), source.get("sha256"), "annual_financial_attachment",
                stock["symbol"])
    bycik = defaultdict(set)
    for item in inputs["identity"]["rows"]:
        if item["symbol"] not in symbols:
            continue
        for registrant in item.get("current_registrants", []):
            bycik[registrant["cik"]].add(item["symbol"])
            add(registrant["evidence"], None, "current_identity_metadata", item["symbol"])
        # Locating-only predecessors are retained separately, never joined.
        for cik in item.get("additional_locating_ciks", []):
            bycik[cik].add(item["symbol"])
    for item in inputs["inventory"]["candidates"]:
        for symbol in sorted(bycik.get(item.get("cik"), set())):
            add(item["relative_path"], item["sha256"], "companyfacts_metadata", symbol)
            sidecar = item.get("origin_sidecar", {})
            if sidecar.get("exists"):
                add(sidecar["sidecar_path"], sidecar.get("sidecar_sha256"),
                    "companyfacts_origin_sidecar", symbol)
    for item in inputs["msft_review"]["reports"]:
        add(item["source_path"], item["sha256"], "independently_reviewed_revenue_report", "MSFT")
    result = []
    for path, item in sorted(refs.items()):
        item = {key: sorted(value) if isinstance(value, set) else value
                for key, value in item.items()}
        item.update(inspect_file(path))
        item["expected_hashes_match"] = (item["status"] == "present" and
            bool(item["expected_sha256s"]) and
            item["expected_sha256s"] == [item["sha256"]])
        result.append(item)
    return result


def freeze():
    require(not (OUT / "SPEC.json").exists(), "Already frozen; do not overwrite")
    inputs = {k: read_json(v) for k, v in INPUTS.items()}
    symbols = check_scope(inputs)
    refs = build_source_refs(inputs, symbols)
    spec = {
        "schema_version": 1, "stage": "FROZEN_METADATA_ONLY_REVENUE_CENSUS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": {"path": CODE, "sha256": digest(raw_read(CODE))},
        "inputs": {k: {"path": v, "sha256": digest(raw_read(v))} for k, v in INPUTS.items()},
        "scope": {"slots": 3904, "issuers": sorted(symbols), "months": 122},
        "source_refs": refs,
        "policy": {
            "information_cutoff": "slot.signal_date; formation_window retained, not substituted",
            "primary_annual": "catalog Q4 plus known annual attachments; one fiscal year counted once",
            "primary_revenue_presence": "unknown except independently reviewed matching MSFT files",
            "companyfacts": "metadata leads only; val removed from parsed objects; never amounts or original statements",
            "accepted_total_revenue_tag_leads": TOTAL_TAGS,
            "annual_duration_days_inclusive": [330, 400],
            "annual_forms": sorted(ANNUAL_FORMS),
            "companyfacts_clock": "next frozen session strictly after filed date, not original earnings-release clock",
            "companyfacts_current_filing_period": "latest annual duration end within accession; no backward-dated comparative rows",
            "four_report_definition": "four distinct annual report fiscal years present by cutoff; longest consecutive fiscal-year run separately shown",
            "companyfacts_historical_identity": "current identity and locating predecessor kept separate, never pooled",
            "accounting_basis": "no comparability or four-level chain admitted; no amount equality inference",
            "body_work": "hash and format signature only; no new original-report text extraction",
            "network_requests": 0, "strategy_trials": 0, "features_fits_returns": 0,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ref = write_once("SPEC.json", spec)
    print(json.dumps({"spec": ref, "bounded_source_files": len(refs)}))


def refs_valid(spec):
    require(digest(raw_read(CODE)) == spec["code"]["sha256"], "Code changed after freeze")
    for item in spec["inputs"].values():
        require(digest(raw_read(item["path"])) == item["sha256"], "Input changed: " + item["path"])
    refs = {}
    for ref in spec["source_refs"]:
        observed = inspect_file(ref["path"])
        for key in ("status", "sha256", "bytes", "format"):
            require(observed[key] == ref[key], "Source changed after freeze: " + ref["path"])
        refs[ref["path"]] = ref
    return refs


def source_ok(refs, path, expected=None):
    if not path:
        return False
    ref = refs.get(relative(path), {})
    return (ref.get("status") == "present" and
            (expected is None or ref.get("sha256") == expected) and
            len(ref.get("expected_sha256s", [])) <= 1)


def primary_reports(inputs, refs, symbols):
    reports = []
    checked = {(relative(r["source_path"]), r["sha256"]): r
               for r in inputs["msft_review"]["reports"]}
    olddocs = {r["event_id"]: r for r in inputs["old_census"]["annual_documents"]}
    for event in inputs["events"]:
        symbol = event.get("symbol")
        if symbol not in symbols or event.get("fiscal_quarter") != 4:
            continue
        path = relative(event["evidence_path"]) if event.get("evidence_path") else None
        sha = event.get("evidence_sha256")
        old = olddocs.get(event["event_id"], {})
        review = checked.get((path, sha)) if symbol == "MSFT" else None
        item = {k: event.get(k) for k in ("event_id", "symbol", "fiscal_year",
            "official_actual_period_end", "publication_date", "available_session", "status",
            "source_kind", "source_url", "first_public_time_verified")}
        item.update(source_id="event:" + event["event_id"], path=path, sha256=sha,
                    present_hash_verified=source_ok(refs, path, sha),
                    catalog_claims_full_body=full_body(event.get("source_kind")),
                    format=refs.get(path, {}).get("format"),
                    previous_text_scan=old.get("text_scanned", False),
                    total_revenue_source_cells_independently_verified=bool(review),
                    revenue_tag_presence="independently_verified" if review else "not_examined",
                    accounting_chain_admitted=False, financial_values_extracted=False)
        if review:
            require(review["fiscal_year"] == event["fiscal_year"], "MSFT fiscal year mismatch")
            require(review["first_conservative_session"] == event["available_session"], "MSFT clock mismatch")
        reports.append(item)
    existing = {(r["symbol"], r["path"], r["sha256"]) for r in reports}
    for stock in inputs["old_census"]["stocks"]:
        if stock["symbol"] not in symbols:
            continue
        for entry in stock.get("previous_admission_records", []):
            source = entry.get("financial_attachment_source") or {}
            if not source.get("path"):
                continue
            path, sha = relative(source["path"]), source.get("sha256")
            key = (stock["symbol"], path, sha)
            if key in existing:
                continue
            year = re.search(r":FY(\d{4}):", entry["event_id"])
            if not year:
                continue
            existing.add(key)
            review = checked.get((path, sha)) if stock["symbol"] == "MSFT" else None
            reports.append({"source_id": "attachment:" + entry["event_id"],
                "event_id": entry["event_id"], "symbol": stock["symbol"],
                "fiscal_year": int(year.group(1)), "available_session": entry.get("available_from"),
                "publication_date": entry["event_id"].split(":")[-1],
                "official_actual_period_end": review["current_period_end"] if review else None,
                "status": "ready", "source_kind": "previous_annual_financial_attachment_metadata",
                "source_url": source.get("url"), "path": path, "sha256": sha,
                "present_hash_verified": source_ok(refs, path, sha), "catalog_claims_full_body": True,
                "format": refs.get(path, {}).get("format"), "previous_text_scan": None,
                "total_revenue_source_cells_independently_verified": bool(review),
                "revenue_tag_presence": "independently_verified" if review else "not_examined",
                "first_public_time_verified": False, "accounting_chain_admitted": False,
                "financial_values_extracted": False})
    return sorted(reports, key=lambda r: (r["symbol"], r["fiscal_year"], r["source_id"]))


def annual_metadata(facts, sessions):
    """Project no amount fields; one current annual period per accession."""
    filings = defaultdict(list)
    rejects = Counter()
    observed_tags, other_revenue_names = [], []
    for taxonomy, concepts in facts.get("facts", {}).items():
        for name in concepts:
            if re.search("revenue|sales", name, flags=re.I) and name not in TOTAL_TAGS.get(taxonomy, []):
                other_revenue_names.append(taxonomy + ":" + name)
        for tag in TOTAL_TAGS.get(taxonomy, []):
            concept = concepts.get(tag)
            if not concept:
                continue
            observed_tags.append(taxonomy + ":" + tag)
            for unit, rows in concept.get("units", {}).items():
                for row in rows:
                    if row.get("form") not in ANNUAL_FORMS:
                        rejects["nonannual_form"] += 1
                        continue
                    if not all(valid_date(row.get(k)) for k in ("start", "end", "filed")):
                        rejects["missing_or_invalid_period_filed_metadata"] += 1
                        continue
                    duration = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days + 1
                    if not 330 <= duration <= 400:
                        rejects["nonannual_duration"] += 1
                        continue
                    if row.get("fp") not in (None, "FY"):
                        rejects["nonannual_fiscal_period"] += 1
                        continue
                    if not row.get("accn") or row["filed"] < row["end"]:
                        rejects["invalid_accession_or_filing_precedes_end"] += 1
                        continue
                    available = next_session(row["filed"], sessions)
                    if available is None:
                        rejects["filing_outside_calendar"] += 1
                        continue
                    metadata = {k: row.get(k) for k in ("start", "end", "filed", "form", "accn", "fy", "fp")}
                    metadata.update(tag=taxonomy + ":" + tag, unit=unit,
                                    duration_days_inclusive=duration, available_session=available)
                    filings[row["accn"]].append(metadata)
    records = []
    for accession, rows in sorted(filings.items()):
        last_end = max(r["end"] for r in rows)
        current = [r for r in rows if r["end"] == last_end]
        filed = sorted({r["filed"] for r in current})
        years = sorted({r["fy"] for r in current if isinstance(r["fy"], int)})
        if len(filed) != 1 or len(years) > 1:
            rejects["ambiguous_accession_filing_or_fiscal_year"] += 1
            continue
        if years and not years[0] <= int(last_end[:4]) <= years[0] + 1:
            # A later annual filing may carry only a discontinued old tag.
            # Its old comparative period does not establish current revenue.
            rejects["no_current_fiscal_year_total_revenue_period"] += 1
            continue
        unique = {encode(r): r for r in current}
        current = [unique[k] for k in sorted(unique)]
        records.append({"accession": accession, "filed": filed[0],
            "available_session": next_session(filed[0], sessions),
            "fiscal_year": years[0] if years else int(last_end[:4]),
            "fiscal_year_basis": "filed_fy" if years else "end_calendar_year_fallback_metadata_only",
            "current_annual_period_end": last_end, "metadata": current,
            "comparative_rows_not_promoted_to_prior_release": len(rows) - len(current),
            "original_report_locally_verified": False, "source_values_admitted": False,
            "historical_identity_admitted": False, "accounting_chain_admitted": False})
    return {"filings": records, "observed_total_revenue_tag_leads": sorted(observed_tags),
            "unaccepted_revenue_or_sales_tag_names": sorted(other_revenue_names),
            "rejection_counts": dict(sorted(rejects.items()))}


def year_coverage(items, cutoff):
    eligible = [r for r in items if valid_date(r.get("available_session")) and
                r["available_session"] <= cutoff and isinstance(r.get("fiscal_year"), int)]
    years = sorted({r["fiscal_year"] for r in eligible})
    longest, run, previous = 0, 0, None
    for year in years:
        run = run + 1 if previous is not None and year == previous + 1 else 1
        longest, previous = max(longest, run), year
    return {"fiscal_years": years, "distinct_annual_years": len(years),
            "at_least_four_annual_years": len(years) >= 4,
            "longest_consecutive_annual_year_run": longest,
            "at_least_four_consecutive_annual_years": longest >= 4,
            "known_source_ids": sorted({r.get("source_id", r.get("accession")) for r in eligible})}


def collect(spec, inputs, refs):
    symbols = check_scope(inputs)
    reports = primary_reports(inputs, refs, symbols)
    identities = {r["symbol"]: r for r in inputs["identity"]["rows"]}
    sources, filings = [], []
    for item in inputs["inventory"]["candidates"]:
        path = relative(item["relative_path"])
        ref = refs.get(path)
        if not ref or "companyfacts_metadata" not in ref["roles"]:
            continue
        source = {"path": path, "sha256": item["sha256"], "cik": item["cik"],
            "symbols": ref["symbols"], "present_hash_verified": source_ok(refs, path, item["sha256"]),
            "origin_sidecar_verified_in_prior_inventory": item.get("origin_sidecar", {}).get("verified_url_hash_time", False),
            "origin_sidecar": item.get("origin_sidecar", {}),
            "original_statement": False, "values_examined": False}
        if source["present_hash_verified"]:
            data = read_json(path, drop_amounts=True)
            cik = str(data.get("cik", "")).zfill(10)
            source["content_cik_matches_inventory"] = cik == item["cik"]
            if source["content_cik_matches_inventory"]:
                meta = annual_metadata(data, inputs["calendar"]["sessions"])
                source.update({k: v for k, v in meta.items() if k != "filings"})
                source["annual_filing_metadata_count"] = len(meta["filings"])
                for filing in meta["filings"]:
                    filings.append(dict(filing, source_path=path, source_sha256=item["sha256"],
                                        cik=item["cik"], symbols=ref["symbols"]))
        sources.append(source)
    rows, slotrows = [], []
    for symbol in sorted(symbols):
        slots = [s for s in inputs["slots"] if s["symbol"] == symbol]
        identity = identities[symbol]
        currentciks = sorted({r["cik"] for r in identity.get("current_registrants", [])})
        locating = sorted(set(identity.get("additional_locating_ciks", [])))
        candidates = [r for r in reports if r["symbol"] == symbol]
        good = [r for r in candidates if r["present_hash_verified"] and r["catalog_claims_full_body"]
                and r.get("status") == "ready"]
        checked = [r for r in good if r["total_revenue_source_cells_independently_verified"]]
        companysources = [r for r in sources if symbol in r["symbols"]]
        bycik = {cik: [f for f in filings if f["cik"] == cik] for cik in currentciks + locating}
        for slot in slots:
            cutoff = slot["signal_date"]
            pcov, vcov = year_coverage(good, cutoff), year_coverage(checked, cutoff)
            ccov = {cik: year_coverage(values, cutoff) for cik, values in bycik.items()}
            slotrows.append({"slot_id": slot["slot_id"], "symbol": symbol,
                "signal_index": slot["signal_index"], "information_cutoff": cutoff,
                "formation_window": slot["formation_window"],
                "primary_report_presence": pcov, "independently_verified_revenue_reports": vcov,
                "companyfacts_filing_metadata_by_cik_not_joined": ccov,
                "any_current_cik_has_four_metadata_years": any(ccov[c]["at_least_four_annual_years"] for c in currentciks),
                "any_locating_cik_has_four_metadata_years": any(ccov[c]["at_least_four_annual_years"] for c in locating),
                "four_report_total_revenue_presence_established": vcov["at_least_four_annual_years"],
                "four_report_same_basis_chain_admitted": False,
                "factor_admitted": False, "strategy_admitted": False})
        ownslots = [s for s in slotrows if s["symbol"] == symbol]
        issues = ["four_report_accounting_basis_and_historical_vintage_not_admitted"]
        if not checked:
            issues.append("annual_total_revenue_values_and_statement_scope_not_independently_verified")
        if not currentciks:
            issues.append("current_cik_not_established_in_bounded_identity_review")
        if not companysources:
            issues.append("no_matching_companyfacts_cache_in_bounded_inventory")
        elif not any(s.get("annual_filing_metadata_count", 0) for s in companysources):
            issues.append("no_accepted_total_revenue_annual_filing_metadata")
        if "conflict_or_gap" in identity.get("status", ""):
            issues.append("historical_issuer_bridge_requires_separate_review")
        if any(not r["present_hash_verified"] for r in candidates):
            issues.append("missing_or_changed_primary_source")
        if any(not r["catalog_claims_full_body"] for r in candidates):
            issues.append("some_catalog_sources_are_excerpts_not_full_original_reports")
        if any((r["format"] or "").startswith("pdf") and not r["total_revenue_source_cells_independently_verified"] for r in candidates):
            issues.append("pdf_total_revenue_and_period_columns_require_extraction_or_visual_review")
        rows.append({"symbol": symbol, "candidate_slots": len(slots),
            "first_cutoff": min(s["signal_date"] for s in slots),
            "last_cutoff": max(s["signal_date"] for s in slots),
            "current_ciks": currentciks, "additional_locating_only_ciks": locating,
            "identity_review_status": identity["status"],
            "identity_note": identity.get("note"), "historical_identity_admitted": False,
            "annual_source_candidates": len(candidates), "primary_full_body_files_verified": len(good),
            "primary_source_ids": [r["source_id"] for r in candidates],
            "source_format_counts": dict(Counter(r["format"] or "unknown" for r in candidates)),
            "companyfacts_source_paths": [s["path"] for s in companysources],
            "slots_with_four_primary_report_years": sum(s["primary_report_presence"]["at_least_four_annual_years"] for s in ownslots),
            "slots_with_four_current_cik_revenue_metadata_years": sum(s["any_current_cik_has_four_metadata_years"] for s in ownslots),
            "slots_with_four_verified_revenue_report_years": sum(s["four_report_total_revenue_presence_established"] for s in ownslots),
            "slots_with_four_same_basis_admitted_years": 0,
            "next_source_action": "Extract annual consolidated total revenue and both period identities from catalog-pinned full statements; inspect accounting notes and original release time before admitting a chain",
            "gaps": issues})
    slotrows.sort(key=lambda s: (s["signal_index"], s["slot_id"]))
    counts = {"issuers": len(rows), "slots": len(slotrows), "annual_source_candidates": len(reports),
        "bounded_source_files": len(refs), "source_files_present": sum(r["status"] == "present" for r in refs.values()),
        "missing_or_unreadable_source_files": sum(r["status"] != "present" for r in refs.values()),
        "source_expected_hash_conflicts": sum(bool(r["expected_sha256s"]) and not r["expected_hashes_match"] for r in refs.values()),
        "matching_companyfacts_sources": len(sources), "annual_total_revenue_filing_metadata_records": len(filings),
        "issuers_with_four_primary_years_in_any_slot": sum(r["slots_with_four_primary_report_years"] > 0 for r in rows),
        "slots_with_four_primary_report_years": sum(r["slots_with_four_primary_report_years"] for r in rows),
        "issuers_with_four_current_cik_metadata_years_in_any_slot": sum(r["slots_with_four_current_cik_revenue_metadata_years"] > 0 for r in rows),
        "slots_with_four_current_cik_revenue_metadata_years": sum(r["slots_with_four_current_cik_revenue_metadata_years"] for r in rows),
        "issuers_with_four_verified_revenue_report_years": sum(r["slots_with_four_verified_revenue_report_years"] > 0 for r in rows),
        "slots_with_four_verified_revenue_report_years": sum(r["slots_with_four_verified_revenue_report_years"] for r in rows),
        "slots_with_four_same_basis_admitted_years": 0,
        "financial_amounts_inspected": 0, "new_value_extractions": 0, "features_computed": 0,
        "models_fit": 0, "new_strategy_trials": 0, "network_requests": 0}
    return reports, sources, filings, rows, slotrows, counts


def run(spec_sha):
    require(digest(raw_read(OUT / "SPEC.json")) == spec_sha, "Wrong frozen spec SHA")
    require(not (OUT / "SUMMARY.json").exists(), "Census already published")
    spec = read_json(OUT / "SPEC.json")
    refs = refs_valid(spec)
    inputs = {k: read_json(v["path"]) for k, v in spec["inputs"].items()}
    reports, sources, filings, rows, slots, counts = collect(spec, inputs, refs)
    outputs = {}
    for name, value in (("ANNUAL_REPORTS.json", reports), ("COMPANYFACTS_SOURCES.json", sources),
                        ("REVENUE_FILING_METADATA.json", filings), ("ISSUERS.json", rows),
                        ("SLOTS.json", slots)):
        outputs[name] = write_once(name, value)
    summary = {"schema_version": 1, "status": "SOURCE_METADATA_CENSUS_COMPLETE_NOT_CHAIN_ADMISSION",
        "spec_sha256": spec_sha, "outputs": outputs, "counts": counts,
        "cutoff_policy": spec["policy"]["information_cutoff"],
        "important_distinctions": [
            "Primary report coverage is a source-presence lead, not confirmation of total revenue in that report.",
            "Companyfacts revenue tags are filing metadata, not original statements, equal accounting bases, or first earnings release times.",
            "Counts never splice current and locating predecessor CIKs; each CIK is evaluated independently.",
            "Four verified MSFT reports can cross its 2018 accounting transition; counts do not certify a chain.",
            "Only own-year reports count here; later comparative columns are not reassigned to earlier release dates.",
            "All 3904 slots remain in scope, including those with no source leads; no universe or observation changes.",
        ], "next_step": "Predeclare a bounded alphabetical source set and revenue extraction/restatement policy; independently verify annual periods, currency, scope and timing; separately assess four-level comparability."}
    write_once("SUMMARY.json", summary)
    md = ["# Annual revenue source coverage census", "", "Metadata and source presence only; no strategy or accounting-chain admission.", "",
          "| Coverage level | Issuers with coverage in at least one slot | Covered slots / 3904 |", "|---|---:|---:|",
          f"| Four primary annual report years present | {counts['issuers_with_four_primary_years_in_any_slot']} | {counts['slots_with_four_primary_report_years']} |",
          f"| Four current-CIK revenue-tag filing metadata years | {counts['issuers_with_four_current_cik_metadata_years_in_any_slot']} | {counts['slots_with_four_current_cik_revenue_metadata_years']} |",
          f"| Four independently checked revenue report years | {counts['issuers_with_four_verified_revenue_report_years']} | {counts['slots_with_four_verified_revenue_report_years']} |",
          "| Four-year comparable chains admitted by this census | 0 | 0 |", "",
          "Cutoff: each original slot's signal date. The formation window is retained separately.", "",
          "## Per-issuer coverage", "", "| Issuer | Slots | Primary reports ≥4 | SEC revenue metadata ≥4 | Verified revenue reports ≥4 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        md.append(f"| {row['symbol']} | {row['candidate_slots']} | {row['slots_with_four_primary_report_years']} | {row['slots_with_four_current_cik_revenue_metadata_years']} | {row['slots_with_four_verified_revenue_report_years']} |")
    md.extend(["", "## Limits", ""] + ["- " + s for s in summary["important_distinctions"]])
    with (OUT / "SUMMARY.md").open("x") as handle:
        handle.write("\n".join(md) + "\n")
    print(json.dumps({"summary_sha256": digest(raw_read(OUT / "SUMMARY.json")), "counts": counts}))


class SyntheticTests(unittest.TestCase):
    sessions = ["2018-01-02", "2018-02-01", "2019-02-01", "2020-02-03", "2021-02-01", "2022-02-01"]

    def fixture(self, **update):
        row = {"start": "2017-01-01", "end": "2017-12-31", "filed": "2018-01-31",
               "form": "10-K", "accn": "a", "fy": 2017, "fp": "FY", "val": object()}
        row.update(update)
        return row

    def facts(self, rows, tag="Revenues"):
        return {"facts": {"us-gaap": {tag: {"units": {"USD": rows}}}}}

    def test_date_only_next_session(self):
        self.assertEqual(next_session("2018-02-01", self.sessions), "2019-02-01")

    def test_weekend_next_session(self):
        self.assertEqual(next_session("2020-02-01", self.sessions), "2020-02-03")

    def test_metadata_never_accesses_amount(self):
        result = annual_metadata(self.facts([self.fixture()]), self.sessions)
        self.assertEqual(len(result["filings"]), 1)
        self.assertNotIn("val", result["filings"][0]["metadata"][0])

    def test_quarter_duration_rejected(self):
        self.assertFalse(annual_metadata(self.facts([self.fixture(start="2017-10-01")]), self.sessions)["filings"])

    def test_quarter_form_rejected(self):
        self.assertFalse(annual_metadata(self.facts([self.fixture(form="10-Q")]), self.sessions)["filings"])

    def test_component_tag_not_admitted(self):
        result = annual_metadata(self.facts([self.fixture()], "SalesRevenueGoodsNet"), self.sessions)
        self.assertEqual(result["filings"], [])
        self.assertEqual(result["unaccepted_revenue_or_sales_tag_names"], ["us-gaap:SalesRevenueGoodsNet"])

    def test_comparative_not_backdated(self):
        old = self.fixture(start="2016-01-01", end="2016-12-31")
        result = annual_metadata(self.facts([old, self.fixture()]), self.sessions)["filings"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["current_annual_period_end"], "2017-12-31")
        self.assertEqual(result[0]["available_session"], "2018-02-01")

    def test_duplicate_rows_do_not_make_four_years(self):
        r = annual_metadata(self.facts([self.fixture()] * 4), self.sessions)["filings"]
        self.assertFalse(year_coverage(r, "2022-02-01")["at_least_four_annual_years"])

    def test_old_tag_only_comparative_is_not_current_filing(self):
        rows = [self.fixture(fy=2019, filed="2020-01-31")]
        self.assertFalse(annual_metadata(self.facts(rows), self.sessions)["filings"])

    def test_future_report_excluded(self):
        rows = [{"fiscal_year": y, "available_session": f"{y+1}-02-01", "source_id": str(y)} for y in range(2017, 2021)]
        self.assertFalse(year_coverage(rows, "2020-12-31")["at_least_four_annual_years"])
        self.assertTrue(year_coverage(rows, "2021-02-01")["at_least_four_annual_years"])

    def test_nonconsecutive_not_consecutive(self):
        rows = [{"fiscal_year": y, "available_session": "2022-02-01", "source_id": str(y)} for y in [2015, 2017, 2019, 2021]]
        result = year_coverage(rows, "2022-02-01")
        self.assertTrue(result["at_least_four_annual_years"])
        self.assertFalse(result["at_least_four_consecutive_annual_years"])

    def test_invalid_dates_excluded(self):
        self.assertFalse(annual_metadata(self.facts([self.fixture(filed="notdate")]), self.sessions)["filings"])

    def test_issuer_excerpt_not_full(self):
        self.assertFalse(full_body("issuer_original_release_public_index_excerpt"))
        self.assertTrue(full_body("issuer_original_release_pdf_full_bytes"))

    def test_pdf_signature_not_extension(self):
        self.assertTrue(format_from_head(b"%PDF-1.7").startswith("pdf"))

    def test_empty_coverage(self):
        self.assertEqual(year_coverage([], "2022-02-01")["distinct_annual_years"], 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.freeze:
        freeze()
    else:
        require(args.spec_sha, "--spec-sha is required")
        run(args.spec_sha)


if __name__ == "__main__":
    main()

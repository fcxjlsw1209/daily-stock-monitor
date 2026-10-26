"""Bounded INTC/KLAC source metadata review; monetary cells are never emitted.

Run only with run_local.sh.  Reads the frozen census and its 19 full HTML
originals.  All 32 catalog events remain in the report.  No admission or
strategy computation is performed.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid, external_context

ROOT = Path(__file__).resolve().parents[1]
CENSUS = "outputs/monthly_financial_coverage_census_v1/CENSUS.json"
CONCEPT = "work/after149_coverage_extension_concept.json"
OUTPUT = "work/after150_next_source_context.json"
EXPECTED_CENSUS = "de95906ef3d752890d28027a9b52cfd7decc7aeb1bc0f09f2785f1826a3500fc"
TARGET = re.compile(
    r"^(?:net\s+(?:income|loss)(?:\s*\(loss\))?(?:\s+attributable\s+to.+)?"
    r"|total\s+assets|stock[\s-]based compensation|share[\s-]based compensation"
    r"|net\s+cash\s+.*(?:operating activities|operations)"
    r"|adjustments to reconcile net.+|cash flows? from operating activities)"
    r"\s*(?:\(\d+\))?:?$", re.I)
TITLE = re.compile(r"(?:condensed\s+)?consolidated\s+"
                   r"(?:(?:unaudited|condensed|summary)\s+)*(?:balance\s+sheets?"
                   r"|statements?\s+of\s+(?:income|operations|cash\s+flows?))", re.I)
BODY = re.compile(r"^(?:assets|current assets|cash(?: and cash equivalents)?"
                  r"|net revenues?|revenues?|net sales|sales|operating activities"
                  r"|cash flows? from operating activities|net income(?:\s*\(loss\))?)"
                  r"(?:\s*\(\d+\))?:?$", re.I)
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
         r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
HEADER = re.compile(
    r"^(?:\(?unaudited\)?|fiscal years?|(?:fiscal\s+)?years?(?:\s+ended)?"
    r"|(?:three|six|nine|twelve|3|6|9|12)\s+months?(?:\s+ended)?"
    r"|(?:52|53)\s+weeks?(?:\s+ended)?|quarter(?:s)?(?:\s+ended)?"
    r"|(?:fiscal\s+)?(?:20\d{2}|19\d{2})"
    r"|" + MONTH + r"(?:\s+\d{1,2})?,?(?:\s+(?:20\d{2}|19\d{2}))?"
    r"|(?:\d{1,2},?\s+)?(?:20\d{2}|19\d{2})"
    r"|(?:as of|for the|for|ended|and|at)"
    r"|(?:.*(?:years?|months?|weeks?) ended\s+)?" + MONTH +
    r"\s+\d{1,2},?(?:\s+(?:20\d{2}|19\d{2}))?)$", re.I)
UNIT = re.compile(r"^\(?\s*(?:(?:amounts|dollars|shares)\s+)?(?:in\s+)?"
                  r"(?:millions|thousands|billions)(?:,?\s+except[^$%\d]*)?\s*\)?$", re.I)
ACCOUNTING = re.compile(r"\b(?:reclassif\w*|restat\w*|recast\w*|retrospect\w*"
                        r"|adopt\w*|accounting (?:standard|principle|change)\w*"
                        r"|discontinued operations|noncontrolling interests?"
                        r"|held[\s-]for[\s-]sale|tax reform|transition tax|deferred tax)\b", re.I)
IDENTITY = {
    "INTC": re.compile(r"Intel Corporation|(?:NASDAQ|Nasdaq)\s*[:(]\s*INTC\)?", re.I),
    "KLAC": re.compile(r"KLA(?:-|\s+)Tencor Corporation|KLA Corporation"
                       r"|(?:NASDAQ|Nasdaq)\s*[:(]\s*KLAC\)?", re.I),
}
CIKS = {"INTC": "0000050863", "KLAC": "0000319201"}


def norm(text):
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_local(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Non-local source path")
    return path.read_bytes()


def metadata_cell(c):
    return {"row": c["row"], "physical_cell": c["physical_cell"],
            "columns": c["columns"], "rowspan": c["rowspan"], "text": norm(c["text"])}


def is_metadata(text):
    return bool(HEADER.fullmatch(text) or UNIT.fullmatch(text)
                or (len(text) < 220 and TITLE.search(text)
                    and not re.search(r"[$%]", text)))


def inspect(event):
    raw = read_local(event["evidence_path"])
    if sha(raw) != event["evidence_sha256"]:
        raise ValueError("Source hash mismatch " + event["event_id"])
    rp = str(Path(event["evidence_path"]).with_suffix(".meta.json"))
    receipt_raw = read_local(rp)
    receipt = json.loads(receipt_raw)
    if (receipt.get("sha256") != sha(raw)
            or receipt.get("url") != event["source_url"]
            or receipt.get("status", receipt.get("http_status")) != 200
            or not receipt.get("retrieved_at")):
        raise ValueError("Receipt mismatch " + event["event_id"])
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    tables, rejected = [], []
    for ti, table in enumerate(soup.find_all("table")):
        if table.find("table"):
            rejected.append({"table": ti, "reason": "nested_table_container"})
            continue
        try:
            rows, grid, physical = table_grid(table)
        except ValueError:
            rejected.append({"table": ti, "reason": "unsupported_grid"})
            continue
        targets = [c for c in physical.values() if TARGET.fullmatch(norm(c["text"]))]
        if not targets:
            continue
        first_target = min(c["row"] for c in targets)
        starts = [c["row"] for c in physical.values()
                  if c["row"] <= first_target and BODY.fullmatch(norm(c["text"]))]
        header_stop = min(starts) if starts else first_target
        headers = [metadata_cell(c) for c in physical.values()
                   if c["row"] < header_stop and is_metadata(norm(c["text"]))]
        # Outside context is bounded at the preceding table. Retain only
        # metadata strings; no body amounts or prose financial summaries.
        external = [norm(t) for t in external_context(table) if is_metadata(norm(t))]
        target_rows = []
        for c in targets:
            coords = []
            for other in physical.values():
                if other["row"] != c["row"]:
                    continue
                text = norm(other["text"])
                role = {"": "empty", "$": "currency", "(": "open_parenthesis",
                        ")": "close_parenthesis", "-": "dash", "—": "dash"}.get(text, "opaque")
                coords.append({"physical_cell": other["physical_cell"],
                               "columns": other["columns"], "rowspan": other["rowspan"],
                               "role": "row_label" if other is c else role})
            target_rows.append({"row": c["row"], "label_cell": c["physical_cell"],
                                "label": norm(c["text"]), "cells": coords})
        tables.append({"table": ti, "rows": len(rows), "header_stop_row": header_stop,
                       "headers": headers, "external_metadata": external,
                       "target_rows": target_rows,
                       "dollar_symbol_present": "$" in table.get_text(),
                       "other_currency_tokens": sorted(set(re.findall(
                           r"€|£|¥|\b(?:EUR|GBP|JPY|CAD|HKD|AUD)\b", table.get_text())))})
    text = norm(soup.get_text(" ", strip=True))
    flags, excerpts = set(), []
    for match in ACCOUNTING.finditer(text):
        flags.add(match.group(0).lower())
        snippet = text[max(0, match.start()-90):match.end()+230]
        # No financial numerals leave the reader; dates and standard numbers
        # in these prose excerpts are deliberately redacted too.
        snippet = re.sub(r"\d[\d,.\s]*", "[number]", snippet)
        if snippet not in excerpts:
            excerpts.append(snippet)
    pre_metadata = []
    for pi, pre in enumerate(soup.find_all("pre")):
        records = []
        for li, line in enumerate(pre.get_text().splitlines()):
            text_line = norm(line)
            if is_metadata(text_line):
                records.append({"line": li, "kind": "header", "text": text_line})
                continue
            # Prefix only: never emit the remainder of a fixed-width value row.
            match = re.match(r"^(Net income(?: \(loss\))?|Total assets)\b", text_line, re.I)
            if match:
                records.append({"line": li, "kind": "label_only", "text": match.group(1)})
        pre_metadata.append({"pre_block": pi, "metadata_lines": records,
                             "numeric_columns_not_reviewed": True})
    return {"source_sha_verified": True,
            "receipt": {"path": rp, "sha256": sha(receipt_raw),
                        "status": 200, "retrieved_at": receipt["retrieved_at"],
                        "url_and_raw_sha_match": True},
            "identity_anchors": sorted(set(m.group(0) for m in IDENTITY[event["symbol"]].finditer(text))),
            "html_table_count": len(soup.find_all("table")),
            "preformatted_metadata": pre_metadata,
            "statement_tables": tables, "table_structure_gaps": rejected,
            "accounting_review_terms": sorted(flags),
            "accounting_standard_identifiers": sorted(set(m.group(0) for m in re.finditer(
                r"\b(?:ASC|ASU|Accounting Standards Update)\s+\d{3,4}(?:-\d{2})?\b", text, re.I))),
            "accounting_contexts_numbers_redacted": excerpts,
            "financial_values_extracted": 0}


def kla_summary(event, context):
    """Metadata-only locator conclusion, not cell extraction or admission."""
    year = event["fiscal_year"]
    if year == 2011:
        return {
            "status": "preformatted_original_requires_separate_column_geometry_review",
            "net_income_locator": {"pre_block": 3, "candidate_lines": [63, 67]},
            "assets_locator": {"pre_block": 2, "candidate_lines": [31]},
            "period_status": "Annual and balance-sheet headings present; precise fixed-width column ownership not certified.",
            "gaps": ["No HTML table grid; distinguish net-income total from repeated net-income-per-share prefix.",
                     "Annual date/column geometry, signed cells and units require an independently tested preformatted reader."],
            "fact_admitted": False}
    tables = context["statement_tables"]
    income = [t for t in tables if any(
        "statements of operations" in h["text"].lower() for h in t["headers"])]
    balance = [t for t in tables if any(
        "balance sheets" in h["text"].lower() for h in t["headers"])]
    if len(income) != 1 or len(balance) != 1:
        raise ValueError("Ambiguous KLAC formal table metadata")
    income, balance = income[0], balance[0]
    ni = [r for r in income["target_rows"] if r["label"].lower() == "net income"]
    assets = [r for r in balance["target_rows"] if r["label"].lower() == "total assets"]
    annual = [h for h in income["headers"] if "twelve months ended" in h["text"].lower()]
    if len(ni) != 1 or len(assets) != 1 or len(annual) != 1:
        raise ValueError("Ambiguous KLAC annual row metadata")
    annual_cols = set(annual[0]["columns"])

    def cols(table, yr, allowed=None):
        result = set()
        for h in table["headers"]:
            if re.search(r"\b" + str(yr) + r"\b", h["text"]):
                result.update(h["columns"])
        return sorted(result if allowed is None else result & allowed)

    flags = []
    if year == 2019:
        flags.append("ASC606_current_and_comparison_not_retrospectively_restated__hold_pending_purpose_review")
    if year <= 2013:
        flags.append("Restatement_related_charges_in_nonGAAP_reconciliation_are_not_proof_of_selected_GAAP_total_restatement")
    return {
        "status": "formal_annual_NI_and_two_assets_endpoints_located_no_values_extracted",
        "current_annual_period": {"start": str(year-1)+"-07-01", "end": str(year)+"-06-30",
                                  "basis": "Interpretation of explicit Twelve months ended June30 current-year header."},
        "comparison_assets_date": str(year-1)+"-06-30",
        "net_income_locator": {"table": income["table"], "row": ni[0]["row"],
                               "label_cell": ni[0]["label_cell"], "label": ni[0]["label"],
                               "annual_header": annual[0],
                               "current_header_columns": cols(income, year, annual_cols),
                               "comparison_header_columns": cols(income, year-1, annual_cols)},
        "assets_locator": {"table": balance["table"], "row": assets[0]["row"],
                           "label_cell": assets[0]["label_cell"], "label": assets[0]["label"],
                           "current_header_columns": cols(balance, year),
                           "comparison_header_columns": cols(balance, year-1)},
        "units": {"scale_literal": "In thousands", "scale": "1000",
                  "evidence": [h for t in (income, balance) for h in t["headers"]
                               if UNIT.fullmatch(h["text"])],
                  "currency_literal": "$" if income["dollar_symbol_present"] and balance["dollar_symbol_present"] else None,
                  "currency_iso": None, "interpretation": "Same-report common-dollar-unit assumption only."},
        "net_income_scope": "Consolidated Net income; exclude the separate Net income attributable to KLA and per-share or non-GAAP lines.",
        "cashflow_warning": "The formal cashflow statement here has Three months ended headers; it cannot supply current annual NI merely because the row label matches.",
        "accounting_attention": flags,
        "gaps": ["Monetary cells, split signs and exclusive year/row ownership have not been parsed or verified.",
                 "Source version, complete correction history and historical security continuity remain conditional.",
                 "No exhaustive accounting-change clearance is implied by absence of keyword matches."],
        "fact_admitted": False}


def intel_summary(event, context):
    year = event["fiscal_year"]
    spec = {
        2017: (20, 18, 21, 23, "2016-12-31",
               "US tax reform transition tax/deferred-tax remeasurement; acquisition-related non-GAAP adjustments and held-for-sale lines. Processor adoption wording is not accounting-standard adoption."),
        2020: (5, 14, 7, 22, "2019-12-28",
               "Proposed NAND disposal, held-for-sale assets and related capital expenditure context. Tax-reform risk wording does not establish a newly adopted accounting standard."),
        2021: (5, 14, 7, 22, "2020-12-26",
               "Held-for-sale NAND assets and first-closing context. A 53-week statement in the next-year Business Outlook is not a current-year duration certificate."),
    }[year]
    ni_table, ni_row, bs_table, bs_row, prior_end, note = spec
    lookup = {t["table"]: t for t in context["statement_tables"]}
    income, balance = lookup[ni_table], lookup[bs_table]
    ni = next(r for r in income["target_rows"] if r["row"] == ni_row)
    assets = next(r for r in balance["target_rows"] if r["row"] == bs_row)
    annual = [h for h in income["headers"] if "twelve months ended" in h["text"].lower()]
    if len(annual) != 1:
        raise ValueError("INTC annual header missing")
    def cols(table, yr, allowed=None):
        result = set()
        for h in table["headers"]:
            if re.search(r"\b" + str(yr) + r"\b", h["text"]):
                result.update(h["columns"])
        return sorted(result if allowed is None else result & allowed)
    return {
        "status": "formal_annual_NI_and_two_assets_endpoints_located_start_date_is_not_directly_quoted",
        "current_annual_period": {
            "start": None, "end": event["official_actual_period_end"],
            "candidate_start_from_prior_end_plus_one": (date.fromisoformat(prior_end)+timedelta(days=1)).isoformat(),
            "basis": "Twelve Months Ended is explicit; deriving exact fiscal-year start from the prior balance-sheet date requires a separately stated consecutive-year-boundary assumption."},
        "comparison_assets_date": prior_end,
        "net_income_locator": {"table": ni_table, "row": ni_row, "label_cell": ni["label_cell"],
                               "label": ni["label"], "annual_header": annual[0],
                               "current_header_columns": cols(income, year, set(annual[0]["columns"])),
                               "comparison_header_columns": cols(income, year-1, set(annual[0]["columns"]))},
        "assets_locator": {"table": bs_table, "row": bs_row, "label_cell": assets["label_cell"],
                           "label": assets["label"], "current_header_columns": cols(balance, year),
                           "comparison_header_columns": cols(balance, year-1)},
        "units": {"scale_literal": "In Millions", "scale": "1000000",
                  "evidence": [h for t in (income, balance) for h in t["headers"]
                               if UNIT.fullmatch(h["text"])],
                  "currency_literal": "$" if income["dollar_symbol_present"] and balance["dollar_symbol_present"] else None,
                  "currency_iso": None, "interpretation": "Same-report common-dollar-unit assumption only."},
        "net_income_scope": "Consolidated Net income or NET INCOME (LOSS), no attributable-to-parent modifier in the selected label.",
        "accounting_attention": [note],
        "gaps": ["Exact annual start is not directly printed; prior-end-plus-one must not be called a quoted fact.",
                 "Monetary cells and sign/year ownership remain untested; no financial admission.",
                 "No complete correction-history, constant-accounting or continuous-security certificate."],
        "fact_admitted": False}


def build():
    census_raw = read_local(CENSUS)
    if sha(census_raw) != EXPECTED_CENSUS:
        raise ValueError("Frozen census mismatch")
    census = json.loads(census_raw)
    events = [e for e in census["annual_documents"] if e["symbol"] in CIKS]
    if len(events) != 32 or len({e["event_id"] for e in events}) != 32:
        raise ValueError("Unexpected event census")
    documents = []
    for event in events:
        item = dict(event)
        selected = event["catalog_claims_issuer_full_body"] and event["format"] == "html_or_text"
        item["html_context_reviewed"] = selected
        if selected:
            item["context"] = inspect(event)
            if event["symbol"] == "KLAC":
                item["context_summary"] = kla_summary(event, item["context"])
            else:
                item["context_summary"] = intel_summary(event, item["context"])
        else:
            item["gap"] = ("pdf_not_read_in_this_html_context_review" if event["format"] == "pdf"
                           else "catalog_not_full_body_no_statement_context_claim")
        documents.append(item)
    counts = Counter(e["symbol"] for e in documents if e["html_context_reviewed"])
    if dict(counts) != {"INTC": 3, "KLAC": 16}:
        raise ValueError("Unexpected HTML scope")
    dependencies = [CENSUS, CONCEPT, "outputs/monthly_financial_coverage_census_v1/PLAN.json",
                    "outputs/monthly_profitability141_sources/DATA_PLAN.json",
                    "work/after141_amd_source_extract.py",
                    "work/after150_next_source_context.py"]
    hashes = {p: sha(read_local(p)) for p in dependencies}
    for item in documents:
        if item["html_context_reviewed"]:
            hashes[item["evidence_path"]] = item["evidence_sha256"]
            receipt = item["context"]["receipt"]
            hashes[receipt["path"]] = receipt["sha256"]
    return {"schema_version": 1, "stage": "bounded_original_report_context_review",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "All32 INTC/KLAC annual catalog events retained; all19 full-body HTML inspected. No monetary cell parsing.",
            "selection": "Next two alphabetical issuers in the frozen coverage concept, not selected by outcomes.",
            "input_hashes": hashes,
            "counts": {"catalog_events": 32, "html_originals_reviewed": 19,
                       "html_originals_by_symbol": dict(counts), "pdfs_not_read": 7,
                       "non_full_body_catalog_events": 6, "financial_values_extracted": 0,
                       "ratios_computed": 0, "models_fitted": 0, "ranks_orders_returns": 0,
                       "network_requests": 0, "new_admissions": 0, "new_trials": 0},
            "coordinate_convention": "Zero-based BeautifulSoup document table index; direct TR rows, direct TD/TH physical cell indices, expanded logical columns. Opaque cells contain no emitted literal.",
            "identity_leads": {s: {"cik": c, "status": "locating_evidence_only_not_daily_security_master"}
                               for s, c in CIKS.items()},
            "next_step": "Freeze a separate source-only extraction plan for these fixed documents. Use the annual consolidated operations NI and same-report assets endpoints. Keep KLAC2011 preformatted geometry and KLAC2019 accounting comparability unresolved until independently reviewed; preserve all other catalog gaps.",
            "proposed_pure_interface": "extract_document(raw_bytes, context_document) -> fields{annual_net_income,current_total_assets,prior_total_assets}, with signed literal/decimal/scale/source/locator/period metadata and explicit null+gap; this report does not implement or authorize numeric extraction.",
            "availability": "publication_date and available_session are inherited catalog metadata; both comparative and current values belong to the current release. This scan does not independently certify first publication or daily calendar coverage.",
            "review_findings": {
                "table_grid_originals": 18,
                "preformatted_originals": 1,
                "KLAC": "Fifteen FY2012–2026 HTML table grids have annual consolidated NI and two June30 balance-sheet endpoints. FY2011 is fixed-width PRE, not an absent report. FY2019 explicitly warns of ASC606 adoption without prior-period restatement; keep a comparability hold. Formal CF statements are quarterly and are not a substitute for annual NI.",
                "INTC": "FY2017/2020/2021 have current/prior formal annual income and balance-sheet context in millions. The 52/53-week fiscal-end convention must not be replaced by December31; precise start dates are not directly quoted.",
                "unchanged_prior_scope": "No original source admission, old event, rejected event, frozen policy, strategy, account or trial counter is changed."},
            "not_certified": ["Financial values, signs, positive assets or correct cell ownership",
                              "First-public versions or complete correction history",
                              "Continuous security identity",
                              "Financial field admission or latest-event fallback eligibility"],
            "fact_admitted": False, "model_admitted": False, "strategy_admitted": False,
            "documents": documents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build()
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        with (ROOT / OUTPUT).open("x") as handle:
            handle.write(payload)
        print(json.dumps({"path": OUTPUT, "sha256": sha(payload.encode()),
                          "counts": result["counts"]}, sort_keys=True))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

"""Pure INTC annual NI/assets extraction, without I/O or source admission.

The context argument is one INTC document from after150_next_source_context.
Only FY2017/2020/2021 are in this bounded scope.  Fiscal-year starts are
explicitly derived candidates, never silently certified as quoted dates.
"""
import copy
import datetime as dt
import re
from decimal import Decimal
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import after149_amzn_extract as old
from after141_amd_source_extract import DATE, dates, external_context, table_grid

FIELDS = old.FIELDS
YEARS = (2017, 2020, 2021)
SourceError = old.SourceError
FieldGap = old.FieldGap
sha256 = old.sha256
_text = old._text
_value = old._value
_amount = old._amount
NI = re.compile(r"net (?:income(?:\s*\(loss\))?|loss)", re.I)
ASSETS = re.compile(r"total assets", re.I)
INCOME_TITLE = re.compile(
    r"consolidated (?:(?:condensed|summary) )?statements? of income"
    r"(?: data| and other information)?", re.I)
BALANCE_TITLE = re.compile(
    r"consolidated (?:(?:condensed|summary) )?balance sheets?(?: data)?", re.I)
ANNUAL = re.compile(r"(?:twelve|12) months ended", re.I)
SHORT = re.compile(r"(?:three|six|nine|3|6|9) months ended", re.I)
BODY = re.compile(
    r"(?:net revenues?|revenues?|net sales|sales|assets|current assets"
    r"|cash(?: and|,|$).*|net income(?:\s*\(loss\))?)", re.I)
BOUNDARY_HOLD = "annual_start_not_directly_quoted_requires_source_use_assumption"


def _table(tables, spec, kind):
    if not isinstance(spec, dict):
        raise FieldGap("invalid_statement_locator")
    ti, row = spec.get("table"), spec.get("row")
    if (type(ti) is not int or type(row) is not int or ti < 0 or row < 0
            or ti >= len(tables)):
        raise FieldGap("invalid_table_or_row_locator")
    table = tables[ti]
    if table.find("table") is not None:
        raise FieldGap("nested_statement_table")
    try:
        rows, grid, physical = table_grid(table)
    except ValueError as exc:
        raise FieldGap("invalid_table_geometry") from exc
    if row >= len(rows):
        raise FieldGap("invalid_table_or_row_locator")
    for (r, p), cell in physical.items():
        cell["original_text"] = rows[r].find_all(["td", "th"], recursive=False)[p].get_text()
    starts = [r for (r, p), c in physical.items()
              if r < row and p == 0 and BODY.fullmatch(_text(c["text"]))]
    stop = min(starts + [row])
    headers = [c for (r, _), c in physical.items() if r < stop]
    outside = list(dict.fromkeys(_text(t) for t in external_context(table)))
    title_pattern = INCOME_TITLE if kind == "income" else BALANCE_TITLE
    titles = [c["text"] for c in headers if title_pattern.fullmatch(_text(c["text"]))]
    titles += [t for t in outside if title_pattern.fullmatch(t)]
    if len(set(titles)) != 1:
        raise FieldGap("formal_consolidated_title_not_unique")
    if any(re.search(r"non[\s-]*gaap|pro forma", t, re.I)
           for t in [c["text"] for c in headers] + outside):
        raise FieldGap("non_gaap_statement")
    units = set()
    scale_evidence = []
    for c in headers:
        found = re.findall(r"\bin (millions|thousands)\b", c["text"], re.I)
        if found:
            units.update(x.lower() for x in found)
            scale_evidence.append(copy.deepcopy(c))
    # The three reviewed originals state units in the selected table itself.
    if len(units) != 1:
        raise FieldGap("missing_or_conflicting_scale")
    text = table.get_text(" ", strip=True)
    if "$" not in text or old.OTHER_CURRENCY.search(text):
        raise FieldGap("missing_or_conflicting_currency")
    scale = "1000000" if units == {"millions"} else "1000"
    return grid, physical, headers, scale, scale_evidence, titles, [
        t for t in outside if title_pattern.fullmatch(t)]


def _column_evidence(headers, columns, endpoint, annual):
    if (not isinstance(columns, list) or not columns
            or any(type(c) is not int or c < 0 for c in columns)
            or sorted(set(columns)) != columns):
        raise FieldGap("invalid_header_columns")
    evidence = old._unique(c for c in headers if set(c["columns"]) & set(columns))
    for col in columns:
        texts = [c["text"] for c in evidence if col in c["columns"]]
        try:
            actual_dates = dates(texts)
        except ValueError as exc:
            raise FieldGap("invalid_header_date") from exc
        if actual_dates != [endpoint]:
            raise FieldGap("missing_or_conflicting_fiscal_end_header")
        text = " ".join(texts)
        if annual and (not ANNUAL.search(text) or SHORT.search(text)):
            raise FieldGap("not_exclusively_annual_column")
    return evidence


def _owned_columns(headers, physical, row, columns, endpoint, annual):
    if annual:
        parents = [set(c["columns"]) for c in headers
                   if ANNUAL.search(c["text"]) and set(columns) <= set(c["columns"])]
        if len(parents) != 1:
            raise FieldGap("annual_parent_span_not_unique")
        domain = parents[0]
    else:
        label = physical.get((row, 0))
        if label is None:
            raise FieldGap("row_label_missing")
        domain = set(range(max(label["columns"])+1,
                           max(max(c["columns"]) for c in old._cells(physical, row))+1))
    anchors = []
    for c in headers:
        cols = set(c["columns"])
        if DATE.fullmatch(_text(c["text"])) and cols <= domain:
            try:
                endpoints = dates([c["text"]])
            except ValueError as exc:
                raise FieldGap("invalid_header_date") from exc
            if len(endpoints) == 1:
                anchors.append((min(cols), max(cols), endpoints[0]))
    # Repeated date rows are not silently collapsed: ownership is ambiguous.
    anchors.sort()
    if (len(anchors) != 2 or anchors[0][1] >= anchors[1][0]
            or anchors[0][2] == anchors[1][2]):
        raise FieldGap("ambiguous_date_group_geometry")
    chosen = [i for i, (lo, hi, day) in enumerate(anchors)
              if day == endpoint and set(columns) <= set(range(lo, hi+1))]
    if len(chosen) != 1:
        raise FieldGap("header_anchor_not_single_period")
    midpoint = (anchors[0][1]+anchors[1][0])//2
    return {c for c in domain if (c <= midpoint) == (chosen[0] == 0)}


def _field(tables, spec, endpoint, which, kind, source, available, period):
    grid, physical, headers, scale, units, titles, outside = _table(tables, spec, kind)
    row = spec["row"]
    target = NI if kind == "income" else ASSETS
    matches = [c for (r, p), c in physical.items()
               if p == 0 and target.fullmatch(_text(c["text"]))]
    if len(matches) != 1 or matches[0]["row"] != row:
        raise FieldGap("exact_unique_row_label_not_found")
    if spec.get("label_cell") != 0 or not target.fullmatch(_text(spec.get("label", ""))):
        raise FieldGap("context_row_label_mismatch")
    columns = spec[which+"_header_columns"]
    evidence = _column_evidence(headers, columns, endpoint, kind == "income")
    owned = _owned_columns(headers, physical, row, columns, endpoint, kind == "income")
    if any((row, c) not in grid for c in columns):
        raise FieldGap("incomplete_amount_geometry")
    value, display, cells = _value(physical, grid, row, columns, owned)
    return {
        "displayed_amount": display, "value_decimal": value, "scale": scale,
        "currency_literal": "$", "currency_iso": None,
        "currency_basis": "same_statement_dollar_symbol_not_ISO_certification",
        "period": copy.deepcopy(period) if kind == "income" else None,
        "as_of": endpoint, "available_from": available, "source": copy.deepcopy(source),
        "fact_admitted": False,
        "locator": {"table": spec["table"], "row": row, "label": copy.deepcopy(matches[0]),
                    "header_columns": list(columns), "owned_logical_columns": sorted(owned),
                    "physical_cells": copy.deepcopy(cells), "header_evidence": copy.deepcopy(evidence),
                    "scale_evidence": units, "statement_title_evidence": titles,
                    "external_context": outside},
    }


def extract_document(raw_bytes, context_document):
    """Return three raw-field candidates, preserving all context holds.

    A raw-hash/identity/reference error is fatal. Missing dates, units, headers
    or ambiguous geometry become explicit null field gaps. The function never
    certifies the next-session calendar or reads files.
    """
    if not isinstance(raw_bytes, bytes) or not isinstance(context_document, dict):
        raise SourceError("bytes_and_context_object_required")
    c = copy.deepcopy(context_document)
    expected_sha = c.get("evidence_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)) or sha256(raw_bytes) != expected_sha:
        raise SourceError("raw_source_sha256_mismatch")
    year = c.get("fiscal_year")
    if c.get("symbol") != "INTC" or type(year) is not int or year not in YEARS:
        raise SourceError("outside_fixed_issuer_year_scope")
    if c.get("event_id") != "INTC_"+str(c.get("publication_date")):
        raise SourceError("event_identity_mismatch")
    try:
        receipt = c["context"]["receipt"]
        summary = c["context_summary"]
        path, url = c["evidence_path"], c["source_url"]
        parsed_url = urlsplit(url)
        if (not isinstance(path, str) or not path or not isinstance(receipt["path"], str)
                or not receipt["path"] or parsed_url.scheme != "https"
                or parsed_url.hostname not in ("www.intc.com", "intc.com")
                or parsed_url.username or parsed_url.password):
            raise ValueError()
        if not isinstance(summary, dict):
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceError("missing_or_invalid_source_context") from exc
    soup = BeautifulSoup(raw_bytes, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    text = soup.get_text(" ", strip=True)
    if (not re.search(r"\bIntel Corporation\b", text, re.I)
            or not re.search(r"\bNasdaq\s*:\s*INTC\b", text, re.I)):
        raise SourceError("raw_issuer_security_anchors_missing")
    source = {"path": path, "sha256": expected_sha, "receipt": receipt["path"], "url": url}
    holds = c.get("context_holds", [])
    if not isinstance(holds, list) or any(not isinstance(h, str) for h in holds):
        raise SourceError("invalid_context_holds")
    result = {
        "schema_version": 1, "stage": "source_only_annual_NI_assets_candidates",
        "symbol": "INTC", "event_id": c["event_id"], "fiscal_year": year,
        "source": source, "publication_date": c.get("publication_date"),
        "available_from": c.get("available_session"), "periods": {},
        "fields": dict.fromkeys(FIELDS), "gaps": [],
        "context_holds": list(dict.fromkeys(holds+[BOUNDARY_HOLD])),
        "accounting_review_flags": copy.deepcopy(summary.get("accounting_attention", [])),
        "fact_admitted": False, "model_admitted": False, "strategy_admitted": False,
        "first_public_version_verified": False, "complete_correction_history_verified": False,
        "certified_daily_security_master": False, "financial_calculations": 0,
        "notes": ["All fields, including prior assets, use this current report's availability.",
                  "The caller must verify receipts and the pinned next trading session.",
                  "Derived fiscal-year boundaries and same-report dollar units require separate source-use review."],
    }
    try:
        current = old._date(c.get("official_actual_period_end"))
        current_meta = summary["current_annual_period"]
        if not isinstance(current_meta, dict):
            raise FieldGap("invalid_current_annual_period")
        if old._date(current_meta.get("end")) != current:
            raise FieldGap("context_current_end_mismatch")
        prior = old._date(summary.get("comparison_assets_date"))
        start = prior+dt.timedelta(days=1)
        if (current.year != year or prior.year != year-1
                or not 330 <= (current-start).days+1 <= 400):
            raise FieldGap("invalid_annual_endpoints")
        if old._date(current_meta.get("candidate_start_from_prior_end_plus_one")) != start:
            raise FieldGap("derived_start_mismatch")
        if current_meta.get("start") is not None:
            raise FieldGap("unexpected_direct_start_claim")
        publication, available = old._date(c.get("publication_date")), old._date(c.get("available_session"))
        if not current < publication < available:
            raise FieldGap("invalid_publication_availability_order")
        result["periods"] = {
            "current": {"period_start": str(start), "period_end": str(current),
                        "days_inclusive": (current-start).days+1},
            "comparison": {"period_start": None, "period_end": str(prior), "days_inclusive": None},
            "start_basis": "derived_same_report_prior_annual_end_plus_one_not_directly_quoted",
            "start_assumption_certified": False}
    except (FieldGap, KeyError, TypeError) as exc:
        result["gaps"].extend({"field": f, "reason": str(exc)} for f in FIELDS)
        result["context_holds"].append(str(exc))
        result["complete_required_fields"] = False
        return result
    tables = soup.find_all("table")
    for name, kind, spec_key, which, endpoint in (
            ("annual_net_income", "income", "net_income_locator", "current", str(current)),
            ("current_total_assets", "assets", "assets_locator", "current", str(current)),
            ("prior_total_assets", "assets", "assets_locator", "comparison", str(prior))):
        try:
            result["fields"][name] = _field(
                tables, summary[spec_key], endpoint, which, kind, source, c["available_session"],
                result["periods"]["current"])
        except (FieldGap, KeyError, TypeError) as exc:
            result["gaps"].append({"field": name, "reason": str(exc)})
    scales = {cell["scale"] for cell in result["fields"].values() if cell is not None}
    if len(scales) > 1:
        result["context_holds"].append("cross_statement_scale_mismatch")
    for name in ("current_total_assets", "prior_total_assets"):
        cell = result["fields"][name]
        if cell is not None and Decimal(cell["value_decimal"]) <= 0:
            result["context_holds"].append(name+"_nonpositive")
    result["complete_required_fields"] = not result["gaps"]
    return result

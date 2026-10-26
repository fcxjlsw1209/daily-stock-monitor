"""Pure AMZN annual NI/assets candidate extraction; no I/O or admission.

Coordinates use the frozen table_grid convention (zero-based physical tables,
rows and cells, expanded logical columns). The caller must freeze/verify the
context and its receipt, and validate available_session against its pinned
calendar. This function verifies the supplied raw bytes and table evidence.
"""
import copy
import datetime as dt
import hashlib
import re
from decimal import Decimal

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid, norm

FIELDS = ("annual_net_income", "current_total_assets", "prior_total_assets")
YEARS = (2010, 2011, 2012, 2014, 2016, 2017, 2018, 2019, 2020, 2021)
NI = re.compile(r"net (?:income(?:\s*\(loss\))?|loss)", re.I)
CFO = re.compile(r"net cash provided by(?:\s*\(used in\))? operating activities(?:\s*\(\d+\))?", re.I)
CF_TITLE = re.compile(r"consolidated statements? of cash flows", re.I)
BS_TITLE = re.compile(r"consolidated balance sheets?", re.I)
ANNUAL = re.compile(r"(?:twelve|12) months ended|years? ended", re.I)
SHORT = re.compile(r"(?:three|six|nine|3|6|9) months ended", re.I)
BODY = re.compile(r"(?:operating activities|assets|current assets|cash(?: and|,|$).*)", re.I)
OTHER_CURRENCY = re.compile(r"€|£|¥|\b(?:EUR|GBP|JPY|CAD|HKD|AUD)\b", re.I)


class SourceError(ValueError):
    """Source bytes or fixed issuer/event identity failed validation."""


class FieldGap(ValueError):
    """Explicit field-level absence or ambiguity, never inferred zero."""


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(s):
    return norm(s).rstrip(":").strip()


def _date(s):
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise FieldGap("invalid_date")
    try:
        return dt.date.fromisoformat(s)
    except ValueError as exc:
        raise FieldGap("invalid_date") from exc


def _amount(text):
    """Exact decimal syntax only; normalization is not financial arithmetic."""
    clean = norm(text).replace("\u2212", "-")
    clean = re.sub(r"\s*,\s*", ",", clean)
    # A dollar sign may precede or follow an opening parenthesis, once only.
    if clean.count("$") > 1:
        raise FieldGap("ambiguous_amount")
    if "$" in clean and not re.match(r"^(?:\$\s*|\(\s*\$)", clean):
        raise FieldGap("ambiguous_amount")
    clean = clean.replace("$", "").strip()
    negative = clean.startswith("(") and clean.endswith(")")
    if negative:
        clean = clean[1:-1].strip()
    if not re.fullmatch(r"-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", clean):
        raise FieldGap("missing_or_ambiguous_amount")
    if negative and clean.startswith("-"):
        raise FieldGap("conflicting_signs")
    # Do not apply Decimal unary minus: its active precision can round.
    digits = clean.replace(",", "").replace(" ", "")
    value = Decimal(("-" if negative else "") + digits)
    return format(value, "f")


def _cells(physical, row):
    return sorted((c for (r, _), c in physical.items() if r == row),
                  key=lambda c: c["physical_cell"])


def _unique(items):
    return list({(c["row"], c["physical_cell"]): c for c in items}.values())


def _headers(physical, stop):
    # Financial rows can precede Net income and Total assets. Bound the header
    # before the first body label rather than treating body amounts as years.
    body = [r for (r, p), c in physical.items()
            if r < stop and p == 0 and BODY.fullmatch(_text(c["text"]))]
    end = min(body + [stop])
    return [c for (r, _), c in physical.items() if r < end], end


def _table(tables, ti, row, kind):
    if type(ti) is not int or type(row) is not int or ti < 0 or row < 0 or ti >= len(tables):
        raise FieldGap("invalid_table_or_row_locator")
    table = tables[ti]
    if table.find("table") is not None:
        raise FieldGap("nested_statement_table")
    try:
        rows, grid, physical = table_grid(table)
    except ValueError as exc:
        raise FieldGap("invalid_table_geometry") from exc
    for (r, p), cell in physical.items():
        cell["original_text"] = rows[r].find_all(["td", "th"], recursive=False)[p].get_text()
    headers, stop = _headers(physical, row)
    title = CF_TITLE if kind == "cf" else BS_TITLE
    titles = [c for c in headers if title.fullmatch(_text(c["text"]))]
    if len(titles) != 1:
        raise FieldGap("formal_consolidated_title_not_unique")
    if any(re.search(r"non[\s-]*gaap|pro forma", c["text"], re.I) for c in headers):
        raise FieldGap("non_gaap_statement")
    scales = set()
    scale_cells = []
    for c in headers:
        for unit in re.findall(r"\bin (millions|thousands)\b", c["text"], re.I):
            scales.add(unit.lower())
            scale_cells.append(c)
    if len(scales) != 1:
        raise FieldGap("missing_or_conflicting_scale")
    scale = "1000000" if scales == {"millions"} else "1000"
    table_text = table.get_text(" ", strip=True)
    if "$" not in table_text or OTHER_CURRENCY.search(table_text):
        raise FieldGap("missing_or_conflicting_currency")
    return grid, physical, headers, stop, scale, _unique(scale_cells)


def _column_evidence(headers, columns, year, annual):
    if (not isinstance(columns, list) or not columns or
            any(type(c) is not int or c < 0 for c in columns) or
            sorted(set(columns)) != columns):
        raise FieldGap("invalid_header_columns")
    evidence = _unique(c for c in headers if set(c["columns"]) & set(columns))
    # Every supplied anchor must independently identify the same full period.
    for col in columns:
        texts = [c["text"] for c in evidence if col in c["columns"]]
        date_text = " ".join(texts)
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", date_text))
        if years != {str(year)} or not re.search(r"December\s+31\b", date_text, re.I):
            raise FieldGap("missing_or_conflicting_year_end_header")
        if annual and (not ANNUAL.search(date_text) or SHORT.search(date_text)):
            raise FieldGap("not_exclusively_annual_column")
    return evidence


def _owned_columns(headers, physical, row, columns, year, annual):
    """Partition one statement's year groups before reading monetary cells.

    Wide year anchors retain their entire span. Empty separators between
    anchors are split at the integer midpoint; punctuation on the wrong side
    is rejected rather than borrowed. Annual and quarterly parent spans are
    separate domains.
    """
    if annual:
        parents = [set(c["columns"]) for c in headers
                   if ANNUAL.search(c["text"]) and set(columns) <= set(c["columns"])]
        if not parents:
            raise FieldGap("annual_parent_span_missing")
        domain = min(parents, key=len)
    else:
        label = physical.get((row, 0))
        if label is None:
            raise FieldGap("row_label_missing")
        domain = set(range(max(label["columns"]) + 1,
                           max(max(c["columns"]) for c in _cells(physical, row)) + 1))
    anchors = []
    for c in headers:
        match = re.fullmatch(r"(?:December\s+31,?\s*)?((?:19|20)\d{2})", _text(c["text"]), re.I)
        cols = set(c["columns"])
        if match and cols <= domain:
            anchors.append((min(cols), max(cols), int(match[1])))
    anchors = sorted(set(anchors))
    if len(anchors) != 2 or anchors[0][1] >= anchors[1][0] or anchors[0][2] == anchors[1][2]:
        raise FieldGap("ambiguous_year_group_geometry")
    selected = [i for i, (lo, hi, y) in enumerate(anchors)
                if y == year and set(columns) <= set(range(lo, hi+1))]
    if len(selected) != 1:
        raise FieldGap("header_anchor_not_single_year")
    boundary = (anchors[0][1] + anchors[1][0]) // 2
    return {c for c in domain if (c <= boundary) == (selected[0] == 0)}


def _value(physical, grid, row, columns, owned):
    selected = _unique(grid[(row, c)] for c in columns if (row, c) in grid)
    if not selected or any(c["row"] != row for c in selected):
        raise FieldGap("missing_or_rowspan_amount")
    # Single-column year anchors may leave '$', '(' or ')' in the immediately
    # adjacent physical cell. Only punctuation, never another amount, expands.
    ordered = _cells(physical, row)
    indexes = [i for i, c in enumerate(ordered) if c in selected]
    lo, hi = min(indexes), max(indexes)
    if lo and _text(ordered[lo-1]["text"]) in ("$", "(", "$ (", "($", "( $", "-"):
        selected.append(ordered[lo-1])
    if hi + 1 < len(ordered) and _text(ordered[hi+1]["text"]) == ")":
        selected.append(ordered[hi+1])
    selected = sorted(_unique(selected), key=lambda c: c["physical_cell"])
    if any(c.get("rowspan", 1) != 1 for c in selected):
        raise FieldGap("amount_cell_crosses_rows")
    if any(not set(c["columns"]) <= owned for c in selected):
        raise FieldGap("amount_cell_crosses_year_group")
    display = " ".join(c["text"] for c in selected).strip()
    return _amount(display), display, selected


def _field(tables, spec, row, columns, year, kind, source, available, period):
    grid, physical, headers, _, scale, scale_cells = _table(tables, spec["table"], row, kind)
    labels = [c for c in _cells(physical, row) if c["physical_cell"] == 0]
    target = NI if kind == "cf" else re.compile(r"total assets", re.I)
    if len(labels) != 1 or not target.fullmatch(_text(labels[0]["text"])):
        raise FieldGap("exact_row_label_not_found")
    matches = [r for (r, p), c in physical.items()
               if p == 0 and target.fullmatch(_text(c["text"]))]
    if matches != [row]:
        raise FieldGap("duplicate_target_rows")
    if kind == "cf":
        cfo_row = spec.get("CFO_row")
        if type(cfo_row) is not int or not row < cfo_row:
            raise FieldGap("invalid_operating_reconciliation")
        before = [_text(c["text"]) for (r, p), c in physical.items()
                  if p == 0 and r < row and _text(c["text"])]
        between = [_text(c["text"]) for (r, p), c in physical.items()
                   if p == 0 and row < r < cfo_row]
        end = [_text(c["text"]) for c in _cells(physical, cfo_row) if c["physical_cell"] == 0]
        if (not before or before[-1].lower() != "operating activities" or
                not any(re.match(r"adjustments to reconcile net (?:income|loss).*operating", t, re.I)
                        for t in between) or
                len(end) != 1 or not CFO.fullmatch(end[0])):
            raise FieldGap("invalid_operating_reconciliation")
    evidence = _column_evidence(headers, columns, year, kind == "cf")
    owned = _owned_columns(headers, physical, row, columns, year, kind == "cf")
    value, display, cells = _value(physical, grid, row, columns, owned)
    return {
        "displayed_amount": display, "value_decimal": value, "scale": scale,
        "currency_literal": "$", "currency_iso": None,
        "currency_basis": "same_statement_dollar_symbol_not_ISO_certification",
        "period": copy.deepcopy(period) if kind == "cf" else None,
        "as_of": period["period_end"],
        "available_from": available, "source": copy.deepcopy(source),
        "fact_admitted": False,
        "locator": {"table": spec["table"], "row": row,
                    "label": copy.deepcopy(labels[0]), "header_columns": list(columns),
                    "owned_logical_columns": sorted(owned),
                    "physical_cells": copy.deepcopy(cells),
                    "header_evidence": copy.deepcopy(evidence),
                    "scale_evidence": copy.deepcopy(scale_cells)},
    }


def extract_document(raw_bytes, context_document):
    """Extract only three current-report candidates. Does not mutate inputs."""
    if not isinstance(raw_bytes, bytes) or not isinstance(context_document, dict):
        raise SourceError("bytes_and_context_object_required")
    c = copy.deepcopy(context_document)
    if not re.fullmatch(r"[0-9a-f]{64}", str(c.get("sha256", ""))) or sha256(raw_bytes) != c["sha256"]:
        raise SourceError("raw_source_sha256_mismatch")
    year = c.get("fiscal_year")
    if type(year) is not int or year not in YEARS:
        raise SourceError("outside_fixed_context_years")
    if c.get("event_id") != "AMZN_" + str(c.get("publication_date")):
        raise SourceError("event_identity_mismatch")
    if not isinstance(c.get("path"), str) or not c["path"] or not isinstance(c.get("receipt"), str):
        raise SourceError("missing_source_reference")
    soup = BeautifulSoup(raw_bytes, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    text = soup.get_text(" ", strip=True)
    if not re.search(r"Amazon\.com,?\s+Inc\.?", text, re.I) or not re.search(r"NASDAQ\s*:\s*AMZN\b", text, re.I):
        raise SourceError("raw_issuer_security_anchors_missing")
    source = {k: c[k] for k in ("path", "sha256", "receipt")}
    result = {
        "schema_version": 1, "stage": "source_only_annual_NI_assets_candidates",
        "symbol": "AMZN", "event_id": c["event_id"], "fiscal_year": year,
        "source": source, "publication_date": c.get("publication_date"),
        "available_from": c.get("available_session"), "periods": {},
        "fields": dict.fromkeys(FIELDS), "gaps": [], "context_holds": [],
        "accounting_review_flags": [c.get("accounting_context", "No accounting context supplied")],
        "fact_admitted": False, "model_admitted": False, "strategy_admitted": False,
        "first_public_version_verified": False, "certified_daily_security_master": False,
        "financial_calculations": 0,
        "notes": [
            "Only current report availability applies, including the prior asset endpoint.",
            "Availability is inherited from the frozen context; caller must verify its pinned calendar.",
            "Dollar symbols do not independently certify ISO currency; no admission is conferred.",
        ],
    }
    if year == 2019:
        result["context_holds"].append("FY2019_lease_adoption_prior_period_not_retrospectively_adjusted")
    try:
        current = _date(c.get("current_end"))
        prior = _date(c.get("comparison_end"))
        publication = _date(c.get("publication_date"))
        available = _date(c.get("available_session"))
        if current != dt.date(year, 12, 31) or prior != dt.date(year-1, 12, 31):
            raise FieldGap("context_year_end_mismatch")
        if not current < publication < available:
            raise FieldGap("invalid_publication_availability_order")
        result["periods"] = {
            "current": {"period_start": f"{year}-01-01", "period_end": str(current),
                        "days_inclusive": (current-dt.date(year, 1, 1)).days+1},
            "comparison": {"period_start": f"{year-1}-01-01", "period_end": str(prior),
                           "days_inclusive": (prior-dt.date(year-1, 1, 1)).days+1},
            "start_basis": "derived_from_explicit_twelve_calendar_months_ended_December_31",
        }
    except FieldGap as exc:
        result["gaps"].extend({"field": f, "reason": str(exc)} for f in FIELDS)
        result["context_holds"].append(str(exc))
        result["complete_required_fields"] = False
        return result
    tables = soup.find_all("table")
    for name, kind, endyear, which in (
            ("annual_net_income", "cf", year, "current"),
            ("current_total_assets", "assets", year, "current"),
            ("prior_total_assets", "assets", year-1, "comparison")):
        try:
            spec = c[kind]
            result["fields"][name] = _field(
                tables, spec, spec["NI_row" if kind == "cf" else "row"],
                spec[which + "_header_columns"], endyear, kind, source,
                c["available_session"], result["periods"][which])
        except (FieldGap, KeyError, TypeError) as exc:
            result["gaps"].append({"field": name, "reason": str(exc)})
    scales = {v["scale"] for v in result["fields"].values() if v is not None}
    if len(scales) > 1:
        result["context_holds"].append("cross_statement_scale_mismatch")
    for name in ("current_total_assets", "prior_total_assets"):
        cell = result["fields"][name]
        if cell is not None and Decimal(cell["value_decimal"]) <= 0:
            result["context_holds"].append(name + "_nonpositive")
    result["complete_required_fields"] = not result["gaps"]
    return result

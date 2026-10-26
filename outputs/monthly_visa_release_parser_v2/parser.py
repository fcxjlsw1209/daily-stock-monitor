"""Parse Visa's issuer-distributed quarterly release, never candidate values.

This module establishes printed facts only. Catalog/accession binding and
historical eligibility are separate admission checks.
"""
import calendar
from datetime import date
from decimal import Decimal
import re


class ParseError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ParseError(reason)


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def human(value):
    match = re.fullmatch(r"([A-Za-z]+) (\d{1,2}), (20\d\d)", clean(value))
    require(match is not None, "date_shape")
    month = {calendar.month_name[i]: i for i in range(1, 13)}
    require(match[1] in month, "date_month")
    try:
        return date(int(match[3]), month[match[1]], int(match[2]))
    except ValueError as exc:
        raise ParseError("invalid_calendar_date") from exc


def identity(text):
    flat = clean(text)
    headings = re.findall(
        r"Visa Inc\. Reports Fiscal (First|Second|Third|Fourth) Quarter"
        r"(?: and Full-Year)? (20\d\d) Results", flat)
    require(len(headings) == 1, "unique_issuer_heading")
    quarter = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4}[headings[0][0]]
    fiscal_year = int(headings[0][1])
    datelines = re.findall(
        r"San Francisco, CA, ([A-Za-z]+ \d{1,2}, 20\d\d) [–—-] "
        r"Visa Inc\. \(NYSE: V\)(?!\w)", flat)
    require(len(datelines) == 1, "unique_issuer_dateline_and_symbol")
    require(re.search(r"\bUSD\b", flat) is not None, "usd_reporting_indicator")
    release = human(datelines[0])
    year = fiscal_year - 1 if quarter == 1 else fiscal_year
    month = {1: 12, 2: 3, 3: 6, 4: 9}[quarter]
    report = date(year, month, calendar.monthrange(year, month)[1])
    require(report < release and (release - report).days <= 100, "release_report_sequence")
    return {"issuer": "Visa Inc.", "symbol": "V", "exchange": "NYSE",
            "currency": "USD", "fiscal_year": fiscal_year, "fiscal_quarter": quarter,
            "report_end": report.isoformat(), "release_date": release.isoformat(),
            "release_date_evidence": "printed first-page issuer dateline",
            "publication_timestamp_verified": False,
            "continuous_security_identity_verified": False}


def printed_amount(token):
    require(re.fullmatch(r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)", token) is not None,
            "invalid_amount_token")
    return int(token.replace(",", "")) * 1_000_000


def statement(text, ident, page_number):
    lines = text.replace("\r\n", "\n").replace("\r", "\n")
    flat = clean(lines)
    title = "Visa Inc. Consolidated Statements of Operations (unaudited)"
    require(flat.count(title) == 1, "statement_identity_or_title")
    body = flat.split(title, 1)[1]
    # Only the observed word split in this exact unit phrase is normalized.
    # Numeric tokens and period text are never repaired.
    units = list(re.finditer(r"\(in millions, (?:except|excep t) per share data\)", body))
    require(len(units) == 1, "statement_unit")
    unit = units[0]
    header = clean(body[:unit.start()])
    durations = r"(Three|Six|Nine|Twelve) Months Ended"
    endings = r"(March 31|June 30|September 30|December 31),"
    columns = r"(20\d\d 20\d\d(?: 20\d\d 20\d\d)?)"
    # Enumerate complete header layouts instead of searching for whichever
    # duration/date pair happens to be adjacent in extracted text.
    single = re.fullmatch(durations + " " + endings + " " + columns, header)
    interleaved = re.fullmatch(durations + " " + endings + " " + durations +
                              " " + endings + " " + columns, header)
    rowwise = re.fullmatch(durations + " " + durations + " " + endings +
                          " " + endings + " " + columns, header)
    require(sum(m is not None for m in (single, interleaved, rowwise)) == 1,
            "complete_period_header_layout")
    if single:
        dates = [(single[1], single[2])]
        year_text, header_layout = single[3], "single"
    elif interleaved:
        dates = [(interleaved[1], interleaved[2]), (interleaved[3], interleaved[4])]
        year_text, header_layout = interleaved[5], "interleaved"
    else:
        dates = [(rowwise[1], rowwise[3]), (rowwise[2], rowwise[4])]
        year_text, header_layout = rowwise[5], "rowwise"
    require(dates[0][0] == "Three", "first_period_is_quarter")
    years = [int(y) for y in year_text.split()]
    require(len(years) == len(dates) * 2, "year_column_count")
    require(years[0] == years[1] + 1 and
            (len(years) == 2 or years[:2] == years[2:]), "year_column_order")
    expected_length = {1: None, 2: "Six", 3: "Nine", 4: "Twelve"}[ident["fiscal_quarter"]]
    require((expected_length is None and len(dates) == 1) or
            (len(dates) == 2 and dates[1][0] == expected_length), "quarter_ytd_header")
    require(all(d[1] == dates[0][1] for d in dates), "inconsistent_period_ends")
    # The revenue row must be complete on one printed text line. Refuse partial
    # tokens, negative parentheses, percentages, and extra numeric columns.
    revenue = re.findall(r"^\s*(Net (?:operating )?revenues)\s+([^\n]+)$", lines, re.M | re.I)
    require(len(revenue) == 1, "unique_net_revenue_row")
    label, raw = revenue[0]
    tokens = raw.replace("$", " ").split()
    require(len(tokens) == len(years), "revenue_column_count")
    values = [printed_amount(token) for token in tokens]
    periods = []
    lengths = {"Three": 3, "Six": 6, "Nine": 9, "Twelve": 12}
    for column, (length_text, ending) in enumerate(dates):
        months = lengths[length_text]
        sides = {}
        for offset, side in enumerate(("current", "prior")):
            idx = 2 * column + offset
            end = human(f"{ending}, {years[idx]}")
            sm = end.month - months + 1
            sy = end.year
            if sm <= 0:
                sm += 12
                sy -= 1
            start = date(sy, sm, 1)
            require(end.day == calendar.monthrange(end.year, end.month)[1], "non_month_end")
            sides[side] = {"start": start.isoformat(), "end": end.isoformat(),
                           "duration_days": (end - start).days + 1,
                           "revenue_usd": values[idx], "printed_token": tokens[idx],
                           "start_directly_printed": False, "end_directly_printed": True}
        require(sides["current"]["end"] == ident["report_end"], "statement_report_end")
        periods.append({"months_printed": months, "row_label": clean(label),
                        "semantic_concept": "consolidated_net_revenues_as_reported",
                        "unit": "USD", "scale": 1_000_000, "header": clean(header),
                        "locator": {"page": page_number, "title": title}, **sides})
    return {"page": page_number, "title": title, "periods": periods,
            "header_layout": header_layout, "unit_phrase_raw": unit[0],
            "unit_word_split_normalized": "excep t" in unit[0],
            "gross_revenues_not_certified": True,
            "constant_economic_perimeter_claimed": False}


def parse_pdf(payload):
    require(payload.get("engine") == "pdfplumber" and payload.get("version") == "0.11.9",
            "extractor_version")
    n = payload.get("total_pages")
    pages = payload.get("pages", [])
    require(type(n) is int and 2 <= n <= 40 and len(pages) == n, "document_extent")
    require([p.get("page") for p in pages] == list(range(1, n + 1)), "complete_page_order")
    require(all(type(p.get("text")) is str for p in pages), "page_text_type")
    ident = identity(pages[0]["text"])
    matching = [p for p in pages if
                "Visa Inc. Consolidated Statements of Operations (unaudited)" in clean(p["text"])]
    require(len(matching) == 1, "unique_statement_page")
    table = statement(matching[0]["text"], ident, matching[0]["page"])
    versions = {}
    for key in ("CreationDate", "ModDate"):
        raw = payload.get("metadata", {}).get(key)
        found = re.fullmatch(r"D:(\d{4})(\d{2})(\d{2})(?:\d{6}(?:Z|[+-]\d{2}'?\d{2}'?)?)?", raw or "")
        require(found is not None, "metadata_date_shape")
        try:
            day = date(*map(int, found.groups()))
        except ValueError as exc:
            raise ParseError("metadata_calendar_date") from exc
        release = date.fromisoformat(ident["release_date"])
        require(date.fromisoformat(ident["report_end"]) < day <= release,
                "document_version_after_release_or_before_period")
        versions[key] = {"raw": raw, "date": day.isoformat()}
    return {"identity": ident, "statement": table,
            "version_consistency": {"dates": versions, "metadata_is_publication_evidence": False},
            "extractor": {"engine": payload["engine"], "version": payload["version"],
                          "pages_reconstructed": n, "total_pages_reported": n},
            "catalog_bound": False, "semantic_admission": False}


def compare_pair(pair, parsed):
    """Compare dates and values without assuming all XBRL revenue tags mean net."""
    require(pair.get("taxonomy") == "us-gaap" and pair.get("unit") == "USD", "pair_taxonomy_or_unit")
    # Caller must separately establish whether the original tag denotes net
    # revenue. This matcher intentionally never marks a record as admitted.
    matches = []
    for period in parsed["statement"]["periods"]:
        if all(pair[side][key] == period[side][key]
               for side in ("current", "prior") for key in ("start", "end", "duration_days")):
            require(all(Decimal(str(pair[side]["val"])).is_finite() for side in ("current", "prior")),
                    "nonfinite_pair_amount")
            if all(Decimal(str(pair[side]["val"])) == Decimal(period[side]["revenue_usd"])
                   for side in ("current", "prior")):
                matches.append(period)
    require(len(matches) <= 1, "ambiguous_pair_period")
    return {"numeric_and_period_match": bool(matches), "semantic_admission": False,
            "matched_period": matches[0] if matches else None}

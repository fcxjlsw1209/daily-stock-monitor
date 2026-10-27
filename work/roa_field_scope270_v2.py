"""Pure, conditional source assessment for three same-report ROA cells.

No files are opened and no ratio is calculated. Hashes bind supplied JSON
objects; the caller must authenticate raw bytes and independent review receipts.
This separate determination never changes a legacy admission or approves a chain.
V2 validates complete rich source metadata without stripping the input event.
Only the independent review raw_source uses a path/SHA projection. URLs and
receipt declarations are syntax-checked, not authenticated or opened.
"""
from bisect import bisect_right
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
import math
from urllib.parse import urlsplit

SCHEMA = "roa_field_scope270_assessment_v2"
REVIEW_SCHEMA = "roa_field_scope270_review_v2"
DEFINITION = "current_annual_net_income_average_assets_field_scope_v1"
REQUIRED = (("current", "net_income"), ("current", "total_assets"),
            ("comparison", "total_assets"))
CELL_NAMES = tuple(side + "." + name for side, name in REQUIRED)
INAPPLICABLE_CODES = ("unknown_comparison_duration", "unknown_comparison_period_start")
REVIEWABLE_FLAG = "new_fields_independent_source_review_required"
REVIEW_KEYS = ("schema", "event_id", "event_sha256", "raw_source", "cell_sha256",
               "review_receipts", "independently_reviewed")
OUTPUT_KEYS = ("schema", "definition", "admitted", "reasons", "event_id",
               "event_sha256", "field_review_sha256", "required_cell_sha256",
               "raw_source", "review_receipts", "evidence_binding_passed",
               "scope_inapplicable_inherited_codes", "scope_satisfied_accounting_flags",
               "current_report_expires", "same_report_literal_currency_assumption",
               "currency_iso_certified", "source_only", "strict_pit",
               "source_bytes_verified", "review_receipt_bytes_verified",
               "first_public_version_verified", "complete_correction_history_verified",
               "ratios_computed", "chain_admitted", "original_rejections_preserved",
               "existing_admissions_modified", "inputs_preserved")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _json(value):
    """Exclude coerced keys, foreign numeric types and non-finite JSON numbers."""
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON string keys required")
        for child in value.values():
            _json(child)
    elif type(value) is list:
        for child in value:
            _json(child)
    else:
        require(type(value) in (str, int, float, bool, type(None)), "Native JSON required")
        require(type(value) is not float or math.isfinite(value), "Finite JSON number required")


def canonical_sha256(value):
    _json(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Canonical SHA256 required")


def _reference(ref):
    require(type(ref) is dict and set(ref) == {"path", "sha256"}, "Exact reference keys required")
    require(type(ref["path"]) is str and bool(ref["path"].strip()), "Reference path required")
    _sha(ref["sha256"])


def _source(ref):
    """Validate the intact source record; authenticate its bytes outside this module."""
    required = {"path", "sha256", "receipt", "url"}
    require(type(ref) is dict and required <= set(ref)
            and set(ref) <= required | {"receipt_entry_key"}, "Exact rich source keys required")
    _reference({"path": ref["path"], "sha256": ref["sha256"]})
    _reference(ref["receipt"])
    if "receipt_entry_key" in ref:
        key = ref["receipt_entry_key"]
        require(key is None or type(key) is str and bool(key.strip()),
                "Receipt entry key must be null or nonempty string")
    url = ref["url"]
    require(type(url) is str and bool(url) and not any(c.isspace() or ord(c) < 32 for c in url),
            "Source URL string required without whitespace or controls")
    parsed = urlsplit(url)
    require(parsed.scheme == "https" and bool(parsed.hostname)
            and parsed.username is None and parsed.password is None and "#" not in url,
            "Source URL must be HTTPS with host, without credentials or fragment")
    # Accessing port also rejects malformed/out-of-range URL port syntax.
    parsed.port


def _day(value):
    require(type(value) is str, "Date string required")
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, "Canonical date required")
    return parsed


def _decimal(value):
    # Original amount strings retain their own scale, without float conversion.
    require(type(value) in (str, int), "Decimal string or integer required")
    number = Decimal(value)
    require(number.is_finite(), "Finite Decimal required")
    return number


def _code_list(value, label):
    require(type(value) is list and all(type(c) is str and c for c in value)
            and len(value) == len(set(value)), "Malformed " + label)
    return value


def _locator(locator, scale):
    require(type(locator.get("row_label")) is str and bool(locator["row_label"].strip()),
            "Physical row label required")
    require(locator.get("currency_literal") == "$" and _decimal(locator["scale"]) == scale,
            "Locator currency or scale mismatch")
    if locator.get("format") == "html":
        for key in ("table_index_zero_based", "row_index_zero_based"):
            require(type(locator.get(key)) is int and locator[key] >= 0, "HTML physical index required")
        ids = locator.get("physical_cell_indices_zero_based")
        require(type(ids) is list and bool(ids) and all(type(i) is int and i >= 0 for i in ids)
                and ids == sorted(set(ids)), "HTML physical cell indices required")
    elif locator.get("format") == "pdf":
        for key in ("page_1based", "line_1based"):
            require(type(locator.get(key)) is int and locator[key] > 0, "PDF physical index required")
        col = locator.get("financial_column_index_0based")
        require(type(col) is int and col >= 0, "PDF financial column required")
    else:
        raise ValueError("Recognized HTML or PDF physical locator required")


def _review(event, review, event_sha, cells):
    require(type(review) is dict and set(review) == set(REVIEW_KEYS), "Exact field review schema required")
    require(review["schema"] == REVIEW_SCHEMA, "Wrong field review schema")
    require(review["independently_reviewed"] is True, "Independent field review declaration required")
    require(review["event_id"] == event["event_id"], "Review event identity mismatch")
    _sha(review["event_sha256"])
    require(review["event_sha256"] == event_sha, "Complete event digest mismatch")
    _source(event["source"])
    _reference(review["raw_source"])
    require(review["raw_source"] == {"path": event["source"]["path"],
                                      "sha256": event["source"]["sha256"]},
            "Raw source reference mismatch")
    pins = review["cell_sha256"]
    require(type(pins) is dict and set(pins) == set(CELL_NAMES), "Exact three cell digests required")
    for name in CELL_NAMES:
        _sha(pins[name])
        require(pins[name] == cells.get(name), "Required cell digest mismatch: " + name)
    refs = review["review_receipts"]
    require(type(refs) is list and bool(refs), "Independent review receipt references required")
    for ref in refs:
        _reference(ref)
    require(len({r["path"] for r in refs}) == len(refs), "Duplicate review receipt path")


def assess(event, sessions, field_review):
    """Return an explicit held/admitted source-scope decision without I/O.

    The full event, including unused fields and nulls, is hashed for provenance.
    Only current NI and the two asset dates/amounts enter semantic assessment.
    Comparison flow starts/durations are not filled, checked or used as a ratio.
    """
    reasons, cells = [], {}
    event_sha = review_sha = expiry = None
    binding = False
    inapplicable, satisfied = [], []
    raw_source, references = None, []
    event_id = event.get("event_id") if type(event) is dict else None
    try:
        require(type(event) is dict, "Event object required")
        require(type(event_id) is str and bool(event_id), "Event ID required")
        event_sha = canonical_sha256(event)
        review_sha = canonical_sha256(field_review)
        for side, name in REQUIRED:
            cell = event.get("fields", {}).get(side, {}).get(name)
            if type(cell) is dict:
                cells[side + "." + name] = canonical_sha256(cell)
        _review(event, field_review, event_sha, cells)
        binding = True
        raw_source = dict(field_review["raw_source"])
        references = [dict(ref) for ref in field_review["review_receipts"]]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        reasons.append("field_evidence:" + str(exc))

    if type(event) is not dict:
        event = {}  # Local fallback only, never repairs the caller's input.
    try:
        prior = event["prior_field_admitted"]
        require(type(prior) is bool, "Legacy prior-field admission flag required")
        codes = _code_list(event["inherited_rejection_codes"], "inherited rejection codes")
        gaps = _code_list(event["metadata_gaps"], "metadata gaps")
        require(not gaps, "Unresolved source metadata")
        require(prior or bool(codes), "Unexplained prior-field rejection")
        unknown = [c for c in codes if c not in INAPPLICABLE_CODES]
        require(not unknown, "Preserved inherited source hold: " + ",".join(unknown))
        # Mark a reason inapplicable only after its required independent binding.
        if binding:
            inapplicable = list(codes)
        for flag in ("fact_admitted", "model_admitted", "strategy_admitted"):
            require(flag not in event or event[flag] is False, "Raw admission flag changed: " + flag)
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append("legacy_source_hold:" + str(exc))
    try:
        flags = event.get("accounting_review_flags", [])
        require(type(flags) is list, "Accounting flags list required")
        for flag in flags:
            require(type(flag) is dict and type(flag.get("code")) is str,
                    "Malformed accounting review flag")
            require(flag["code"] == REVIEWABLE_FLAG, "Preserved accounting review flag: " + flag["code"])
            require(binding, "Independent field review requirement unsatisfied")
            satisfied.append(flag["code"])
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append("accounting_source_hold:" + str(exc))

    start = end = None
    try:
        require(type(sessions) in (list, tuple) and bool(sessions), "Nonempty pinned session sequence required")
        previous = None
        for session in sessions:
            _day(session)
            require(previous is None or previous < session, "Strict sorted unique sessions required")
            previous = session
        released, available = _day(event["release_date"]), _day(event["available_from"])
        j = bisect_right(sessions, released.isoformat())
        require(0 < j < len(sessions) and sessions[j] == available.isoformat(),
                "Calendar does not bracket release and first later session")
        symbol, identity = event["symbol"], event["identity"]
        require(type(symbol) is str and bool(symbol) and identity["security_id"] == symbol
                and type(identity["issuer_id"]) is str and bool(identity["issuer_id"]),
                "Missing or conflicting identity")
        require(identity["identity_valid_from"] == available.isoformat(), "Identity availability mismatch")
        current = event["periods"]["current"]
        start, end = _day(current["period_start"]), _day(current["period_end"])
        days = (end - start).days + 1
        require(330 <= days <= 400 and type(current["days_inclusive"]) is int
                and current["days_inclusive"] == days, "Current annual duration invalid")
        require(end <= released, "Release precedes fiscal end")
        expiry = (end + timedelta(days=550)).isoformat()
        require(identity["identity_valid_through"] == expiry, "Identity expiry mismatch")
        require(available.isoformat() <= expiry, "Availability after report expiry")
        # Comparison period_start/days are flow metadata, intentionally unused.
        require(_day(event["periods"]["comparison"]["period_end"]) == start - timedelta(days=1),
                "Comparison asset date not opening instant")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        reasons.append("period_identity_or_clock:" + str(exc))

    currencies = []
    for side, name in REQUIRED:
        try:
            cell = event["fields"][side][name]
            require(type(cell) is dict, "Required cell object missing")
            _source(cell["source"])
            require(cell["source"] == event["source"], "Different release source")
            require(cell["available_from"] == event["available_from"], "Different availability")
            require(cell["currency_literal"] == "$", "Unrecognized literal currency")
            require("currency_iso" in cell and cell["currency_iso"] in (None, "USD"),
                    "Missing or conflicting explicit currency")
            currencies.append(cell["currency_iso"])
            value, scale = _decimal(cell["value_decimal"]), _decimal(cell["scale"])
            require(scale > 0, "Nonpositive amount scale")
            require(name != "total_assets" or value > 0, "Nonpositive assets")
            require(cell["fact_admitted"] is False, "Raw admission flag changed")
            period, locator = cell["period"], cell["locator"]
            require(type(period) is dict and type(locator) is dict, "Period and physical locator required")
            _locator(locator, scale)
            if name == "net_income":
                require(canonical_sha256(period) == canonical_sha256(event["periods"]["current"]),
                        "Current NI period differs from current annual period")
                require(locator["ni_scope"] == "indirect_method_consolidated_start",
                        "Income scope not consolidated reconciliation start")
                expected = end
            else:
                require(start is not None and end is not None, "Valid current annual dates required")
                expected = end if side == "current" else start - timedelta(days=1)
            require(expected is not None and _day(period["period_end"]) == expected,
                    "Required cell instant or period end mismatch")
            require(_day(locator["period_end"]) == expected, "Locator end mismatch")
            if cell.get("as_of") is not None:
                require(_day(cell["as_of"]) == expected, "Explicit as_of date mismatch")
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            reasons.append("invalid_required:" + side + "." + name + ":" + str(exc))
    if len(set(currencies)) > 1:
        reasons.append("invalid_required:mixed_same_report_currency_iso")

    return dict(schema=SCHEMA, definition=DEFINITION, admitted=not reasons, reasons=reasons,
                event_id=event_id, event_sha256=event_sha, field_review_sha256=review_sha,
                required_cell_sha256=cells, raw_source=raw_source, review_receipts=references,
                evidence_binding_passed=binding, scope_inapplicable_inherited_codes=inapplicable,
                scope_satisfied_accounting_flags=satisfied, current_report_expires=expiry,
                same_report_literal_currency_assumption=True, currency_iso_certified=False,
                source_only=True, strict_pit=False, source_bytes_verified=False,
                review_receipt_bytes_verified=False, first_public_version_verified=False,
                complete_correction_history_verified=False, ratios_computed=False,
                chain_admitted=False, original_rejections_preserved=True,
                existing_admissions_modified=False, inputs_preserved=True)

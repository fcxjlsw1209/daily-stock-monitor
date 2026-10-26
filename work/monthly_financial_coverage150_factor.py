"""Pure, matched-dose earnings coverage extension. No I/O or portfolio work."""
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction

import numpy as np

from monthly_cash_quality146_factor import _fraction, bounded

METHOD = "expanded_annual_earnings_coverage_v1"
DEFINITION = "current_annual_net_income_average_assets_v1"
NEW_SYMBOLS = frozenset(("AMZN", "CSCO"))
OLD_SYMBOLS = frozenset(("AAPL", "AMD", "MSFT", "NVDA"))
FIELDS = ("annual_net_income", "current_total_assets", "prior_total_assets")
MAX_AGE_DAYS = 550


def require(ok, message):
    if not ok:
        raise ValueError(message)


def day(value):
    require(type(value) is str, "Explicit date string required")
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, "Canonical date required")
    return parsed


def unavailable(reason, event_id=None):
    return dict(available=False, reason=reason, event_id=event_id,
                earnings_bounded=None)


def evaluate(event, signal):
    """Evaluate only a release selected from the complete annual catalog."""
    result = unavailable("unavailable", event.get("event_id"))

    def fail(reason):
        result["reason"] = reason
        return result

    gate = event.get("ni_assets_admission")
    if not isinstance(gate, dict) or gate.get("definition") != DEFINITION:
        return fail("missing_or_wrong_admission")
    if gate.get("admitted") is not True or gate.get("reasons") != []:
        return fail("latest_event_held")
    if event.get("symbol") not in NEW_SYMBOLS:
        return fail("unsupported_symbol")
    try:
        signal_day, available = day(signal), day(event["available_from"])
        released = day(event.get("release_date", event.get("publication_date")))
        if not released < available <= signal_day:
            return fail("invalid_available_time")
        source = event["source"]
        fields = event["fields"]
        require(set(fields) == set(FIELDS), "Three exact fields required")
        period = fields["annual_net_income"]["period"]
        start, end = day(period["period_start"]), day(period["period_end"])
        duration = (end - start).days + 1
        if not 330 <= duration <= 400 or period["days_inclusive"] != duration:
            return fail("invalid_annual_period")
        if end > released:
            return fail("release_before_period_end")
        identity = event["identity"]
        require(isinstance(identity["issuer_id"], str) and identity["issuer_id"].strip(),
                "Explicit source issuer identity required")
        require(identity["security_id"] == event["symbol"], "Security identity mismatch")
        identity_start = day(identity["identity_valid_from"])
        identity_end = day(identity["identity_valid_through"])
        require(identity_start == available and identity_end == end + timedelta(days=MAX_AGE_DAYS),
                "Source-use identity window mismatch")
        if (signal_day - end).days > MAX_AGE_DAYS:
            return fail("stale_latest_event")
        require(identity_start <= signal_day <= identity_end, "Outside source-use window")
        values = {}
        for name in FIELDS:
            cell = fields[name]
            require(isinstance(cell, dict), "Required value missing")
            require(cell["source"]["path"] == source["path"] and
                    cell["source"]["sha256"] == source["sha256"],
                    "Source mismatch")
            require(cell["available_from"] == event["available_from"],
                    "Availability mismatch")
            require(cell["currency_literal"] == "$", "Common dollar basis required")
            expected_end = start - timedelta(days=1) if name == "prior_total_assets" else end
            require(day(cell["as_of"]) == expected_end, "Endpoint mismatch")
            value, scale = _fraction(cell["value_decimal"]), _fraction(cell["scale"])
            require(scale > 0, "Positive declared scale required")
            values[name] = value * scale
        currencies = {fields[name].get("currency_iso") for name in FIELDS}
        require(len(currencies) == 1, "Inconsistent source currency")
        current, prior = values["current_total_assets"], values["prior_total_assets"]
        require(current > 0 and prior > 0, "Positive asset endpoints required")
        earnings = values["annual_net_income"] / ((current + prior) / 2)
        require(isinstance(earnings, Fraction), "Exact ratio required")
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return fail("invalid_required_source_field")
    result.update(available=True, reason="available",
                  earnings_bounded=bounded(earnings),
                  available_from=event["available_from"],
                  period_end=end.isoformat())
    return result


def latest(catalog, events, symbol, signal):
    """Unknown/unprocessed/held latest releases block older valid releases."""
    known = {}
    signal_day = day(signal)
    for row in catalog:
        if row.get("symbol") != symbol:
            continue
        key = row["event_id"]
        availability = row.get("available_from", row.get("available_session"))
        released = day(row.get("release_date", row.get("publication_date")))
        clock = row.get("clock_review", {})
        if clock.get("verified") is True:
            require(day(availability) > released and
                    clock.get("first_pinned_session") == availability,
                    "Verified catalog clock mismatch")
            is_known = day(availability) <= signal_day
        else:
            # An unverified clock never permits a value, but a prior release
            # may conservatively block fallback to an older valid report.
            is_known = released < signal_day
        if not is_known:
            continue
        if key in known and known[key] != row:
            return unavailable("conflicting_catalog_event", key)
        known[key] = row
    if not known:
        return unavailable("no_known_annual_event")
    last_release = max(row.get("release_date", row.get("publication_date"))
                       for row in known.values())
    selected = [row for row in known.values()
                if row.get("release_date", row.get("publication_date")) == last_release]
    if len(selected) != 1:
        return unavailable("ambiguous_latest_annual_event")
    row = selected[0]
    key = row["event_id"]
    if row.get("clock_review", {}).get("verified") is not True:
        return unavailable("latest_annual_event_clock_unverified", key)
    if key not in events:
        return unavailable("latest_annual_event_unprocessed", key)
    event = events[key]
    last = row.get("available_from", row.get("available_session"))
    require(event["symbol"] == symbol and event["available_from"] == last,
            "Extracted/catalog identity or time mismatch")
    require(event.get("release_date", event.get("publication_date")) ==
            row.get("release_date", row.get("publication_date")),
            "Extracted/catalog release date mismatch")
    return evaluate(event, signal)


def build(meta, original_earnings, dataset):
    """Keep every old cell exact; add only source-admitted AMZN/CSCO cells."""
    require(dataset.get("schema_version") == 1 and dataset.get("method") == METHOD,
            "Explicit expanded coverage dataset required")
    dates, symbols, etfs = meta["dates"], meta["symbols"], set(meta["etfs"])
    require(len(etfs) == 17 and not etfs & (NEW_SYMBOLS | OLD_SYMBOLS),
            "Original ETF population required")
    parsed = [day(value) for value in dates]
    require(parsed == sorted(set(parsed)), "Sorted unique dates required")
    original = np.asarray(original_earnings)
    require(original.shape == (len(dates), 49) and original.dtype == np.float64
            and not np.isinf(original).any(), "Original earnings array required")
    require(len(symbols) == len(dates), "Original symbol axes required")
    raw_events = dataset["events"]
    events = {event["event_id"]: event for event in raw_events}
    require(len(events) == len(raw_events), "Duplicate extracted event")
    require(all(event["symbol"] in NEW_SYMBOLS for event in raw_events),
            "Unexpected new issuer")
    catalog = dataset["latest_event_catalog"]
    require(isinstance(catalog, list) and
            all(row.get("symbol") in NEW_SYMBOLS for row in catalog),
            "Complete declared issuer catalog required")
    before = original.tobytes()
    expanded, details = original.copy(), []
    for i, (signal, names) in enumerate(zip(dates, symbols)):
        require(len(names) == len(set(names)) == 49 and etfs <= set(names),
                "Original 32+17 axes required")
        for j, symbol in enumerate(names):
            require(not np.isfinite(original[i, j]) or symbol in OLD_SYMBOLS,
                    "Original coverage has unexpected issuer")
            if symbol in NEW_SYMBOLS:
                detail = latest(catalog, events, symbol, signal)
                if detail["available"]:
                    expanded[i, j] = detail["earnings_bounded"]
            else:
                detail = unavailable("etf_not_applicable" if symbol in etfs else
                                     "preserved_original_coverage" if symbol in OLD_SYMBOLS else
                                     "no_new_source_coverage")
            details.append(dict(signal_index=i, pool_index=j, signal_date=signal,
                                symbol=symbol, **detail))
    require(original.tobytes() == before, "Original input changed")
    return dict(
        arrays=dict(earnings_original=original.copy(), earnings_expanded=expanded,
                    original_available=np.isfinite(original),
                    expanded_available=np.isfinite(expanded)),
        audit=dict(method=METHOD, original_cells=int(np.isfinite(original).sum()),
                   expanded_cells=int(np.isfinite(expanded).sum()),
                   added_cells=int(np.isfinite(expanded).sum()-np.isfinite(original).sum()),
                   original_input_unchanged=True,
                   new_model_fits=0, returns_computed=False,
                   reasons=dict(Counter(row["reason"] for row in details)),
                   cells=details))

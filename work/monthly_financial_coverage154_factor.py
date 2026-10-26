"""Pure, matched-dose earnings coverage extension. No I/O or portfolio work."""
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction

import numpy as np

from monthly_cash_quality146_factor import _fraction, bounded

METHOD = "expanded_annual_earnings_coverage_v1"
DEFINITION = "current_annual_net_income_average_assets_v1"
NEW_SYMBOLS = frozenset(("PLTR", "PYPL", "TMO"))
BATCH4_SYMBOLS = frozenset(("MU", "PEP"))
BATCH3_SYMBOLS = frozenset(("KO", "LRCX"))
BATCH2_SYMBOLS = frozenset(("INTC", "KLAC"))
BATCH1_SYMBOLS = frozenset(("AMZN", "CSCO"))
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
        if (duration not in (365, 366) or period["days_inclusive"] != duration
                or start != date(end.year, 1, 1) or end != date(end.year, 12, 31)):
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


def build(meta, original_earnings, batch1_earnings, batch2_earnings, batch3_earnings, batch4_earnings, dataset):
    """Retain frozen149/150/151/152/153 controls; add only PLTR/PYPL/TMO.

    The caller authenticates the source package and full annual catalog. Old
    issuer events are never re-evaluated: the five frozen controls own them.
    New periods must be complete calendar years, without annualization.
    """
    require(dataset.get("schema_version") == 1 and dataset.get("method") == METHOD,
            "Explicit cumulative source dataset required")
    dates, symbols, etfs = meta["dates"], meta["symbols"], set(meta["etfs"])
    all_issuers = OLD_SYMBOLS | BATCH1_SYMBOLS | BATCH2_SYMBOLS | BATCH3_SYMBOLS | BATCH4_SYMBOLS | NEW_SYMBOLS
    require(len(etfs) == 17 and not etfs & all_issuers, "Original ETF population required")
    parsed = [day(value) for value in dates]
    require(parsed == sorted(set(parsed)) and len(symbols) == len(dates), "Sorted unique axes required")
    original, batch1, batch2, batch3, batch4 = map(np.asarray,
        (original_earnings, batch1_earnings, batch2_earnings, batch3_earnings, batch4_earnings))
    for array in (original, batch1, batch2, batch3, batch4):
        require(array.shape == (len(dates), 49) and array.dtype == np.float64 and not np.isinf(array).any(),
                "Frozen float64 earnings array required")
    events = {event["event_id"]: event for event in dataset["events"]}
    catalog = dataset["latest_event_catalog"]
    require(len(events) == len(dataset["events"]) and len({row["event_id"] for row in catalog}) == len(catalog),
            "Duplicate source event")
    source_issuers = BATCH1_SYMBOLS | BATCH2_SYMBOLS | BATCH3_SYMBOLS | BATCH4_SYMBOLS | NEW_SYMBOLS
    require(all(event["symbol"] in source_issuers for event in events.values()) and
            all(row["symbol"] in source_issuers for row in catalog), "Unexpected cumulative source issuer")
    before = tuple(array.tobytes() for array in (original, batch1, batch2, batch3, batch4))
    expanded, details = batch4.copy(), []
    for i, (signal, names) in enumerate(zip(dates, symbols)):
        require(len(names) == len(set(names)) == 49 and etfs <= set(names), "Original32+17 axes required")
        for j, symbol in enumerate(names):
            require(not np.isfinite(original[i,j]) or symbol in OLD_SYMBOLS, "Original149 scope changed")
            require(not np.isfinite(batch1[i,j]) or symbol in OLD_SYMBOLS | BATCH1_SYMBOLS, "Prior150 scope changed")
            require(not np.isfinite(batch2[i,j]) or symbol in OLD_SYMBOLS | BATCH1_SYMBOLS | BATCH2_SYMBOLS,
                    "Prior151 scope changed")
            require(not np.isfinite(batch3[i,j]) or symbol in OLD_SYMBOLS | BATCH1_SYMBOLS | BATCH2_SYMBOLS | BATCH3_SYMBOLS,
                    "Prior152 scope changed")
            require(not np.isfinite(batch4[i,j]) or symbol in OLD_SYMBOLS | BATCH1_SYMBOLS | BATCH2_SYMBOLS | BATCH3_SYMBOLS | BATCH4_SYMBOLS,
                    "Prior153 scope changed")
            if symbol not in BATCH1_SYMBOLS:
                require(original[i:i+1,j:j+1].tobytes() == batch1[i:i+1,j:j+1].tobytes(),
                        "Prior150 modified a nonbatch1 cell")
            if symbol not in BATCH2_SYMBOLS:
                require(batch1[i:i+1,j:j+1].tobytes() == batch2[i:i+1,j:j+1].tobytes(),
                        "Prior151 modified a nonbatch2 cell")
            if symbol not in BATCH3_SYMBOLS:
                require(batch2[i:i+1,j:j+1].tobytes() == batch3[i:i+1,j:j+1].tobytes(),
                        "Prior152 modified a nonbatch3 cell")
            if symbol not in BATCH4_SYMBOLS:
                require(batch3[i:i+1,j:j+1].tobytes() == batch4[i:i+1,j:j+1].tobytes(),
                        "Prior153 modified a nonbatch4 cell")
            if symbol in NEW_SYMBOLS:
                detail = latest(catalog, events, symbol, signal)
                if detail["available"]:
                    expanded[i,j] = detail["earnings_bounded"]
            else:
                detail = unavailable("etf_not_applicable" if symbol in etfs else
                    "preserved_original_coverage" if symbol in OLD_SYMBOLS else
                    "preserved_batch1_coverage" if symbol in BATCH1_SYMBOLS else
                    "preserved_batch2_coverage" if symbol in BATCH2_SYMBOLS else
                    "preserved_batch3_coverage" if symbol in BATCH3_SYMBOLS else
                    "preserved_batch4_coverage" if symbol in BATCH4_SYMBOLS else "no_new_source_coverage")
            details.append(dict(signal_index=i,pool_index=j,signal_date=signal,symbol=symbol,**detail))
    require(before == tuple(array.tobytes() for array in (original, batch1, batch2, batch3, batch4)),
            "Frozen control inputs changed")
    return dict(arrays=dict(earnings_original=original.copy(),earnings_batch1=batch1.copy(),
            earnings_batch2=batch2.copy(),earnings_batch3=batch3.copy(),earnings_batch4=batch4.copy(),earnings_expanded=expanded,
            original_available=np.isfinite(original),batch1_available=np.isfinite(batch1),
            batch2_available=np.isfinite(batch2),batch3_available=np.isfinite(batch3),batch4_available=np.isfinite(batch4),expanded_available=np.isfinite(expanded)),
        audit=dict(method=METHOD,original_cells=int(np.isfinite(original).sum()),
            batch1_cells=int(np.isfinite(batch1).sum()),batch2_cells=int(np.isfinite(batch2).sum()),
            batch3_cells=int(np.isfinite(batch3).sum()),batch4_cells=int(np.isfinite(batch4).sum()),expanded_cells=int(np.isfinite(expanded).sum()),
            added_cells=int(np.isfinite(expanded).sum()-np.isfinite(batch4).sum()),
            original_input_unchanged=True,batch1_input_unchanged=True,batch2_input_unchanged=True,
            batch3_input_unchanged=True,batch4_input_unchanged=True,new_model_fits=0,returns_computed=False,
            reasons=dict(Counter(row["reason"] for row in details)),cells=details))
